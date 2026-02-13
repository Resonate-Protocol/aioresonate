"""PlayerV1Role implementation for audio playback (v1).

This PlayerV1Role implementation uses hook-based streaming:
- on_stream_start(): Send stream/start message
- on_audio_chunk(): Pack and send binary audio
- on_stream_clear(): Send stream/clear message
- on_stream_end(): Send stream/end message
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiosendspin.models import AudioCodec, BinaryMessageType, pack_binary_header_raw
from aiosendspin.models.core import (
    ClientStatePayload,
    ServerCommandMessage,
    ServerCommandPayload,
    StreamClearMessage,
    StreamClearPayload,
    StreamEndMessage,
    StreamEndPayload,
    StreamRequestFormatPayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.player import PlayerCommandPayload, StreamStartPlayer, SupportedAudioFormat
from aiosendspin.models.types import PlayerCommand
from aiosendspin.server.audio import AudioFormat, BufferTracker
from aiosendspin.server.roles.base import (
    AudioChunk,
    AudioRequirements,
    BinaryHandling,
    Role,
    StreamRequirements,
)
from aiosendspin.server.roles.player.audio_transformers import (
    FlacEncoder,
    OpusEncoder,
    PcmPassthrough,
)
from aiosendspin.server.roles.player.capabilities import can_encode_format, filter_encodable_formats
from aiosendspin.server.roles.player.events import VolumeChangedEvent
from aiosendspin.util import create_task

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient


@dataclass
class PlayerPersistentState:
    """Persistent player state stored on the SendspinClient."""

    volume: int = 100
    muted: bool = False
    preferred_format: AudioFormat | None = None
    preferred_codec: AudioCodec | None = None
    buffer_tracker: BufferTracker | None = None
    buffer_capacity_scale: float = 1.0
    max_duration_us: int = 30_000_000
    disconnect_time_us: int | None = None
    buffer_reset_handle: asyncio.TimerHandle | None = None


class PlayerV1Role(Role):
    """Role implementation for audio playback.

    Hook-based streaming:
    - on_stream_start(): Send stream/start message
    - on_audio_chunk(): Pack and send binary audio
    - on_stream_clear(): Send stream/clear message
    - on_stream_end(): Send stream/end message
    """

    def __init__(
        self,
        client: SendspinClient | None = None,
        *,
        preferred_format: AudioFormat | None = None,
        audio_requirements: AudioRequirements | None = None,
    ) -> None:
        """Initialize PlayerV1Role.

        Args:
            client: The owning SendspinClient.
            preferred_format: Preferred audio format for this player.
            audio_requirements: Audio requirements for hook-based streaming.
        """
        if client is None:
            msg = "PlayerV1Role requires a client"
            raise ValueError(msg)
        self._client = client
        self._preferred_format_override = preferred_format
        self._audio_requirements = audio_requirements
        self._stream_started = False
        self._buffer_tracker = None
        # Initialize timing state for binary handling
        self._stream_start_time_us = None
        self._last_late_log_s = 0.0
        self._late_skips_since_log = 0
        # Cached state reference (avoids repeated dict lookup + isinstance check)
        self._cached_state: PlayerPersistentState | None = None
        # Deferred stream start: set True by on_stream_start(), sent on first audio chunk
        self._pending_stream_start = False

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "player@v1"

    @property
    def role_family(self) -> str:
        """Role family name for protocol messages."""
        return "player"

    @property
    def preferred_format(self) -> AudioFormat | None:
        """Return the preferred audio format for this player."""
        state = self._state()
        return state.preferred_format or self._preferred_format_override

    @preferred_format.setter
    def preferred_format(self, value: AudioFormat | None) -> None:
        self._state().preferred_format = value

    @property
    def preferred_codec(self) -> AudioCodec | None:
        """Return the preferred audio codec for this player."""
        return self._state().preferred_codec

    @preferred_codec.setter
    def preferred_codec(self, value: AudioCodec | None) -> None:
        self._state().preferred_codec = value

    # --- Declarations ---

    def get_stream_requirements(self) -> StreamRequirements:
        """Player role sends binary audio streams."""
        return StreamRequirements()

    def get_audio_requirements(self) -> AudioRequirements | None:
        """Return audio requirements for hook-based streaming."""
        req = self._audio_requirements
        if req is None:
            return None

        # Legacy/manually-injected requirements may omit channel assignment.
        # In that case, preserve existing behavior.
        if req.channel_id is None:
            return req

        # Channel routing may change at stream start when a custom channel_resolver
        # is installed. Refresh cached requirements if the resolved channel changed.
        channel_id = self._client.group.get_channel_for_player(self._client.client_id)

        if channel_id != req.channel_id:
            self._ensure_audio_requirements(self._state(), force=True)

        return self._audio_requirements

    def get_binary_handling(self, message_type: int) -> BinaryHandling | None:
        """Return handling policy for AUDIO_CHUNK messages."""
        if message_type == BinaryMessageType.AUDIO_CHUNK.value:
            return BinaryHandling(
                drop_late=True,
                grace_period_us=2_000_000,  # 2 seconds grace for initial buffering
                buffer_track=True,
            )
        return None

    def get_buffer_tracker(self) -> BufferTracker | None:
        """Return the role-owned buffer tracker."""
        return self._state().buffer_tracker

    def get_join_delay_s(self) -> float:
        """Delay joins briefly to allow time sync to stabilize."""
        return 1.0

    # --- Lifecycle hooks ---

    def on_connect(self) -> None:
        """Reset stream state and subscribe to PlayerGroupRole."""
        self._subscribe_to_group_role()
        self._stream_started = False
        state = self._state()
        if state.buffer_reset_handle is not None:
            state.buffer_reset_handle.cancel()
            state.buffer_reset_handle = None
        state.disconnect_time_us = None
        self._ensure_buffer_tracker(state)
        # Reset buffer tracker on (re)connect - client buffer is empty after reconnect
        if state.buffer_tracker is not None:
            state.buffer_tracker.reset()
        self._ensure_preferred_format(state)
        self._ensure_audio_requirements(state)

    def on_disconnect(self) -> None:
        """Clean up, apply delayed buffer reset policy, and unsubscribe from PlayerGroupRole."""
        self._unsubscribe_from_group_role()
        self._stream_started = False

        state = self._state()
        state.disconnect_time_us = self._client._server.clock.now_us()  # noqa: SLF001
        if state.buffer_tracker is None:
            return

        disconnect_time_us = state.disconnect_time_us

        def _maybe_reset() -> None:
            state.buffer_reset_handle = None
            if self._client.connection is not None:
                return
            if disconnect_time_us != state.disconnect_time_us:
                return
            if state.buffer_tracker is None:
                return
            state.buffer_tracker.reset()

        reset_after_s = 2.0
        if state.buffer_reset_handle is not None:
            state.buffer_reset_handle.cancel()
        state.buffer_reset_handle = self._client._server.loop.call_later(  # noqa: SLF001
            reset_after_s, _maybe_reset
        )

    def requires_initial_state(self) -> bool:
        """Player role requires initial state with volume/mute info."""
        return True

    def on_group_changed(self, group: object) -> None:
        """Refresh transformer selection when group changes."""
        super().on_group_changed(group)
        state = self._state()
        self._ensure_buffer_tracker(state)
        if state.buffer_tracker is not None:
            # Group switches imply a stream boundary for this player; any previously
            # tracked buffered audio belongs to the old group timeline.
            state.buffer_tracker.reset()
        self._stream_started = False
        self._pending_stream_start = False
        self.reset_binary_timing()
        self._ensure_audio_requirements(state, force=True)

    # --- Stream lifecycle hooks ---

    def on_stream_start(self) -> None:
        """Mark stream start as pending - actual message sent on first audio chunk.

        This defers the stream/start message until the first audio chunk arrives,
        ensuring the codec header is available (FLAC generates header on first encode).
        """
        req = self.get_audio_requirements()
        if req is None:
            self._ensure_audio_requirements(self._state())
            req = self.get_audio_requirements()
        if req is None:
            return

        if not self.has_connection():
            return

        # New stream boundary: clear prior stream timing/log state so
        # late-drop grace period is measured from this stream's first chunk.
        self.reset_binary_timing()
        self._pending_stream_start = True

    def _send_stream_start_message(self) -> None:
        """Send stream/start message with codec header from transformer."""
        req = self.get_audio_requirements()
        if req is None or not self.has_connection():
            return

        transformer = req.transformer
        header = transformer.get_header() if isinstance(transformer, FlacEncoder) else None
        header_b64 = base64.b64encode(header).decode() if header else None

        # Determine codec from transformer type
        if isinstance(transformer, FlacEncoder):
            codec = AudioCodec.FLAC
        elif isinstance(transformer, OpusEncoder):
            codec = AudioCodec.OPUS
        else:
            codec = AudioCodec.PCM

        stream_start = StreamStartMessage(
            payload=StreamStartPayload(
                player=StreamStartPlayer(
                    codec=codec,
                    sample_rate=req.sample_rate,
                    channels=req.channels,
                    bit_depth=req.bit_depth,
                    codec_header=header_b64,
                )
            )
        )
        self.send_message(stream_start)
        is_initial = not self._stream_started
        self._stream_started = True

        # Allow client to process stream/start before first binary audio (initial only).
        if is_initial and self._buffer_tracker is not None:
            self._buffer_tracker.set_send_blocked(200_000)

    def on_audio_chunk(self, chunk: AudioChunk) -> None:
        """Pack and send binary audio. Late audio is discarded by connection."""
        # Send deferred stream/start on first chunk (ensures encoder header is available)
        if self._pending_stream_start:
            self._send_stream_start_message()
            self._pending_stream_start = False

        # Pack binary header and send
        message_type = BinaryMessageType.AUDIO_CHUNK.value
        header = pack_binary_header_raw(message_type, chunk.timestamp_us)
        packed_data = header + chunk.data
        chunk_end_us = chunk.timestamp_us + chunk.duration_us

        self._client.send_binary(
            packed_data,
            role_family=self.role_family,
            timestamp_us=chunk.timestamp_us,
            message_type=message_type,
            buffer_end_time_us=chunk_end_us,
            buffer_byte_count=chunk.byte_count,
            duration_us=chunk.duration_us,
        )

    def on_stream_clear(self) -> None:
        """Send stream/clear and reset state."""
        if not self.has_connection():
            return

        stream_clear = StreamClearMessage(payload=StreamClearPayload(roles=["player"]))
        self.send_message(stream_clear)
        self._stream_started = False
        self._pending_stream_start = False
        self.reset_binary_timing()

        if self._buffer_tracker is not None:
            self._buffer_tracker.reset()

    def on_stream_end(self) -> None:
        """Send stream/end and reset state."""
        if not self.has_connection():
            return

        # End all streams (roles omitted) for best client compatibility.
        stream_end = StreamEndMessage(payload=StreamEndPayload(roles=None))
        self.send_message(stream_end)
        self._stream_started = False
        self._pending_stream_start = False
        self.reset_binary_timing()

        if self._buffer_tracker is not None:
            self._buffer_tracker.reset()

    @property
    def stream_started(self) -> bool:
        """Whether stream/start has been sent."""
        return self._stream_started

    # ---- Volume/mute state and commands ----

    @property
    def volume(self) -> int:
        """Current volume of this player (0-100)."""
        return self._state().volume

    @volume.setter
    def volume(self, value: int) -> None:
        self._state().volume = value

    @property
    def muted(self) -> bool:
        """Current mute state of this player."""
        return self._state().muted

    @muted.setter
    def muted(self, value: bool) -> None:
        self._state().muted = value

    def get_player_volume(self) -> int | None:
        """Return current volume for group aggregation."""
        return self.volume

    def get_player_muted(self) -> bool | None:
        """Return current mute state for group aggregation."""
        return self.muted

    def set_player_volume(self, volume: int) -> None:
        """Set player volume via role API."""
        self.set_volume(volume)

    def set_player_mute(self, muted: bool) -> None:  # noqa: FBT001
        """Set player mute via role API."""
        self.set_mute(muted)

    def get_supported_formats(self) -> list[SupportedAudioFormat] | None:
        """Return formats both client and server support, in client priority order."""
        support = self._client.info.player_support
        if support is None:
            return None
        return filter_encodable_formats(support.supported_formats)

    def set_preferred_format(self, audio_format: AudioFormat, codec: AudioCodec) -> bool:
        """Set preferred format if compatible with both client and server.

        Args:
            audio_format: The audio format to set (sample_rate, bit_depth, channels).
            codec: The codec to use.

        Returns:
            True if the format was set, False if incompatible.
        """
        support = self._client.info.player_support
        if support is None:
            return False

        # Check if format is in client's supported list
        client_format = SupportedAudioFormat(
            codec=codec,
            sample_rate=audio_format.sample_rate,
            bit_depth=audio_format.bit_depth,
            channels=audio_format.channels,
        )
        is_client_supported = any(
            fmt.codec == codec
            and fmt.sample_rate == audio_format.sample_rate
            and fmt.bit_depth == audio_format.bit_depth
            and fmt.channels == audio_format.channels
            for fmt in support.supported_formats
        )
        if not is_client_supported:
            return False

        # Check if server can encode this format
        if not can_encode_format(client_format):
            return False

        # Set the preferred format
        state = self._state()
        state.preferred_format = audio_format
        state.preferred_codec = codec

        # Rebuild audio requirements with the new format
        self._ensure_audio_requirements(state, force=True)

        return True

    def set_volume(self, volume: int) -> None:
        """Set the volume of this player."""
        support = self._client.info.player_support
        if not support or PlayerCommand.VOLUME not in support.supported_commands:
            return

        self._client.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(
                    player=PlayerCommandPayload(
                        command=PlayerCommand.VOLUME,
                        volume=volume,
                    )
                )
            )
        )

    def set_mute(self, muted: bool) -> None:  # noqa: FBT001
        """Set the mute state of this player."""
        support = self._client.info.player_support
        if not support or PlayerCommand.MUTE not in support.supported_commands:
            return

        self._client.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(
                    player=PlayerCommandPayload(
                        command=PlayerCommand.MUTE,
                        mute=muted,
                    )
                )
            )
        )

    # ---- Client message handling ----

    def on_client_state(self, payload: ClientStatePayload) -> None:
        """Handle player-specific fields in client/state."""
        state = payload.player
        if state is None:
            return

        # DEPRECATED(before-spec-pr-50): fall back to player.state for older clients.
        if payload.state is None and state.state is not None:
            create_task(self._client.handle_state_transition(state.state))

        support = self._client.info.player_support
        changed = False

        if state.volume is not None:
            if not support or PlayerCommand.VOLUME not in support.supported_commands:
                self._client._logger.warning(  # noqa: SLF001
                    "Client sent volume field without declaring 'volume' in supported_commands"
                )
            elif self.volume != state.volume:
                self.volume = state.volume
                changed = True

        if state.muted is not None:
            if not support or PlayerCommand.MUTE not in support.supported_commands:
                self._client._logger.warning(  # noqa: SLF001
                    "Client sent muted field without declaring 'mute' in supported_commands"
                )
            elif self.muted != state.muted:
                self.muted = state.muted
                changed = True

        if changed:
            self._client._signal_event(  # noqa: SLF001
                VolumeChangedEvent(volume=self.volume, muted=self.muted)
            )

    def on_stream_request_format(self, payload: StreamRequestFormatPayload) -> None:
        """Handle stream/request-format for player role."""
        player_req = payload.player
        if player_req is None:
            return

        support = self._client.info.player_support
        if support is None:
            raise ValueError(
                f"Client {self._client.client_id} sent player format request "
                "but has no player support"
            )

        supported = filter_encodable_formats(support.supported_formats)
        if not supported:
            self._client._logger.warning(  # noqa: SLF001
                "Client %s requested format change but has no server-compatible formats",
                self._client.client_id,
            )
            return

        preferred_supported = supported[0]
        base_format = self.preferred_format or AudioFormat(
            sample_rate=preferred_supported.sample_rate,
            bit_depth=preferred_supported.bit_depth,
            channels=preferred_supported.channels,
        )
        base_codec = self.preferred_codec or preferred_supported.codec

        requested_codec = player_req.codec or base_codec
        requested_format = AudioFormat(
            sample_rate=player_req.sample_rate or base_format.sample_rate,
            bit_depth=player_req.bit_depth or base_format.bit_depth,
            channels=player_req.channels or base_format.channels,
        )

        if not any(
            fmt.codec == requested_codec
            and fmt.sample_rate == requested_format.sample_rate
            and fmt.bit_depth == requested_format.bit_depth
            and fmt.channels == requested_format.channels
            for fmt in supported
        ):
            self._client._logger.warning(  # noqa: SLF001
                "Client %s requested unsupported format %s codec=%s, falling back to %s",
                self._client.client_id,
                requested_format,
                requested_codec,
                base_format,
            )
            requested_format = base_format
            requested_codec = base_codec

        self.preferred_format = requested_format
        self.preferred_codec = requested_codec

        state = self._state()

        stream_active = self._client.group.has_active_stream
        if stream_active:
            # Mid-stream format change: rebuild requirements and defer stream/start
            # until the next audio chunk (which provides the codec header).
            self._ensure_audio_requirements(state, force=True)
            self._pending_stream_start = True
            self._client.group.on_role_format_changed(self)
        else:
            # No active stream: also defer stream/start via _pending_stream_start
            # so codec header is included when the first chunk arrives.
            self._ensure_audio_requirements(state, force=True)
            self._pending_stream_start = True

    # ---- Internal helpers ----

    def _state(self) -> PlayerPersistentState:
        if self._cached_state is None:
            self._cached_state = self._client.get_or_create_role_state(
                "player", PlayerPersistentState
            )
        return self._cached_state

    def _ensure_buffer_tracker(self, state: PlayerPersistentState) -> None:
        support = self._client.info.player_support
        if support is None:
            self._buffer_tracker = None
            return

        capacity = int(support.buffer_capacity * state.buffer_capacity_scale)
        capacity = max(1, capacity)
        max_duration_us = state.max_duration_us

        if state.buffer_tracker is None:
            state.buffer_tracker = BufferTracker(
                clock=self._client._server.clock,  # noqa: SLF001
                client_id=self._client.client_id,
                capacity_bytes=capacity,
                max_duration_us=max_duration_us,
            )
        else:
            state.buffer_tracker.capacity_bytes = capacity
            state.buffer_tracker.max_duration_us = max_duration_us
        self._buffer_tracker = state.buffer_tracker

    def _ensure_preferred_format(self, state: PlayerPersistentState) -> None:
        support = self._client.info.player_support
        if support is None:
            return

        if state.preferred_format is None and self._preferred_format_override is not None:
            state.preferred_format = self._preferred_format_override

        # Filter to formats the server can actually encode
        compatible = filter_encodable_formats(support.supported_formats)
        if not compatible:
            self._client._logger.warning(  # noqa: SLF001
                "Client %s has no server-compatible formats",
                self._client.client_id,
            )
            return

        # Use client's first compatible format (client priority order)
        preferred_supported = compatible[0]
        default_format = AudioFormat(
            sample_rate=preferred_supported.sample_rate,
            bit_depth=preferred_supported.bit_depth,
            channels=preferred_supported.channels,
        )
        default_codec = preferred_supported.codec

        current_codec = state.preferred_codec or default_codec
        current_format = state.preferred_format
        is_supported = current_format is not None and any(
            fmt.codec == current_codec
            and fmt.sample_rate == current_format.sample_rate
            and fmt.bit_depth == current_format.bit_depth
            and fmt.channels == current_format.channels
            for fmt in compatible
        )
        if not is_supported:
            state.preferred_format = default_format
            state.preferred_codec = default_codec

    def _ensure_audio_requirements(
        self, state: PlayerPersistentState, *, force: bool = False
    ) -> None:
        if self._audio_requirements is not None and not force:
            return

        support = self._client.info.player_support
        if support is None:
            self._audio_requirements = None
            return

        audio_format = state.preferred_format
        audio_codec = state.preferred_codec
        if audio_format is None or audio_codec is None:
            self._audio_requirements = None
            return

        group = self._client.group
        frame_duration_us = 25_000
        channel_id = group.get_channel_for_player(self._client.client_id)
        channel_id_int = channel_id.int
        transformer: FlacEncoder | OpusEncoder | PcmPassthrough
        if audio_codec == AudioCodec.FLAC:
            transformer = group.transformer_pool.get_or_create(
                FlacEncoder,
                channel_id=channel_id_int,
                sample_rate=audio_format.sample_rate,
                bit_depth=audio_format.bit_depth,
                channels=audio_format.channels,
                frame_duration_us=frame_duration_us,
            )
        elif audio_codec == AudioCodec.OPUS:
            transformer = group.transformer_pool.get_or_create(
                OpusEncoder,
                channel_id=channel_id_int,
                sample_rate=audio_format.sample_rate,
                bit_depth=audio_format.bit_depth,
                channels=audio_format.channels,
                frame_duration_us=frame_duration_us,
            )
        else:
            transformer = group.transformer_pool.get_or_create(
                PcmPassthrough,
                channel_id=channel_id_int,
                sample_rate=audio_format.sample_rate,
                bit_depth=audio_format.bit_depth,
                channels=audio_format.channels,
                frame_duration_us=frame_duration_us,
            )

        self._audio_requirements = AudioRequirements(
            sample_rate=audio_format.sample_rate,
            bit_depth=audio_format.bit_depth,
            channels=audio_format.channels,
            transformer=transformer,
            channel_id=channel_id,
            frame_duration_us=frame_duration_us,
        )
