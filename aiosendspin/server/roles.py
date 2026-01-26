"""Role implementations for connection-specific behavior.

Roles encapsulate per-connection behavior for different client capabilities.
The PlayerRole handles audio streaming lifecycle, binary message packing,
and stream state management.
"""

from __future__ import annotations

import base64
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from aiosendspin.models import AudioCodec, BinaryMessageType, pack_binary_header_raw
from aiosendspin.models.core import (
    StreamClearMessage,
    StreamClearPayload,
    StreamEndMessage,
    StreamEndPayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.player import StreamStartPlayer
from aiosendspin.server.transformers import FlacEncoder

if TYPE_CHECKING:
    from aiosendspin.models.types import ServerMessage
    from aiosendspin.server.audio import AudioFormat, BufferTracker
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.pipeline import EncodedChunk
    from aiosendspin.server.transformers import AudioTransformer


@dataclass(frozen=True)
class StreamRequirements:
    """Declaration that a role sends binary streams.

    Roles that return this from get_stream_requirements() will have a
    BufferTracker injected by the framework.
    """


@dataclass(frozen=True)
class AudioChunk:
    """Audio chunk delivered to roles."""

    data: bytes
    """Transformed audio bytes (PCM or encoded, depending on transformer)."""

    timestamp_us: int
    """Playback timestamp in microseconds."""

    duration_us: int
    """Duration of this chunk in microseconds."""

    byte_count: int
    """Size of data (for buffer tracking)."""


@dataclass(frozen=True)
class AudioRequirements:
    """Declaration that a role needs audio chunks.

    Roles that return this from get_audio_requirements() will receive
    audio via on_audio_chunk() calls.
    """

    sample_rate: int
    """Target sample rate in Hz."""

    bit_depth: int
    """Target bit depth (8, 16, 24, 32)."""

    channels: int
    """Number of audio channels."""

    transformer: AudioTransformer | None = None
    """Optional transformer for encoding. None means raw PCM."""

    channel_id: UUID | None = None
    """Channel to receive audio from. None means main channel."""


class Role(ABC):
    """Base class for all roles.

    Roles encapsulate per-connection behavior for different client capabilities.
    Each role can declare its streaming requirements and receive framework-injected
    resources like BufferTracker.
    """

    _client: SendspinClient
    """Reference to the owning client."""

    _buffer_tracker: BufferTracker | None = None
    """Framework-injected buffer tracker for roles that stream binary data."""

    _stream_started: bool = False
    """Whether stream/start has been sent for this role."""

    _has_transport: bool = False
    """Whether this role has an active WebSocket transport."""

    @property
    @abstractmethod
    def role_family(self) -> str:
        """Role family name for protocol messages (e.g., 'player', 'artwork')."""
        ...

    # --- Declarations ---

    def get_stream_requirements(self) -> StreamRequirements | None:
        """Return StreamRequirements if role sends binary streams, else None.

        Roles that return StreamRequirements will have a BufferTracker injected
        by the framework.
        """
        return None

    def get_audio_requirements(self) -> AudioRequirements | None:
        """Return AudioRequirements if role needs audio, else None.

        Roles that return AudioRequirements will receive audio chunks via
        on_audio_chunk() calls from PushStream.
        """
        return None

    # --- Framework-provided send methods ---

    def send_message(self, message: ServerMessage) -> None:
        """Send JSON message to the client. Drop silently if no transport."""
        if not self._has_transport:
            return
        self._client.send_message(message)

    # --- Stream lifecycle hooks (optional) ---

    def on_stream_start(self) -> None:  # noqa: B027
        """Handle stream start before first audio chunk."""

    def on_audio_chunk(self, chunk: AudioChunk) -> bool:  # noqa: ARG002
        """Receive audio chunk. Return True if accepted, False for backpressure."""
        return True

    def on_stream_clear(self) -> None:  # noqa: B027
        """Handle seek/clear by discarding buffered audio."""

    def on_stream_end(self) -> None:  # noqa: B027
        """Handle stream stop."""

    # --- Lifecycle hooks ---

    def on_transport_attach(self) -> None:
        """Handle WebSocket connect/reconnect."""
        self._has_transport = True

    def on_transport_detach(self) -> None:
        """Handle WebSocket disconnect."""
        self._has_transport = False

    @abstractmethod
    def on_connect(self) -> None:
        """Handle connection establishment."""

    @abstractmethod
    def on_disconnect(self) -> None:
        """Handle connection close."""


@dataclass
class PlayerSendState:
    """Current state of a player's audio delivery."""

    healthy: bool = True
    """Whether the player is receiving audio normally."""
    needs_resync: bool = False
    """Whether the player needs to be resynced (dropped and needs catch-up)."""
    dropped_commits: int = 0
    """Number of commits dropped for this player."""
    last_sent_timestamp_us: int | None = None
    """Timestamp of the last audio chunk sent to this player."""


@dataclass
class PlayerRole(Role):
    """
    Role implementation for audio playback.

    Handles:
    - Stream lifecycle (stream/start, stream/clear, stream/end)
    - Binary audio message packing
    - Format tracking
    - Send state management
    """

    _client: SendspinClient
    """Persistent client/device state."""

    role_family: str = field(default="player", init=False)
    """Role family name for protocol messages."""

    _stream_started: bool = field(default=False, init=False)
    """Whether stream/start has been sent for the current format."""
    _current_format: AudioFormat | None = field(default=None, init=False)
    """The current audio format being streamed."""
    _codec_header_b64: str | None = field(default=None, init=False)
    """Base64-encoded codec header (e.g., for FLAC)."""
    _send_state: PlayerSendState = field(default_factory=PlayerSendState, init=False)
    """Current send state for this player."""
    _last_drop_log_s: float = field(default=0.0, init=False)
    _drops_since_log: int = field(default=0, init=False)
    """Rate-limited drop logging for diagnosing stutter."""

    _audio_requirements: AudioRequirements | None = field(default=None, init=False)
    """Audio requirements for the new hook-based streaming."""

    def get_stream_requirements(self) -> StreamRequirements:
        """Player role sends binary audio streams."""
        return StreamRequirements()

    def get_audio_requirements(self) -> AudioRequirements | None:
        """Return audio requirements for hook-based streaming."""
        return self._audio_requirements

    def on_connect(self) -> None:
        """Reset stream state on new connection."""
        self._stream_started = False
        self._current_format = None
        self._codec_header_b64 = None
        self._send_state = PlayerSendState()

    def on_disconnect(self) -> None:
        """Clean up on disconnect.

        Note: BufferTracker reset semantics are handled at the persistent client layer
        (goodbye immediate, otherwise duration-based). Role-level clear/end
        always resets, but plain disconnect does not unconditionally reset.
        """
        self._stream_started = False
        self._current_format = None

    @property
    def stream_started(self) -> bool:
        """Whether stream/start has been sent."""
        return self._stream_started

    @property
    def current_format(self) -> AudioFormat | None:
        """The current audio format, or None if no stream active."""
        return self._current_format

    def get_send_state(self) -> PlayerSendState:
        """Return current send state (healthy, needs_resync, dropped_count)."""
        return self._send_state

    def send_stream_start(
        self,
        audio_format: AudioFormat,
        codec: AudioCodec,
        codec_header_b64: str | None = None,
    ) -> None:
        """
        Send stream/start message to the player.

        Args:
            audio_format: Audio format for this stream.
            codec: Audio codec for encoding.
            codec_header_b64: Optional base64-encoded codec header.
        """
        stream_start = StreamStartMessage(
            payload=StreamStartPayload(
                player=StreamStartPlayer(
                    codec=codec,
                    sample_rate=audio_format.sample_rate,
                    channels=audio_format.channels,
                    bit_depth=audio_format.bit_depth,
                    codec_header=codec_header_b64,
                )
            )
        )
        self.send_message(stream_start)
        self._stream_started = True
        self._current_format = audio_format
        self._codec_header_b64 = codec_header_b64

    def send_audio(
        self,
        chunk: EncodedChunk,
        timestamp_us: int,
        audio_format: AudioFormat,
        codec: AudioCodec,
        codec_header_b64: str | None = None,
    ) -> bool:
        """
        Send audio chunk with timestamp.

        Automatically sends stream/start if needed (format change or first chunk).
        For non-blocking players, checks queue high water and drops if needed.

        Args:
            chunk: Encoded audio chunk to send.
            timestamp_us: Playback timestamp in microseconds.
            audio_format: Audio format for this chunk.
            codec: Audio codec for encoding.
            codec_header_b64: Optional base64-encoded codec header.

        Returns:
            True if sent successfully, False if dropped (queue high water).
        """
        # Avoid building large per-connection backlogs: binary audio is droppable for all
        # players. If the outgoing queue is congested, drop and schedule a resync.
        if self._client.queue_high_water(threshold=0.5):
            self._drops_since_log += 1
            now_s = time.monotonic()
            if now_s - self._last_drop_log_s >= 1.0:
                qsize, qmax = self._client.queue_status()
                self._client._logger.warning(  # noqa: SLF001
                    "Dropping audio due to queue high water: drops=%s queue=%s/%s",
                    self._drops_since_log,
                    qsize,
                    qmax,
                )
                self._drops_since_log = 0
                self._last_drop_log_s = now_s
            self.mark_needs_resync()
            return False

        # Check if we need to send stream/start (first chunk or format change)
        if not self._stream_started or self._current_format != audio_format:
            self.send_stream_start(audio_format, codec, codec_header_b64)

        # Pack binary header and send
        header = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, timestamp_us)
        packed_data = header + chunk.data
        chunk_end_us = timestamp_us + chunk.duration_us
        if not self._client.try_send_binary(
            packed_data,
            buffer_end_time_us=chunk_end_us,
            buffer_byte_count=chunk.byte_count,
            duration_us=chunk.duration_us,
        ):
            self._drops_since_log += 1
            now_s = time.monotonic()
            if now_s - self._last_drop_log_s >= 1.0:
                qsize, qmax = self._client.queue_status()
                self._client._logger.warning(  # noqa: SLF001
                    "Dropping audio due to enqueue failure: drops=%s queue=%s/%s",
                    self._drops_since_log,
                    qsize,
                    qmax,
                )
                self._drops_since_log = 0
                self._last_drop_log_s = now_s
            self.mark_needs_resync()
            return False

        # Update send state
        self._send_state.last_sent_timestamp_us = timestamp_us
        self._send_state.healthy = True

        return True

    def send_cached_chunk(
        self,
        payload: bytes,
        timestamp_us: int,
        duration_us: int,
        byte_count: int,
    ) -> bool:
        """
        Send a cached chunk (for late joiner catch-up).

        Args:
            payload: Encoded audio payload bytes (without binary header).
            timestamp_us: Playback timestamp for this chunk.
            duration_us: Duration of this chunk in microseconds.
            byte_count: Size of the encoded audio (for buffer tracking).

        Returns:
            True if sent successfully, False if dropped.
        """
        # Catch-up is also droppable binary data.
        if self._client.queue_high_water(threshold=0.5):
            self._drops_since_log += 1
            self.mark_needs_resync()
            return False

        header = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, timestamp_us)
        packed_data = header + payload
        chunk_end_us = timestamp_us + duration_us
        if not self._client.try_send_binary(
            packed_data,
            buffer_end_time_us=chunk_end_us,
            buffer_byte_count=byte_count,
            duration_us=duration_us,
        ):
            self._drops_since_log += 1
            self.mark_needs_resync()
            return False

        # Update send state
        self._send_state.last_sent_timestamp_us = timestamp_us

        return True

    def clear_stream(self) -> None:
        """
        Send stream/clear and reset stream state.

        Used for seek operations to discard buffered audio.
        """
        stream_clear = StreamClearMessage(payload=StreamClearPayload(roles=["player"]))
        self.send_message(stream_clear)

        # Reset stream state (stream/start will be re-sent)
        self._stream_started = False
        self._current_format = None

        # Reset buffer tracker
        if self._client.buffer_tracker is not None:
            self._client.buffer_tracker.reset()

    def end_stream(self) -> None:
        """
        Send stream/end and reset stream state.

        Used when playback stops completely.
        """
        # End all streams (roles omitted) for best client compatibility.
        stream_end = StreamEndMessage(payload=StreamEndPayload(roles=None))
        self.send_message(stream_end)

        # Reset stream state
        self._stream_started = False
        self._current_format = None

        # Reset buffer tracker
        if self._client.buffer_tracker is not None:
            self._client.buffer_tracker.reset()

    def on_format_change(self, new_format: AudioFormat) -> None:  # noqa: ARG002
        """
        Handle format change request.

        Marks stream as needing new stream/start on next audio send.
        The new_format will be applied when send_audio() is called.

        Args:
            new_format: The new audio format (applied on next send_audio).
        """
        # Clear stream state so next send_audio() will send stream/start
        self._stream_started = False
        self._current_format = None

    def mark_needs_resync(self) -> None:
        """Mark this player as needing resync (dropped audio)."""
        self._send_state.healthy = False
        self._send_state.needs_resync = True
        self._send_state.dropped_commits += 1

    def clear_resync_needed(self) -> None:
        """Clear the needs_resync flag after successful resync."""
        self._send_state.needs_resync = False
        self._send_state.healthy = True

    def resync(self) -> None:
        """
        Resync a dropped player by clearing and restarting the stream.

        Sends stream/clear, resets buffer tracker, and prepares for new stream/start.
        The next audio send will automatically send stream/start with the correct format.
        """
        # Send stream/clear to discard buffered audio on client
        self.clear_stream()

        # Prepare for new stream/start on next audio send
        # The send_audio method will send stream/start automatically
        # We just need to clear the state so it knows to send it
        self._stream_started = False
        self._current_format = None

        # Clear the resync flag
        self.clear_resync_needed()

    # --- New hook-based streaming methods ---
    # These will be called by PushStream after Task 5 refactoring.
    # The old methods (send_audio, send_stream_start, etc.) remain until Task 6.

    def on_stream_start(self) -> None:
        """Send stream/start message using transformer header."""
        req = self.get_audio_requirements()
        if req is None:
            return

        transformer = req.transformer
        header = transformer.get_header() if transformer else None
        header_b64 = base64.b64encode(header).decode() if header else None

        # Determine codec from transformer type
        codec = AudioCodec.FLAC if isinstance(transformer, FlacEncoder) else AudioCodec.PCM

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
        self._stream_started = True

    def on_audio_chunk(self, chunk: AudioChunk) -> bool:
        """Pack and send binary audio. Return False for backpressure."""
        # Check backpressure
        if self._client.queue_high_water(threshold=0.5):
            self.mark_needs_resync()
            return False

        # Pack binary header and send
        header = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, chunk.timestamp_us)
        packed_data = header + chunk.data
        chunk_end_us = chunk.timestamp_us + chunk.duration_us

        if not self._client.try_send_binary(
            packed_data,
            buffer_end_time_us=chunk_end_us,
            buffer_byte_count=chunk.byte_count,
            duration_us=chunk.duration_us,
        ):
            self.mark_needs_resync()
            return False

        self._send_state.last_sent_timestamp_us = chunk.timestamp_us
        self._send_state.healthy = True
        return True

    def on_stream_clear(self) -> None:
        """Send stream/clear and reset state."""
        stream_clear = StreamClearMessage(payload=StreamClearPayload(roles=["player"]))
        self.send_message(stream_clear)
        self._stream_started = False
        if self._buffer_tracker is not None:
            self._buffer_tracker.reset()

    def on_stream_end(self) -> None:
        """Send stream/end and reset state."""
        stream_end = StreamEndMessage(payload=StreamEndPayload(roles=None))
        self.send_message(stream_end)
        self._stream_started = False
        if self._buffer_tracker is not None:
            self._buffer_tracker.reset()
