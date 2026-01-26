"""PlayerRole implementation for audio playback (v1).

This PlayerRole implementation supports both:
- Hook-based streaming (on_stream_start, on_audio_chunk, etc.)
- Legacy direct methods (send_audio, send_stream_start, etc.)

The hook-based approach is preferred for new code. Legacy methods are retained
for backward compatibility with existing PushStream code paths.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
from aiosendspin.server.roles.base import (
    AudioChunk,
    AudioRequirements,
    Role,
    StreamRequirements,
)
from aiosendspin.server.transformers import FlacEncoder

if TYPE_CHECKING:
    from aiosendspin.server.audio import AudioFormat
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.pipeline import EncodedChunk


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


class PlayerRole(Role):
    """Role implementation for audio playback.

    This implementation supports both hook-based and legacy direct methods:

    Hook-based streaming (preferred):
    - on_stream_start(): Send stream/start message
    - on_audio_chunk(): Pack and send binary audio
    - on_stream_clear(): Send stream/clear message
    - on_stream_end(): Send stream/end message

    Legacy direct methods (for backward compatibility):
    - send_stream_start(): Send stream/start with explicit format
    - send_audio(): Send audio chunk with explicit format
    - send_cached_chunk(): Send cached chunk for late joiner catch-up
    - clear_stream(): Send stream/clear and reset state
    - end_stream(): Send stream/end and reset state
    """

    def __init__(
        self,
        client: SendspinClient | None = None,
        *,
        preferred_format: AudioFormat | None = None,
        blocking: bool = True,
        audio_requirements: AudioRequirements | None = None,
        # Legacy parameter name for backward compatibility
        _client: SendspinClient | None = None,
    ) -> None:
        """Initialize PlayerRole.

        Args:
            client: The owning SendspinClient.
            preferred_format: Preferred audio format for this player.
            blocking: Whether this player participates in backpressure timing.
            audio_requirements: Audio requirements for hook-based streaming.
            _client: Legacy parameter name (use 'client' instead).
        """
        # Support both 'client' and '_client' parameter names
        actual_client = client if client is not None else _client
        if actual_client is None:
            msg = "PlayerRole requires a client"
            raise ValueError(msg)
        self._client = actual_client
        self._preferred_format = preferred_format
        self._blocking = blocking
        self._audio_requirements = audio_requirements
        self._has_transport = False
        self._stream_started = False
        self._buffer_tracker = None
        # Legacy state for direct method approach
        self._current_format: AudioFormat | None = None
        self._codec_header_b64: str | None = None
        self._send_state = PlayerSendState()
        self._last_drop_log_s: float = 0.0
        self._drops_since_log: int = 0

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
        return self._preferred_format

    @property
    def blocking(self) -> bool:
        """Return whether this player participates in backpressure timing."""
        return self._blocking

    # --- Declarations ---

    def get_stream_requirements(self) -> StreamRequirements:
        """Player role sends binary audio streams."""
        return StreamRequirements()

    def get_audio_requirements(self) -> AudioRequirements | None:
        """Return audio requirements for hook-based streaming."""
        return self._audio_requirements

    # --- Lifecycle hooks ---

    def on_connect(self) -> None:
        """Reset stream state on new connection."""
        self._stream_started = False

    def on_disconnect(self) -> None:
        """Clean up on disconnect."""
        self._stream_started = False

    # --- Stream lifecycle hooks ---

    def on_stream_start(self) -> None:
        """Send stream/start message using transformer header."""
        req = self.get_audio_requirements()
        if req is None:
            return

        if not self._has_transport:
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
            return False

        # Pack binary header and send
        header = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, chunk.timestamp_us)
        packed_data = header + chunk.data
        chunk_end_us = chunk.timestamp_us + chunk.duration_us

        return self._client.try_send_binary(
            packed_data,
            buffer_end_time_us=chunk_end_us,
            buffer_byte_count=chunk.byte_count,
            duration_us=chunk.duration_us,
        )

    def on_stream_clear(self) -> None:
        """Send stream/clear and reset state."""
        if not self._has_transport:
            return

        stream_clear = StreamClearMessage(payload=StreamClearPayload(roles=["player"]))
        self.send_message(stream_clear)
        self._stream_started = False

        if self._buffer_tracker is not None:
            self._buffer_tracker.reset()

    def on_stream_end(self) -> None:
        """Send stream/end and reset state."""
        if not self._has_transport:
            return

        # End all streams (roles omitted) for best client compatibility.
        stream_end = StreamEndMessage(payload=StreamEndPayload(roles=None))
        self.send_message(stream_end)
        self._stream_started = False

        if self._buffer_tracker is not None:
            self._buffer_tracker.reset()

    # --- Legacy properties (for backward compatibility) ---

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

    # --- Legacy methods (for backward compatibility) ---

    def send_stream_start(
        self,
        audio_format: AudioFormat,
        codec: AudioCodec,
        codec_header_b64: str | None = None,
    ) -> None:
        """Send stream/start message to the player.

        Legacy method for explicit format specification.

        Args:
            audio_format: Audio format for this stream.
            codec: Audio codec for encoding.
            codec_header_b64: Optional base64-encoded codec header.
        """
        if not self._has_transport:
            return

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
        """Send audio chunk with timestamp.

        Legacy method - automatically sends stream/start if needed.

        Args:
            chunk: Encoded audio chunk to send.
            timestamp_us: Playback timestamp in microseconds.
            audio_format: Audio format for this chunk.
            codec: Audio codec for encoding.
            codec_header_b64: Optional base64-encoded codec header.

        Returns:
            True if sent successfully, False if dropped (queue high water).
        """
        # Avoid building large per-connection backlogs
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
        """Send a cached chunk (for late joiner catch-up).

        Args:
            payload: Encoded audio payload bytes (without binary header).
            timestamp_us: Playback timestamp for this chunk.
            duration_us: Duration of this chunk in microseconds.
            byte_count: Size of the encoded audio (for buffer tracking).

        Returns:
            True if sent successfully, False if dropped.
        """
        # Catch-up is also droppable binary data
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
        """Send stream/clear and reset stream state.

        Legacy method - used for seek operations.
        """
        if not self._has_transport:
            return

        stream_clear = StreamClearMessage(payload=StreamClearPayload(roles=["player"]))
        self.send_message(stream_clear)

        # Reset stream state
        self._stream_started = False
        self._current_format = None

        # Reset buffer tracker
        if self._client.buffer_tracker is not None:
            self._client.buffer_tracker.reset()

    def end_stream(self) -> None:
        """Send stream/end and reset stream state.

        Legacy method - used when playback stops completely.
        """
        if not self._has_transport:
            return

        # End all streams (roles omitted) for best client compatibility
        stream_end = StreamEndMessage(payload=StreamEndPayload(roles=None))
        self.send_message(stream_end)

        # Reset stream state
        self._stream_started = False
        self._current_format = None

        # Reset buffer tracker
        if self._client.buffer_tracker is not None:
            self._client.buffer_tracker.reset()

    def on_format_change(self, new_format: AudioFormat) -> None:  # noqa: ARG002
        """Handle format change request.

        Marks stream as needing new stream/start on next audio send.

        Args:
            new_format: The new audio format (applied on next send_audio).
        """
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
        """Resync a dropped player by clearing and restarting the stream.

        Sends stream/clear, resets buffer tracker, and prepares for new stream/start.
        """
        # Send stream/clear to discard buffered audio on client
        self.clear_stream()

        # Prepare for new stream/start on next audio send
        self._stream_started = False
        self._current_format = None

        # Clear the resync flag
        self.clear_resync_needed()
