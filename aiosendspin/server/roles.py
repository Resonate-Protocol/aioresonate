"""Role implementations for connection-specific behavior.

Roles encapsulate per-connection behavior for different client capabilities.
The PlayerRole handles audio streaming lifecycle, binary message packing,
and stream state management.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aiosendspin.models import BinaryMessageType, pack_binary_header_raw
from aiosendspin.models.core import (
    StreamClearMessage,
    StreamClearPayload,
    StreamEndMessage,
    StreamEndPayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.player import StreamStartPlayer

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.pipeline import EncodedChunk
    from aiosendspin.server.player_state import PlayerRecord
    from aiosendspin.server.stream import AudioFormat


class Role(ABC):
    """Base class for connection-specific role behavior."""

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

    _record: PlayerRecord
    """The connection-independent player record."""
    _connection: SendspinClient
    """The current WebSocket connection."""
    _stream_started: bool = field(default=False, init=False)
    """Whether stream/start has been sent for the current format."""
    _current_format: AudioFormat | None = field(default=None, init=False)
    """The current audio format being streamed."""
    _codec_header_b64: str | None = field(default=None, init=False)
    """Base64-encoded codec header (e.g., for FLAC)."""
    _send_state: PlayerSendState = field(default_factory=PlayerSendState, init=False)
    """Current send state for this player."""

    def on_connect(self) -> None:
        """Reset stream state on new connection."""
        self._stream_started = False
        self._current_format = None
        self._codec_header_b64 = None
        self._send_state = PlayerSendState()

    def on_disconnect(self) -> None:
        """Clean up on disconnect.

        Note: BufferTracker reset semantics are handled at the server layer
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
        codec_header_b64: str | None = None,
    ) -> None:
        """
        Send stream/start message to the player.

        Args:
            audio_format: Audio format for this stream.
            codec_header_b64: Optional base64-encoded codec header.
        """
        stream_start = StreamStartMessage(
            payload=StreamStartPayload(
                player=StreamStartPlayer(
                    codec=audio_format.codec,
                    sample_rate=audio_format.sample_rate,
                    channels=audio_format.channels,
                    bit_depth=audio_format.bit_depth,
                    codec_header=codec_header_b64,
                )
            )
        )
        self._connection.send_message(stream_start)
        self._stream_started = True
        self._current_format = audio_format
        self._codec_header_b64 = codec_header_b64

    def send_audio(
        self,
        chunk: EncodedChunk,
        timestamp_us: int,
        audio_format: AudioFormat,
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
            codec_header_b64: Optional base64-encoded codec header.

        Returns:
            True if sent successfully, False if dropped (queue high water).
        """
        # For non-blocking players, check queue before sending
        if not self._record.blocking and self._connection.queue_high_water():
            self.mark_needs_resync()
            return False

        # Check if we need to send stream/start (first chunk or format change)
        if not self._stream_started or self._current_format != audio_format:
            self.send_stream_start(audio_format, codec_header_b64)

        # Pack binary header and send
        header = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, timestamp_us)
        packed_data = header + chunk.data
        if not self._connection.try_send_binary(packed_data):
            self.mark_needs_resync()
            return False

        # Update send state
        self._send_state.last_sent_timestamp_us = timestamp_us
        self._send_state.healthy = True

        # Register with buffer tracker
        chunk_end_us = timestamp_us + chunk.duration_us
        if self._record.buffer_tracker is not None:
            self._record.buffer_tracker.register(chunk_end_us, chunk.byte_count)

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
        if not self._record.blocking and self._connection.queue_high_water():
            self.mark_needs_resync()
            return False

        header = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, timestamp_us)
        packed_data = header + payload
        if not self._connection.try_send_binary(packed_data):
            self.mark_needs_resync()
            return False

        # Update send state
        self._send_state.last_sent_timestamp_us = timestamp_us

        # Register with buffer tracker using the real duration.
        chunk_end_us = timestamp_us + duration_us
        if self._record.buffer_tracker is not None:
            self._record.buffer_tracker.register(chunk_end_us, byte_count)

        return True

    def clear_stream(self) -> None:
        """
        Send stream/clear and reset stream state.

        Used for seek operations to discard buffered audio.
        """
        stream_clear = StreamClearMessage(payload=StreamClearPayload(roles=["player"]))
        self._connection.send_message(stream_clear)

        # Reset stream state (stream/start will be re-sent)
        self._stream_started = False
        self._current_format = None

        # Reset buffer tracker
        if self._record.buffer_tracker is not None:
            self._record.buffer_tracker.reset()

    def end_stream(self) -> None:
        """
        Send stream/end and reset stream state.

        Used when playback stops completely.
        """
        stream_end = StreamEndMessage(payload=StreamEndPayload(roles=["player"]))
        self._connection.send_message(stream_end)

        # Reset stream state
        self._stream_started = False
        self._current_format = None

        # Reset buffer tracker
        if self._record.buffer_tracker is not None:
            self._record.buffer_tracker.reset()

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
