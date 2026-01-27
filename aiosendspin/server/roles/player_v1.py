"""PlayerRole implementation for audio playback (v1).

This PlayerRole implementation uses hook-based streaming:
- on_stream_start(): Send stream/start message
- on_audio_chunk(): Pack and send binary audio
- on_stream_clear(): Send stream/clear message
- on_stream_end(): Send stream/end message
"""

from __future__ import annotations

import base64
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


class PlayerRole(Role):
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
        blocking: bool = True,
        audio_requirements: AudioRequirements | None = None,
    ) -> None:
        """Initialize PlayerRole.

        Args:
            client: The owning SendspinClient.
            preferred_format: Preferred audio format for this player.
            blocking: Whether this player participates in backpressure timing.
            audio_requirements: Audio requirements for hook-based streaming.
        """
        if client is None:
            msg = "PlayerRole requires a client"
            raise ValueError(msg)
        self._client = client
        self._preferred_format = preferred_format
        self._blocking = blocking
        self._audio_requirements = audio_requirements
        self._has_transport = False
        self._stream_started = False
        self._buffer_tracker = None

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

    def requires_initial_state(self) -> bool:
        """Player role requires initial state with volume/mute info."""
        return True

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
        """Pack and send binary audio. Late audio is discarded by connection."""
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

    @property
    def stream_started(self) -> bool:
        """Whether stream/start has been sent."""
        return self._stream_started
