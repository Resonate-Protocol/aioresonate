"""Base role classes and dataclasses.

This module contains:
- StreamRequirements: Declaration that a role sends binary streams
- AudioChunk: Audio data delivered to roles
- AudioRequirements: Declaration that a role needs audio chunks
- Role: Abstract base class for all roles
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from aiosendspin.models.types import ServerMessage
    from aiosendspin.server.audio import BufferTracker
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.transformers import AudioTransformer


@dataclass(frozen=True)
class BinaryHandling:
    """Policy for how binary messages should be handled by connection.

    Roles return this from get_binary_handling() to declare how the connection
    should handle their binary messages (late detection, rate limiting, etc).
    """

    drop_late: bool = False
    """Drop binary messages whose timestamp is in the past."""

    grace_period_us: int = 0
    """Grace period after stream start before dropping late messages."""

    rate_limit: bool = False
    """Rate-limit delivery based on duration_us to avoid bursty sends."""

    rate_limit_factor: float = 1.1
    """Send at this multiple of real-time (1.1 = 110% speed)."""

    buffer_track: bool = False
    """Track sent bytes in the role's buffer tracker."""


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
    audio via on_audio_chunk() calls from PushStream.
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

    # Timing state for binary handling (used by connection)
    _stream_start_time_us: int | None = None
    """Timestamp when stream started, for grace period calculation."""

    _last_late_log_s: float = 0.0
    """Monotonic time of last late-message log (for rate limiting logs)."""

    _late_skips_since_log: int = 0
    """Count of skipped late messages since last log."""

    @property
    @abstractmethod
    def role_id(self) -> str:
        """Versioned role identifier (e.g., 'player@v1')."""
        ...

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

    def get_binary_handling(self, message_type: int) -> BinaryHandling | None:  # noqa: ARG002
        """Return handling policy for a binary message type, or None if not handled.

        The connection calls this to determine how to handle binary messages:
        - Whether to drop late messages
        - Whether to rate-limit delivery
        - Whether to track in buffer tracker
        """
        return None

    def reset_binary_timing(self) -> None:
        """Reset timing state for binary handling (called on stream clear/end)."""
        self._stream_start_time_us = None

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

    def requires_initial_state(self) -> bool:
        """Whether this role requires initial client/state before being 'connected'.

        Roles that return True will block the connection's "connected" status
        until their initial state subobject is received in client/state.
        """
        return False
