"""Player state management for connection-independent player tracking."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aiosendspin.server.stream import AudioFormat, BufferTracker

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient


class PlayerRecord:
    """
    Connection-independent player state keyed by client_id.

    This class tracks player state that persists across connections,
    enabling reconnection with preserved volume, mute state, and group membership.
    Each PlayerRecord owns its own BufferTracker for backpressure management.
    """

    def __init__(
        self,
        client_id: str,
        *,
        loop: asyncio.AbstractEventLoop,
        buffer_capacity_bytes: int,
    ) -> None:
        """
        Create a new PlayerRecord.

        Args:
            client_id: Unique identifier for this player.
            loop: Event loop for the BufferTracker.
            buffer_capacity_bytes: Buffer capacity for backpressure tracking.
        """
        self._client_id = client_id
        self._volume: int = 100
        self._muted: bool = False
        self._group_id: str | None = None
        self._connection: SendspinClient | None = None
        self._preferred_format: AudioFormat | None = None
        self._disconnect_time_us: int | None = None
        self._buffer_tracker = BufferTracker(
            loop=loop,
            client_id=client_id,
            capacity_bytes=buffer_capacity_bytes,
        )

    @property
    def client_id(self) -> str:
        """Unique identifier for this player."""
        return self._client_id

    @property
    def volume(self) -> int:
        """Current volume level (0-100)."""
        return self._volume

    @volume.setter
    def volume(self, value: int) -> None:
        """Set volume level (0-100)."""
        self._volume = max(0, min(100, value))

    @property
    def muted(self) -> bool:
        """Current mute state."""
        return self._muted

    @muted.setter
    def muted(self, value: bool) -> None:
        """Set mute state."""
        self._muted = value

    @property
    def group_id(self) -> str | None:
        """Group ID this player belongs to, or None if not in a group."""
        return self._group_id

    @group_id.setter
    def group_id(self, value: str | None) -> None:
        """Set group membership."""
        self._group_id = value

    @property
    def connection(self) -> SendspinClient | None:
        """Current WebSocket connection, or None if disconnected."""
        return self._connection

    @connection.setter
    def connection(self, value: SendspinClient | None) -> None:
        """Set the WebSocket connection."""
        self._connection = value

    @property
    def is_connected(self) -> bool:
        """Whether this player currently has an active connection."""
        return self._connection is not None

    @property
    def preferred_format(self) -> AudioFormat | None:
        """Preferred audio format for this player."""
        return self._preferred_format

    @preferred_format.setter
    def preferred_format(self, value: AudioFormat | None) -> None:
        """Set preferred audio format."""
        self._preferred_format = value

    @property
    def buffer_tracker(self) -> BufferTracker:
        """BufferTracker owned by this player for backpressure management."""
        return self._buffer_tracker

    @property
    def disconnect_time_us(self) -> int | None:
        """Timestamp when player disconnected, or None if connected or never disconnected."""
        return self._disconnect_time_us

    def mark_disconnected(self, time_us: int) -> None:
        """Record when this player disconnected for cleanup decisions."""
        self._disconnect_time_us = time_us
