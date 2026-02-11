"""VisualizerV1Role implementation (v1).

This is a placeholder implementation. The role exists and registers with
the group, but does not send any stream messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.server.roles.base import Role

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient


class VisualizerV1Role(Role):
    """Placeholder role implementation for visualizers.

    Currently does nothing. Real implementation would:
    - Send stream/start with visualizer config
    - Send timestamped visualization data (FFT, etc.)
    - Handle stream/request-format for visualization preferences
    """

    def __init__(self, client: SendspinClient | None = None) -> None:
        """Initialize VisualizerV1Role.

        Args:
            client: The owning SendspinClient.
        """
        if client is None:
            msg = "VisualizerV1Role requires a client"
            raise ValueError(msg)
        self._client = client
        self._stream_started = False
        self._buffer_tracker = None
        self._group_role = None

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "visualizer@v1"

    @property
    def role_family(self) -> str:
        """Role family name for protocol messages."""
        return "visualizer"

    def on_connect(self) -> None:
        """Subscribe to VisualizerGroupRole."""
        self._subscribe_to_group_role()

    def on_disconnect(self) -> None:
        """Unsubscribe from VisualizerGroupRole."""
        self._unsubscribe_from_group_role()
