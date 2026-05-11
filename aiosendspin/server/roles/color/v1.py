"""ColorV1Role implementation (v1).

This role handles outbound server/state messages with color palettes for display clients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.server.roles.base import Role

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient


class ColorV1Role(Role):
    """Role implementation for color palette display.

    Receives color updates from ColorGroupRole and sends server/state
    messages to the client. This role is outbound-only.
    """

    def __init__(self, client: SendspinClient | None = None) -> None:
        """Initialize ColorV1Role."""
        if client is None:
            msg = "ColorV1Role requires a client"
            raise ValueError(msg)
        self._client = client
        self._stream_started = False
        self._buffer_tracker = None
        self._group_role = None

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "color@v1"

    @property
    def role_family(self) -> str:
        """Role family name for protocol messages."""
        return "color"

    def on_connect(self) -> None:
        """Subscribe to ColorGroupRole for state updates."""
        self._subscribe_to_group_role()

    def on_disconnect(self) -> None:
        """Unsubscribe from ColorGroupRole."""
        self._unsubscribe_from_group_role()
