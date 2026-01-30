"""ControllerRole implementation (v1).

This role handles bidirectional communication:
- Inbound: client/command controller messages
- Outbound: server/state controller messages
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.models.controller import ControllerCommandPayload
from aiosendspin.server.roles.base import Role

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.roles.controller.group import ControllerGroupRole


class ControllerRole(Role):
    """Role implementation for controller clients.

    Receives controller state from ControllerGroupRole and forwards commands
    from the client to the group.
    """

    def __init__(self, client: SendspinClient | None = None) -> None:
        """Initialize ControllerRole.

        Args:
            client: The owning SendspinClient.
        """
        if client is None:
            msg = "ControllerRole requires a client"
            raise ValueError(msg)
        self._client = client
        self._has_transport = False
        self._stream_started = False
        self._buffer_tracker = None
        self._group_role: ControllerGroupRole | None = None

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "controller@v1"

    @property
    def role_family(self) -> str:
        """Role family name for protocol messages."""
        return "controller"

    def on_connect(self) -> None:
        """Subscribe to ControllerGroupRole for state updates."""
        self._subscribe_to_group_role()

    def on_disconnect(self) -> None:
        """Unsubscribe from ControllerGroupRole."""
        self._unsubscribe_from_group_role()

    def handle_command(self, cmd: ControllerCommandPayload) -> None:
        """Forward a controller command to the group role.

        Args:
            cmd: The controller command from the client.
        """
        if self._group_role is not None:
            self._group_role.handle_command(cmd)
