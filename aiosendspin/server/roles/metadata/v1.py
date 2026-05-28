"""MetadataV1Role implementation (v1).

This role handles outbound server/state messages with metadata for display clients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.server.roles.base import Role
from aiosendspin.server.roles.metadata.group import MetadataGroupRole

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient


class MetadataV1Role(Role):
    """Role implementation for metadata display.

    Receives metadata updates from MetadataGroupRole and sends server/state
    messages to the client. This role is outbound-only.
    """

    def __init__(self, client: SendspinClient | None = None) -> None:
        """Initialize MetadataV1Role.

        Args:
            client: The owning SendspinClient.
        """
        if client is None:
            msg = "MetadataV1Role requires a client"
            raise ValueError(msg)
        self._client = client
        self._stream_started = False
        self._buffer_tracker = None
        self._group_role = None

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "metadata@v1"

    @property
    def role_family(self) -> str:
        """Role family name for protocol messages."""
        return "metadata"

    def on_connect(self) -> None:
        """Subscribe to MetadataGroupRole for state updates."""
        self._subscribe_to_group_role()
        if isinstance(self._group_role, MetadataGroupRole):
            self._group_role._send_state_to_role(self)  # noqa: SLF001

    def on_disconnect(self) -> None:
        """Unsubscribe from MetadataGroupRole."""
        self._unsubscribe_from_group_role()
