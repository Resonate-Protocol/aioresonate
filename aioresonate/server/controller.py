"""Helpers for clients supporting the controller role."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aioresonate.models.controller import (
    GroupCommandClientPayload,
    GroupGetListClientMessage,
    GroupJoinClientPayload,
    GroupUnjoinClientMessage,
)

if TYPE_CHECKING:
    from .client import ResonateClient


class ControllerClient:
    """Encapsulates controller role behaviour for a client."""

    def __init__(self, client: ResonateClient) -> None:
        """Attach to a client that exposes controller capabilities."""
        self.client = client
        self._logger = client._logger.getChild("controller")  # noqa: SLF001

    def handle_get_list(self, _message: GroupGetListClientMessage) -> None:
        """Handle a request for the list of groups."""
        raise NotImplementedError("Group listing is not supported yet")

    def handle_join(self, _message: GroupJoinClientPayload) -> None:
        """Handle a request to join a group."""
        raise NotImplementedError("Joining groups is not supported yet")

    def handle_unjoin(self, _message: GroupUnjoinClientMessage) -> None:
        """Handle a request to leave the current group."""
        raise NotImplementedError("Leaving groups is not supported yet")

    def handle_command(self, message: GroupCommandClientPayload) -> None:
        """Forward a playback command to the owning group."""
        self.client.group._handle_group_command(message)  # noqa: SLF001
