"""Helpers for clients supporting the controller role."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aioresonate.models.controller import ControllerCommandPayload

if TYPE_CHECKING:
    from .client import ResonateClient


class ControllerClient:
    """Encapsulates controller role behaviour for a client."""

    def __init__(self, client: ResonateClient) -> None:
        """Attach to a client that exposes controller capabilities."""
        self.client = client
        self._logger = client._logger.getChild("controller")  # noqa: SLF001

    def handle_command(self, payload: ControllerCommandPayload) -> None:
        """Forward a playback command to the owning group."""
        self.client.group._handle_group_command(payload)  # noqa: SLF001
