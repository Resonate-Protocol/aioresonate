"""Shared controller role protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aiosendspin.models.types import ServerMessage


@runtime_checkable
class ControllerRoleProtocol(Protocol):
    """Protocol for controller role implementations."""

    @property
    def role_id(self) -> str:
        """Return the versioned role identifier."""
        ...

    def send_message(self, message: ServerMessage) -> None:
        """Send a JSON message to the client."""
        ...
