"""Event types for the Sendspin server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .group import SendspinGroup


class ClientEvent:
    """Base event type used by Client.add_event_listener()."""


@dataclass
class VolumeChangedEvent(ClientEvent):
    """The volume or mute status of the player was changed."""

    volume: int
    muted: bool


@dataclass
class ClientGroupChangedEvent(ClientEvent):
    """The client was moved to a different group."""

    new_group: SendspinGroup
    """The new group the client is now part of."""
