"""Client event base types."""

# TODO: delete this file and move to client.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiosendspin.server.group import SendspinGroup


class ClientEvent:
    """Base event type used by Client.add_event_listener()."""


@dataclass
class ClientGroupChangedEvent(ClientEvent):
    """The client was moved to a different group."""

    new_group: SendspinGroup
    """The new group the client is now part of."""
