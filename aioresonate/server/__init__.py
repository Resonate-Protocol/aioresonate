"""
Resonate Server implementation to connect to and manage Resonate Clients.

ResonateServer is the core of the music listening experience, responsible for:
- Managing connected clients
- Orchestrating synchronized grouped playback
"""

__all__ = [
    "AudioFormat",
    "Client",
    "ClientAddedEvent",
    "ClientEvent",
    "ClientGroup",
    "ClientGroupChangedEvent",
    "ClientRemovedEvent",
    "GroupCommandEvent",
    "GroupDeletedEvent",
    "GroupEvent",
    "GroupMemberAddedEvent",
    "GroupMemberRemovedEvent",
    "GroupStateChangedEvent",
    "ResonateEvent",
    "ResonateServer",
    "VolumeChangedEvent",
]

from .client import (
    Client,
    ClientEvent,
    ClientGroupChangedEvent,
    VolumeChangedEvent,
)
from .group import (
    AudioFormat,
    ClientGroup,
    GroupCommandEvent,
    GroupDeletedEvent,
    GroupEvent,
    GroupMemberAddedEvent,
    GroupMemberRemovedEvent,
    GroupStateChangedEvent,
)
from .server import ClientAddedEvent, ClientRemovedEvent, ResonateEvent, ResonateServer
