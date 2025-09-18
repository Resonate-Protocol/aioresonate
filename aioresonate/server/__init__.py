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
    "ClientRemovedEvent",
    "GroupCommandEvent",
    "GroupDeletedEvent",
    "GroupEvent",
    "GroupMemberAddedEvent",
    "GroupMemberRemovedEvent",
    "GroupState",
    "GroupStateChangedEvent",
    "PlayerGroup",
    "PlayerGroupChangedEvent",
    "ResonateEvent",
    "ResonateServer",
    "VolumeChangedEvent",
]

from .client import (
    Client,
    ClientEvent,
    PlayerGroupChangedEvent,
    VolumeChangedEvent,
)
from .group import (
    AudioFormat,
    GroupCommandEvent,
    GroupDeletedEvent,
    GroupEvent,
    GroupMemberAddedEvent,
    GroupMemberRemovedEvent,
    GroupState,
    GroupStateChangedEvent,
    PlayerGroup,
)
from .server import ClientAddedEvent, ClientRemovedEvent, ResonateEvent, ResonateServer
