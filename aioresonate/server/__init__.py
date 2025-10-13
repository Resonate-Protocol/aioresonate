"""
Resonate Server implementation to connect to and manage Resonate Clients.

ResonateServer is the core of the music listening experience, responsible for:
- Managing connected clients
- Orchestrating synchronized grouped playback
"""

__all__ = [
    "AudioCodec",
    "AudioFormat",
    "ClientAddedEvent",
    "ClientEvent",
    "ClientGroupChangedEvent",
    "ClientRemovedEvent",
    "DisconnectBehaviour",
    "GroupCommandEvent",
    "GroupDeletedEvent",
    "GroupEvent",
    "GroupMemberAddedEvent",
    "GroupMemberRemovedEvent",
    "GroupStateChangedEvent",
    "ResonateClient",
    "ResonateEvent",
    "ResonateGroup",
    "ResonateServer",
    "VolumeChangedEvent",
]

from .client import DisconnectBehaviour, ResonateClient
from .events import ClientEvent, ClientGroupChangedEvent, VolumeChangedEvent
from .group import (
    GroupCommandEvent,
    GroupDeletedEvent,
    GroupEvent,
    GroupMemberAddedEvent,
    GroupMemberRemovedEvent,
    GroupStateChangedEvent,
    ResonateGroup,
)
from .server import ClientAddedEvent, ClientRemovedEvent, ResonateEvent, ResonateServer
from .stream import AudioCodec, AudioFormat
