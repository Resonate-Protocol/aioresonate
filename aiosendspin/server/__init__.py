"""
Sendspin Server implementation to connect to and manage Sendspin Clients.

SendspinServer is the core of the music listening experience, responsible for:
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
    "SendspinClient",
    "SendspinEvent",
    "SendspinGroup",
    "SendspinServer",
    "VolumeChangedEvent",
]

from .client import DisconnectBehaviour, SendspinClient
from .events import ClientEvent, ClientGroupChangedEvent, VolumeChangedEvent
from .group import (
    GroupCommandEvent,
    GroupDeletedEvent,
    GroupEvent,
    GroupMemberAddedEvent,
    GroupMemberRemovedEvent,
    GroupStateChangedEvent,
    SendspinGroup,
)
from .server import ClientAddedEvent, ClientRemovedEvent, SendspinEvent, SendspinServer
from .stream import AudioCodec, AudioFormat
