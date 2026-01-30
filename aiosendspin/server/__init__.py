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

from aiosendspin.models.types import AudioCodec

from .audio import AudioFormat
from .client import DisconnectBehaviour, SendspinClient
from .events import ClientEvent, ClientGroupChangedEvent, VolumeChangedEvent
from .group import (
    GroupDeletedEvent,
    GroupEvent,
    GroupMemberAddedEvent,
    GroupMemberRemovedEvent,
    GroupStateChangedEvent,
    SendspinGroup,
)
from .server import ClientAddedEvent, ClientRemovedEvent, SendspinEvent, SendspinServer
