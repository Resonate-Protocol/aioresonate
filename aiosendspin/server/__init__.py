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
    "ClientConnectedEvent",
    "ClientDisconnectedEvent",
    "ClientEvent",
    "ClientGroupChangedEvent",
    "ClientReconnectedEvent",
    "ClientRemovedEvent",
    "ClientRoleEvent",
    "ClientUpdatedEvent",
    "ConnectionSecurity",
    "DisconnectBehaviour",
    "ExternalStreamStartCallback",
    "ExternalStreamStartRequest",
    "GroupDeletedEvent",
    "GroupEvent",
    "GroupMemberAddedEvent",
    "GroupMemberRemovedEvent",
    "GroupRoleEvent",
    "GroupStateChangedEvent",
    "MinBufferChangedEvent",
    "RequiredLeadTimeChangedEvent",
    "SendspinClient",
    "SendspinEvent",
    "SendspinGroup",
    "SendspinServer",
    "StaticDelayChangedEvent",
    "VolumeChangedEvent",
]

from aiosendspin.models.types import AudioCodec

from .audio import AudioFormat
from .client import ConnectionSecurity, DisconnectBehaviour, SendspinClient
from .events import (
    ClientEvent,
    ClientGroupChangedEvent,
    ClientRoleEvent,
    GroupDeletedEvent,
    GroupEvent,
    GroupMemberAddedEvent,
    GroupMemberRemovedEvent,
    GroupRoleEvent,
    GroupStateChangedEvent,
)
from .group import (
    SendspinGroup,
)
from .roles.player.events import (
    MinBufferChangedEvent,
    RequiredLeadTimeChangedEvent,
    StaticDelayChangedEvent,
    VolumeChangedEvent,
)
from .server import (
    ClientAddedEvent,
    ClientConnectedEvent,
    ClientDisconnectedEvent,
    ClientReconnectedEvent,
    ClientRemovedEvent,
    ClientUpdatedEvent,
    ExternalStreamStartCallback,
    ExternalStreamStartRequest,
    SendspinEvent,
    SendspinServer,
)
