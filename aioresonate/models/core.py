"""Core messages for the Resonate protocol.

This module contains the fundamental messages that establish communication between
clients and the server. These messages handle initial handshakes, ongoing clock
synchronization, and stream lifecycle management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .metadata import (
    MetadataSupportClientPayload,
    StreamStartMetadataServerPayload,
    StreamUpdateMetadataServerPayload,
)
from .player import (
    PlayerSupportClientPayload,
    StreamStartPlayerServerPayload,
    StreamUpdatePlayerServerPayload,
)
from .types import ClientMessage, Roles, ServerMessage


# Client -> Server: client/hello
@dataclass
class ClientHelloClientPayload(DataClassORJSONMixin):
    """Information about a connected client."""

    client_id: str
    """Uniquely identifies the client for groups and de-duplication."""
    name: str
    """Friendly name of the client."""
    version: int
    """Version that the Resonate client implements."""
    supported_roles: list[Roles | str]
    """List of roles the client supports."""
    player_support: PlayerSupportClientPayload | None = None
    """Player support configuration - only if player role is in supported_roles."""
    metadata_support: MetadataSupportClientPayload | None = None
    """Metadata support configuration - only if metadata role is in supported_roles."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ClientHelloClientMessage(ClientMessage):
    """Message sent by the client to identify itself."""

    payload: ClientHelloClientPayload
    type: Literal["client/hello"] = "client/hello"


# Client -> Server: client/time
@dataclass
class ClientTimeClientPayload(DataClassORJSONMixin):
    """Timing information from the client."""

    client_transmitted: int
    """Client's internal clock timestamp in microseconds."""


@dataclass
class ClientTimeClientMessage(ClientMessage):
    """Message sent by the client for time synchronization."""

    payload: ClientTimeClientPayload
    type: Literal["client/time"] = "client/time"


# Server -> Client: server/hello
@dataclass
class ServerHelloServerPayload(DataClassORJSONMixin):
    """Information about the server."""

    server_id: str
    """Identifier of the server."""
    name: str
    """Friendly name of the server"""
    version: int
    """Latest supported version of Resonate."""


@dataclass
class ServerHelloServerMessage(ServerMessage):
    """Message sent by the server to identify itself."""

    payload: ServerHelloServerPayload
    type: Literal["server/hello"] = "server/hello"


# Server -> Client: server/time
@dataclass
class ServerTimeServerPayload(DataClassORJSONMixin):
    """Timing information from the server."""

    client_transmitted: int
    """Client's internal clock timestamp received in the client/time message"""
    server_received: int
    """Timestamp that the server received the client/time message in microseconds"""
    server_transmitted: int
    """Timestamp that the server transmitted this message in microseconds"""


@dataclass
class ServerTimeServerMessage(ServerMessage):
    """Message sent by the server for time synchronization."""

    payload: ServerTimeServerPayload
    type: Literal["server/time"] = "server/time"


# Client -> Server: stream/start
@dataclass
class StreamStartServerPayload(DataClassORJSONMixin):
    """Information about an active streaming session."""

    player: StreamStartPlayerServerPayload | None = None
    """Information about the player."""
    metadata: StreamStartMetadataServerPayload | None = None
    """Metadata information (sent to clients that specified supported picture formats)."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamStartServerMessage(ServerMessage):
    """Message sent by the server to start a stream."""

    payload: StreamStartServerPayload
    type: Literal["stream/start"] = "stream/start"


# Server -> Client: stream/update
@dataclass
class StreamUpdateServerPayload(DataClassORJSONMixin):
    """Delta updates for the ongoing stream."""

    player: StreamUpdatePlayerServerPayload | None = None
    """Player updates."""
    metadata: StreamUpdateMetadataServerPayload | None = None
    """Metadata updates."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamUpdateServerMessage(ServerMessage):
    """Message sent by the server to update stream format."""

    payload: StreamUpdateServerPayload
    type: Literal["stream/update"] = "stream/update"


# Server -> Client: stream/end
@dataclass
class StreamEndServerMessage(ServerMessage):
    """Message sent by the server to end a stream."""

    type: Literal["stream/end"] = "stream/end"
