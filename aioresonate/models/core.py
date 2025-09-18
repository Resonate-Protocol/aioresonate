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
    ClientHelloMetadataSupport,
    SessionUpdateMetadata,
    StreamStartMetadata,
    StreamUpdateMetadata,
)
from .player import (
    ClientHelloPlayerSupport,
    StreamStartPlayer,
    StreamUpdatePlayer,
)
from .types import ClientMessage, PlaybackStateType, Roles, ServerMessage
from .visualizer import (
    ClientHelloVisualizerSupport,
    StreamStartVisualizer,
    StreamUpdateVisualizer,
)


# Client -> Server: client/hello
@dataclass
class ClientHelloPayload(DataClassORJSONMixin):
    """Information about a connected client."""

    client_id: str
    """Uniquely identifies the client for groups and de-duplication."""
    name: str
    """Friendly name of the client."""
    version: int
    """Version that the Resonate client implements."""
    supported_roles: list[Roles]
    """List of roles the client supports."""
    player_support: ClientHelloPlayerSupport | None = None
    """Player support configuration - only if player role is in supported_roles."""
    metadata_support: ClientHelloMetadataSupport | None = None
    """Metadata support configuration - only if metadata role is in supported_roles."""
    visualizer_support: ClientHelloVisualizerSupport | None = None
    """Visualizer support configuration - only if visualizer role is in supported_roles."""

    def __post_init__(self) -> None:
        """Enforce that support configs match supported roles."""
        # Validate player role and support configuration
        player_role_supported = Roles.PLAYER in self.supported_roles
        if player_role_supported and self.player_support is None:
            raise ValueError(
                "player_support must be provided when 'player' role is in supported_roles"
            )
        if not player_role_supported:
            self.player_support = None

        # Validate metadata role and support configuration
        metadata_role_supported = Roles.METADATA in self.supported_roles
        if metadata_role_supported and self.metadata_support is None:
            raise ValueError(
                "metadata_support must be provided when 'metadata' role is in supported_roles"
            )
        if not metadata_role_supported:
            self.metadata_support = None

        # Validate visualizer role and support configuration
        visualizer_role_supported = Roles.VISUALIZER in self.supported_roles
        if visualizer_role_supported and self.visualizer_support is None:
            raise ValueError(
                "visualizer_support must be provided when 'visualizer' role is in supported_roles"
            )
        if not visualizer_role_supported:
            self.visualizer_support = None

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ClientHelloMessage(ClientMessage):
    """Message sent by the client to identify itself."""

    payload: ClientHelloPayload
    type: Literal["client/hello"] = "client/hello"


# Client -> Server: client/time
@dataclass
class ClientTimePayload(DataClassORJSONMixin):
    """Timing information from the client."""

    client_transmitted: int
    """Client's internal clock timestamp in microseconds."""


@dataclass
class ClientTimeMessage(ClientMessage):
    """Message sent by the client for time synchronization."""

    payload: ClientTimePayload
    type: Literal["client/time"] = "client/time"


# Server -> Client: server/hello
@dataclass
class ServerHelloPayload(DataClassORJSONMixin):
    """Information about the server."""

    server_id: str
    """Identifier of the server."""
    name: str
    """Friendly name of the server"""
    version: int
    """Latest supported version of Resonate."""


@dataclass
class ServerHelloMessage(ServerMessage):
    """Message sent by the server to identify itself."""

    payload: ServerHelloPayload
    type: Literal["server/hello"] = "server/hello"


# Server -> Client: server/time
@dataclass
class ServerTimePayload(DataClassORJSONMixin):
    """Timing information from the server."""

    client_transmitted: int
    """Client's internal clock timestamp received in the client/time message"""
    server_received: int
    """Timestamp that the server received the client/time message in microseconds"""
    server_transmitted: int
    """Timestamp that the server transmitted this message in microseconds"""


@dataclass
class ServerTimeMessage(ServerMessage):
    """Message sent by the server for time synchronization."""

    payload: ServerTimePayload
    type: Literal["server/time"] = "server/time"


# Server -> Client: stream/start
@dataclass
class StreamStartPayload(DataClassORJSONMixin):
    """Information about an active streaming session."""

    player: StreamStartPlayer | None = None
    """Information about the player."""
    metadata: StreamStartMetadata | None = None
    """Metadata information (sent to clients that specified supported picture formats)."""
    visualizer: StreamStartVisualizer | None = None
    """Visualizer information (sent to clients with visualizer role)."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamStartMessage(ServerMessage):
    """Message sent by the server to start a stream."""

    payload: StreamStartPayload
    type: Literal["stream/start"] = "stream/start"


# Server -> Client: stream/update
@dataclass
class StreamUpdatePayload(DataClassORJSONMixin):
    """Delta updates for the ongoing stream."""

    player: StreamUpdatePlayer | None = None
    """Player updates."""
    metadata: StreamUpdateMetadata | None = None
    """Metadata updates."""
    visualizer: StreamUpdateVisualizer | None = None
    """Visualizer updates."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamUpdateMessage(ServerMessage):
    """Message sent by the server to update stream format."""

    payload: StreamUpdatePayload
    type: Literal["stream/update"] = "stream/update"


# Server -> Client: stream/end
@dataclass
class StreamEndMessage(ServerMessage):
    """Message sent by the server to end a stream."""

    type: Literal["stream/end"] = "stream/end"


# Server -> Client: session/update
@dataclass
class SessionUpdatePayload(DataClassORJSONMixin):
    """Delta updates for session state."""

    group_id: str
    """Group identifier."""
    playback_state: PlaybackStateType | None = None
    """Only sent to clients with controller or metadata roles."""
    metadata: SessionUpdateMetadata | None = None
    """Only sent to clients with metadata role."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class SessionUpdateMessage(ServerMessage):
    """Message sent by the server to update session state."""

    payload: SessionUpdatePayload
    type: Literal["session/update"] = "session/update"
