"""
Core messages for the Resonate protocol.

This module contains the fundamental messages that establish communication between
clients and the server. These messages handle initial handshakes, ongoing clock
synchronization, and stream lifecycle management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .artwork import (
    ClientHelloArtworkSupport,
    StreamRequestFormatArtwork,
    StreamStartArtwork,
    StreamUpdateArtwork,
)
from .metadata import SessionUpdateMetadata
from .player import (
    ClientHelloPlayerSupport,
    StreamRequestFormatPlayer,
    StreamStartPlayer,
    StreamUpdatePlayer,
)
from .types import ClientMessage, PlaybackStateType, Roles, ServerMessage
from .visualizer import (
    ClientHelloVisualizerSupport,
    StreamStartVisualizer,
    StreamUpdateVisualizer,
)


@dataclass
class DeviceInfo(DataClassORJSONMixin):
    """Optional information about the device."""

    product_name: str | None = None
    """Device model/product name."""
    manufacturer: str | None = None
    """Device manufacturer name."""
    software_version: str | None = None
    """Software version of the client (not the Resonate version)."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


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
    device_info: DeviceInfo | None = None
    """Optional information about the device."""
    player_support: ClientHelloPlayerSupport | None = None
    """Player support configuration - only if player role is in supported_roles."""
    artwork_support: ClientHelloArtworkSupport | None = None
    """Artwork support configuration - only if artwork role is in supported_roles."""
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

        # Validate artwork role and support configuration
        artwork_role_supported = Roles.ARTWORK in self.supported_roles
        if artwork_role_supported and self.artwork_support is None:
            raise ValueError(
                "artwork_support must be provided when 'artwork' role is in supported_roles"
            )
        if not artwork_role_supported:
            self.artwork_support = None

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
    artwork: StreamStartArtwork | None = None
    """Artwork information (sent to clients with artwork role)."""
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
    artwork: StreamUpdateArtwork | None = None
    """Artwork updates."""
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


# Client -> Server: stream/request-format
@dataclass
class StreamRequestFormatPayload(DataClassORJSONMixin):
    """Request different stream format (upgrade or downgrade)."""

    player: StreamRequestFormatPlayer | None = None
    """Player format request (only for clients with player role)."""
    artwork: StreamRequestFormatArtwork | None = None
    """Artwork format request (only for clients with artwork role)."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamRequestFormatMessage(ClientMessage):
    """Message sent by the client to request different stream format."""

    payload: StreamRequestFormatPayload
    type: Literal["stream/request-format"] = "stream/request-format"


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
