"""
Core messages for the Sendspin protocol.

This module contains the fundamental messages that establish communication between
clients and the server. These messages handle initial handshakes, ongoing clock
synchronization, stream lifecycle management, and role-based state updates and commands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin
from mashumaro.types import Alias

from .artwork import (
    ClientHelloArtworkSupport,
    StreamRequestFormatArtwork,
    StreamStartArtwork,
)
from .controller import ControllerCommandPayload, ControllerStatePayload
from .metadata import SessionUpdateMetadata
from .player import (
    ClientHelloPlayerSupport,
    PlayerCommandPayload,
    PlayerStatePayload,
    StreamRequestFormatPlayer,
    StreamStartPlayer,
)
from .types import (
    ClientMessage,
    ClientStateType,
    ConnectionReason,
    GoodbyeReason,
    PlaybackStateType,
    Roles,
    ServerMessage,
)
from .visualizer import (
    ClientHelloVisualizerSupport,
    StreamStartVisualizer,
)

logger = logging.getLogger(__name__)


# This server implementation accidentally used incorrect field names (player_support, etc.)
# instead of the spec-compliant names (player@v1_support, etc.). This alias mapping allows
# deserialization of JSON using either the old or new field names.
# DEPRECATED(before-spec-1.0): Remove this mapping once all clients use spec-compliant names.
_CLIENT_HELLO_LEGACY_FIELD_ALIASES: dict[str, str] = {
    "player_support": "player@v1_support",
    "artwork_support": "artwork@v1_support",
    "visualizer_support": "visualizer@v1_support",
}


@dataclass
class DeviceInfo(DataClassORJSONMixin):
    """Optional information about the device."""

    product_name: str | None = None
    """Device model/product name."""
    manufacturer: str | None = None
    """Device manufacturer name."""
    software_version: str | None = None
    """Software version of the client (not the Sendspin version)."""

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
    """Version that the Sendspin client implements."""
    supported_roles: list[Roles]
    """List of roles the client supports."""
    device_info: DeviceInfo | None = None
    """Optional information about the device."""
    player_support: Annotated[ClientHelloPlayerSupport | None, Alias("player@v1_support")] = None
    """Player support configuration - only if player role is in supported_roles."""
    artwork_support: Annotated[ClientHelloArtworkSupport | None, Alias("artwork@v1_support")] = None
    """Artwork support configuration - only if artwork role is in supported_roles."""
    visualizer_support: Annotated[
        ClientHelloVisualizerSupport | None, Alias("visualizer@v1_support")
    ] = None
    """Visualizer support configuration - only if visualizer role is in supported_roles."""

    @classmethod
    def __pre_deserialize__(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Accept both legacy and spec-compliant field names."""
        legacy_fields_used: list[tuple[str, str]] = []
        for legacy_name, spec_name in _CLIENT_HELLO_LEGACY_FIELD_ALIASES.items():
            if legacy_name in d and spec_name not in d:
                legacy_fields_used.append((legacy_name, spec_name))
                d[spec_name] = d.pop(legacy_name)
        if legacy_fields_used:
            old_names = ", ".join(old for old, _ in legacy_fields_used)
            new_names = ", ".join(new for _, new in legacy_fields_used)
            logger.info(
                "client/hello message used deprecated field names (%s), "
                "please update client to use (%s) instead",
                old_names,
                new_names,
            )
        return d

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


# Client -> Server: client/state
@dataclass
class ClientStatePayload(DataClassORJSONMixin):
    """Client sends state updates to the server."""

    state: ClientStateType | None = None
    """
    Client operational state.

    - 'synchronized': Client is operational and synchronized with server timestamps.
    - 'error': Client has a problem preventing normal operation.
    - 'external_source': Client is in use by an external system and cannot participate
      in Sendspin playback.
    """
    player: PlayerStatePayload | None = None
    """Player state - only if client has player role."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ClientStateMessage(ClientMessage):
    """Message sent by the client to report state changes."""

    payload: ClientStatePayload
    type: Literal["client/state"] = "client/state"


# Client -> Server: client/command
@dataclass
class ClientCommandPayload(DataClassORJSONMixin):
    """Client sends commands to the server."""

    controller: ControllerCommandPayload | None = None
    """Controller commands - only if client has controller role."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ClientCommandMessage(ClientMessage):
    """Message sent by the client to send commands."""

    payload: ClientCommandPayload
    type: Literal["client/command"] = "client/command"


# Client -> Server: client/goodbye
@dataclass
class ClientGoodbyePayload(DataClassORJSONMixin):
    """Payload for client goodbye message."""

    reason: GoodbyeReason
    """Reason for disconnecting."""


@dataclass
class ClientGoodbyeMessage(ClientMessage):
    """Message sent by the client before gracefully closing the connection."""

    payload: ClientGoodbyePayload
    type: Literal["client/goodbye"] = "client/goodbye"


# Server -> Client: server/hello
@dataclass
class ServerHelloPayload(DataClassORJSONMixin):
    """Information about the server."""

    server_id: str
    """Identifier of the server."""
    name: str
    """Friendly name of the server"""
    version: int
    """Version of the core message format (independent of role versions)."""
    active_roles: list[Roles]
    """Versioned roles that are active for this client (e.g., player@v1, controller@v1)."""
    connection_reason: ConnectionReason
    """Reason for this connection (relevant for multi-server environments)."""


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


# Server -> Client: server/state
@dataclass
class ServerStatePayload(DataClassORJSONMixin):
    """Server sends state updates to the client."""

    metadata: SessionUpdateMetadata | None = None
    """Metadata state - only sent to clients with metadata role."""
    controller: ControllerStatePayload | None = None
    """Controller state - only sent to clients with controller role."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ServerStateMessage(ServerMessage):
    """Message sent by the server to send state updates."""

    payload: ServerStatePayload
    type: Literal["server/state"] = "server/state"


# Server -> Client: group/update
@dataclass
class GroupUpdateServerPayload(DataClassORJSONMixin):
    """State update of the group this client is part of."""

    playback_state: PlaybackStateType | None = None
    """Playback state of the group."""
    group_id: str | None = None
    """Group identifier."""
    group_name: str | None = None
    """Friendly name of the group."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class GroupUpdateServerMessage(ServerMessage):
    """Message sent by the server to update group state."""

    payload: GroupUpdateServerPayload
    type: Literal["group/update"] = "group/update"


# Server -> Client: server/command
@dataclass
class ServerCommandPayload(DataClassORJSONMixin):
    """Server sends commands to the client."""

    player: PlayerCommandPayload | None = None
    """Player commands - only sent to clients with player role."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ServerCommandMessage(ServerMessage):
    """Message sent by the server to send commands to the client."""

    payload: ServerCommandPayload
    type: Literal["server/command"] = "server/command"


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


# Roles that support stream/clear (have buffers to clear)
STREAM_CLEAR_ROLES = frozenset({Roles.PLAYER, Roles.VISUALIZER})


# Server -> Client: stream/clear
@dataclass
class StreamClearPayload(DataClassORJSONMixin):
    """Instructs clients to clear buffers without ending the stream."""

    roles: list[Roles] | None = None
    """Roles to clear: player, visualizer, or both. If omitted, clears both roles."""

    def __post_init__(self) -> None:
        """Validate that only player and visualizer roles are specified."""
        if self.roles is not None:
            invalid_roles = set(self.roles) - STREAM_CLEAR_ROLES
            if invalid_roles:
                raise ValueError(
                    f"stream/clear only supports roles {[r.value for r in STREAM_CLEAR_ROLES]}, "
                    f"got invalid roles: {[r.value for r in invalid_roles]}"
                )

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamClearMessage(ServerMessage):
    """Message sent by the server to clear stream buffers (e.g., for seek operations)."""

    payload: StreamClearPayload
    type: Literal["stream/clear"] = "stream/clear"


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
class StreamEndPayload(DataClassORJSONMixin):
    """Payload for stream/end message."""

    roles: list[Roles] | None = None
    """Roles to end streams for. If omitted, ends all active streams."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamEndMessage(ServerMessage):
    """Message sent by the server to end a stream."""

    payload: StreamEndPayload
    type: Literal["stream/end"] = "stream/end"
