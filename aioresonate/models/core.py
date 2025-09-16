"""Core messages for the Resonate protocol.

This module contains the fundamental messages that establish communication between
clients and the server. These messages handle initial handshakes, ongoing clock
synchronization, and stream lifecycle management.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin
from mashumaro.types import Discriminator


class Roles(Enum):
    """Client roles."""

    PLAYER = "player"
    """
    Receives audio and plays it in sync.

    Has its own volume and mute state and preferred format settings.
    """
    METADATA = "metadata"
    """
    Displays metadata.

    Has preferred format for cover art.
    """
    CONTROLLER = "controller"
    """Controls Resonate groups."""


# Base message classes
@dataclass
class ClientMessage(DataClassORJSONMixin):
    """Base class for client messages."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclass
class ServerMessage(DataClassORJSONMixin):
    """Base class for server messages."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        discriminator = Discriminator(field="type", include_subtypes=True)


# Support payloads for client hello
@dataclass
class PlayerSupportClientPayload(DataClassORJSONMixin):
    """Player support configuration - only if player role is set."""

    support_codecs: list[str]
    """Supported codecs in priority order."""
    support_channels: list[int]
    """Number of channels in priority order."""
    support_sample_rates: list[int]
    """Supported sample rates in priority order."""
    support_bit_depth: list[int]
    """Bit depth in priority order."""
    buffer_capacity: int
    """Buffer capacity size in bytes."""


@dataclass
class MetadataSupportClientPayload(DataClassORJSONMixin):
    """Metadata support configuration - only if metadata role is set."""

    support_picture_formats: list[str]
    """Supported media art image formats (empty array if no art desired)."""
    media_width: int | None = None
    """Max width in pixels (if only width set, scales preserving aspect ratio)."""
    media_height: int | None = None
    """Max height in pixels (if only height set, scales preserving aspect ratio)."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


# Client → Server core messages
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


# Server → Client core messages
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


# Stream lifecycle messages
@dataclass
class StreamStartPlayerServerPayload(DataClassORJSONMixin):
    """Player object in stream/start message."""

    codec: str
    """Codec to be used."""
    sample_rate: int
    """Sample rate to be used."""
    channels: int
    """Channels to be used."""
    bit_depth: int
    """Bit depth to be used."""
    codec_header: str | None = None
    """Base64 encoded codec header (if necessary; e.g., FLAC)."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamStartMetadataServerPayload(DataClassORJSONMixin):
    """Metadata object in stream/start message.

    Sent to clients that specified supported picture formats.
    """

    art_format: Literal["bmp", "jpeg", "png"]
    """Format of the encoded image."""


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


@dataclass
class StreamUpdatePlayerServerPayload(DataClassORJSONMixin):
    """Player object in stream/update message with delta updates."""

    codec: str | None = None
    """Codec to be used."""
    sample_rate: int | None = None
    """Sample rate to be used."""
    channels: int | None = None
    """Channels to be used."""
    bit_depth: int | None = None
    """Bit depth to be used."""
    codec_header: str | None = None
    """Base64 encoded codec header (if necessary; e.g., FLAC)."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamUpdateMetadataServerPayload(DataClassORJSONMixin):
    """Metadata object in stream/update message with delta updates."""

    art_format: Literal["bmp", "jpeg", "png"] | None = None
    """Format of the encoded image."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


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


@dataclass
class StreamEndServerMessage(ServerMessage):
    """Message sent by the server to end a stream."""

    type: Literal["stream/end"] = "stream/end"
