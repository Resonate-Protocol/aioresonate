"""Models for messages sent by the client."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin
from mashumaro.types import Discriminator

from .types import MediaCommand, PlayerStateType


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


@dataclass
class PlayerSupportPayload(DataClassORJSONMixin):
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
class MetadataSupportPayload(DataClassORJSONMixin):
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


@dataclass
class ClientMessage(DataClassORJSONMixin):
    """Client Message type used by resonate."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclass
class ClientHelloPayload(DataClassORJSONMixin):
    """Information about a connected client."""

    client_id: str
    """Uniquely identifies the client for groups and de-duplication."""
    name: str
    """Friendly name of the client."""
    version: int
    """Version that the Resonate client implements."""
    supported_roles: list[Roles | str]
    """List of roles the client supports."""
    player_support: PlayerSupportPayload | None = None
    """Player support configuration - only if player role is in supported_roles."""
    metadata_support: MetadataSupportPayload | None = None
    """Metadata support configuration - only if metadata role is in supported_roles."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ClientHelloMessage(ClientMessage):
    """Message sent by the client to identify itself."""

    payload: ClientHelloPayload
    type: Literal["client/hello"] = "client/hello"


@dataclass
class PlayerUpdatePayload(DataClassORJSONMixin):
    """State information of the player."""

    state: PlayerStateType
    """Playing if active stream, idle if no active stream."""
    volume: int
    """Volume range 0-100."""
    muted: bool
    """Mute state."""


@dataclass
class PlayerUpdateMessage(ClientMessage):
    """Message sent by the player to report its state changes."""

    payload: PlayerUpdatePayload
    type: Literal["player/update"] = "player/update"


@dataclass
class StreamRequestFormatPayload(DataClassORJSONMixin):
    """Request different stream format (upgrade or downgrade)."""

    codec: str | None = None
    """Requested codec."""
    sample_rate: int | None = None
    """Requested sample rate."""
    channels: int | None = None
    """Requested channels."""
    bit_depth: int | None = None
    """Requested bit depth."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamRequestFormatMessage(ClientMessage):
    """Message sent by the client to request different stream format."""

    payload: StreamRequestFormatPayload
    type: Literal["stream/request-format"] = "stream/request-format"


@dataclass
class GroupGetListMessage(ClientMessage):
    """Message sent by the client to request all groups available to join."""

    type: Literal["group/get-list"] = "group/get-list"


@dataclass
class GroupJoinPayload(DataClassORJSONMixin):
    """Payload for joining a group."""

    group_id: str
    """Identifier of group to join."""


@dataclass
class GroupJoinMessage(ClientMessage):
    """Message sent by the client to join a group."""

    payload: GroupJoinPayload
    type: Literal["group/join"] = "group/join"


@dataclass
class GroupUnjoinMessage(ClientMessage):
    """Message sent by the client to leave current group."""

    type: Literal["group/unjoin"] = "group/unjoin"


@dataclass
class GroupCommandPayload(DataClassORJSONMixin):
    """Control the group that's playing."""

    command: MediaCommand
    """Command must be one of the values listed in group/update field supported_commands."""
    volume: int | None = None
    """Volume range 0-100, only set if command is volume."""
    mute: bool | None = None
    """True to mute, false to unmute, only set if command is mute."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class GroupCommandMessage(ClientMessage):
    """Message sent by the client to control the group."""

    payload: GroupCommandPayload
    type: Literal["group/command"] = "group/command"


@dataclass
class ClientTimePayload(DataClassORJSONMixin):
    """Timing information from the client."""

    client_transmitted: int


@dataclass
class ClientTimeMessage(ClientMessage):
    """Message sent by the client for time synchronization."""

    payload: ClientTimePayload
    type: Literal["client/time"] = "client/time"
