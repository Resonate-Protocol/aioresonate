"""Models for messages sent by the client."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from mashumaro.mixins.orjson import DataClassORJSONMixin

from . import MediaCommand, PlayerStateType, RepeatMode


class BinaryMessageType(Enum):
    """Enum for Binary Message Types."""

    PlayAudioChunk = 1


@dataclass
class PlayerInfo(DataClassORJSONMixin):
    """Information about a connected player."""

    player_id: str
    name: str
    role: str
    buffer_capacity: int
    support_codecs: list[str]
    support_channels: list[int]
    support_sample_rates: list[int]
    support_bit_depth: list[int]
    support_streams: list[str]
    support_picture_formats: list[str]
    media_display_size: str | None = None


@dataclass
class PlayerTimeInfo(DataClassORJSONMixin):
    """Timing information from the player."""

    player_transmitted: int


@dataclass
class Metadata(DataClassORJSONMixin):
    """Metadata about the currently playing media."""

    title: str | None = None
    artist: str | None = None
    album: str | None = None
    year: int | None = None
    track: int | None = None
    group_members: list[str] = field(default_factory=list)
    support_commands: list[MediaCommand] = field(default_factory=list)
    repeat: RepeatMode = RepeatMode.OFF
    shuffle: bool = False


@dataclass
class StreamCommandPayload(DataClassORJSONMixin):
    """Payload for stream commands."""

    command: MediaCommand


@dataclass
class PlayerState(DataClassORJSONMixin):
    """State information of the player."""

    state: PlayerStateType
    volume: int
    muted: bool


# Client -> Server Messages
@dataclass
class PlayerHelloMessage(DataClassORJSONMixin):
    """Message sent by the player to identify itself."""

    payload: PlayerInfo
    type: Literal["player/hello"] = "player/hello"


@dataclass
class StreamCommandMessage(DataClassORJSONMixin):
    """Message sent by the client to issue a stream command (e.g., play/pause)."""

    payload: StreamCommandPayload
    type: Literal["stream/command"] = "stream/command"


@dataclass
class PlayerStateMessage(DataClassORJSONMixin):
    """Message sent by the player to report its state."""

    payload: PlayerState
    type: Literal["player/state"] = "player/state"


@dataclass
class PlayerTimeMessage(DataClassORJSONMixin):
    """Message sent by the player for time synchronization."""

    payload: PlayerTimeInfo
    type: Literal["player/time"] = "player/time"


ClientMessages = PlayerHelloMessage | StreamCommandMessage | PlayerStateMessage | PlayerTimeMessage
