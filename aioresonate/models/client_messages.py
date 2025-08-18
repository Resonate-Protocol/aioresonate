"""Models for messages sent by the client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin
from mashumaro.types import Discriminator

from .types import MediaCommand, PlayerStateType


@dataclass
class ClientMessage(DataClassORJSONMixin):
    """Client Message type used by resonate."""

    class Config(BaseConfig):
        """Config for parsing client messages."""

        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclass
class PlayerHelloPayload(DataClassORJSONMixin):
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
class PlayerHelloMessage(ClientMessage):
    """Message sent by the player to identify itself."""

    payload: PlayerHelloPayload
    type: Literal["player/hello"] = "player/hello"


@dataclass
class StreamCommandPayload(DataClassORJSONMixin):
    """Payload for stream commands."""

    command: MediaCommand


@dataclass
class StreamCommandMessage(ClientMessage):
    """Message sent by the client to issue a stream command (e.g., play/pause)."""

    payload: StreamCommandPayload
    type: Literal["stream/command"] = "stream/command"


@dataclass
class PlayerStatePayload(DataClassORJSONMixin):
    """State information of the player."""

    state: PlayerStateType
    volume: int
    muted: bool


@dataclass
class PlayerStateMessage(ClientMessage):
    """Message sent by the player to report its state."""

    payload: PlayerStatePayload
    type: Literal["player/state"] = "player/state"


@dataclass
class PlayerTimePayload(DataClassORJSONMixin):
    """Timing information from the player."""

    player_transmitted: int


@dataclass
class PlayerTimeMessage(ClientMessage):
    """Message sent by the player for time synchronization."""

    payload: PlayerTimePayload
    type: Literal["player/time"] = "player/time"
