"""Models for messages sent by the server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin
from mashumaro.types import Discriminator

from .types import MediaCommand, RepeatMode


@dataclass
class ServerMessage(DataClassORJSONMixin):
    """Server Message type used by resonate."""

    class Config(BaseConfig):
        """Config for parsing server messages."""

        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclass
class SessionStartPayload(DataClassORJSONMixin):
    """Information about an active streaming session."""

    session_id: str
    codec: str
    sample_rate: int
    channels: int
    bit_depth: int
    now: int
    codec_header: str | None = None


@dataclass
class SessionStartMessage(ServerMessage):
    """Message sent by the server to start a session."""

    payload: SessionStartPayload
    type: Literal["session/start"] = "session/start"


@dataclass
class ServerHelloPayload(DataClassORJSONMixin):
    """Information about the server (e.g., Music Assistant)."""

    server_id: str
    name: str


@dataclass
class ServerHelloMessage(ServerMessage):
    """Message sent by the server to identify itself."""

    payload: ServerHelloPayload
    type: Literal["server/hello"] = "server/hello"


@dataclass
class SessionEndPayload(DataClassORJSONMixin):
    """Payload for the session/end message."""

    sessionId: str  # noqa: N815


@dataclass
class SessionEndMessage(ServerMessage):
    """Message sent by the server to end a session."""

    payload: SessionEndPayload
    type: Literal["session/end"] = "session/end"


@dataclass
class MetadataUpdatePayload(DataClassORJSONMixin):
    """Represents a partial update to Metadata."""

    title: str | None = None
    artist: str | None = None
    album: str | None = None
    year: int | None = None
    track: int | None = None
    group_members: list[str] | None = None
    support_commands: list[MediaCommand] | None = None
    repeat: RepeatMode | None = None
    shuffle: bool | None = None


@dataclass
class MetadataUpdateMessage(ServerMessage):
    """Message sent by the server to update metadata."""

    payload: MetadataUpdatePayload
    type: Literal["metadata/update"] = "metadata/update"


@dataclass
class ServerTimePayload(DataClassORJSONMixin):
    """Timing information from the server."""

    client_transmitted: int
    server_received: int
    server_transmitted: int


@dataclass
class ServerTimeMessage(ServerMessage):
    """Message sent by the server for time synchronization."""

    payload: ServerTimePayload
    type: Literal["server/time"] = "server/time"


@dataclass
class VolumeSetPayload(DataClassORJSONMixin):
    """Payload for the set volume command."""

    volume: int


@dataclass
class VolumeSetMessage(ServerMessage):
    """Message sent by the server to set the volume."""

    payload: VolumeSetPayload
    type: Literal["volume/set"] = "volume/set"


@dataclass
class MuteSetPayload(DataClassORJSONMixin):
    """Payload for the set mute command."""

    mute: bool


@dataclass
class MuteSetMessage(ServerMessage):
    """Message sent by the server to set the mute mode."""

    payload: MuteSetPayload
    type: Literal["mute/set"] = "mute/set"
