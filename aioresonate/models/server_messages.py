"""Models for messages sent by the server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import MediaCommand, RepeatMode


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
class SessionStartMessage(DataClassORJSONMixin):
    """Message sent by the server to start a session."""

    payload: SessionStartPayload
    type: Literal["session/start"] = "session/start"


@dataclass
class SourceHelloPayload(DataClassORJSONMixin):
    """Information about the source (e.g., Music Assistant)."""

    source_id: str
    name: str


@dataclass
class SourceHelloMessage(DataClassORJSONMixin):
    """Message sent by the server to identify itself."""

    payload: SourceHelloPayload
    type: Literal["source/hello"] = "source/hello"


@dataclass
class SessionEndPayload(DataClassORJSONMixin):
    """Payload for the session/end message."""

    sessionId: str  # noqa: N815


@dataclass
class SessionEndMessage(DataClassORJSONMixin):
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
class MetadataUpdateMessage(DataClassORJSONMixin):
    """Message sent by the server to update metadata."""

    payload: MetadataUpdatePayload
    type: Literal["metadata/update"] = "metadata/update"


@dataclass
class SourceTimePayload(DataClassORJSONMixin):
    """Timing information from the source."""

    player_transmitted: int
    source_received: int
    source_transmitted: int


@dataclass
class SourceTimeMessage(DataClassORJSONMixin):
    """Message sent by the server for time synchronization."""

    payload: SourceTimePayload
    type: Literal["source/time"] = "source/time"


@dataclass
class VolumeSetPayload(DataClassORJSONMixin):
    """Payload for the set volume command."""

    volume: int


@dataclass
class VolumeSetMessage(DataClassORJSONMixin):
    """Message sent by the source to set the volume."""

    payload: VolumeSetPayload
    type: Literal["volume/set"] = "volume/set"


@dataclass
class MuteSetPayload(DataClassORJSONMixin):
    """Payload for the set mute command."""

    mute: bool


@dataclass
class MuteSetMessage(DataClassORJSONMixin):
    """Message sent by the source to set the mute mode."""

    payload: MuteSetPayload
    type: Literal["mute/set"] = "mute/set"


ServerMessages = (
    SessionStartMessage
    | SourceHelloMessage
    | SessionEndMessage
    | MetadataUpdateMessage
    | SourceTimeMessage
    | VolumeSetMessage
    | MuteSetMessage
)
