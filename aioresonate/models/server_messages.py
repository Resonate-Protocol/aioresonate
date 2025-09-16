"""Models for messages sent by the server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin
from mashumaro.types import Discriminator

from .types import RepeatMode


@dataclass
class StreamStartPlayerPayload(DataClassORJSONMixin):
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
class StreamStartMetadataPayload(DataClassORJSONMixin):
    """Metadata object in stream/start message.

    Sent to clients that specified supported picture formats.
    """

    art_format: Literal["bmp", "jpeg", "png"]
    """Format of the encoded image."""


@dataclass
class ServerMessage(DataClassORJSONMixin):
    """Server Message type used by resonate."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclass
class StreamStartPayload(DataClassORJSONMixin):
    """Information about an active streaming session."""

    player: StreamStartPlayerPayload | None = None
    """Information about the player."""
    metadata: StreamStartMetadataPayload | None = None
    """Metadata information (sent to clients that specified supported picture formats)."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamStartMessage(ServerMessage):
    """Message sent by the server to start a stream."""

    payload: StreamStartPayload
    type: Literal["stream/start"] = "stream/start"


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


@dataclass
class StreamEndMessage(ServerMessage):
    """Message sent by the server to end a stream."""

    type: Literal["stream/end"] = "stream/end"


@dataclass
class StreamUpdatePlayerPayload(DataClassORJSONMixin):
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
class StreamUpdateMetadataPayload(DataClassORJSONMixin):
    """Metadata object in stream/update message with delta updates."""

    art_format: Literal["bmp", "jpeg", "png"] | None = None
    """Format of the encoded image."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamUpdatePayload(DataClassORJSONMixin):
    """Delta updates for the ongoing stream."""

    player: StreamUpdatePlayerPayload | None = None
    """Player updates."""
    metadata: StreamUpdateMetadataPayload | None = None
    """Metadata updates."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamUpdateMessage(ServerMessage):
    """Message sent by the server to update stream format."""

    payload: StreamUpdatePayload
    type: Literal["stream/update"] = "stream/update"


@dataclass
class SessionMetadataPayload(DataClassORJSONMixin):
    """Metadata object in session/update message."""

    timestamp: int
    """Server timestamp for when this metadata is valid."""
    title: str | None = None
    artist: str | None = None
    album_artist: str | None = None
    album: str | None = None
    artwork_url: str | None = None
    year: int | None = None
    track: int | None = None
    track_progress: float | None = None
    """Track progress in seconds."""
    track_duration: float | None = None
    """Track duration in seconds."""
    playback_speed: float | None = None
    """Speed factor."""
    repeat: RepeatMode | None = None
    shuffle: bool | None = None

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class SessionUpdatePayload(DataClassORJSONMixin):
    """Delta updates for session state."""

    group_id: str
    """Group identifier."""
    playback_state: Literal["playing", "paused", "stopped"] | None = None
    """Only sent to clients with controller or metadata roles."""
    metadata: SessionMetadataPayload | None = None
    """Only sent to clients with metadata role."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class SessionUpdateMessage(ServerMessage):
    """Message sent by the server to update session state."""

    payload: SessionUpdatePayload
    type: Literal["session/update"] = "session/update"


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


@dataclass
class GroupMember(DataClassORJSONMixin):
    """Represents a group member."""

    client_id: str
    """Client identifier."""
    name: str
    """Client friendly name."""


@dataclass
class GroupInfo(DataClassORJSONMixin):
    """Information about a group."""

    group_id: str
    """Group identifier."""
    name: str
    """Group name."""
    state: Literal["playing", "paused", "idle"]
    """Group state."""
    member_count: int
    """Number of clients in group."""


@dataclass
class GroupListPayload(DataClassORJSONMixin):
    """All groups available to join on the server."""

    groups: list[GroupInfo]
    """List of available groups."""


@dataclass
class GroupListMessage(ServerMessage):
    """Message sent by the server with list of available groups."""

    payload: GroupListPayload
    type: Literal["group/list"] = "group/list"


@dataclass
class GroupUpdatePayload(DataClassORJSONMixin):
    """Group state update."""

    supported_commands: list[str]
    """Subset of: play, pause, stop, next, previous, seek, volume, mute."""
    members: list[GroupMember]
    """List of group members."""
    session_id: str | None
    """Null if no active session."""
    volume: int
    """Volume range 0-100."""
    muted: bool
    """Mute state."""


@dataclass
class GroupUpdateMessage(ServerMessage):
    """Message sent by the server to update group state."""

    payload: GroupUpdatePayload
    type: Literal["group/update"] = "group/update"
