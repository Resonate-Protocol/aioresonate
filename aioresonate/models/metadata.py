"""Metadata messages for the Resonate protocol.

This module contains messages specific to clients with the metadata role, which
handle display of track information, artwork, and playback state. Metadata clients
receive session updates with track details and can optionally receive artwork in
their preferred format and resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import RepeatMode, ServerMessage, UndefinedField, undefined_field


# Client -> Server: client/hello metadata support object
@dataclass
class ClientHelloMetadataSupport(DataClassORJSONMixin):
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


# Server -> Client: stream/start metadata object
@dataclass
class StreamStartMetadata(DataClassORJSONMixin):
    """Metadata object in stream/start message.

    Sent to clients that specified supported picture formats.
    """

    art_format: Literal["bmp", "jpeg", "png"]
    """Format of the encoded image."""


# Server -> Client: stream/update metadata object
@dataclass
class StreamUpdateMetadata(DataClassORJSONMixin):
    """Metadata object in stream/update message with delta updates."""

    art_format: Literal["bmp", "jpeg", "png"]
    """Format of the encoded image."""


# Server -> Client: session/update metadata object
@dataclass
class SessionUpdateMetadata(DataClassORJSONMixin):
    """Metadata object in session/update message."""

    timestamp: int
    """Server timestamp for when this metadata is valid."""
    title: str | None | UndefinedField = field(default_factory=undefined_field)
    artist: str | None | UndefinedField = field(default_factory=undefined_field)
    album_artist: str | None | UndefinedField = field(default_factory=undefined_field)
    album: str | None | UndefinedField = field(default_factory=undefined_field)
    artwork_url: str | None | UndefinedField = field(default_factory=undefined_field)
    year: int | None | UndefinedField = field(default_factory=undefined_field)
    track: int | None | UndefinedField = field(default_factory=undefined_field)
    track_progress: float | None | UndefinedField = field(default_factory=undefined_field)
    """Track progress in seconds."""
    track_duration: float | None | UndefinedField = field(default_factory=undefined_field)
    """Track duration in seconds."""
    playback_speed: float | None | UndefinedField = field(default_factory=undefined_field)
    """Speed factor."""
    repeat: RepeatMode | None | UndefinedField = field(default_factory=undefined_field)
    shuffle: bool | None | UndefinedField = field(default_factory=undefined_field)

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_default = True


@dataclass
class SessionUpdatePayload(DataClassORJSONMixin):
    """Delta updates for session state."""

    group_id: str
    """Group identifier."""
    playback_state: Literal["playing", "paused", "stopped"] | None = None
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
