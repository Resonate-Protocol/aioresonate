"""Metadata messages for the Resonate protocol.

This module contains messages specific to clients with the metadata role, which
handle display of track information, artwork, and playback state. Metadata clients
receive session updates with track details and can optionally receive artwork in
their preferred format and resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from . import ServerMessage
from .types import RepeatMode


# Client -> Server: client/hello metadata support object
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


# Server -> Client: stream/start metadata object
@dataclass
class StreamStartMetadataServerPayload(DataClassORJSONMixin):
    """Metadata object in stream/start message.

    Sent to clients that specified supported picture formats.
    """

    art_format: Literal["bmp", "jpeg", "png"]
    """Format of the encoded image."""


# Server -> Client: stream/update metadata object
@dataclass
class StreamUpdateMetadataServerPayload(DataClassORJSONMixin):
    """Metadata object in stream/update message with delta updates."""

    art_format: Literal["bmp", "jpeg", "png"]
    """Format of the encoded image."""


# Server -> Client: session/update metadata object
@dataclass
class SessionMetadataServerPayload(DataClassORJSONMixin):
    """Metadata object in session/update message."""

    timestamp: int
    """Server timestamp for when this metadata is valid."""
    title: str | None | Literal["__UNDEFINED__MARKER__"] = "__UNDEFINED__MARKER__"
    artist: str | None | Literal["__UNDEFINED__MARKER__"] = "__UNDEFINED__MARKER__"
    album_artist: str | None | Literal["__UNDEFINED__MARKER__"] = "__UNDEFINED__MARKER__"
    album: str | None | Literal["__UNDEFINED__MARKER__"] = "__UNDEFINED__MARKER__"
    artwork_url: str | None | Literal["__UNDEFINED__MARKER__"] = "__UNDEFINED__MARKER__"
    year: int | None | Literal["__UNDEFINED__MARKER__"] = "__UNDEFINED__MARKER__"
    track: int | None | Literal["__UNDEFINED__MARKER__"] = "__UNDEFINED__MARKER__"
    track_progress: float | None | Literal["__UNDEFINED__MARKER__"] = "__UNDEFINED__MARKER__"
    """Track progress in seconds."""
    track_duration: float | None | Literal["__UNDEFINED__MARKER__"] = "__UNDEFINED__MARKER__"
    """Track duration in seconds."""
    playback_speed: float | None | Literal["__UNDEFINED__MARKER__"] = "__UNDEFINED__MARKER__"
    """Speed factor."""
    repeat: RepeatMode | None | Literal["__UNDEFINED__MARKER__"] = "__UNDEFINED__MARKER__"
    shuffle: bool | None | Literal["__UNDEFINED__MARKER__"] = "__UNDEFINED__MARKER__"

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_default = True


@dataclass
class SessionUpdateServerPayload(DataClassORJSONMixin):
    """Delta updates for session state."""

    group_id: str
    """Group identifier."""
    playback_state: Literal["playing", "paused", "stopped"] | None = None
    """Only sent to clients with controller or metadata roles."""
    metadata: SessionMetadataServerPayload | None = None
    """Only sent to clients with metadata role."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class SessionUpdateServerMessage(ServerMessage):
    """Message sent by the server to update session state."""

    payload: SessionUpdateServerPayload
    type: Literal["session/update"] = "session/update"
