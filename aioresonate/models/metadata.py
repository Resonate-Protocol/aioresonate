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

from .core import ServerMessage
from .types import RepeatMode


# Server → Client metadata messages
@dataclass
class SessionMetadataServerPayload(DataClassORJSONMixin):
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
