"""Metadata messages for the Resonate protocol.

This module contains messages specific to clients with the metadata role, which
handle display of track information, artwork, and playback state. Metadata clients
receive session updates with track details and can optionally receive artwork in
their preferred format and resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import PictureFormat, RepeatMode, UndefinedField, undefined_field


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

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.media_width is not None and self.media_width <= 0:
            raise ValueError(f"media_width must be positive, got {self.media_width}")

        if self.media_height is not None and self.media_height <= 0:
            raise ValueError(f"media_height must be positive, got {self.media_height}")

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


# Server -> Client: stream/start metadata object
@dataclass
class StreamStartMetadata(DataClassORJSONMixin):
    """Metadata object in stream/start message.

    Sent to clients that specified supported picture formats.
    """

    art_format: PictureFormat
    """Format of the encoded image."""


# Server -> Client: stream/update metadata object
@dataclass
class StreamUpdateMetadata(DataClassORJSONMixin):
    """Metadata object in stream/update message with delta updates."""

    art_format: PictureFormat
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
    track_progress: int | None | UndefinedField = field(default_factory=undefined_field)
    """Track progress in seconds."""
    track_duration: int | None | UndefinedField = field(default_factory=undefined_field)
    """Track duration in seconds."""
    playback_speed: int | None | UndefinedField = field(default_factory=undefined_field)
    """Speed factor."""
    repeat: RepeatMode | None | UndefinedField = field(default_factory=undefined_field)
    shuffle: bool | None | UndefinedField = field(default_factory=undefined_field)

    def __post_init__(self) -> None:
        """Validate field values."""
        # Validate track_progress is non-negative
        if (
            not isinstance(self.track_progress, UndefinedField)
            and self.track_progress is not None
            and self.track_progress < 0
        ):
            raise ValueError(f"track_progress must be non-negative, got {self.track_progress}")

        # Validate track_duration is positive
        if (
            not isinstance(self.track_duration, UndefinedField)
            and self.track_duration is not None
            and self.track_duration <= 0
        ):
            raise ValueError(f"track_duration must be positive, got {self.track_duration}")

        # Validate playback_speed is positive
        if (
            not isinstance(self.playback_speed, UndefinedField)
            and self.playback_speed is not None
            and self.playback_speed <= 0
        ):
            raise ValueError(f"playback_speed must be positive, got {self.playback_speed}")

        # Validate year is reasonable (between 1000 and current year + 10)
        if (
            not isinstance(self.year, UndefinedField)
            and self.year is not None
            and not (1000 <= self.year <= 2040)
        ):
            raise ValueError(f"year must be between 1000 and 2040, got {self.year}")

        # Validate track number is positive
        if (
            not isinstance(self.track, UndefinedField)
            and self.track is not None
            and self.track <= 0
        ):
            raise ValueError(f"track must be positive, got {self.track}")

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_default = True
