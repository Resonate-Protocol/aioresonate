"""
Metadata messages for the Sendspin protocol.

This module contains messages specific to clients with the metadata role, which
handle display of track information and playback progress. Metadata clients
receive state updates with track details.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import RepeatMode, UndefinedField, undefined_field


@dataclass
class Progress(DataClassORJSONMixin):
    """Playback progress information."""

    track_progress: int
    """Track progress in milliseconds, since start of track."""
    track_duration: int
    """Track duration in milliseconds. 0 for unlimited/unknown duration (e.g., live streams)."""
    playback_speed: int
    """Playback speed multiplier * 1000 (e.g., 1000 = normal, 1500 = 1.5x, 0 = paused)."""

    def __post_init__(self) -> None:
        """Validate field values."""
        # Validate track_progress is non-negative
        if self.track_progress < 0:
            raise ValueError(f"track_progress must be non-negative, got {self.track_progress}")

        # Validate track_duration is non-negative (0 allowed for live streams)
        if self.track_duration < 0:
            raise ValueError(f"track_duration must be non-negative, got {self.track_duration}")

        # Validate playback_speed is non-negative
        if self.playback_speed < 0:
            raise ValueError(f"playback_speed must be non-negative, got {self.playback_speed}")

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_default = True


# Server -> Client: server/state metadata object
@dataclass
class SessionUpdateMetadata(DataClassORJSONMixin):
    """Metadata object in server/state message."""

    timestamp: int
    """Server clock time in microseconds for when this metadata is valid."""
    title: str | None | UndefinedField = field(default_factory=undefined_field)
    artist: str | None | UndefinedField = field(default_factory=undefined_field)
    album_artist: str | None | UndefinedField = field(default_factory=undefined_field)
    album: str | None | UndefinedField = field(default_factory=undefined_field)
    artwork_url: str | None | UndefinedField = field(default_factory=undefined_field)
    year: int | None | UndefinedField = field(default_factory=undefined_field)
    track: int | None | UndefinedField = field(default_factory=undefined_field)
    progress: Progress | None | UndefinedField = field(default_factory=undefined_field)
    """
    Playback progress information.

    The server must send this object whenever playback state changes.
    """
    repeat: RepeatMode | None | UndefinedField = field(default_factory=undefined_field)
    shuffle: bool | None | UndefinedField = field(default_factory=undefined_field)

    def __post_init__(self) -> None:
        """Validate field values."""
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
