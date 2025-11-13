"""
Metadata messages for the Resonate protocol.

This module contains messages specific to clients with the metadata role, which
handle display of track information and playback progress. Metadata clients
receive state updates with track details.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import RepeatMode, UndefinedField, undefined_field


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
    track_progress: int | None | UndefinedField = field(default_factory=undefined_field)
    """Track progress in milliseconds, since start of track, at the given timestamp."""
    track_duration: int | None | UndefinedField = field(default_factory=undefined_field)
    """Track duration in milliseconds."""
    playback_speed: int | None | UndefinedField = field(default_factory=undefined_field)
    """Playback speed multiplier * 1000."""
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
