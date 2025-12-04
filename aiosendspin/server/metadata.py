"""Helpers for clients supporting the metadata role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiosendspin.models.metadata import Progress, SessionUpdateMetadata
from aiosendspin.models.types import RepeatMode

if TYPE_CHECKING:
    from .client import SendspinClient


@dataclass
class Metadata:
    """Metadata for media playback."""

    title: str | None = None
    """Title of the current media."""
    artist: str | None = None
    """Artist of the current media."""
    album_artist: str | None = None
    """Album artist of the current media."""
    album: str | None = None
    """Album of the current media."""
    artwork_url: str | None = None
    """Artwork URL of the current media."""
    year: int | None = None
    """Release year of the current media."""
    track: int | None = None
    """Track number of the current media."""
    repeat: RepeatMode | None = None
    """Current repeat mode."""
    shuffle: bool | None = None
    """Whether shuffle is enabled."""

    # Progress fields:
    # When sending to clients, all three fields must be set or none will be sent
    track_progress: int | None = None
    """Track progress in milliseconds at the last update time."""
    track_duration: int | None = None
    """Track duration in milliseconds. Use 0 for unlimited/unknown duration (e.g., live streams)."""
    playback_speed: int | None = None
    """Playback speed multiplier * 1000 (e.g., 1000 = normal, 1500 = 1.5x, 0 = paused)."""

    timestamp_us: int | None = None
    """
    Timestamp in microseconds when this metadata was captured.

    You don't need to set this, since it will be set automatically by set_metadata() if not
    provided.
    """

    def equals(self, other: Metadata | None, progress_tolerance_ms: int = 500) -> bool:
        """
        Check if metadata is meaningfully equal.

        Args:
            other: The other Metadata object to compare with.
            progress_tolerance_ms: Tolerance in milliseconds for track progress comparison.

        Returns:
            True if metadata is meaningfully equal, False otherwise.
        """
        if other is None:
            return False

        # Compare all non-progress fields
        if not (
            self.title == other.title
            and self.artist == other.artist
            and self.album_artist == other.album_artist
            and self.album == other.album
            and self.artwork_url == other.artwork_url
            and self.year == other.year
            and self.track == other.track
            and self.track_duration == other.track_duration
            and self.playback_speed == other.playback_speed
            and self.repeat == other.repeat
            and self.shuffle == other.shuffle
        ):
            return False

        # If both have no progress info, they're equal
        if self.track_progress is None and other.track_progress is None:
            return True

        # If only one has progress info, they're different
        if self.track_progress is None or other.track_progress is None:
            return False

        # If we don't have timestamps, fall back to simple tolerance check
        if self.timestamp_us is None or other.timestamp_us is None:
            return abs(self.track_progress - other.track_progress) <= progress_tolerance_ms

        # Calculate expected progress change based on elapsed time and playback speed
        time_diff_ms = (other.timestamp_us - self.timestamp_us) / 1000
        playback_speed = (self.playback_speed or 1000) / 1000  # Convert to float multiplier
        expected_progress_change = time_diff_ms * playback_speed

        # Calculate actual progress change
        actual_progress_change = other.track_progress - self.track_progress

        # Check if the difference between expected and actual is within tolerance
        progress_drift = abs(actual_progress_change - expected_progress_change)
        return progress_drift <= progress_tolerance_ms

    def diff_update(self, last: Metadata | None, timestamp: int) -> SessionUpdateMetadata:
        """Build a SessionUpdateMetadata containing only changed fields compared to last."""
        metadata_update = SessionUpdateMetadata(timestamp=timestamp)

        # Only include fields that have changed since the last metadata update
        if last is None or last.title != self.title:
            metadata_update.title = self.title
        if last is None or last.artist != self.artist:
            metadata_update.artist = self.artist
        if last is None or last.album_artist != self.album_artist:
            metadata_update.album_artist = self.album_artist
        if last is None or last.album != self.album:
            metadata_update.album = self.album
        if last is None or last.artwork_url != self.artwork_url:
            metadata_update.artwork_url = self.artwork_url
        if last is None or last.year != self.year:
            metadata_update.year = self.year
        if last is None or last.track != self.track:
            metadata_update.track = self.track
        if last is None or last.repeat != self.repeat:
            metadata_update.repeat = self.repeat
        if last is None or last.shuffle != self.shuffle:
            metadata_update.shuffle = self.shuffle

        # Send progress object if any progress field changed or if track_progress is set
        # (clients need fresh timestamp for progress calculation)
        progress_changed = (
            last is None
            or last.track_duration != self.track_duration
            or last.playback_speed != self.playback_speed
            or self.track_progress is not None
        )
        if (
            progress_changed
            and self.track_progress is not None
            and self.track_duration is not None
            and self.playback_speed is not None
        ):
            metadata_update.progress = Progress(
                track_progress=self.track_progress,
                track_duration=self.track_duration,
                playback_speed=self.playback_speed,
            )

        return metadata_update

    @staticmethod
    def cleared_update(timestamp: int) -> SessionUpdateMetadata:
        """Build a SessionUpdateMetadata that clears all metadata fields."""
        metadata_update = SessionUpdateMetadata(timestamp=timestamp)
        metadata_update.title = None
        metadata_update.artist = None
        metadata_update.album_artist = None
        metadata_update.album = None
        metadata_update.artwork_url = None
        metadata_update.year = None
        metadata_update.track = None
        metadata_update.progress = None
        metadata_update.repeat = None
        metadata_update.shuffle = None
        return metadata_update

    def snapshot_update(self, timestamp: int) -> SessionUpdateMetadata:
        """Build a SessionUpdateMetadata snapshot with all current values."""
        metadata_update = SessionUpdateMetadata(timestamp=timestamp)
        metadata_update.title = self.title
        metadata_update.artist = self.artist
        metadata_update.album_artist = self.album_artist
        metadata_update.album = self.album
        metadata_update.artwork_url = self.artwork_url
        metadata_update.year = self.year
        metadata_update.track = self.track
        metadata_update.repeat = self.repeat
        metadata_update.shuffle = self.shuffle
        # Build progress object if all progress fields are set
        if (
            self.track_progress is not None
            and self.track_duration is not None
            and self.playback_speed is not None
        ):
            metadata_update.progress = Progress(
                track_progress=self.track_progress,
                track_duration=self.track_duration,
                playback_speed=self.playback_speed,
            )
        return metadata_update


class MetadataClient:
    """Expose metadata capabilities reported by the client."""

    def __init__(self, client: SendspinClient) -> None:
        """Attach to a client that exposes metadata capabilities."""
        self.client = client
        self._logger = client._logger.getChild("metadata")  # noqa: SLF001
