"""Helpers for clients supporting the metadata role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aioresonate.models.metadata import SessionUpdateMetadata
from aioresonate.models.types import RepeatMode

if TYPE_CHECKING:
    from aioresonate.models.metadata import ClientHelloMetadataSupport

    from .client import ResonateClient


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
    track_duration: int | None = None
    """Track duration in seconds."""
    playback_speed: int | None = 1
    """Speed factor, 1.0 is normal speed."""
    repeat: RepeatMode | None = None
    """Current repeat mode."""
    shuffle: bool | None = None
    """Whether shuffle is enabled."""

    # TODO: inject track_progress and timestamp when sending updates?

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
        if last is None or last.track_duration != self.track_duration:
            metadata_update.track_duration = self.track_duration
        if last is None or last.playback_speed != self.playback_speed:
            metadata_update.playback_speed = self.playback_speed
        if last is None or last.repeat != self.repeat:
            metadata_update.repeat = self.repeat
        if last is None or last.shuffle != self.shuffle:
            metadata_update.shuffle = self.shuffle

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
        metadata_update.track_duration = None
        metadata_update.playback_speed = None
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
        metadata_update.track_duration = self.track_duration
        metadata_update.playback_speed = self.playback_speed
        metadata_update.repeat = self.repeat
        metadata_update.shuffle = self.shuffle
        return metadata_update


class MetadataClient:
    """Expose metadata capabilities reported by the client."""

    def __init__(self, client: ResonateClient) -> None:
        """Attach to a client that exposes metadata capabilities."""
        self.client = client
        self._logger = client._logger.getChild("metadata")  # noqa: SLF001

    @property
    def support(self) -> ClientHelloMetadataSupport | None:
        """Return metadata capabilities advertised in the hello payload."""
        return self.client.info.metadata_support
