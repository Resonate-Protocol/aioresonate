"""MetadataGroupRole - group-level metadata coordination."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from aiosendspin.models.core import ServerStateMessage, ServerStatePayload
from aiosendspin.models.metadata import Progress
from aiosendspin.models.types import RepeatMode
from aiosendspin.server.roles.base import GroupRole, Role
from aiosendspin.server.roles.metadata.state import Metadata

if TYPE_CHECKING:
    from aiosendspin.server.group import SendspinGroup


class MetadataGroupRole(GroupRole):
    """Coordinate metadata across a group.

    Stores current metadata state and pushes updates to subscribed MetadataRoles.
    """

    role_family = "metadata"

    def __init__(self, group: SendspinGroup) -> None:
        """Initialize MetadataGroupRole."""
        super().__init__(group)
        self._current_metadata: Metadata | None = None
        self._track_progress_timestamp_us: int | None = None

    @property
    def metadata(self) -> Metadata | None:
        """Return current metadata."""
        return self._current_metadata

    def on_member_join(self, role: Role) -> None:
        """Send current metadata to newly joined member."""
        self._send_state_to_role(role)

    def _send_state_to_role(self, role: Role) -> None:
        """Send current metadata state to a single role."""
        # TODO: refactor to guard clause: if metadata is None, send clear and return
        timestamp = self._group._server.clock.now_us()  # noqa: SLF001

        if self._current_metadata is not None:
            metadata_update = self._current_metadata.snapshot_update(timestamp)
            current_progress = self._get_current_track_progress()
            if (
                current_progress is not None
                and self._current_metadata.track_duration is not None
                and self._current_metadata.playback_speed is not None
            ):
                metadata_update.progress = Progress(
                    track_progress=current_progress,
                    track_duration=self._current_metadata.track_duration,
                    playback_speed=self._current_metadata.playback_speed,
                )
        else:
            metadata_update = Metadata.cleared_update(timestamp)

        state_message = ServerStateMessage(ServerStatePayload(metadata=metadata_update))
        role.send_message(state_message)

    def _get_current_track_progress(self) -> int | None:
        """Calculate current track progress in milliseconds."""
        if self._current_metadata is None or self._current_metadata.track_progress is None:
            return None

        if (
            self._track_progress_timestamp_us is not None
            and self._group.has_active_stream
            and self._current_metadata.playback_speed is not None
        ):
            current_time_us = self._group._server.clock.now_us()  # noqa: SLF001
            elapsed_us = current_time_us - self._track_progress_timestamp_us
            elapsed_ms = (elapsed_us * self._current_metadata.playback_speed) // 1_000_000
            calculated_progress = self._current_metadata.track_progress + elapsed_ms

            if (
                self._current_metadata.track_duration is not None
                and self._current_metadata.track_duration > 0
            ):
                calculated_progress = max(
                    0, min(calculated_progress, self._current_metadata.track_duration)
                )
            else:
                calculated_progress = max(0, calculated_progress)

            return calculated_progress

        return self._current_metadata.track_progress

    def set_metadata(self, metadata: Metadata | None) -> None:
        """Set metadata and push updates to all subscribed roles.

        Only sends updates for fields that have changed.
        """
        timestamp = self._group._server.clock.now_us()  # noqa: SLF001

        if metadata is not None:
            if metadata.timestamp_us is None:
                metadata = replace(metadata, timestamp_us=timestamp)
            else:
                timestamp = metadata.timestamp_us

        if metadata is not None and metadata.equals(self._current_metadata):
            return

        last_metadata = self._current_metadata
        if metadata is None:
            metadata_update = Metadata.cleared_update(timestamp)
        else:
            metadata_update = metadata.diff_update(last_metadata, timestamp)

        self._current_metadata = metadata

        if metadata is not None and metadata.track_progress is not None:
            self._track_progress_timestamp_us = timestamp

        for role in self._members:
            state_message = ServerStateMessage(ServerStatePayload(metadata=metadata_update))
            role.send_message(state_message)

    # TODO: consider single method with optional kwargs instead of split methods
    def set_title(self, title: str | None) -> None:
        """Update title field."""
        self._update_field("title", title)

    def set_artist(self, artist: str | None) -> None:
        """Update artist field."""
        self._update_field("artist", artist)

    def set_album_artist(self, album_artist: str | None) -> None:
        """Update album_artist field."""
        self._update_field("album_artist", album_artist)

    def set_album(self, album: str | None) -> None:
        """Update album field."""
        self._update_field("album", album)

    def set_artwork_url(self, url: str | None) -> None:
        """Update artwork_url field."""
        self._update_field("artwork_url", url)

    def set_year(self, year: int | None) -> None:
        """Update year field."""
        self._update_field("year", year)

    def set_track(self, track: int | None) -> None:
        """Update track field."""
        self._update_field("track", track)

    def set_repeat(self, mode: RepeatMode | None) -> None:
        """Update repeat mode."""
        self._update_field("repeat", mode)

    def set_shuffle(self, shuffle: bool | None) -> None:  # noqa: FBT001
        """Update shuffle state."""
        self._update_field("shuffle", shuffle)

    def set_progress(
        self,
        track_progress_ms: int,
        track_duration_ms: int,
        playback_speed: int = 1000,
    ) -> None:
        """Update progress fields.

        Args:
            track_progress_ms: Current track progress in milliseconds.
            track_duration_ms: Track duration in milliseconds (0 for live streams).
            playback_speed: Playback speed * 1000 (e.g., 1000 = 1x, 1500 = 1.5x, 0 = paused).
        """
        current = self._current_metadata or Metadata()
        new_metadata = replace(
            current,
            track_progress=track_progress_ms,
            track_duration=track_duration_ms,
            playback_speed=playback_speed,
        )
        self.set_metadata(new_metadata)

    # TODO: use SENTINEL pattern to support partial updates (see HA entity registry)
    def update(  # noqa: PLR0913
        self,
        *,
        title: str | None = None,
        artist: str | None = None,
        album_artist: str | None = None,
        album: str | None = None,
        artwork_url: str | None = None,
        year: int | None = None,
        track: int | None = None,
        repeat: RepeatMode | None = None,
        shuffle: bool | None = None,
        track_progress: int | None = None,
        track_duration: int | None = None,
        playback_speed: int | None = None,
    ) -> None:
        """Batch update multiple metadata fields.

        Only fields with non-None values will be updated.
        """
        current = self._current_metadata or Metadata()
        new_metadata = Metadata(
            title=title if title is not None else current.title,
            artist=artist if artist is not None else current.artist,
            album_artist=album_artist if album_artist is not None else current.album_artist,
            album=album if album is not None else current.album,
            artwork_url=artwork_url if artwork_url is not None else current.artwork_url,
            year=year if year is not None else current.year,
            track=track if track is not None else current.track,
            repeat=repeat if repeat is not None else current.repeat,
            shuffle=shuffle if shuffle is not None else current.shuffle,
            track_progress=track_progress if track_progress is not None else current.track_progress,
            track_duration=track_duration if track_duration is not None else current.track_duration,
            playback_speed=playback_speed if playback_speed is not None else current.playback_speed,
        )
        self.set_metadata(new_metadata)

    def clear(self) -> None:
        """Clear all metadata."""
        self.set_metadata(None)

    def _update_field(self, field: str, value: object) -> None:
        """Update a single metadata field."""
        current = self._current_metadata or Metadata()
        # Use explicit field mapping to satisfy type checker
        kwargs: dict[str, object] = {field: value}
        new_metadata = replace(current, **kwargs)  # type: ignore[arg-type]
        self.set_metadata(new_metadata)
