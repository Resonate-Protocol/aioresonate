"""MetadataGroupRole - group-level metadata coordination."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from aiosendspin.models.core import ServerStateMessage, ServerStatePayload
from aiosendspin.models.metadata import Progress, SessionUpdateMetadata
from aiosendspin.models.types import UndefinedField
from aiosendspin.server.roles.base import GroupRole, Role
from aiosendspin.server.roles.metadata.events import MetadataClearedEvent, MetadataUpdatedEvent
from aiosendspin.server.roles.metadata.state import Metadata
from aiosendspin.server.roles.scheduled_state import ScheduledRoleState

if TYPE_CHECKING:
    from aiosendspin.server.group import SendspinGroup

_UNSET = object()


class MetadataGroupRole(GroupRole):
    """Coordinate metadata across a group.

    Stores current metadata state and pushes updates to subscribed MetadataRoles.
    """

    role_family = "metadata"

    def __init__(self, group: SendspinGroup) -> None:
        """Initialize MetadataGroupRole."""
        super().__init__(group)
        self._state: ScheduledRoleState[Metadata, SessionUpdateMetadata] = ScheduledRoleState(
            self._on_state_commit
        )
        self._track_progress_timestamp_us: int | None = None

    @property
    def metadata(self) -> Metadata | None:
        """Return current metadata."""
        return self._state.current(self._now_us())

    def on_member_join(self, role: Role) -> None:
        """Send current metadata to newly joined member."""
        self._send_state_to_role(role)

    def _send_state_to_role(self, role: Role) -> None:
        """Send current metadata state to a single role."""
        # TODO: refactor to guard clause: if metadata is None, send clear and return
        timestamp = self._now_us()
        current = self._state.current(timestamp)

        if current is not None:
            metadata_update = current.snapshot_update(timestamp)
            current_progress = self._get_track_progress_at(current, timestamp)
            if (
                current_progress is not None
                and current.track_duration is not None
                and current.playback_speed is not None
            ):
                metadata_update.progress = Progress(
                    track_progress=current_progress,
                    track_duration=current.track_duration,
                    playback_speed=current.playback_speed,
                )
        else:
            metadata_update = Metadata.cleared_update(timestamp)

        state_message = ServerStateMessage(ServerStatePayload(metadata=metadata_update))
        role.send_message(state_message)
        if (pending_update := self._state.pending_update) is not None:
            role.send_message(ServerStateMessage(ServerStatePayload(metadata=pending_update)))

    def _get_current_track_progress(self) -> int | None:
        """Calculate current track progress in milliseconds."""
        now_us = self._now_us()
        return self._get_track_progress_at(self._state.current(now_us), now_us)

    def _get_track_progress_at(self, metadata: Metadata | None, timestamp_us: int) -> int | None:
        if metadata is None or metadata.track_progress is None:
            return None

        if (
            self._track_progress_timestamp_us is not None
            and self._group.has_active_stream
            and metadata.playback_speed is not None
        ):
            elapsed_us = timestamp_us - self._track_progress_timestamp_us
            elapsed_ms = (elapsed_us * metadata.playback_speed) // 1_000_000
            calculated_progress = metadata.track_progress + elapsed_ms

            if metadata.track_duration is not None and metadata.track_duration > 0:
                calculated_progress = max(0, min(calculated_progress, metadata.track_duration))
            else:
                calculated_progress = max(0, calculated_progress)

            return calculated_progress

        return metadata.track_progress

    def freeze_progress(self) -> None:
        """Snapshot current progress and stop further client-side progress extrapolation."""
        now_us = self._now_us()
        metadata = self._state.current(now_us)
        if metadata is None:
            return

        current_progress = self._get_track_progress_at(metadata, now_us)
        if current_progress is None:
            if self._state.has_pending:
                self.set_metadata(replace(metadata, timestamp_us=None))
            return

        self.set_metadata(
            replace(
                metadata,
                track_progress=current_progress,
                playback_speed=0,
                timestamp_us=None,
            )
        )

    def set_metadata(self, metadata: Metadata | None, *, timestamp_us: int | None = None) -> None:
        """Set or schedule metadata and push updates to all subscribed roles.

        Only sends updates for fields that have changed.
        """
        now_us = self._now_us()
        current = self._state.current(now_us)
        timestamp = now_us if timestamp_us is None else timestamp_us

        if metadata is not None:
            if timestamp_us is not None:
                metadata = replace(metadata, timestamp_us=timestamp_us)
            elif metadata.timestamp_us is None:
                metadata = replace(metadata, timestamp_us=timestamp)
            else:
                timestamp = metadata.timestamp_us

        if not self._state.has_pending and not self._state.scheduled_fields:
            if metadata is None and current is None:
                return
            if metadata is not None and metadata.equals(current):
                return

        last_metadata = current
        scheduled_fields = self._state.scheduled_fields
        if metadata is None:
            metadata_update = Metadata.cleared_update(timestamp)
        else:
            metadata_update = metadata.diff_update(
                last_metadata, timestamp, include=scheduled_fields - {"progress"}
            )
            if "progress" in scheduled_fields and isinstance(
                metadata_update.progress, UndefinedField
            ):
                metadata_update.progress = self._scheduled_progress_value(
                    last_metadata, metadata, timestamp
                )

        if timestamp > now_us:
            self._state.schedule(
                metadata, metadata_update, timestamp, set(metadata_update.to_dict()) - {"timestamp"}
            )
        else:
            self._state.apply(metadata, timestamp)

        for role in self._members:
            state_message = ServerStateMessage(ServerStatePayload(metadata=metadata_update))
            role.send_message(state_message)

        if metadata is None:
            self.emit_group_event(
                MetadataClearedEvent(previous_metadata=last_metadata, timestamp_us=timestamp)
            )
            return
        self.emit_group_event(
            MetadataUpdatedEvent(
                metadata=metadata,
                previous_metadata=last_metadata,
                timestamp_us=timestamp,
            )
        )

    def _on_state_commit(self, metadata: Metadata | None, timestamp_us: int) -> None:
        self._track_progress_timestamp_us = (
            timestamp_us if metadata is not None and metadata.track_progress is not None else None
        )

    def _scheduled_progress_value(
        self,
        last_metadata: Metadata | None,
        metadata: Metadata,
        timestamp_us: int,
    ) -> Progress | None:
        """Progress a scheduled diff must restate when it would otherwise omit it."""
        progress = self._get_track_progress_at(last_metadata, timestamp_us)
        if (
            progress is not None
            and metadata.track_duration is not None
            and metadata.playback_speed is not None
        ):
            return Progress(
                track_progress=progress,
                track_duration=metadata.track_duration,
                playback_speed=metadata.playback_speed,
            )
        if (
            metadata.track_progress is not None
            and metadata.track_duration is not None
            and metadata.playback_speed is not None
        ):
            return Progress(
                track_progress=metadata.track_progress,
                track_duration=metadata.track_duration,
                playback_speed=metadata.playback_speed,
            )
        return None

    def update(
        self,
        *,
        title: str | None | object = _UNSET,
        artist: str | None | object = _UNSET,
        album_artist: str | None | object = _UNSET,
        album: str | None | object = _UNSET,
        artwork_url: str | None | object = _UNSET,
        year: int | None | object = _UNSET,
        track: int | None | object = _UNSET,
        track_progress: int | None | object = _UNSET,
        track_duration: int | None | object = _UNSET,
        playback_speed: int | None | object = _UNSET,
    ) -> None:
        """Batch update multiple metadata fields.

        Fields set to `_UNSET` are left unchanged. Passing `None` clears a field.
        """
        current = self.metadata or Metadata()
        kwargs: dict[str, object] = {}
        if title is not _UNSET:
            kwargs["title"] = title
        if artist is not _UNSET:
            kwargs["artist"] = artist
        if album_artist is not _UNSET:
            kwargs["album_artist"] = album_artist
        if album is not _UNSET:
            kwargs["album"] = album
        if artwork_url is not _UNSET:
            kwargs["artwork_url"] = artwork_url
        if year is not _UNSET:
            kwargs["year"] = year
        if track is not _UNSET:
            kwargs["track"] = track
        if track_progress is not _UNSET:
            kwargs["track_progress"] = track_progress
        if track_duration is not _UNSET:
            kwargs["track_duration"] = track_duration
        if playback_speed is not _UNSET:
            kwargs["playback_speed"] = playback_speed

        if not kwargs:
            return

        new_metadata = replace(current, **kwargs)  # type: ignore[arg-type]
        self.set_metadata(new_metadata)

    def clear(self, *, timestamp_us: int | None = None) -> None:
        """Clear all metadata."""
        self.set_metadata(None, timestamp_us=timestamp_us)
