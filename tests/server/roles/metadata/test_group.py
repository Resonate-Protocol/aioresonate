"""Tests for MetadataGroupRole."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.models.core import ServerStateMessage
from aiosendspin.models.types import UndefinedField
from aiosendspin.server.roles.metadata import Metadata, MetadataClearedEvent, MetadataUpdatedEvent
from aiosendspin.server.roles.metadata.group import MetadataGroupRole


def _make_group_stub() -> MagicMock:
    """Create a mock group for testing."""
    group = MagicMock()
    group._server = MagicMock()  # noqa: SLF001
    group._server.clock.now_us.return_value = 1_000_000  # noqa: SLF001
    group.has_active_stream = False
    return group


def test_metadata_group_role_family() -> None:
    """MetadataGroupRole has role_family of 'metadata'."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    assert mgr.role_family == "metadata"


def test_metadata_group_role_initial_metadata_is_none() -> None:
    """Initial metadata is None."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    assert mgr.metadata is None


def test_metadata_group_role_set_metadata_stores_value() -> None:
    """set_metadata() stores the metadata."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    metadata = Metadata(title="Test Song", artist="Test Artist")
    mgr.set_metadata(metadata)

    assert mgr.metadata is not None
    assert mgr.metadata.title == "Test Song"
    assert mgr.metadata.artist == "Test Artist"
    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, MetadataUpdatedEvent)
    assert event.metadata.title == "Test Song"
    assert event.previous_metadata is None


def test_metadata_group_role_set_metadata_sends_to_members() -> None:
    """set_metadata() sends update to all subscribed members."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    member = MagicMock()
    mgr._members = [member]  # noqa: SLF001

    metadata = Metadata(title="Test Song")
    mgr.set_metadata(metadata)

    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.metadata is not None
    assert msg.payload.metadata.title == "Test Song"


def test_metadata_group_role_clear_metadata() -> None:
    """clear() sets metadata to None and sends clear update."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    member = MagicMock()
    mgr._members = [member]  # noqa: SLF001

    mgr.set_metadata(Metadata(title="Test"))
    member.reset_mock()

    mgr.clear()

    assert mgr.metadata is None
    member.send_message.assert_called_once()
    group._signal_event.assert_called()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, MetadataClearedEvent)


def test_metadata_group_role_clear_when_already_cleared_is_noop() -> None:
    """Clearing already-cleared metadata sends nothing and emits no event."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    member = MagicMock()
    mgr._members = [member]  # noqa: SLF001

    mgr.clear()

    member.send_message.assert_not_called()
    group._signal_event.assert_not_called()  # noqa: SLF001


def test_metadata_group_role_update_title() -> None:
    """update() updates only the title field."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    mgr.update(title="New Title")

    assert mgr.metadata is not None
    assert mgr.metadata.title == "New Title"


def test_metadata_group_role_update_artist() -> None:
    """update() updates only the artist field."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    mgr.update(artist="New Artist")

    assert mgr.metadata is not None
    assert mgr.metadata.artist == "New Artist"


def test_metadata_group_role_update_progress() -> None:
    """update() updates progress fields."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    mgr.update(track_progress=30000, track_duration=180000, playback_speed=1000)

    assert mgr.metadata is not None
    assert mgr.metadata.track_progress == 30000
    assert mgr.metadata.track_duration == 180000
    assert mgr.metadata.playback_speed == 1000


def test_metadata_group_role_update_batch() -> None:
    """update() can set multiple fields at once."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    mgr.update(title="Song", artist="Artist", year=2024)

    assert mgr.metadata is not None
    assert mgr.metadata.title == "Song"
    assert mgr.metadata.artist == "Artist"
    assert mgr.metadata.year == 2024


def test_metadata_group_role_update_can_clear_field_with_none() -> None:
    """update() should allow clearing a field via explicit None."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    mgr.set_metadata(Metadata(title="Song", artist="Artist"))

    mgr.update(title=None)

    assert mgr.metadata is not None
    assert mgr.metadata.title is None
    assert mgr.metadata.artist == "Artist"


def test_metadata_group_role_on_member_join_sends_current_state() -> None:
    """on_member_join() sends current metadata to new member."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    mgr.set_metadata(Metadata(title="Test Song"))

    new_member = MagicMock()
    mgr.on_member_join(new_member)

    new_member.send_message.assert_called_once()
    msg = new_member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.metadata is not None
    assert msg.payload.metadata.title == "Test Song"


def test_metadata_group_role_on_member_join_no_metadata() -> None:
    """on_member_join() sends cleared metadata when no metadata set."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    new_member = MagicMock()
    mgr.on_member_join(new_member)

    new_member.send_message.assert_called_once()
    msg = new_member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    # Cleared update has explicit None values
    assert msg.payload.metadata is not None


def test_metadata_group_role_skips_unchanged() -> None:
    """set_metadata() skips sending if metadata is equivalent."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    member = MagicMock()
    mgr._members = [member]  # noqa: SLF001

    metadata = Metadata(title="Test")
    mgr.set_metadata(metadata)
    member.reset_mock()

    # Set same metadata again
    same_metadata = Metadata(title="Test")
    mgr.set_metadata(same_metadata)

    # Should not have sent again
    member.send_message.assert_not_called()
    group._signal_event.assert_called_once()  # noqa: SLF001


def test_metadata_group_role_freeze_progress_snapshots_elapsed_position() -> None:
    """freeze_progress() should snapshot live progress and stop extrapolation."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    group.has_active_stream = True

    mgr.set_metadata(
        Metadata(
            title="Test",
            track_progress=30_000,
            track_duration=180_000,
            playback_speed=1000,
        )
    )

    group._server.clock.now_us.return_value = 11_000_000  # noqa: SLF001
    mgr.freeze_progress()

    assert mgr.metadata is not None
    assert mgr.metadata.track_progress == 40_000
    assert mgr.metadata.playback_speed == 0


def test_metadata_group_role_member_join_does_not_rewind_after_freeze() -> None:
    """Frozen progress should be sent unchanged after the stream becomes inactive."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    group.has_active_stream = True

    mgr.set_metadata(
        Metadata(
            title="Test",
            track_progress=30_000,
            track_duration=180_000,
            playback_speed=1000,
        )
    )

    group._server.clock.now_us.return_value = 11_000_000  # noqa: SLF001
    mgr.freeze_progress()
    group.has_active_stream = False

    new_member = MagicMock()
    mgr.on_member_join(new_member)

    msg = new_member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.metadata is not None
    assert msg.payload.metadata.progress is not None
    assert msg.payload.metadata.progress.track_progress == 40_000
    assert msg.payload.metadata.progress.playback_speed == 0


def test_future_metadata_keeps_current_and_replays_both_states() -> None:
    """A future update remains pending and is replayed after current state."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    mgr.set_metadata(Metadata(title="Current"))
    mgr.set_metadata(Metadata(title="Next", timestamp_us=2_000_000))

    assert mgr.metadata is not None
    assert mgr.metadata.title == "Current"

    member = MagicMock()
    mgr.on_member_join(member)

    messages = [call.args[0].payload.metadata for call in member.send_message.call_args_list]
    assert [message.timestamp for message in messages] == [1_000_000, 2_000_000]
    assert messages[0].title == "Current"
    assert messages[1].title == "Next"


def test_earlier_metadata_replaces_pending_without_committing_it() -> None:
    """An earlier timestamp discards the prior pending update."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    mgr.set_metadata(Metadata(title="Current"))
    mgr.set_metadata(Metadata(title="Later", timestamp_us=3_000_000))
    mgr.set_metadata(Metadata(title="Earlier", timestamp_us=2_000_000))

    assert mgr.metadata is not None
    assert mgr.metadata.title == "Current"
    assert mgr._pending_metadata is not None  # noqa: SLF001
    assert mgr._pending_metadata.title == "Earlier"  # noqa: SLF001


def test_later_metadata_commits_pending_before_storing_replacement() -> None:
    """A later timestamp commits pending before becoming the replacement."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    mgr.set_metadata(Metadata(title="Current"))
    mgr.set_metadata(Metadata(title="Pending", artist="One", timestamp_us=2_000_000))
    mgr.set_metadata(Metadata(title="Replacement", artist="One", timestamp_us=3_000_000))

    assert mgr.metadata is not None
    assert mgr.metadata.title == "Pending"
    assert mgr._pending_metadata is not None  # noqa: SLF001
    assert mgr._pending_metadata.title == "Replacement"  # noqa: SLF001


def test_present_metadata_cancels_pending_even_when_current_is_unchanged() -> None:
    """A present timestamp cancels pending without requiring changed fields."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    member = MagicMock()
    mgr._members = [member]  # noqa: SLF001
    mgr.set_metadata(Metadata(title="Current"))
    mgr.set_metadata(Metadata(title="Pending", timestamp_us=2_000_000))
    member.reset_mock()

    mgr.set_metadata(Metadata(title="Current"))

    assert mgr._pending_update is None  # noqa: SLF001
    update = member.send_message.call_args.args[0].payload.metadata
    assert update.timestamp == 1_000_000
    assert "title" not in update.to_dict()


def test_freeze_progress_uses_current_and_discards_pending() -> None:
    """Freezing snapshots confirmed progress and cancels pending metadata."""
    group = _make_group_stub()
    group.has_active_stream = True
    mgr = MetadataGroupRole(group)
    mgr.set_metadata(
        Metadata(
            title="Current",
            track_progress=1_000,
            track_duration=10_000,
            playback_speed=1000,
        )
    )
    mgr.set_metadata(
        Metadata(
            title="Pending",
            track_progress=0,
            track_duration=20_000,
            playback_speed=1000,
            timestamp_us=5_000_000,
        )
    )
    group._server.clock.now_us.return_value = 2_000_000  # noqa: SLF001

    mgr.freeze_progress()

    assert mgr.metadata is not None
    assert mgr.metadata.title == "Current"
    assert mgr.metadata.track_progress == 2_000
    assert mgr.metadata.playback_speed == 0
    assert mgr._pending_update is None  # noqa: SLF001


def test_equal_metadata_commits_pending_before_storing_replacement() -> None:
    """An equal timestamp commits pending before retaining the new update."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    mgr.set_metadata(Metadata(title="Current"))
    mgr.set_metadata(Metadata(title="Pending", timestamp_us=2_000_000))
    mgr.set_metadata(Metadata(title="Replacement", timestamp_us=2_000_000))

    assert mgr.metadata is not None
    assert mgr.metadata.title == "Pending"
    assert mgr._pending_metadata is not None  # noqa: SLF001
    assert mgr._pending_metadata.title == "Replacement"  # noqa: SLF001


def test_present_update_commits_pending_that_has_taken_effect() -> None:
    """A later present update commits a pending update before diffing."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)
    mgr.set_metadata(Metadata(title="Current", artist="Artist"))
    mgr.set_metadata(Metadata(title="Pending", artist="Artist", timestamp_us=2_000_000))
    group._server.clock.now_us.return_value = 3_000_000  # noqa: SLF001
    member = MagicMock()
    mgr._members = [member]  # noqa: SLF001

    mgr.set_metadata(Metadata(title="Pending", artist="Updated"))

    update = member.send_message.call_args.args[0].payload.metadata
    assert "title" not in update.to_dict()
    assert update.artist == "Updated"


def test_progress_set_on_first_update() -> None:
    """diff_update emits a full Progress object on the first update."""
    current = Metadata(
        title="Song",
        track_progress=5_000,
        track_duration=180_000,
        playback_speed=1000,
    )

    update = current.diff_update(None, timestamp=1_000_000)

    assert update.progress is not None
    assert not isinstance(update.progress, UndefinedField)
    assert update.progress.track_progress == 5_000
    assert update.progress.track_duration == 180_000
    assert update.progress.playback_speed == 1000


def test_progress_cleared_when_track_progress_becomes_none() -> None:
    """diff_update emits progress=null when previous state had progress and new doesn't."""
    last = Metadata(
        title="Song",
        track_progress=12_345,
        track_duration=180_000,
        playback_speed=1000,
    )
    current = Metadata(title="Loading next track...")

    update = current.diff_update(last, timestamp=2_000_000)

    assert update.progress is None
    assert update.to_dict()["progress"] is None
    assert '"progress":null' in update.to_json()


def test_progress_omitted_when_unchanged() -> None:
    """diff_update omits progress when no progress field changed."""
    last = Metadata(
        title="Song",
        track_progress=5_000,
        track_duration=180_000,
        playback_speed=1000,
    )
    current = Metadata(
        title="New Title",
        track_progress=5_000,
        track_duration=180_000,
        playback_speed=1000,
    )

    update = current.diff_update(last, timestamp=2_000_000)

    assert isinstance(update.progress, UndefinedField)
    assert "progress" not in update.to_json()
