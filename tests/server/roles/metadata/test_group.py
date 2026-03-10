"""Tests for MetadataGroupRole."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.models.core import ServerStateMessage
from aiosendspin.models.types import RepeatMode
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


def test_metadata_group_role_update_repeat() -> None:
    """update() updates repeat mode."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    mgr.update(repeat=RepeatMode.ALL)

    assert mgr.metadata is not None
    assert mgr.metadata.repeat == RepeatMode.ALL


def test_metadata_group_role_update_shuffle() -> None:
    """update() updates shuffle state."""
    group = _make_group_stub()
    mgr = MetadataGroupRole(group)

    mgr.update(shuffle=True)

    assert mgr.metadata is not None
    assert mgr.metadata.shuffle is True


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
