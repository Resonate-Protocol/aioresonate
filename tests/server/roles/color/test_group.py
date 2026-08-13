"""Tests for ColorGroupRole."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.models.core import ServerStateMessage
from aiosendspin.server.roles.color import ColorClearedEvent, ColorUpdatedEvent
from aiosendspin.server.roles.color.group import ColorGroupRole
from aiosendspin.server.roles.color.state import Color


def _make_group_stub() -> MagicMock:
    group = MagicMock()
    group._server = MagicMock()  # noqa: SLF001
    group._server.clock.now_us.return_value = 1_000_000  # noqa: SLF001
    return group


def test_color_group_role_family() -> None:
    """ColorGroupRole has role_family of 'color'."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)
    assert cgr.role_family == "color"


def test_color_group_role_initial_color_is_none() -> None:
    """Initial color is None."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)
    assert cgr.color is None


def test_set_color_stores_and_broadcasts() -> None:
    """set_color() stores the color and sends update to members."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)

    member = MagicMock()
    cgr._members = [member]  # noqa: SLF001

    color = Color(primary=(255, 0, 0), accent=(0, 255, 0))
    cgr.set_color(color)

    assert cgr.color is not None
    assert cgr.color.primary == (255, 0, 0)

    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.color is not None
    assert msg.payload.color.primary == (255, 0, 0)

    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ColorUpdatedEvent)
    assert event.color.primary == (255, 0, 0)
    assert event.previous_color is None


def test_set_color_no_op_when_equal() -> None:
    """set_color() with the same color does nothing."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)

    color = Color(primary=(255, 0, 0))
    cgr.set_color(color)
    group._signal_event.reset_mock()  # noqa: SLF001

    cgr.set_color(Color(primary=(255, 0, 0)))

    group._signal_event.assert_not_called()  # noqa: SLF001


def test_clear_color() -> None:
    """clear() sets color to None and sends cleared update."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)

    member = MagicMock()
    cgr._members = [member]  # noqa: SLF001

    cgr.set_color(Color(primary=(255, 0, 0)))
    member.send_message.reset_mock()
    group._signal_event.reset_mock()  # noqa: SLF001

    cgr.clear()

    assert cgr.color is None
    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.color is not None
    assert msg.payload.color.primary is None

    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ColorClearedEvent)


def test_on_member_join_sends_current_color() -> None:
    """on_member_join sends a snapshot to the new member."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)
    cgr._current_color = Color(primary=(100, 150, 200))  # noqa: SLF001

    member = MagicMock()
    cgr.on_member_join(member)

    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.color is not None
    assert msg.payload.color.primary == (100, 150, 200)


def test_on_member_join_sends_cleared_when_no_color() -> None:
    """on_member_join sends a cleared update when no color is set."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)

    member = MagicMock()
    cgr.on_member_join(member)

    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.color is not None
    assert msg.payload.color.primary is None


def test_future_color_keeps_current_and_replays_both_states() -> None:
    """A future palette remains pending and replays after current state."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)
    cgr.set_color(Color(primary=(1, 2, 3)))
    cgr.set_color(Color(primary=(4, 5, 6)), timestamp_us=2_000_000)

    assert cgr.color == Color(primary=(1, 2, 3))

    member = MagicMock()
    cgr.on_member_join(member)

    updates = [call.args[0].payload.color for call in member.send_message.call_args_list]
    assert [update.timestamp for update in updates] == [1_000_000, 2_000_000]
    assert updates[0].primary == (1, 2, 3)
    assert updates[1].primary == (4, 5, 6)


def test_late_join_initial_color_is_current_after_pending_becomes_due() -> None:
    """A due palette becomes the present-timestamped initial state."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)
    cgr.set_color(Color(primary=(1, 2, 3)))
    cgr.set_color(Color(primary=(4, 5, 6)), timestamp_us=2_000_000)
    group._server.clock.now_us.return_value = 3_000_000  # noqa: SLF001
    member = MagicMock()

    cgr.on_member_join(member)

    updates = [call.args[0].payload.color for call in member.send_message.call_args_list]
    assert len(updates) == 1
    assert updates[0].timestamp == 3_000_000
    assert updates[0].primary == (4, 5, 6)


def test_later_arrival_replaces_color_when_timestamp_goes_backwards() -> None:
    """Future color replacement follows arrival order, not timestamp order."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)
    current = Color(primary=(1, 2, 3))
    earlier = Color(primary=(4, 5, 6))
    cgr.set_color(current)
    cgr.set_color(Color(primary=(7, 8, 9)), timestamp_us=3_000_000)
    cgr.set_color(earlier, timestamp_us=2_000_000)

    assert cgr.color == current
    assert cgr._pending_color == earlier  # noqa: SLF001


def test_chained_scheduled_color_updates_carry_prior_fields() -> None:
    """Each scheduled palette includes every field carried by its predecessor."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)
    member = MagicMock()
    cgr._members = [member]  # noqa: SLF001
    cgr.set_color(Color(primary=(1, 2, 3), accent=(2, 3, 4)))
    member.reset_mock()

    cgr.set_color(Color(primary=(4, 5, 6), accent=(2, 3, 4)), timestamp_us=3_000_000)
    cgr.set_color(Color(primary=(1, 2, 3), accent=(5, 6, 7)), timestamp_us=2_500_000)
    cgr.set_color(Color(primary=(7, 8, 9), accent=(2, 3, 4)), timestamp_us=2_000_000)

    updates = [call.args[0].payload.color.to_dict() for call in member.send_message.call_args_list]
    assert updates[0] == {"timestamp": 3_000_000, "primary": [4, 5, 6]}
    assert updates[1] == {
        "timestamp": 2_500_000,
        "primary": [1, 2, 3],
        "accent": [5, 6, 7],
    }
    assert updates[2] == {
        "timestamp": 2_000_000,
        "primary": [7, 8, 9],
        "accent": [2, 3, 4],
    }
    assert cgr.color == Color(primary=(1, 2, 3), accent=(2, 3, 4))


def test_present_color_cancels_pending_with_timestamp_only_update() -> None:
    """A present unchanged palette cancels pending with only a timestamp."""
    group = _make_group_stub()
    cgr = ColorGroupRole(group)
    member = MagicMock()
    cgr._members = [member]  # noqa: SLF001
    current = Color(primary=(1, 2, 3))
    cgr.set_color(current)
    cgr.set_color(Color(primary=(4, 5, 6)), timestamp_us=2_000_000)
    member.reset_mock()

    cgr.set_color(current)

    assert cgr._pending_update is None  # noqa: SLF001
    update = member.send_message.call_args.args[0].payload.color
    assert update.timestamp == 1_000_000
    assert update.primary == (1, 2, 3)
