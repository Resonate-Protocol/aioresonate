"""Tests for PlayerGroupRole volume/mute coordination."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.server.roles.player.events import (
    PlayerGroupMuteChangedEvent,
    PlayerGroupVolumeChangedEvent,
)
from aiosendspin.server.roles.player.group import PlayerGroupRole


def test_player_group_role_volume_empty() -> None:
    """Return 100 when no players are subscribed."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    assert pgr.volume == 100


def test_player_group_role_volume_single_player() -> None:
    """Return player volume when single player is subscribed."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    player = MagicMock()
    player.get_player_volume.return_value = 75
    pgr._members = [player]  # noqa: SLF001

    assert pgr.volume == 75


def test_player_group_role_volume_average() -> None:
    """Return average of player volumes."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_volume.return_value = 80
    p2 = MagicMock()
    p2.get_player_volume.return_value = 60
    pgr._members = [p1, p2]  # noqa: SLF001

    assert pgr.volume == 70


def test_player_group_role_volume_skips_none() -> None:
    """Skip players that return None for volume."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_volume.return_value = 80
    p2 = MagicMock()
    p2.get_player_volume.return_value = None
    pgr._members = [p1, p2]  # noqa: SLF001

    assert pgr.volume == 80


def test_player_group_role_muted_empty() -> None:
    """Return False when no players are subscribed."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    assert pgr.muted is False


def test_player_group_role_muted_all_muted() -> None:
    """Return True when all players are muted."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_muted.return_value = True
    p2 = MagicMock()
    p2.get_player_muted.return_value = True
    pgr._members = [p1, p2]  # noqa: SLF001

    assert pgr.muted is True


def test_player_group_role_muted_one_unmuted() -> None:
    """Return False when any player is unmuted."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_muted.return_value = True
    p2 = MagicMock()
    p2.get_player_muted.return_value = False
    pgr._members = [p1, p2]  # noqa: SLF001

    assert pgr.muted is False


def test_player_group_role_muted_none_returns_false() -> None:
    """Return False when any player returns None for muted."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_muted.return_value = True
    p2 = MagicMock()
    p2.get_player_muted.return_value = None
    pgr._members = [p1, p2]  # noqa: SLF001

    assert pgr.muted is False


def test_player_group_role_set_mute_propagates() -> None:
    """Propagate mute state to all players."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p2 = MagicMock()
    muted_state = {"p1": False, "p2": False}
    p1.get_player_muted.side_effect = lambda: muted_state["p1"]
    p2.get_player_muted.side_effect = lambda: muted_state["p2"]
    p1.set_player_mute.side_effect = lambda muted: muted_state.__setitem__("p1", muted)
    p2.set_player_mute.side_effect = lambda muted: muted_state.__setitem__("p2", muted)
    pgr._members = [p1, p2]  # noqa: SLF001

    pgr.set_mute(muted=True)

    p1.set_player_mute.assert_called_once_with(True)  # noqa: FBT003
    p2.set_player_mute.assert_called_once_with(True)  # noqa: FBT003
    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, PlayerGroupMuteChangedEvent)
    assert event.previous_muted is False
    assert event.muted is True


def test_player_group_role_set_volume_empty() -> None:
    """Set volume on empty group is a no-op."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    pgr.set_volume(50)  # Should not raise


def test_player_group_role_set_volume_single_player() -> None:
    """Set volume on single player sets that player's volume."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    player = MagicMock()
    state = {"volume": 100}
    player.get_player_volume.side_effect = lambda: state["volume"]
    player.set_player_volume.side_effect = lambda volume: state.__setitem__("volume", volume)
    pgr._members = [player]  # noqa: SLF001

    pgr.set_volume(50)

    player.set_player_volume.assert_called_once_with(50)
    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, PlayerGroupVolumeChangedEvent)
    assert event.previous_volume == 100
    assert event.volume == 50


def test_player_group_role_set_volume_redistributes() -> None:
    """Set volume redistributes across multiple players."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_volume.return_value = 100
    p2 = MagicMock()
    p2.get_player_volume.return_value = 100
    pgr._members = [p1, p2]  # noqa: SLF001

    pgr.set_volume(50)

    # Both should be set to 50 (average target is 50, both at 100 -> delta -50)
    p1.set_player_volume.assert_called_once_with(50)
    p2.set_player_volume.assert_called_once_with(50)


def test_player_group_role_set_volume_clamps_to_zero() -> None:
    """Set volume clamps player volumes to 0 minimum."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    player = MagicMock()
    player.get_player_volume.return_value = 10
    pgr._members = [player]  # noqa: SLF001

    pgr.set_volume(0)

    player.set_player_volume.assert_called_once_with(0)


def test_player_group_role_set_volume_clamps_to_100() -> None:
    """Set volume clamps player volumes to 100 maximum."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    player = MagicMock()
    player.get_player_volume.return_value = 90
    pgr._members = [player]  # noqa: SLF001

    pgr.set_volume(100)

    player.set_player_volume.assert_called_once_with(100)


def test_player_group_role_set_volume_skips_none() -> None:
    """Skip players that return None for volume."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    p1 = MagicMock()
    p1.get_player_volume.return_value = 50
    p2 = MagicMock()
    p2.get_player_volume.return_value = None
    pgr._members = [p1, p2]  # noqa: SLF001

    pgr.set_volume(75)

    p1.set_player_volume.assert_called_once_with(75)
    p2.set_player_volume.assert_not_called()


def test_player_group_role_set_volume_no_change_no_event() -> None:
    """No event is emitted when effective group volume is unchanged."""
    group = MagicMock()
    pgr = PlayerGroupRole(group)

    player = MagicMock()
    player.get_player_volume.return_value = 50
    pgr._members = [player]  # noqa: SLF001

    pgr.set_volume(50)

    group._signal_event.assert_not_called()  # noqa: SLF001
