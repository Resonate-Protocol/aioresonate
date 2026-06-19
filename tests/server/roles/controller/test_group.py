"""Tests for ControllerGroupRole."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiosendspin.models.controller import ControllerCommandPayload
from aiosendspin.models.core import ServerStateMessage
from aiosendspin.models.types import MediaCommand, RepeatMode
from aiosendspin.server.roles.controller.events import (
    ControllerMuteEvent,
    ControllerNextEvent,
    ControllerPauseEvent,
    ControllerPlayEvent,
    ControllerPreviousEvent,
    ControllerRepeatEvent,
    ControllerSeekEvent,
    ControllerSeekRelativeEvent,
    ControllerShuffleEvent,
    ControllerStopEvent,
    ControllerSwitchEvent,
    ControllerVolumeEvent,
)
from aiosendspin.server.roles.controller.group import ControllerGroupRole


def _make_group_stub() -> MagicMock:
    """Create a mock group for testing."""
    group = MagicMock()
    group.group_role.return_value = None
    return group


def test_controller_group_role_family() -> None:
    """ControllerGroupRole has role_family of 'controller'."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)
    assert cgr.role_family == "controller"


def test_controller_group_role_volume_delegates_to_player() -> None:
    """Volume property delegates to PlayerGroupRole."""
    group = _make_group_stub()
    player_group_role = MagicMock()
    player_group_role.get_group_volume.return_value = 75
    group.group_role.return_value = player_group_role

    cgr = ControllerGroupRole(group)
    assert cgr.volume == 75


def test_controller_group_role_volume_default() -> None:
    """Volume returns 100 when no player group role."""
    group = _make_group_stub()
    group.group_role.return_value = None

    cgr = ControllerGroupRole(group)
    assert cgr.volume == 100


def test_controller_group_role_muted_delegates_to_player() -> None:
    """Muted property delegates to PlayerGroupRole."""
    group = _make_group_stub()
    player_group_role = MagicMock()
    player_group_role.get_group_muted.return_value = True
    group.group_role.return_value = player_group_role

    cgr = ControllerGroupRole(group)
    assert cgr.muted is True


def test_controller_group_role_muted_default() -> None:
    """Muted returns False when no player group role."""
    group = _make_group_stub()
    group.group_role.return_value = None

    cgr = ControllerGroupRole(group)
    assert cgr.muted is False


def test_controller_group_role_set_volume_delegates() -> None:
    """set_volume() delegates to PlayerGroupRole."""
    group = _make_group_stub()
    player_group_role = MagicMock()
    player_group_role.get_group_volume.return_value = 50
    player_group_role.get_group_muted.return_value = False
    group.group_role.return_value = player_group_role

    cgr = ControllerGroupRole(group)
    cgr.set_volume(50)

    player_group_role.set_group_volume.assert_called_once_with(50)


def test_controller_group_role_set_mute_delegates() -> None:
    """set_mute() delegates to PlayerGroupRole."""
    group = _make_group_stub()
    player_group_role = MagicMock()
    player_group_role.get_group_volume.return_value = 100
    player_group_role.get_group_muted.return_value = True
    group.group_role.return_value = player_group_role

    cgr = ControllerGroupRole(group)
    cgr.set_mute(True)

    player_group_role.set_group_muted.assert_called_once_with(True)  # noqa: FBT003


def test_controller_group_role_supported_commands_default() -> None:
    """Default supported commands include VOLUME, MUTE, SWITCH."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)

    commands = cgr._get_supported_commands()  # noqa: SLF001

    assert MediaCommand.VOLUME in commands
    assert MediaCommand.MUTE in commands
    assert MediaCommand.SWITCH in commands


def test_controller_group_role_set_supported_commands() -> None:
    """set_supported_commands() adds app commands to list."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)

    cgr.set_supported_commands([MediaCommand.PLAY, MediaCommand.PAUSE])
    commands = cgr._get_supported_commands()  # noqa: SLF001

    # Should have both protocol and app commands
    assert MediaCommand.VOLUME in commands
    assert MediaCommand.PLAY in commands
    assert MediaCommand.PAUSE in commands


def test_controller_group_role_on_member_join_sends_state() -> None:
    """on_member_join() sends current controller state."""
    group = _make_group_stub()
    player_group_role = MagicMock()
    player_group_role.get_group_volume.return_value = 100
    player_group_role.get_group_muted.return_value = False
    group.group_role.return_value = player_group_role

    cgr = ControllerGroupRole(group)

    member = MagicMock()
    cgr.on_member_join(member)

    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert isinstance(msg, ServerStateMessage)
    assert msg.payload.controller is not None


def test_controller_group_role_handle_volume_command() -> None:
    """handle_command() sets volume and emits event for VOLUME command."""
    group = _make_group_stub()
    player_group_role = MagicMock()
    player_group_role.get_group_volume.return_value = 50
    player_group_role.get_group_muted.return_value = False
    group.group_role.return_value = player_group_role

    cgr = ControllerGroupRole(group)

    cmd = ControllerCommandPayload(command=MediaCommand.VOLUME, volume=50)
    cgr.handle_command(cmd)

    player_group_role.set_group_volume.assert_called_once_with(50)
    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ControllerVolumeEvent)
    assert event.volume == 50


def test_controller_group_role_handle_mute_command() -> None:
    """handle_command() sets mute and emits event for MUTE command."""
    group = _make_group_stub()
    player_group_role = MagicMock()
    player_group_role.get_group_volume.return_value = 100
    player_group_role.get_group_muted.return_value = True
    group.group_role.return_value = player_group_role

    cgr = ControllerGroupRole(group)

    cmd = ControllerCommandPayload(command=MediaCommand.MUTE, mute=True)
    cgr.handle_command(cmd)

    player_group_role.set_group_muted.assert_called_once_with(True)  # noqa: FBT003
    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ControllerMuteEvent)
    assert event.muted is True


def test_controller_group_role_handle_play_command() -> None:
    """handle_command() emits PlayEvent for PLAY command."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)
    cgr.set_supported_commands([MediaCommand.PLAY])

    cmd = ControllerCommandPayload(command=MediaCommand.PLAY)
    cgr.handle_command(cmd)

    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ControllerPlayEvent)


def test_controller_group_role_handle_pause_command() -> None:
    """handle_command() emits PauseEvent for PAUSE command."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)
    cgr.set_supported_commands([MediaCommand.PAUSE])

    cmd = ControllerCommandPayload(command=MediaCommand.PAUSE)
    cgr.handle_command(cmd)

    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ControllerPauseEvent)


def test_controller_group_role_handle_stop_command() -> None:
    """handle_command() emits StopEvent for STOP command."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)
    cgr.set_supported_commands([MediaCommand.STOP])

    cmd = ControllerCommandPayload(command=MediaCommand.STOP)
    cgr.handle_command(cmd)

    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ControllerStopEvent)


def test_controller_group_role_handle_next_command() -> None:
    """handle_command() emits NextEvent for NEXT command."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)
    cgr.set_supported_commands([MediaCommand.NEXT])

    cmd = ControllerCommandPayload(command=MediaCommand.NEXT)
    cgr.handle_command(cmd)

    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ControllerNextEvent)


def test_controller_group_role_handle_previous_command() -> None:
    """handle_command() emits PreviousEvent for PREVIOUS command."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)
    cgr.set_supported_commands([MediaCommand.PREVIOUS])

    cmd = ControllerCommandPayload(command=MediaCommand.PREVIOUS)
    cgr.handle_command(cmd)

    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ControllerPreviousEvent)


def test_controller_group_role_handle_switch_command() -> None:
    """handle_command() emits SwitchEvent for SWITCH command."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)

    cmd = ControllerCommandPayload(command=MediaCommand.SWITCH)
    cgr.handle_command(cmd)

    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ControllerSwitchEvent)


def test_controller_group_role_handle_repeat_commands() -> None:
    """handle_command() emits RepeatEvent for repeat commands."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)
    cgr.set_supported_commands(
        [MediaCommand.REPEAT_OFF, MediaCommand.REPEAT_ONE, MediaCommand.REPEAT_ALL]
    )

    for media_cmd, expected_mode in [
        (MediaCommand.REPEAT_OFF, RepeatMode.OFF),
        (MediaCommand.REPEAT_ONE, RepeatMode.ONE),
        (MediaCommand.REPEAT_ALL, RepeatMode.ALL),
    ]:
        group._signal_event.reset_mock()  # noqa: SLF001

        cmd = ControllerCommandPayload(command=media_cmd)
        cgr.handle_command(cmd)

        group._signal_event.assert_called_once()  # noqa: SLF001
        event = group._signal_event.call_args.args[0]  # noqa: SLF001
        assert isinstance(event, ControllerRepeatEvent)
        assert event.mode == expected_mode


def test_controller_group_role_handle_shuffle_commands() -> None:
    """handle_command() emits ShuffleEvent for shuffle commands."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)
    cgr.set_supported_commands([MediaCommand.SHUFFLE, MediaCommand.UNSHUFFLE])

    for media_cmd, expected_shuffle in [
        (MediaCommand.SHUFFLE, True),
        (MediaCommand.UNSHUFFLE, False),
    ]:
        group._signal_event.reset_mock()  # noqa: SLF001

        cmd = ControllerCommandPayload(command=media_cmd)
        cgr.handle_command(cmd)

        group._signal_event.assert_called_once()  # noqa: SLF001
        event = group._signal_event.call_args.args[0]  # noqa: SLF001
        assert isinstance(event, ControllerShuffleEvent)
        assert event.shuffle == expected_shuffle


def test_controller_group_role_on_member_join_includes_repeat_and_shuffle_defaults() -> None:
    """on_member_join() includes default repeat=OFF and shuffle=False in state."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)

    member = MagicMock()
    cgr.on_member_join(member)

    msg = member.send_message.call_args.args[0]
    assert msg.payload.controller.repeat == RepeatMode.OFF
    assert msg.payload.controller.shuffle is False


def test_controller_group_role_set_repeat_pushes_state() -> None:
    """set_repeat() pushes updated controller state to members."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)

    member = MagicMock()
    cgr._members.append(member)  # noqa: SLF001
    member.send_message.reset_mock()

    cgr.set_repeat(RepeatMode.ALL)

    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert msg.payload.controller.repeat == RepeatMode.ALL


def test_controller_group_role_set_shuffle_pushes_state() -> None:
    """set_shuffle() pushes updated controller state to members."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)

    member = MagicMock()
    cgr._members.append(member)  # noqa: SLF001
    member.send_message.reset_mock()

    cgr.set_shuffle(shuffle=True)

    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert msg.payload.controller.shuffle is True


def test_controller_group_role_set_repeat_dedupes_unchanged_value() -> None:
    """set_repeat() with same value does not re-push state."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)

    member = MagicMock()
    cgr._members.append(member)  # noqa: SLF001

    cgr.set_repeat(RepeatMode.ONE)
    member.send_message.reset_mock()

    cgr.set_repeat(RepeatMode.ONE)
    member.send_message.assert_not_called()


def test_controller_group_role_set_shuffle_dedupes_unchanged_value() -> None:
    """set_shuffle() with same value does not re-push state."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)

    member = MagicMock()
    cgr._members.append(member)  # noqa: SLF001

    cgr.set_shuffle(shuffle=True)
    member.send_message.reset_mock()

    cgr.set_shuffle(shuffle=True)
    member.send_message.assert_not_called()


def test_controller_group_role_handle_unsupported_command() -> None:
    """handle_command() ignores unsupported commands."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)
    # Don't add PLAY to supported commands

    cmd = ControllerCommandPayload(command=MediaCommand.PLAY)
    cgr.handle_command(cmd)

    group._signal_event.assert_not_called()  # noqa: SLF001


def test_seek_events_exported_from_roles() -> None:
    """Seek events are importable from the public roles package (MA depends on this)."""
    from aiosendspin.server.roles import (  # noqa: PLC0415
        ControllerSeekEvent,
        ControllerSeekRelativeEvent,
    )
    from aiosendspin.server.roles.controller.events import ControllerEvent  # noqa: PLC0415

    assert issubclass(ControllerSeekEvent, ControllerEvent)
    assert issubclass(ControllerSeekRelativeEvent, ControllerEvent)


def test_get_supported_commands_hides_seek_without_max() -> None:
    """'seek' is dropped from advertised commands while seek_max_ms is unknown."""
    cgr = ControllerGroupRole(_make_group_stub())
    cgr.set_supported_commands([MediaCommand.SEEK, MediaCommand.SEEK_RELATIVE])

    commands = cgr._get_supported_commands()  # noqa: SLF001
    assert MediaCommand.SEEK not in commands
    assert MediaCommand.SEEK_RELATIVE in commands


def test_get_supported_commands_shows_seek_with_max() -> None:
    """'seek' is advertised once seek_max_ms is set."""
    cgr = ControllerGroupRole(_make_group_stub())
    cgr.set_supported_commands([MediaCommand.SEEK])
    cgr.set_seek_max_ms(300_000)

    assert MediaCommand.SEEK in cgr._get_supported_commands()  # noqa: SLF001


def test_set_seek_max_ms_pushes_state() -> None:
    """set_seek_max_ms() pushes controller state carrying seek_max_ms."""
    cgr = ControllerGroupRole(_make_group_stub())
    member = MagicMock()
    cgr._members.append(member)  # noqa: SLF001
    member.send_message.reset_mock()

    cgr.set_seek_max_ms(300_000)

    member.send_message.assert_called_once()
    msg = member.send_message.call_args.args[0]
    assert msg.payload.controller.seek_max_ms == 300_000


def test_set_seek_max_ms_dedupes_unchanged_value() -> None:
    """set_seek_max_ms() with the same value does not re-push state."""
    cgr = ControllerGroupRole(_make_group_stub())
    member = MagicMock()
    cgr._members.append(member)  # noqa: SLF001

    cgr.set_seek_max_ms(300_000)
    member.send_message.reset_mock()

    cgr.set_seek_max_ms(300_000)
    member.send_message.assert_not_called()


def test_handle_seek_command_emits_event_when_in_range() -> None:
    """An in-range 'seek' emits ControllerSeekEvent with the target position."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)
    cgr.set_supported_commands([MediaCommand.SEEK])
    cgr.set_seek_max_ms(300_000)
    group._signal_event.reset_mock()  # noqa: SLF001

    cgr.handle_command(ControllerCommandPayload(command=MediaCommand.SEEK, position_ms=120_000))

    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ControllerSeekEvent)
    assert event.position_ms == 120_000


def test_handle_seek_command_ignored_when_out_of_range() -> None:
    """An out-of-range 'seek' is ignored (no event)."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)
    cgr.set_supported_commands([MediaCommand.SEEK])
    cgr.set_seek_max_ms(300_000)
    group._signal_event.reset_mock()  # noqa: SLF001

    cgr.handle_command(ControllerCommandPayload(command=MediaCommand.SEEK, position_ms=400_000))

    group._signal_event.assert_not_called()  # noqa: SLF001


def test_handle_seek_relative_command_emits_event() -> None:
    """'seek_relative' emits ControllerSeekRelativeEvent without needing seek_max_ms."""
    group = _make_group_stub()
    cgr = ControllerGroupRole(group)
    cgr.set_supported_commands([MediaCommand.SEEK_RELATIVE])
    group._signal_event.reset_mock()  # noqa: SLF001

    cgr.handle_command(
        ControllerCommandPayload(command=MediaCommand.SEEK_RELATIVE, offset_ms=-15_000)
    )

    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ControllerSeekRelativeEvent)
    assert event.offset_ms == -15_000


def test_set_seek_max_ms_rejects_negative() -> None:
    """set_seek_max_ms() rejects a negative bound."""
    cgr = ControllerGroupRole(_make_group_stub())
    with pytest.raises(ValueError, match="non-negative"):
        cgr.set_seek_max_ms(-1)
