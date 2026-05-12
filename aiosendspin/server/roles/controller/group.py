"""ControllerGroupRole - group-level controller coordination."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from aiosendspin.models.controller import ControllerCommandPayload, ControllerStatePayload
from aiosendspin.models.core import ServerStateMessage, ServerStatePayload
from aiosendspin.models.types import MediaCommand, RepeatMode, has_role_family
from aiosendspin.server.events import ClientEvent
from aiosendspin.server.roles.base import GroupRole, Role
from aiosendspin.server.roles.controller.events import (
    ControllerEvent,
    ControllerMuteEvent,
    ControllerNextEvent,
    ControllerPauseEvent,
    ControllerPlayEvent,
    ControllerPreviousEvent,
    ControllerRepeatEvent,
    ControllerShuffleEvent,
    ControllerStopEvent,
    ControllerSwitchEvent,
    ControllerVolumeEvent,
)
from aiosendspin.server.roles.metadata.group import MetadataGroupRole
from aiosendspin.server.roles.metadata.state import Metadata
from aiosendspin.server.roles.player.events import VolumeChangedEvent

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.group import SendspinGroup

logger = logging.getLogger(__name__)


class ControllerGroupRole(GroupRole):
    """Coordinate controller roles across a group.

    Handles incoming commands from controller clients, validates against
    supported_commands, and emits events for application handling.
    """

    role_family = "controller"

    def __init__(self, group: SendspinGroup) -> None:
        """Initialize ControllerGroupRole."""
        super().__init__(group)
        self._supported_commands: list[MediaCommand] = []
        self._repeat: RepeatMode = RepeatMode.OFF
        self._shuffle: bool = False
        self._last_sent_volume: int | None = None
        self._last_sent_muted: bool | None = None
        self._last_sent_supported_commands: list[MediaCommand] | None = None
        self._last_sent_repeat: RepeatMode | None = None
        self._last_sent_shuffle: bool | None = None
        # Track volume event subscriptions for player clients
        self._player_client_unsubs: dict[SendspinClient, Callable[[], None]] = {}

    @property
    def volume(self) -> int:
        """Return current group volume, delegated to PlayerGroupRole."""
        player_group_role = self._group.group_role("player")
        if player_group_role is not None:
            vol = player_group_role.get_group_volume()
            if vol is not None:
                return vol
        return 100

    @property
    def muted(self) -> bool:
        """Return current group mute state, delegated to PlayerGroupRole."""
        player_group_role = self._group.group_role("player")
        if player_group_role is not None:
            m = player_group_role.get_group_muted()
            if m is not None:
                return m
        return False

    def set_volume(self, level: int) -> None:
        """Set group volume, delegated to PlayerGroupRole."""
        player_group_role = self._group.group_role("player")
        if player_group_role is not None:
            player_group_role.set_group_volume(level)
        self._push_state_to_members()

    def set_mute(self, muted: bool) -> None:  # noqa: FBT001
        """Set group mute state, delegated to PlayerGroupRole."""
        player_group_role = self._group.group_role("player")
        if player_group_role is not None:
            player_group_role.set_group_muted(muted)
        self._push_state_to_members()

    @property
    def repeat(self) -> RepeatMode:
        """Return current group repeat mode."""
        return self._repeat

    @property
    def shuffle(self) -> bool:
        """Return current group shuffle state."""
        return self._shuffle

    def set_repeat(self, mode: RepeatMode) -> None:
        """Set group repeat mode and push state to members if changed."""
        self._repeat = mode
        self._push_state_to_members()
        self._mirror_to_metadata_back_compat()

    def set_shuffle(self, shuffle: bool) -> None:  # noqa: FBT001
        """Set group shuffle state and push state to members if changed."""
        self._shuffle = shuffle
        self._push_state_to_members()
        self._mirror_to_metadata_back_compat()

    def _mirror_to_metadata_back_compat(self) -> None:
        """Mirror repeat/shuffle into metadata state for v1 clients."""
        # Deprecated: drop with metadata dual-emit.
        metadata_gr = self._group.group_role("metadata")
        if not isinstance(metadata_gr, MetadataGroupRole):
            return
        current = metadata_gr.metadata or Metadata()
        metadata_gr.set_metadata(replace(current, repeat=self._repeat, shuffle=self._shuffle))

    def set_supported_commands(self, commands: list[MediaCommand]) -> None:
        """Set the commands supported by the application.

        Args:
            commands: List of MediaCommand values the application can handle.
        """
        self._supported_commands = commands
        self._push_state_to_members()

    def on_member_join(self, role: Role) -> None:
        """Send current controller state to newly joined member."""
        self._send_state_to_role(role)

    def _get_supported_commands(self) -> list[MediaCommand]:
        """Get list of commands supported by protocol + application."""
        protocol_commands = [
            MediaCommand.VOLUME,
            MediaCommand.MUTE,
            MediaCommand.SWITCH,
        ]

        if self._supported_commands:
            return list(set(protocol_commands) | set(self._supported_commands))

        return protocol_commands

    def _send_state_to_role(self, role: Role) -> None:
        """Send current controller state to a single role."""
        supported_commands = self._get_supported_commands()
        controller_state = ControllerStatePayload(
            supported_commands=supported_commands,
            volume=self.volume,
            muted=self.muted,
            repeat=self._repeat,
            shuffle=self._shuffle,
        )
        state_message = ServerStateMessage(ServerStatePayload(controller=controller_state))
        role.send_message(state_message)

    def _push_state_to_members(self) -> None:
        """Push controller state to all subscribed members if changed."""
        current_volume = self.volume
        current_muted = self.muted
        current_supported_commands = self._get_supported_commands()
        current_repeat = self._repeat
        current_shuffle = self._shuffle

        if (
            self._last_sent_volume == current_volume
            and self._last_sent_muted == current_muted
            and self._last_sent_supported_commands == current_supported_commands
            and self._last_sent_repeat == current_repeat
            and self._last_sent_shuffle == current_shuffle
        ):
            return

        self._last_sent_volume = current_volume
        self._last_sent_muted = current_muted
        self._last_sent_supported_commands = current_supported_commands
        self._last_sent_repeat = current_repeat
        self._last_sent_shuffle = current_shuffle

        controller_state = ControllerStatePayload(
            supported_commands=current_supported_commands,
            volume=current_volume,
            muted=current_muted,
            repeat=current_repeat,
            shuffle=current_shuffle,
        )
        state_message = ServerStateMessage(ServerStatePayload(controller=controller_state))

        for role in self._members:
            role.send_message(state_message)

    def handle_command(self, cmd: ControllerCommandPayload) -> None:
        """Handle a command from a controller client.

        Validates the command against supported_commands and either handles
        it directly (volume, mute) or emits an event for application handling.
        """
        supported = self._get_supported_commands()
        if cmd.command not in supported:
            logger.warning(
                "Received unsupported command %s (supported: %s)",
                cmd.command,
                supported,
            )
            return

        if cmd.command == MediaCommand.VOLUME and cmd.volume is not None:
            self.set_volume(cmd.volume)
            self.emit_group_event(ControllerVolumeEvent(volume=cmd.volume))
            return
        if cmd.command == MediaCommand.MUTE and cmd.mute is not None:
            self.set_mute(cmd.mute)
            self.emit_group_event(ControllerMuteEvent(muted=cmd.mute))
            return

        event = self._command_to_event(cmd)
        if event is not None:
            self.emit_group_event(event)

    def _command_to_event(self, cmd: ControllerCommandPayload) -> ControllerEvent | None:
        """Convert a command payload to an event."""
        match cmd.command:
            case MediaCommand.PLAY:
                return ControllerPlayEvent()
            case MediaCommand.PAUSE:
                return ControllerPauseEvent()
            case MediaCommand.STOP:
                return ControllerStopEvent()
            case MediaCommand.NEXT:
                return ControllerNextEvent()
            case MediaCommand.PREVIOUS:
                return ControllerPreviousEvent()
            case MediaCommand.SWITCH:
                return ControllerSwitchEvent()
            case MediaCommand.REPEAT_OFF:
                return ControllerRepeatEvent(mode=RepeatMode.OFF)
            case MediaCommand.REPEAT_ONE:
                return ControllerRepeatEvent(mode=RepeatMode.ONE)
            case MediaCommand.REPEAT_ALL:
                return ControllerRepeatEvent(mode=RepeatMode.ALL)
            case MediaCommand.SHUFFLE:
                return ControllerShuffleEvent(shuffle=True)
            case MediaCommand.UNSHUFFLE:
                return ControllerShuffleEvent(shuffle=False)
            case _:
                return None

    # --- Client added/removed hooks ---

    def on_client_added(self, client: SendspinClient) -> None:
        """Subscribe to volume events from player clients."""
        if client in self._player_client_unsubs:
            return
        if not has_role_family("player", client.negotiated_roles):
            return

        def on_client_event(_client: SendspinClient, event: ClientEvent) -> None:
            if isinstance(event, VolumeChangedEvent):
                self._push_state_to_members()

        unsub = client.add_event_listener(on_client_event)
        self._player_client_unsubs[client] = unsub

    def on_client_removed(self, client: SendspinClient) -> None:
        """Unsubscribe from player client events."""
        if client in self._player_client_unsubs:
            self._player_client_unsubs[client]()
            del self._player_client_unsubs[client]
