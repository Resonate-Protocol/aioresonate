"""
Controller messages for the Resonate protocol.

This module contains messages specific to clients with the controller role, which
enables remote control of groups and playback. Controller clients can browse
available groups, join/leave groups, and send playback commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import ClientMessage, MediaCommand, PlaybackStateType, ServerMessage


# Client -> Server: group/get-list
@dataclass
class GroupGetListClientMessage(ClientMessage):
    """Message sent by the client to request all groups available to join."""

    type: Literal["group/get-list"] = "group/get-list"


# Client -> Server: group/join
@dataclass
class GroupJoinClientPayload(DataClassORJSONMixin):
    """Payload for joining a group."""

    group_id: str
    """Identifier of group to join."""


@dataclass
class GroupJoinClientMessage(ClientMessage):
    """Message sent by the client to join a group."""

    payload: GroupJoinClientPayload
    type: Literal["group/join"] = "group/join"


# Client -> Server: group/unjoin
@dataclass
class GroupUnjoinClientMessage(ClientMessage):
    """Message sent by the client to leave current group."""

    type: Literal["group/unjoin"] = "group/unjoin"


# Client -> Server: client/command controller object
@dataclass
class ControllerCommandPayload(DataClassORJSONMixin):
    """Control the group that's playing."""

    command: MediaCommand
    """
    Command must be one of the values listed in supported_commands from server/state controller
    object.
    """
    volume: int | None = None
    """Volume range 0-100, only set if command is volume."""
    mute: bool | None = None
    """True to mute, false to unmute, only set if command is mute."""

    def __post_init__(self) -> None:
        """Validate field values and command consistency."""
        if self.command == MediaCommand.VOLUME:
            if self.volume is None:
                raise ValueError("Volume must be provided when command is 'volume'")
            if not 0 <= self.volume <= 100:
                raise ValueError(f"Volume must be in range 0-100, got {self.volume}")
        elif self.volume is not None:
            raise ValueError(f"Volume should not be provided for command '{self.command.value}'")

        if self.command == MediaCommand.MUTE:
            if self.mute is None:
                raise ValueError("Mute must be provided when command is 'mute'")
        elif self.mute is not None:
            raise ValueError(f"Mute should not be provided for command '{self.command.value}'")

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


# Server -> Client: group/list
@dataclass
class GroupInfoServerPayload(DataClassORJSONMixin):
    """Information about a group."""

    group_id: str
    """Group identifier."""
    name: str
    """Group name."""
    state: Literal["playing", "paused", "idle"]
    """Group state."""
    member_count: int
    """Number of clients in group."""


@dataclass
class GroupListServerPayload(DataClassORJSONMixin):
    """All groups available to join on the server."""

    groups: list[GroupInfoServerPayload]
    """List of available groups."""


@dataclass
class GroupListServerMessage(ServerMessage):
    """Message sent by the server with list of available groups."""

    payload: GroupListServerPayload
    type: Literal["group/list"] = "group/list"


# Server -> Client: group/update
@dataclass
class GroupUpdateServerPayload(DataClassORJSONMixin):
    """State update of the group this client is part of."""

    playback_state: PlaybackStateType | None = None
    """Playback state of the group."""
    group_id: str | None = None
    """Group identifier."""
    group_name: str | None = None
    """Friendly name of the group."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class GroupUpdateServerMessage(ServerMessage):
    """Message sent by the server to update group state."""

    payload: GroupUpdateServerPayload
    type: Literal["group/update"] = "group/update"
