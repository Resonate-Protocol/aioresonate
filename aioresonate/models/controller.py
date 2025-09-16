"""Controller messages for the Resonate protocol.

This module contains messages specific to clients with the controller role, which
enables remote control of groups and playback. Controller clients can browse
available groups, join/leave groups, and send playback commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .core import ClientMessage, ServerMessage
from .types import MediaCommand


# Client → Server controller messages
@dataclass
class GroupGetListClientMessage(ClientMessage):
    """Message sent by the client to request all groups available to join."""

    type: Literal["group/get-list"] = "group/get-list"


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


@dataclass
class GroupUnjoinClientMessage(ClientMessage):
    """Message sent by the client to leave current group."""

    type: Literal["group/unjoin"] = "group/unjoin"


@dataclass
class GroupCommandClientPayload(DataClassORJSONMixin):
    """Control the group that's playing."""

    command: MediaCommand
    """Command must be one of the values listed in group/update field supported_commands."""
    volume: int | None = None
    """Volume range 0-100, only set if command is volume."""
    mute: bool | None = None
    """True to mute, false to unmute, only set if command is mute."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class GroupCommandClientMessage(ClientMessage):
    """Message sent by the client to control the group."""

    payload: GroupCommandClientPayload
    type: Literal["group/command"] = "group/command"


# Server → Client controller messages
@dataclass
class GroupMemberServerPayload(DataClassORJSONMixin):
    """Represents a group member."""

    client_id: str
    """Client identifier."""
    name: str
    """Client friendly name."""


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


@dataclass
class GroupUpdateServerPayload(DataClassORJSONMixin):
    """Group state update."""

    supported_commands: list[str]
    """Subset of: play, pause, stop, next, previous, seek, volume, mute."""
    members: list[GroupMemberServerPayload]
    """List of group members."""
    session_id: str | None
    """Null if no active session."""
    volume: int
    """Volume range 0-100."""
    muted: bool
    """Mute state."""


@dataclass
class GroupUpdateServerMessage(ServerMessage):
    """Message sent by the server to update group state."""

    payload: GroupUpdateServerPayload
    type: Literal["group/update"] = "group/update"
