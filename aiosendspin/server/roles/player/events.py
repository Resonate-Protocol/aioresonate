"""Player role events."""

from __future__ import annotations

from dataclasses import dataclass

from aiosendspin.server.events import ClientRoleEvent, GroupRoleEvent


@dataclass
class VolumeChangedEvent(ClientRoleEvent):
    """The volume or mute status of the player was changed."""

    volume: int
    muted: bool


@dataclass
class StaticDelayChangedEvent(ClientRoleEvent):
    """The static delay of the player was changed."""

    static_delay_ms: int


class PlayerGroupEvent(GroupRoleEvent):
    """Base event type for player group role changes."""


@dataclass
class PlayerGroupVolumeChangedEvent(PlayerGroupEvent):
    """The effective group volume changed."""

    previous_volume: int
    volume: int


@dataclass
class PlayerGroupMuteChangedEvent(PlayerGroupEvent):
    """The effective group mute state changed."""

    previous_muted: bool
    muted: bool
