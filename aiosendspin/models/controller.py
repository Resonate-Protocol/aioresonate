"""
Controller messages for the Sendspin protocol.

This module contains messages specific to clients with the controller role, which
enables the client to control the Sendspin group this client is part of, and switch
between groups.
"""

from __future__ import annotations

from dataclasses import dataclass

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import MediaCommand


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


# Server -> Client: server/state controller object
@dataclass
class ControllerStatePayload(DataClassORJSONMixin):
    """Controller state object in server/state message."""

    supported_commands: list[MediaCommand]
    """
    Subset of: play, pause, stop, next, previous, volume, mute, repeat_off, repeat_one,
    repeat_all, shuffle, unshuffle, switch.
    """
    volume: int
    """Volume of the whole group, range 0-100."""
    muted: bool
    """Mute state of the whole group."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if not 0 <= self.volume <= 100:
            raise ValueError(f"Volume must be in range 0-100, got {self.volume}")
