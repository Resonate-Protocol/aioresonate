"""Models for the resonate audio protocol."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mashumaro.mixins.orjson import DataClassORJSONMixin


class RepeatMode(Enum):
    """Enum for Repeat Modes."""

    OFF = "off"
    ONE = "one"
    ALL = "all"


class PlayerStateType(Enum):
    """Enum for Player States."""

    PLAYING = "playing"
    PAUSED = "paused"
    IDLE = "idle"


class MediaCommand(Enum):
    """Enum for Media Commands."""

    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    SEEK = "seek"
    VOLUME = "volume"


@dataclass
class Message(DataClassORJSONMixin):
    """Message type used by resonate."""

    type: str
    payload: dict[str, Any] | Any


# TODO: check this
BINARY_HEADER_FORMAT = ">BQI"
BINARY_HEADER_SIZE = struct.calcsize(BINARY_HEADER_FORMAT)
