"""Models for enum types used by resonate."""

from enum import Enum


class BinaryMessageType(Enum):
    """Enum for Binary Message Types."""

    PlayAudioChunk = 1


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
