"""Models for enum types used by resonate."""

from enum import Enum


class BinaryMessageType(Enum):
    """Enum for Binary Message Types."""

    AUDIO_CHUNK = 1
    """Audio chunks with timestamps."""
    MEDIA_ART = 2
    """Media art (images)."""
    VISUALIZATION_DATA = 3
    """Visualization data."""


class RepeatMode(Enum):
    """Enum for Repeat Modes."""

    OFF = "off"
    ONE = "one"
    ALL = "all"


class PlayerStateType(Enum):
    """Enum for Player States."""

    PLAYING = "playing"
    IDLE = "idle"


class GroupStateType(Enum):
    """Enum for Group States."""

    PLAYING = "playing"
    PAUSED = "paused"
    IDLE = "idle"


class PlaybackStateType(Enum):
    """Enum for Playback States."""

    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"


class MediaCommand(Enum):
    """Enum for Media Commands."""

    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    NEXT = "next"
    PREVIOUS = "previous"
    SEEK = "seek"
    VOLUME = "volume"
    MUTE = "mute"
