"""Models for enum types used by resonate."""

from dataclasses import dataclass
from enum import Enum

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin
from mashumaro.types import Discriminator


# Base message classes
@dataclass
class ClientMessage(DataClassORJSONMixin):
    """Base class for client messages."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclass
class ServerMessage(DataClassORJSONMixin):
    """Base class for server messages."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        discriminator = Discriminator(field="type", include_subtypes=True)


# Helpers for discerning between null and undefined fields in messages
@dataclass
class UndefinedField(DataClassORJSONMixin):
    """Marker type to indicate undefined fields in messages."""


_UNDEFINED_SINGLETON = UndefinedField()


def undefined_field() -> UndefinedField:
    """Return the singleton UndefinedField instance."""
    return _UNDEFINED_SINGLETON


# Enums


class Roles(Enum):
    """Client roles."""

    PLAYER = "player"
    """
    Receives audio and plays it in sync.

    Has its own volume and mute state and preferred format settings.
    """
    CONTROLLER = "controller"
    """Controls Resonate groups."""
    METADATA = "metadata"
    """
    Displays metadata.

    Has preferred format for cover art.
    """
    VISUALIZER = "visualizer"
    """
    Visualizes music.

    Has preferred format for audio features.
    """


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


class PictureFormat(Enum):
    """Supported image formats for artwork/media art."""

    BMP = "bmp"
    JPEG = "jpeg"
    PNG = "png"
