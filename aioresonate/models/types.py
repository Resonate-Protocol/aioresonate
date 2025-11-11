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
    """Displays text metadata describing the currently playing audio."""
    ARTWORK = "artwork"
    """Displays artwork images."""
    VISUALIZER = "visualizer"
    """
    Visualizes music.

    Has preferred format for audio features.
    """


class BinaryMessageType(Enum):
    """Enum for Binary Message Types."""

    # Player role (bits 000000xx):
    AUDIO_CHUNK = 0
    """Audio chunks with timestamps (Player role, slot 0)."""

    # Artwork role (bits 000001xx):
    ARTWORK_CHANNEL_0 = 4
    """Artwork channel 0 (Artwork role, slot 0)."""
    ARTWORK_CHANNEL_1 = 5
    """Artwork channel 1 (Artwork role, slot 1)."""
    ARTWORK_CHANNEL_2 = 6
    """Artwork channel 2 (Artwork role, slot 2)."""
    ARTWORK_CHANNEL_3 = 7
    """Artwork channel 3 (Artwork role, slot 3)."""

    # Visualizer role (bits 000010xx):
    VISUALIZATION_DATA = 8
    """Visualization data (Visualizer role, slot 0)."""


class RepeatMode(Enum):
    """Enum for Repeat Modes."""

    OFF = "off"
    ONE = "one"
    ALL = "all"


class PlayerStateType(Enum):
    """Enum for Player States."""

    SYNCHRONIZED = "synchronized"
    ERROR = "error"


class PlaybackStateType(Enum):
    """Enum for Playback States."""

    PLAYING = "playing"
    PAUSED = "paused"
    STOPPED = "stopped"


class AudioCodec(Enum):
    """Enum for Audio Codecs."""

    OPUS = "opus"
    FLAC = "flac"
    PCM = "pcm"


class PlayerCommand(Enum):
    """Enum for Player Commands."""

    VOLUME = "volume"
    MUTE = "mute"


class MediaCommand(Enum):
    """Enum for Media Commands."""

    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    NEXT = "next"
    PREVIOUS = "previous"
    VOLUME = "volume"
    MUTE = "mute"
    REPEAT_OFF = "repeat_off"
    REPEAT_ONE = "repeat_one"
    REPEAT_ALL = "repeat_all"
    SHUFFLE = "shuffle"
    UNSHUFFLE = "unshuffle"
    SWITCH = "switch"


class PictureFormat(Enum):
    """Supported image formats for artwork/media art."""

    BMP = "bmp"
    JPEG = "jpeg"
    PNG = "png"


class ArtworkSource(Enum):
    """Artwork source type."""

    ALBUM = "album"
    """Album artwork."""
    ARTIST = "artist"
    """Artist artwork."""
    NONE = "none"
    """No artwork - channel disabled."""
