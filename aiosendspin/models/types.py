"""Models for enum types used by Sendspin."""

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
    """Client roles with explicit versioning."""

    PLAYER = "player@v1"
    """
    Receives audio and plays it in sync.

    Has its own volume and mute state and preferred format settings.
    """
    CONTROLLER = "controller@v1"
    """Controls the Sendspin group this client is part of."""
    METADATA = "metadata@v1"
    """Displays text metadata describing the currently playing audio."""
    ARTWORK = "artwork@v1"
    """Displays artwork images. Has preferred format for images."""
    VISUALIZER = "visualizer@v1"
    """
    Visualizes music.

    Has preferred format for audio features.
    """


class BinaryMessageType(Enum):
    """Enum for Binary Message Types."""

    # Player role (bits 000001xx, IDs 4-7):
    AUDIO_CHUNK = 4
    """Audio chunks with timestamps (Player role, slot 0)."""

    # Artwork role (bits 000010xx, IDs 8-11):
    ARTWORK_CHANNEL_0 = 8
    """Artwork channel 0 (Artwork role, slot 0)."""
    ARTWORK_CHANNEL_1 = 9
    """Artwork channel 1 (Artwork role, slot 1)."""
    ARTWORK_CHANNEL_2 = 10
    """Artwork channel 2 (Artwork role, slot 2)."""
    ARTWORK_CHANNEL_3 = 11
    """Artwork channel 3 (Artwork role, slot 3)."""

    # Visualizer role (bits 00010xxx, IDs 16-23):
    VISUALIZATION_DATA = 16
    """Visualization data (Visualizer role, slot 0)."""


class RepeatMode(Enum):
    """Enum for Repeat Modes."""

    OFF = "off"
    ONE = "one"
    ALL = "all"


class ClientStateType(Enum):
    """Enum for Client States."""

    SYNCHRONIZED = "synchronized"
    """Client is operational and synchronized with server timestamps."""
    ERROR = "error"
    """Client has a problem preventing normal operation."""
    EXTERNAL_SOURCE = "external_source"
    """Client is in use by an external system and cannot participate in Sendspin playback."""


# DEPRECATED(before-spec-pr-50): Remove once all clients use client-level state
PlayerStateType = ClientStateType


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


class ConnectionReason(Enum):
    """Reason for server connection (multi-server support)."""

    DISCOVERY = "discovery"
    """Server is connecting for general availability (e.g., initial discovery, reconnection)."""
    PLAYBACK = "playback"
    """Server needs client for active or upcoming playback."""


class GoodbyeReason(Enum):
    """Reason for client disconnect (multi-server support)."""

    ANOTHER_SERVER = "another_server"
    """Client is switching to a different Sendspin server."""
    SHUTDOWN = "shutdown"
    """Client is shutting down."""
    RESTART = "restart"
    """Client is restarting and will reconnect."""
    USER_REQUEST = "user_request"
    """User explicitly requested to disconnect from this server."""
