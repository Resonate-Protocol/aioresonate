"""Player messages for the Resonate protocol.

This module contains messages specific to clients with the player role, which
handle audio output and synchronized playback. Player clients receive timestamped
audio data, manage their own volume and mute state, and can request different
audio formats based on their capabilities and current conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import ClientMessage, PlayerStateType


# Client -> Server client/hello player support object
@dataclass
class ClientHelloPlayerSupport(DataClassORJSONMixin):
    """Player support configuration - only if player role is set."""

    support_codecs: list[str]
    """Supported codecs in priority order."""
    support_channels: list[int]
    """Number of channels in priority order."""
    support_sample_rates: list[int]
    """Supported sample rates in priority order."""
    support_bit_depth: list[int]
    """Bit depth in priority order."""
    buffer_capacity: int
    """Buffer capacity size in bytes."""


# Client -> Server player/update
@dataclass
class PlayerUpdatePayload(DataClassORJSONMixin):
    """State information of the player."""

    state: PlayerStateType
    """Playing if active stream, idle if no active stream."""
    volume: int
    """Volume range 0-100."""
    muted: bool
    """Mute state."""


@dataclass
class PlayerUpdateMessage(ClientMessage):
    """Message sent by the player to report its state changes."""

    payload: PlayerUpdatePayload
    type: Literal["player/update"] = "player/update"


# Client -> Server stream/request-format
@dataclass
class StreamRequestFormatPayload(DataClassORJSONMixin):
    """Request different stream format (upgrade or downgrade)."""

    codec: str | None = None
    """Requested codec."""
    sample_rate: int | None = None
    """Requested sample rate."""
    channels: int | None = None
    """Requested channels."""
    bit_depth: int | None = None
    """Requested bit depth."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class StreamRequestFormatMessage(ClientMessage):
    """Message sent by the client to request different stream format."""

    payload: StreamRequestFormatPayload
    type: Literal["stream/request-format"] = "stream/request-format"


# Server -> Client stream/start player object
@dataclass
class StreamStartPlayer(DataClassORJSONMixin):
    """Player object in stream/start message."""

    codec: str
    """Codec to be used."""
    sample_rate: int
    """Sample rate to be used."""
    channels: int
    """Channels to be used."""
    bit_depth: int
    """Bit depth to be used."""
    codec_header: str | None = None
    """Base64 encoded codec header (if necessary; e.g., FLAC)."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


# Server -> Client stream/update player object
@dataclass
class StreamUpdatePlayer(DataClassORJSONMixin):
    """Player object in stream/update message with delta updates."""

    codec: str | None = None
    """Codec to be used."""
    sample_rate: int | None = None
    """Sample rate to be used."""
    channels: int | None = None
    """Channels to be used."""
    bit_depth: int | None = None
    """Bit depth to be used."""
    codec_header: str | None = None
    """Base64 encoded codec header (if necessary; e.g., FLAC)."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True
