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

from .core import ClientMessage
from .types import PlayerStateType


# Client → Server player messages
@dataclass
class PlayerUpdateClientPayload(DataClassORJSONMixin):
    """State information of the player."""

    state: PlayerStateType
    """Playing if active stream, idle if no active stream."""
    volume: int
    """Volume range 0-100."""
    muted: bool
    """Mute state."""


@dataclass
class PlayerUpdateClientMessage(ClientMessage):
    """Message sent by the player to report its state changes."""

    payload: PlayerUpdateClientPayload
    type: Literal["player/update"] = "player/update"


@dataclass
class StreamRequestFormatClientPayload(DataClassORJSONMixin):
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
class StreamRequestFormatClientMessage(ClientMessage):
    """Message sent by the client to request different stream format."""

    payload: StreamRequestFormatClientPayload
    type: Literal["stream/request-format"] = "stream/request-format"
