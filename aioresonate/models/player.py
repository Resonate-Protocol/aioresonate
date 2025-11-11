"""
Player messages for the Resonate protocol.

This module contains messages specific to clients with the player role, which
handle audio output and synchronized playback. Player clients receive timestamped
audio data, manage their own volume and mute state, and can request different
audio formats based on their capabilities and current conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import PlayerCommand, PlayerStateType


# Client -> Server client/hello player support object
@dataclass
class SupportedAudioFormat(DataClassORJSONMixin):
    """Supported audio format configuration."""

    codec: str
    """Codec identifier (e.g., 'opus', 'flac', 'pcm')."""
    channels: int
    """Supported number of channels (e.g., 1 = mono, 2 = stereo)."""
    sample_rate: int
    """Sample rate in Hz (e.g., 44100, 48000)."""
    bit_depth: int
    """Bit depth for this format (e.g., 16, 24)."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.bit_depth <= 0:
            raise ValueError(f"bit_depth must be positive, got {self.bit_depth}")


@dataclass
class ClientHelloPlayerSupport(DataClassORJSONMixin):
    """Player support configuration - only if player role is set."""

    support_formats: list[SupportedAudioFormat]
    """List of supported audio formats in priority order (first is preferred)."""
    buffer_capacity: int
    """Max size in bytes of compressed audio messages in the buffer that are yet to be played."""
    supported_commands: list[PlayerCommand]
    """Subset of: 'volume', 'mute'."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.buffer_capacity <= 0:
            raise ValueError(f"buffer_capacity must be positive, got {self.buffer_capacity}")

        if not self.support_formats:
            raise ValueError("support_formats cannot be empty")


# Client -> Server: client/state player object
@dataclass
class PlayerStatePayload(DataClassORJSONMixin):
    """Player object in client/state message."""

    state: PlayerStateType
    """State of the player - synchronized or error."""
    volume: int
    """Volume range 0-100."""
    muted: bool
    """Mute state."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if not 0 <= self.volume <= 100:
            raise ValueError(f"Volume must be in range 0-100, got {self.volume}")


# Server -> Client: server/command player object
@dataclass
class PlayerCommandPayload(DataClassORJSONMixin):
    """Player object in server/command message."""

    command: PlayerCommand
    """Command - must be 'volume' or 'mute'."""
    volume: int | None = None
    """Volume range 0-100, only set if command is volume."""
    mute: bool | None = None
    """True to mute, false to unmute, only set if command is mute."""

    def __post_init__(self) -> None:
        """Validate field values and command consistency."""
        if self.command == PlayerCommand.VOLUME:
            if self.volume is None:
                raise ValueError("Volume must be provided when command is 'volume'")
            if not 0 <= self.volume <= 100:
                raise ValueError(f"Volume must be in range 0-100, got {self.volume}")
        elif self.volume is not None:
            raise ValueError(f"Volume should not be provided for command '{self.command.value}'")

        if self.command == PlayerCommand.MUTE:
            if self.mute is None:
                raise ValueError("Mute must be provided when command is 'mute'")
        elif self.mute is not None:
            raise ValueError(f"Mute should not be provided for command '{self.command.value}'")

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


# Client -> Server stream/request-format player object
@dataclass
class StreamRequestFormatPlayer(DataClassORJSONMixin):
    """Request different player stream format (upgrade or downgrade)."""

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
