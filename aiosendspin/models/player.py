"""
Player messages for the Sendspin protocol.

This module contains messages specific to clients with the player role, which
handle audio output and synchronized playback. Player clients receive timestamped
audio data, manage their own volume and mute state, and can request different
audio formats based on their capabilities and current conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import AudioCodec, PlayerCommand, PlayerStateType


# Client -> Server client/hello player support object
@dataclass
class SupportedAudioFormat(DataClassORJSONMixin):
    """Supported audio format configuration."""

    codec: AudioCodec
    """Codec identifier."""
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

    supported_formats: list[SupportedAudioFormat]
    """List of supported audio formats in priority order (first is preferred)."""
    buffer_capacity: int
    """Max size in bytes of compressed audio messages in the buffer that are yet to be played."""
    supported_commands: list[PlayerCommand]
    """Subset of: 'volume', 'mute'."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.buffer_capacity <= 0:
            raise ValueError(f"buffer_capacity must be positive, got {self.buffer_capacity}")

        if not self.supported_formats:
            raise ValueError("supported_formats cannot be empty")


# Client -> Server: client/state player object
@dataclass
class PlayerStatePayload(DataClassORJSONMixin):
    """Player object in client/state message."""

    # DEPRECATED(before-spec-pr-50): Remove once all clients send state at client level.
    # State should now be sent at the ClientStatePayload level, not in the player object.
    state: PlayerStateType | None = None
    """
    State of the player - should always be 'synchronized' unless there is
    an error preventing current or future playback (unable to keep up,
    issues keeping the clock in sync, etc).

    DEPRECATED: State should now be sent at the client/state level, not here.
    """
    volume: int | None = None
    """Volume range 0-100, only included if 'volume' in supported_commands."""
    muted: bool | None = None
    """Mute state, only included if 'mute' in supported_commands."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.volume is not None and not 0 <= self.volume <= 100:
            raise ValueError(f"Volume must be in range 0-100, got {self.volume}")

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


# Server -> Client: server/command player object
@dataclass
class PlayerCommandPayload(DataClassORJSONMixin):
    """Player object in server/command message."""

    command: PlayerCommand
    """
    Command - must be 'volume' or 'mute', and must be one of the values
    listed in supported_commands from player_support in client/hello.
    """
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

    codec: AudioCodec | None = None
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

    codec: AudioCodec
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
