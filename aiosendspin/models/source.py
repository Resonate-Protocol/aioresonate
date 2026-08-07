"""
Source messages for the Sendspin protocol.

This module contains messages specific to clients with the source role, which
capture audio from a local input (e.g., AUX/line-in, turntable preamp, Bluetooth
receiver, or microphone) and stream it to the server. Unlike other roles, a source
sends audio to the server; the server remains the single place that resamples,
transcodes, mixes, buffers, and distributes audio to output players. Sources stay
simple: they capture and encode audio, optionally report basic signal presence
(line sensing), and stream timestamped audio frames.
"""

from __future__ import annotations

from dataclasses import dataclass

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import AudioCodec, SourceCommand, SourceSignal


# Client -> Server client/hello source support object
@dataclass
class SourceFeatures(DataClassORJSONMixin):
    """Optional source feature hints."""

    line_sense: bool | None = None
    """True if the source reports `signal` (line sensing) in client/state."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


@dataclass
class ClientHelloSourceSupport(DataClassORJSONMixin):
    """Source support configuration - only if source role is set.

    A source announces its input format per stream in `client_stream/start`;
    there is no format pre-negotiation in `client/hello`. Servers must support
    every codec ('opus', 'flac', 'pcm'), so there is nothing to negotiate.
    """

    features: SourceFeatures | None = None
    """Optional feature hints."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


# Client -> Server: client/state source object
@dataclass
class SourceStatePayload(DataClassORJSONMixin):
    """Source object in client/state message."""

    signal: SourceSignal | None = None
    """Optional line sensing/signal presence, only if 'line_sense' is supported."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


# Server -> Client: server/command source object
@dataclass
class SourceCommandPayload(DataClassORJSONMixin):
    """Source object in server/command message."""

    command: SourceCommand
    """Whether this source streams to the server: 'start' or 'stop'."""


# Client -> Server: client_stream/start source object
@dataclass
class ClientStreamStartSource(DataClassORJSONMixin):
    """Source object in client_stream/start message."""

    codec: AudioCodec
    """Codec of the input stream."""
    channels: int
    """Number of channels."""
    sample_rate: int
    """Sample rate in Hz."""
    bit_depth: int
    """Bit depth."""
    codec_header: str | None = None
    """Base64 encoded codec header (if necessary; e.g., FLAC)."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.bit_depth <= 0:
            raise ValueError(f"bit_depth must be positive, got {self.bit_depth}")

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True
