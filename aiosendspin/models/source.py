"""Source role messages for the Sendspin protocol."""

from __future__ import annotations

from dataclasses import dataclass

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import (
    AudioCodec,
    SourceClientCommand,
    SourceCommand,
    SourceControl,
    SourceSignalType,
    SourceStateType,
)


@dataclass
class SourceFormat(DataClassORJSONMixin):
    """Audio format for a source stream."""

    codec: AudioCodec
    channels: int
    sample_rate: int
    bit_depth: int

    def __post_init__(self) -> None:
        """Validate source audio format values."""
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.bit_depth <= 0:
            raise ValueError(f"bit_depth must be positive, got {self.bit_depth}")


@dataclass
class InputStreamStartSource(DataClassORJSONMixin):
    """Source object in input_stream/start message."""

    codec: AudioCodec
    channels: int
    sample_rate: int
    bit_depth: int
    codec_header: str | None = None

    def __post_init__(self) -> None:
        """Validate source input stream values."""
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.bit_depth <= 0:
            raise ValueError(f"bit_depth must be positive, got {self.bit_depth}")

    class Config(BaseConfig):
        """Mashumaro serialization config."""

        omit_none = True


@dataclass
class InputStreamRequestFormatSource(DataClassORJSONMixin):
    """Source object in input_stream/request-format message."""

    codec: AudioCodec | None = None
    channels: int | None = None
    sample_rate: int | None = None
    bit_depth: int | None = None

    class Config(BaseConfig):
        """Mashumaro serialization config."""

        omit_none = True


@dataclass
class SourceFeatures(DataClassORJSONMixin):
    """Source feature hints."""

    level: bool | None = None
    line_sense: bool | None = None

    class Config(BaseConfig):
        """Mashumaro serialization config."""

        omit_none = True


@dataclass
class ClientHelloSourceSupport(DataClassORJSONMixin):
    """Source support configuration - only if source role is set."""

    supported_formats: list[SourceFormat]
    controls: list[SourceControl] | None = None
    features: SourceFeatures | None = None

    def __post_init__(self) -> None:
        """Validate source support payload."""
        if not self.supported_formats:
            raise ValueError("supported_formats cannot be empty")

    class Config(BaseConfig):
        """Mashumaro serialization config."""

        omit_none = True


@dataclass
class SourceStatePayload(DataClassORJSONMixin):
    """Source object in client/state message."""

    state: SourceStateType
    level: float | None = None
    signal: SourceSignalType | None = None

    def __post_init__(self) -> None:
        """Validate optional source level."""
        if self.level is not None and not 0.0 <= self.level <= 1.0:
            raise ValueError(f"level must be in range 0..1, got {self.level}")

    class Config(BaseConfig):
        """Mashumaro serialization config."""

        omit_none = True


@dataclass
class SourceVadSettings(DataClassORJSONMixin):
    """Voice activity detection settings."""

    threshold_db: float | None = None
    hold_ms: int | None = None

    class Config(BaseConfig):
        """Mashumaro serialization config."""

        omit_none = True


@dataclass
class SourceCommandPayload(DataClassORJSONMixin):
    """Source object in server/command message."""

    command: SourceCommand | None = None
    control: SourceControl | None = None
    vad: SourceVadSettings | None = None

    class Config(BaseConfig):
        """Mashumaro serialization config."""

        omit_none = True


@dataclass
class SourceClientCommandPayload(DataClassORJSONMixin):
    """Source object in client/command message."""

    command: SourceClientCommand

    class Config(BaseConfig):
        """Mashumaro serialization config."""

        omit_none = True


@dataclass
class ControllerSourceItem(DataClassORJSONMixin):
    """Controller-facing source listing entry."""

    id: str
    name: str
    state: SourceStateType
    signal: SourceSignalType | None = None
    selected: bool | None = None
    last_event: SourceClientCommand | None = None
    last_event_ts_us: int | None = None

    class Config(BaseConfig):
        """Mashumaro serialization config."""

        omit_none = True
