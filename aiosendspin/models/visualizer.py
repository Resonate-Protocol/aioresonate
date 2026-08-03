"""Visualizer role models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, cast

from .base import SendspinConfig, SendspinModel

VisualizerType = Literal["loudness", "f_peak", "spectrum", "beat", "peak", "pitch"]
SupportedVisualizerType = Literal["loudness", "f_peak", "spectrum", "beat", "peak", "pitch"]
SpectrumScale = Literal["lin", "log", "mel"]

_SUPPORTED_TYPES: tuple[SupportedVisualizerType, ...] = (
    "loudness",
    "f_peak",
    "spectrum",
    "beat",
    "peak",
    "pitch",
)


class BeatAvailability(Enum):
    """Server-side declaration of whether beats will arrive for the source.

    `PENDING` (default): beats may arrive via `append_beat_schedule()`. `beat`
    is deferred from `stream/start.types` until the first schedule lands (when
    the client requested it), at which point the role re-emits `stream/start`
    with `beat` added.

    `UNAVAILABLE`: no beats will arrive for this source. `beat` is excluded
    from the negotiated types, and any `append_beat_schedule()` call is a
    no-op until availability changes back.
    """

    PENDING = "pending"
    UNAVAILABLE = "unavailable"


# Client -> Server: client/hello visualizer support object
@dataclass(frozen=True)
class ClientHelloVisualizerSpectrum(SendspinModel):
    """Spectrum configuration from client/hello visualizer support."""

    n_disp_bins: int
    scale: SpectrumScale
    f_min: int
    f_max: int

    def __post_init__(self) -> None:
        """Validate spectrum config bounds."""
        if self.n_disp_bins <= 0:
            raise ValueError(f"n_disp_bins must be > 0, got {self.n_disp_bins}")
        if self.f_min < 0:
            raise ValueError(f"f_min must be >= 0, got {self.f_min}")
        if self.f_max <= self.f_min:
            raise ValueError(f"f_max must be > f_min, got f_min={self.f_min}, f_max={self.f_max}")

    def to_wire_dict(self) -> dict[str, Any]:
        """Serialize to stream/start visualizer.spectrum payload."""
        return self.to_dict()

    class Config(SendspinConfig):
        """Config for json serialization."""

        omit_none = True


@dataclass
class ClientHelloVisualizerSupport(SendspinModel):
    """Visualizer support payload for client/hello visualizer negotiation."""

    buffer_capacity: int
    rate_max: int
    types: list[str]
    spectrum: ClientHelloVisualizerSpectrum | None = None

    @classmethod
    def __pre_deserialize__(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize incoming support payload before dataclass construction."""
        raw_types = payload.get("types")
        if isinstance(raw_types, list):
            deduped: list[str] = []
            for value in raw_types:
                if isinstance(value, str) and value in _SUPPORTED_TYPES and value not in deduped:
                    deduped.append(value)
            payload = dict(payload)
            payload["types"] = deduped
        return payload

    def __post_init__(self) -> None:
        """Validate support object constraints."""
        if not self.types:
            raise ValueError(
                "visualizer support 'types' must contain at least one supported type "
                f"(supported: {list(_SUPPORTED_TYPES)})"
            )
        if self.buffer_capacity <= 0:
            raise ValueError(f"buffer_capacity must be > 0, got {self.buffer_capacity}")
        if self.rate_max <= 0:
            raise ValueError(f"rate_max must be > 0, got {self.rate_max}")
        if "spectrum" in self.types and self.spectrum is None:
            raise ValueError("visualizer support must include 'spectrum' object")

    class Config(SendspinConfig):
        """Config for json serialization."""

        omit_none = True


# Server -> Client: stream/start visualizer object
@dataclass(frozen=True)
class StreamStartVisualizer(SendspinModel):
    """Negotiated visualizer stream config returned in stream/start."""

    types: tuple[SupportedVisualizerType, ...]
    rate_max: int
    tracks_downbeats: bool | None = None
    spectrum: ClientHelloVisualizerSpectrum | None = None

    @classmethod
    def from_support(
        cls,
        support: ClientHelloVisualizerSupport,
        *,
        tracks_downbeats: bool | None = None,
    ) -> StreamStartVisualizer:
        """Create server stream config from validated client support data."""
        stream_types = cast(
            "tuple[SupportedVisualizerType, ...]",
            tuple(typed for typed in support.types if typed in _SUPPORTED_TYPES),
        )
        if not stream_types:
            raise ValueError("visualizer stream must contain at least one supported type")
        # tracks_downbeats is only meaningful when beat is in types. Preserve
        # the caller's tri-state (None = not declared) instead of coercing to
        # False so an "unknown" stays omitted on the wire.
        effective_tracks_downbeats = tracks_downbeats if "beat" in stream_types else None
        return cls(
            types=stream_types,
            rate_max=support.rate_max,
            spectrum=support.spectrum,
            tracks_downbeats=effective_tracks_downbeats,
        )

    def to_wire_dict(self) -> dict[str, Any]:
        """Serialize to stream/start payload format."""
        return self.to_dict()

    class Config(SendspinConfig):
        """Config for json serialization."""

        omit_none = True


# Client -> Server: stream/request-format visualizer object
@dataclass
class StreamRequestFormatVisualizer(SendspinModel):
    """Visualizer stream format renegotiation payload.

    All fields optional; omitted fields keep the prior value.
    """

    types: list[VisualizerType] | None = None
    rate_max: int | None = None
    buffer_capacity: int | None = None
    spectrum: ClientHelloVisualizerSpectrum | None = None

    class Config(SendspinConfig):
        """Config for json serialization."""

        omit_none = True


@dataclass(slots=True)
class VisualizerFrame:
    """Single visualizer frame parsed by clients from binary payloads.

    On the v1 wire each binary message carries one type's data, so any given
    `VisualizerFrame` has exactly one of the optional fields populated.
    """

    timestamp_us: int
    loudness: int | None = None
    f_peak_freq: int | None = None
    f_peak_amp: int | None = None
    spectrum: list[int] | None = None
    peak_strength: int | None = None
    pitch_midi_q88: int | None = None
    pitch_confidence: int | None = None
    # Set for beat frames (msg 17). True at bar boundaries on the v1 wire.
    is_downbeat: bool | None = None


@dataclass(frozen=True, slots=True)
class BeatTiming:
    """A single beat in a visualizer beat schedule."""

    timestamp_us: int
    is_downbeat: bool = False
