"""Audio transformers for role-specific encoding/processing.

Transformers convert resampled PCM audio into role-specific output formats.
They are managed by TransformerPool for deduplication across roles.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable
from uuid import UUID

from aiosendspin.server.transform_keys import normalize_options

T = TypeVar("T", bound="AudioTransformer")


@runtime_checkable
class AudioTransformer(Protocol):
    """Protocol for audio transformers.

    Transformers process PCM audio into role-specific output.
    Examples: FlacEncoder, OpusEncoder, FFTComputer for visualizer.
    """

    @property
    def frame_duration_us(self) -> int:
        """Duration of each output frame in microseconds."""
        ...

    # TODO: why do we pass in the timestamp_us, but never retrieve it?
    # TODO: maybe remove timestamp_us then?
    def process(self, pcm: bytes, timestamp_us: int, duration_us: int) -> list[bytes]:
        """Transform PCM chunk into output frames.

        Args:
            pcm: Raw PCM audio data (already resampled to target format).
            timestamp_us: Playback timestamp in microseconds.
            duration_us: Duration of this chunk in microseconds.

        Returns:
            List of encoded frames. May be empty if buffering incomplete frame.
            May contain multiple frames if input spans multiple frame boundaries.
        """
        ...

    def flush(self) -> list[bytes]:
        """Flush remaining buffered audio at stream end.

        Returns:
            Final frame(s), possibly padded with silence.
        """
        ...

    # TODO: remove this, this should be specific to each transformer
    def get_header(self) -> bytes | None:
        """Return codec header bytes, or None if not applicable.

        For codecs like FLAC, this returns the streaminfo header.
        For PCM or non-codec transformers, returns None.
        """
        ...

    def reset(self) -> None:
        """Reset internal state.

        Called on stream/clear to discard buffered state.
        """
        ...


@dataclass(frozen=True)
class TransformerKey:
    """Unique identifier for a transformer configuration."""

    channel_id: UUID
    transformer_type: type
    sample_rate: int
    bit_depth: int
    channels: int
    frame_duration_us: int
    options: tuple[tuple[str, str], ...]


# TODO: just checking, do we have any issues reusing transformers, i mean they
# TODO: can only handle one stream pushed per instance, 2 will get mixed up.
class TransformerPool:
    """Manages shared transformer instances.

    Transformers are keyed by (channel_id, type, sample_rate, bit_depth, channels, frame).
    Multiple roles with the same configuration share the same transformer,
    enabling encoding deduplication.
    """

    def __init__(self) -> None:
        """Initialize an empty transformer pool."""
        self._transformers: dict[TransformerKey, AudioTransformer] = {}

    # TODO: maybe generically pass in kwargs to the specific AudioTransformer?
    # TODO: but still key it so we reuse existing instances
    def get_or_create(
        self,
        transformer_type: type[T],
        *,
        channel_id: UUID,
        sample_rate: int,
        bit_depth: int,
        channels: int,
        frame_duration_us: int,
        options: Mapping[str, str] | None = None,
    ) -> T:
        """Get existing transformer or create new one."""
        key = TransformerKey(
            channel_id=channel_id,
            transformer_type=transformer_type,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            channels=channels,
            frame_duration_us=frame_duration_us,
            options=normalize_options(options),
        )
        if key not in self._transformers:
            self._transformers[key] = transformer_type(  # type: ignore[call-arg]
                sample_rate=sample_rate,
                bit_depth=bit_depth,
                channels=channels,
                chunk_duration_us=frame_duration_us,
                options=options,
            )
        return self._transformers[key]  # type: ignore[return-value]

    def reset_all(self) -> None:
        """Reset all transformers (called on stream/clear)."""
        for transformer in self._transformers.values():
            transformer.reset()


# TODO: delete this, no backwards compatibility needed
# Re-export player-specific transformers for backwards compatibility
from aiosendspin.server.roles.player.audio_transformers import (  # noqa: E402
    FlacEncoder,
    PcmPassthrough,
)

__all__ = [
    "AudioTransformer",
    "FlacEncoder",
    "PcmPassthrough",
    "TransformerKey",
    "TransformerPool",
]
