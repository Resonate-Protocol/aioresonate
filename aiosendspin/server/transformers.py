"""Audio transformers for role-specific encoding/processing.

Transformers convert resampled PCM audio into role-specific output formats.
They are managed by TransformerPool for deduplication across roles.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AudioTransformer(Protocol):
    """Protocol for audio transformers.

    Transformers process PCM audio into role-specific output.
    Examples: FlacEncoder, OpusEncoder, FFTComputer for visualizer.
    """

    def process(self, pcm: bytes, timestamp_us: int, duration_us: int) -> bytes:
        """Transform PCM chunk into output format.

        Args:
            pcm: Raw PCM audio data (already resampled to target format).
            timestamp_us: Playback timestamp in microseconds.
            duration_us: Duration of this chunk in microseconds.

        Returns:
            Transformed bytes (encoded audio, frequency bins, etc.).
        """
        ...

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
