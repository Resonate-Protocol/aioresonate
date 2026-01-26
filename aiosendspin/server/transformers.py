"""Audio transformers for role-specific encoding/processing.

Transformers convert resampled PCM audio into role-specific output formats.
They are managed by TransformerPool for deduplication across roles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from aiosendspin.server.audio import AudioFormat, _get_av, _resolve_audio_format

if TYPE_CHECKING:
    import av

T = TypeVar("T", bound="AudioTransformer")


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


@dataclass(frozen=True)
class TransformerKey:
    """Unique identifier for a transformer configuration."""

    transformer_type: type
    sample_rate: int
    bit_depth: int
    channels: int


class TransformerPool:
    """Manages shared transformer instances.

    Transformers are keyed by (type, sample_rate, bit_depth, channels).
    Multiple roles with the same configuration share the same transformer,
    enabling encoding deduplication.
    """

    def __init__(self) -> None:
        """Initialize an empty transformer pool."""
        self._transformers: dict[TransformerKey, AudioTransformer] = {}

    def get_or_create(
        self,
        transformer_type: type[T],
        *,
        sample_rate: int,
        bit_depth: int,
        channels: int,
    ) -> T:
        """Get existing transformer or create new one."""
        key = TransformerKey(
            transformer_type=transformer_type,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            channels=channels,
        )
        if key not in self._transformers:
            self._transformers[key] = transformer_type(  # type: ignore[call-arg]
                sample_rate=sample_rate,
                bit_depth=bit_depth,
                channels=channels,
            )
        return self._transformers[key]  # type: ignore[return-value]

    def reset_all(self) -> None:
        """Reset all transformers (called on stream/clear)."""
        for transformer in self._transformers.values():
            transformer.reset()


class PcmPassthrough:
    """Passthrough transformer that returns PCM unchanged.

    Use when a role wants raw PCM audio without encoding.
    """

    def __init__(self, **kwargs: object) -> None:
        """Accept and ignore PCM format parameters."""

    def process(self, pcm: bytes, timestamp_us: int, duration_us: int) -> bytes:  # noqa: ARG002
        """Return PCM data unchanged."""
        return pcm

    def get_header(self) -> bytes | None:
        """No codec header for raw PCM."""
        return None

    def reset(self) -> None:
        """No state to reset."""


class FlacEncoder:
    """FLAC audio encoder transformer."""

    def __init__(self, *, sample_rate: int, bit_depth: int, channels: int) -> None:
        """Initialize FLAC encoder with audio format parameters."""
        self._sample_rate = sample_rate
        self._bit_depth = bit_depth
        self._channels = channels
        self._encoder: av.AudioCodecContext | None = None
        self._codec_header: bytes | None = None
        self._av_format: str | None = None
        self._av_layout: str | None = None
        self._frame_stride: int = (bit_depth // 8) * channels
        self._chunk_samples = int(sample_rate * 0.025)  # 25ms chunks
        self._buffer = bytearray()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize encoder on first use."""
        if self._initialized:
            return

        av = _get_av()
        audio_format = AudioFormat(
            sample_rate=self._sample_rate,
            bit_depth=self._bit_depth,
            channels=self._channels,
        )
        _, self._av_format, self._av_layout = _resolve_audio_format(audio_format)

        self._encoder = av.AudioCodecContext.create("flac", "w")
        self._encoder.sample_rate = self._sample_rate
        self._encoder.layout = self._av_layout
        self._encoder.format = self._av_format
        self._encoder.options = {"compression_level": "5"}

        with av.logging.Capture():
            self._encoder.open()

        header = bytes(self._encoder.extradata) if self._encoder.extradata else b""
        if header:
            self._codec_header = b"fLaC\x80" + len(header).to_bytes(3, "big") + header
        else:
            self._codec_header = None

        self._initialized = True

    def process(self, pcm: bytes, timestamp_us: int, duration_us: int) -> bytes:  # noqa: ARG002
        """Encode PCM to FLAC."""
        self._ensure_initialized()
        assert self._encoder is not None
        av = _get_av()

        self._buffer.extend(pcm)
        output = bytearray()

        while len(self._buffer) >= self._frame_stride * self._chunk_samples:
            chunk_size = self._chunk_samples * self._frame_stride
            chunk_pcm = bytes(self._buffer[:chunk_size])
            del self._buffer[:chunk_size]

            frame = av.AudioFrame(
                format=self._av_format,
                layout=self._av_layout,
                samples=self._chunk_samples,
            )
            frame.sample_rate = self._sample_rate
            frame.planes[0].update(chunk_pcm)

            packets = self._encoder.encode(frame)
            for packet in packets:
                output.extend(bytes(packet))

        return bytes(output)

    def get_header(self) -> bytes | None:
        """Return FLAC streaminfo header."""
        return self._codec_header

    def reset(self) -> None:
        """Reset encoder state."""
        self._encoder = None
        self._codec_header = None
        self._buffer.clear()
        self._initialized = False
