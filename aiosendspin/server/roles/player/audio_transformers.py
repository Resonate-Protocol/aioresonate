"""Player-specific audio transformers for PCM and FLAC encoding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from aiosendspin.server.audio import AudioFormat, _get_av, _validate_pcm_buffer_length

if TYPE_CHECKING:
    import av


class PcmPassthrough:
    """Passthrough transformer that chunks PCM into fixed-size frames.

    Use when a role wants raw PCM audio in consistent frame sizes.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        bit_depth: int,
        channels: int,
        chunk_duration_us: int = 25_000,
        options: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize with audio format parameters.

        Args:
            sample_rate: Sample rate in Hz (e.g., 48000).
            bit_depth: Bits per sample (e.g., 16).
            channels: Number of channels (e.g., 2 for stereo).
            chunk_duration_us: Duration of each output frame in microseconds.
        """
        self._sample_rate = sample_rate
        self._chunk_duration_us = chunk_duration_us
        self._frame_stride = (bit_depth // 8) * channels
        self._options = options
        # Calculate frame size: samples = sample_rate * duration_s
        # For 48kHz, 25ms: 48000 * 0.025 = 1200 samples
        # Frame size = samples * frame_stride = 1200 * 4 = 4800 bytes
        chunk_samples = int(sample_rate * chunk_duration_us / 1_000_000)
        self._frame_size = chunk_samples * self._frame_stride
        self._buffer = bytearray()
        # Track timestamp of the first sample in the buffer
        self._pending_timestamp_us: int | None = None
        # Track last input timestamp to detect production gaps
        self._last_input_timestamp_us: int | None = None

    @property
    def frame_duration_us(self) -> int:
        """Duration of each output frame in microseconds."""
        return self._chunk_duration_us

    @property
    def pending_timestamp_us(self) -> int | None:
        """Timestamp of the first buffered sample, or None if buffer is empty."""
        return self._pending_timestamp_us

    def process(self, pcm: bytes, timestamp_us: int, duration_us: int) -> list[bytes]:  # noqa: ARG002
        """Chunk PCM into fixed-size frames.

        Args:
            pcm: Raw PCM audio data.
            timestamp_us: Playback timestamp in microseconds.
            duration_us: Duration of this chunk in microseconds (unused).

        Returns:
            List of fixed-size PCM frames. May be empty if buffering.
        """
        # Detect production gaps: if input timestamp jumped by >1.5s, reset timeline
        # This handles cases where audio production stopped and resumed
        if self._last_input_timestamp_us is not None:
            input_gap = timestamp_us - self._last_input_timestamp_us
            if input_gap > 1_500_000:  # 1.5s threshold
                # Production gap detected - reset timestamp tracking
                self._pending_timestamp_us = timestamp_us
        self._last_input_timestamp_us = timestamp_us

        # Track timestamp of first buffered sample (only if not already set)
        if self._pending_timestamp_us is None:
            self._pending_timestamp_us = timestamp_us

        self._buffer.extend(pcm)
        frames: list[bytes] = []

        while len(self._buffer) >= self._frame_size:
            frame = bytes(self._buffer[: self._frame_size])
            del self._buffer[: self._frame_size]
            frames.append(frame)
            # Advance pending timestamp by one frame duration
            if self._pending_timestamp_us is not None:
                self._pending_timestamp_us += self._chunk_duration_us

        return frames

    def flush(self) -> list[bytes]:
        """Flush remaining buffered audio, padded with silence.

        Returns:
            Final frame padded with silence, or empty list if buffer is empty.
        """
        if not self._buffer:
            return []

        # Pad with silence (zeros) to fill the frame
        padding_needed = self._frame_size - len(self._buffer)
        self._buffer.extend(bytes(padding_needed))
        frame = bytes(self._buffer)
        self._buffer.clear()
        self._pending_timestamp_us = None
        return [frame]

    def get_header(self) -> bytes | None:
        """No codec header for raw PCM."""
        return None

    def reset(self) -> None:
        """Reset internal buffer."""
        self._buffer.clear()
        self._pending_timestamp_us = None
        self._last_input_timestamp_us = None


class FlacEncoder:
    """FLAC audio encoder transformer."""

    def __init__(
        self,
        *,
        sample_rate: int,
        bit_depth: int,
        channels: int,
        chunk_duration_us: int = 25_000,
        options: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize FLAC encoder with audio format parameters.

        Args:
            sample_rate: Sample rate in Hz (e.g., 48000).
            bit_depth: Bits per sample (e.g., 16).
            channels: Number of channels (e.g., 2 for stereo).
            chunk_duration_us: Duration of each output frame in microseconds.
        """
        self._sample_rate = sample_rate
        self._bit_depth = bit_depth
        self._channels = channels
        self._chunk_duration_us = chunk_duration_us
        self._options = options or {}
        self._encoder: av.AudioCodecContext | None = None
        self._codec_header: bytes | None = None
        self._av_format: str | None = None
        self._av_layout: str | None = None
        self._frame_stride: int = (bit_depth // 8) * channels
        self._chunk_samples = int(sample_rate * chunk_duration_us / 1_000_000)
        self._buffer = bytearray()
        self._initialized = False
        # Track stream start and output count for timestamp calculation
        # FLAC codec has internal buffering, so we can't track based on input
        self._stream_start_timestamp_us: int | None = None
        self._output_frame_count: int = 0
        self._first_input_timestamp_us: int | None = None
        self._chunks_encoded_total: int = 0
        # Track last input timestamp to detect production gaps
        self._last_input_timestamp_us: int | None = None

    @property
    def frame_duration_us(self) -> int:
        """Duration of each output frame in microseconds."""
        return self._chunk_duration_us

    @property
    def pending_timestamp_us(self) -> int | None:
        """Timestamp of the next output frame, or None if stream not started."""
        if self._stream_start_timestamp_us is None:
            return None
        return self._stream_start_timestamp_us + self._output_frame_count * self._chunk_duration_us

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
        _, self._av_format, self._av_layout, av_bytes_per_sample = audio_format.resolve_av_format()
        self._frame_stride = av_bytes_per_sample * self._channels

        self._encoder = av.AudioCodecContext.create("flac", "w")
        self._encoder.sample_rate = self._sample_rate
        self._encoder.layout = self._av_layout
        self._encoder.format = self._av_format
        self._encoder.options = {"compression_level": self._options.get("compression_level", "5")}

        with av.logging.Capture():
            self._encoder.open()

        # Update chunk duration to match FLAC's actual block size.
        # FLAC determines its own block size (e.g., 4608 samples = 96ms at 48kHz),
        # regardless of what input frame sizes we use.
        if self._encoder.frame_size:
            self._chunk_samples = self._encoder.frame_size
            self._chunk_duration_us = int(self._chunk_samples / self._sample_rate * 1_000_000)

        header = bytes(self._encoder.extradata) if self._encoder.extradata else b""
        if header:
            self._codec_header = b"fLaC\x80" + len(header).to_bytes(3, "big") + header
        else:
            self._codec_header = None

        self._initialized = True

    def _encode_chunk(self, chunk_pcm: bytes) -> bytes:
        """Encode a single chunk of PCM to FLAC."""
        assert self._encoder is not None
        av = _get_av()
        _validate_pcm_buffer_length(
            chunk_pcm,
            expected=self._chunk_samples * self._frame_stride,
            context="FLAC encoder input",
        )

        frame = av.AudioFrame(
            format=self._av_format,
            layout=self._av_layout,
            samples=self._chunk_samples,
        )
        frame.sample_rate = self._sample_rate
        frame.planes[0].update(chunk_pcm)

        output = bytearray()
        packets = self._encoder.encode(frame)
        for packet in packets:
            output.extend(bytes(packet))
        return bytes(output)

    def process(self, pcm: bytes, timestamp_us: int, duration_us: int) -> list[bytes]:  # noqa: ARG002
        """Encode PCM to FLAC frames.

        Args:
            pcm: Raw PCM audio data.
            timestamp_us: Playback timestamp in microseconds.
            duration_us: Duration of this chunk in microseconds (unused).

        Returns:
            List of encoded FLAC frames. May be empty if buffering.
        """
        self._ensure_initialized()

        # Detect production gaps: if input timestamp jumped by >1.5s, reset timeline
        # This handles cases where audio production stopped and resumed
        if self._last_input_timestamp_us is not None:
            input_gap = timestamp_us - self._last_input_timestamp_us
            if input_gap > 1_500_000:  # 1.5s threshold
                # Production gap detected - reset timestamp tracking
                self._stream_start_timestamp_us = None
                self._output_frame_count = 0
                self._first_input_timestamp_us = timestamp_us
                self._chunks_encoded_total = 0
        self._last_input_timestamp_us = timestamp_us

        # Track first input timestamp for encoder-delay compensation
        if self._first_input_timestamp_us is None:
            self._first_input_timestamp_us = timestamp_us

        self._buffer.extend(pcm)
        frames: list[bytes] = []
        chunk_size = self._chunk_samples * self._frame_stride

        while len(self._buffer) >= chunk_size:
            chunk_pcm = bytes(self._buffer[:chunk_size])
            del self._buffer[:chunk_size]
            encoded = self._encode_chunk(chunk_pcm)
            self._chunks_encoded_total += 1
            if encoded:
                if self._stream_start_timestamp_us is None:
                    assert self._first_input_timestamp_us is not None
                    encoder_delay_chunks = max(self._chunks_encoded_total - 1, 0)
                    self._stream_start_timestamp_us = (
                        self._first_input_timestamp_us
                        + encoder_delay_chunks * self._chunk_duration_us
                    )
                frames.append(encoded)
                # Count output frames for timestamp calculation
                self._output_frame_count += 1

        return frames

    def flush(self) -> list[bytes]:
        """Flush remaining buffered audio, padded with silence.

        Returns:
            Final encoded frame, or empty list if buffer is empty.
        """
        if not self._buffer:
            return []

        self._ensure_initialized()
        chunk_size = self._chunk_samples * self._frame_stride

        # Pad with silence (zeros) to fill the frame
        padding_needed = chunk_size - len(self._buffer)
        self._buffer.extend(bytes(padding_needed))
        chunk_pcm = bytes(self._buffer)
        self._buffer.clear()

        encoded = self._encode_chunk(chunk_pcm)
        if encoded:
            self._output_frame_count += 1
            return [encoded]
        return []

    def get_header(self) -> bytes | None:
        """Return FLAC streaminfo header."""
        return self._codec_header

    def reset(self) -> None:
        """Reset encoder state."""
        self._encoder = None
        self._codec_header = None
        self._buffer.clear()
        self._initialized = False
        self._stream_start_timestamp_us = None
        self._output_frame_count = 0
        self._first_input_timestamp_us = None
        self._chunks_encoded_total = 0
        self._last_input_timestamp_us = None


class OpusEncoder:
    """Opus audio encoder transformer."""

    # Opus only supports these sample rates
    VALID_SAMPLE_RATES = frozenset({8000, 12000, 16000, 24000, 48000})

    def __init__(
        self,
        *,
        sample_rate: int,
        bit_depth: int,  # noqa: ARG002 - Opus uses s16 internally
        channels: int,
        chunk_duration_us: int = 25_000,
        options: Mapping[str, str] | None = None,  # noqa: ARG002 - uses libopus defaults
    ) -> None:
        """Initialize Opus encoder with audio format parameters.

        Args:
            sample_rate: Sample rate in Hz. Must be one of: 8000, 12000, 16000, 24000, 48000.
            bit_depth: Bits per sample (ignored - Opus uses s16 internally).
            channels: Number of channels (e.g., 2 for stereo).
            chunk_duration_us: Duration of each output frame in microseconds.
            options: Encoder options (ignored - uses libopus defaults).
        """
        if sample_rate not in self.VALID_SAMPLE_RATES:
            valid = sorted(self.VALID_SAMPLE_RATES)
            msg = f"Opus only supports sample rates {valid}, got {sample_rate}"
            raise ValueError(msg)

        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_duration_us = chunk_duration_us
        self._encoder: av.AudioCodecContext | None = None
        # Opus uses s16 input format
        self._frame_stride: int = 2 * channels  # 16-bit = 2 bytes per sample
        self._chunk_samples = int(sample_rate * chunk_duration_us / 1_000_000)
        self._buffer = bytearray()
        self._initialized = False
        # Track stream start and output count for timestamp calculation
        self._stream_start_timestamp_us: int | None = None
        self._output_frame_count: int = 0
        self._first_input_timestamp_us: int | None = None
        self._chunks_encoded_total: int = 0
        # Track last input timestamp to detect production gaps
        self._last_input_timestamp_us: int | None = None

    @property
    def frame_duration_us(self) -> int:
        """Duration of each output frame in microseconds."""
        return self._chunk_duration_us

    @property
    def pending_timestamp_us(self) -> int | None:
        """Timestamp of the next output frame, or None if stream not started."""
        if self._stream_start_timestamp_us is None:
            return None
        return self._stream_start_timestamp_us + self._output_frame_count * self._chunk_duration_us

    def _ensure_initialized(self) -> None:
        """Lazily initialize encoder on first use."""
        if self._initialized:
            return

        av = _get_av()

        self._encoder = av.AudioCodecContext.create("libopus", "w")
        self._encoder.sample_rate = self._sample_rate
        self._encoder.layout = "stereo" if self._channels == 2 else "mono"
        self._encoder.format = "s16"  # Opus uses s16 input

        with av.logging.Capture():
            self._encoder.open()

        # Update chunk duration to match Opus's actual frame size
        # Opus typically uses 960 samples = 20ms at 48kHz
        if self._encoder.frame_size:
            self._chunk_samples = self._encoder.frame_size
            self._chunk_duration_us = int(self._chunk_samples / self._sample_rate * 1_000_000)

        self._initialized = True

    def _encode_chunk(self, chunk_pcm: bytes) -> bytes:
        """Encode a single chunk of PCM to Opus."""
        assert self._encoder is not None
        av = _get_av()
        _validate_pcm_buffer_length(
            chunk_pcm,
            expected=self._chunk_samples * self._frame_stride,
            context="Opus encoder input",
        )

        frame = av.AudioFrame(
            format="s16",
            layout="stereo" if self._channels == 2 else "mono",
            samples=self._chunk_samples,
        )
        frame.sample_rate = self._sample_rate
        frame.planes[0].update(chunk_pcm)

        output = bytearray()
        packets = self._encoder.encode(frame)
        for packet in packets:
            output.extend(bytes(packet))
        return bytes(output)

    def process(self, pcm: bytes, timestamp_us: int, duration_us: int) -> list[bytes]:  # noqa: ARG002
        """Encode PCM to Opus frames.

        Args:
            pcm: Raw PCM audio data.
            timestamp_us: Playback timestamp in microseconds.
            duration_us: Duration of this chunk in microseconds (unused).

        Returns:
            List of encoded Opus frames. May be empty if buffering.
        """
        self._ensure_initialized()

        # Detect production gaps: if input timestamp jumped by >1.5s, reset timeline
        if self._last_input_timestamp_us is not None:
            input_gap = timestamp_us - self._last_input_timestamp_us
            if input_gap > 1_500_000:  # 1.5s threshold
                # Production gap detected - reset timestamp tracking
                self._stream_start_timestamp_us = None
                self._output_frame_count = 0
                self._first_input_timestamp_us = timestamp_us
                self._chunks_encoded_total = 0
        self._last_input_timestamp_us = timestamp_us

        # Track first input timestamp for encoder-delay compensation
        if self._first_input_timestamp_us is None:
            self._first_input_timestamp_us = timestamp_us

        self._buffer.extend(pcm)
        frames: list[bytes] = []
        chunk_size = self._chunk_samples * self._frame_stride

        while len(self._buffer) >= chunk_size:
            chunk_pcm = bytes(self._buffer[:chunk_size])
            del self._buffer[:chunk_size]
            encoded = self._encode_chunk(chunk_pcm)
            self._chunks_encoded_total += 1
            if encoded:
                if self._stream_start_timestamp_us is None:
                    assert self._first_input_timestamp_us is not None
                    encoder_delay_chunks = max(self._chunks_encoded_total - 1, 0)
                    self._stream_start_timestamp_us = (
                        self._first_input_timestamp_us
                        + encoder_delay_chunks * self._chunk_duration_us
                    )
                frames.append(encoded)
                self._output_frame_count += 1

        return frames

    def flush(self) -> list[bytes]:
        """Flush remaining buffered audio, padded with silence.

        Returns:
            Final encoded frame, or empty list if buffer is empty.
        """
        if not self._buffer:
            return []

        self._ensure_initialized()
        chunk_size = self._chunk_samples * self._frame_stride

        # Pad with silence (zeros) to fill the frame
        padding_needed = chunk_size - len(self._buffer)
        self._buffer.extend(bytes(padding_needed))
        chunk_pcm = bytes(self._buffer)
        self._buffer.clear()

        encoded = self._encode_chunk(chunk_pcm)
        if encoded:
            self._output_frame_count += 1
            return [encoded]
        return []

    def get_header(self) -> bytes | None:
        """Opus doesn't need a header for raw packets."""
        return None

    def reset(self) -> None:
        """Reset encoder state."""
        self._encoder = None
        self._buffer.clear()
        self._initialized = False
        self._stream_start_timestamp_us = None
        self._output_frame_count = 0
        self._first_input_timestamp_us = None
        self._chunks_encoded_total = 0
        self._last_input_timestamp_us = None
