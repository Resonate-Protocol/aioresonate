"""Pipeline management for parallel audio encoding."""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from aiosendspin.models import AudioCodec
from aiosendspin.server.audio import AudioFormat, _get_av, _resolve_audio_format

if TYPE_CHECKING:
    from uuid import UUID

    import av

# Shared thread pool for CPU-bound encoding work
_ENCODER_POOL: ThreadPoolExecutor | None = None


def _get_encoder_pool() -> ThreadPoolExecutor:
    """Get or create the shared encoder thread pool."""
    global _ENCODER_POOL  # noqa: PLW0603
    if _ENCODER_POOL is None:
        # Use a small pool since encoding is CPU-bound and doesn't benefit
        # from too much parallelism on typical hardware
        _ENCODER_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="encoder")
    return _ENCODER_POOL


class PipelineKey(NamedTuple):
    """Unique key for a pipeline: (channel_id, source_format, target_format)."""

    channel_id: UUID
    source_format: AudioFormat
    target_format: AudioFormat


class ResamplerKey(NamedTuple):
    """Key for sharing resamplers: (channel_id, source_format, target PCM params).

    Resamplers convert PCM from source format to target sample rate/channels/bit_depth.
    The codec is irrelevant for resampling, so multiple target formats with different
    codecs but the same PCM parameters can share a resampler.
    """

    channel_id: UUID
    source_format: AudioFormat
    target_sample_rate: int
    target_channels: int
    target_bit_depth: int


class EncoderKey(NamedTuple):
    """Key for sharing encoders within a single stream.

    Encoders are stream-stateful and must not be shared across independent streams
    (e.g., different channels). We key encoders by the resampler output parameters
    for a specific channel + the codec.
    """

    resampler_key: ResamplerKey
    codec: AudioCodec


@dataclass(frozen=True)
class EncodedChunk:
    """Encoded audio chunk ready for delivery (no timestamps assigned yet)."""

    timestamp_us: int
    """Start timestamp for this chunk in microseconds."""
    data: bytes
    """Encoded audio data."""
    byte_count: int
    """Size of data in bytes."""
    sample_count: int
    """Number of samples in this chunk."""
    duration_us: int
    """Duration in microseconds."""


@dataclass
class _ResamplerState:
    """Shared resampler state keyed by ResamplerKey."""

    key: ResamplerKey
    """Resampler key for identification."""
    resampler: av.AudioResampler
    """PyAV audio resampler."""
    source_av_format: str
    """PyAV format string for source."""
    source_av_layout: str
    """PyAV channel layout for source."""
    target_av_format: str
    """PyAV format string for target (after resampling)."""
    target_layout: str
    """PyAV channel layout for target."""
    target_frame_stride: int
    """Bytes per frame in target format."""
    pending_timestamp_us: int | None = None
    """Timestamp of the earliest audio sample not yet emitted by this resampler."""


@dataclass
class _EncoderState:
    """Shared encoder state keyed by EncoderKey."""

    key: EncoderKey
    """Encoder key for identification."""
    encoder: av.AudioCodecContext | None
    """PyAV encoder (None for PCM)."""
    codec_header: bytes | None
    """Codec header bytes (e.g., FLAC streaminfo)."""
    chunk_samples: int
    """Number of samples per output chunk."""


@dataclass
class _PipelineState:
    """Internal state for a single encoding pipeline.

    Each pipeline references shared resampler and encoder states.
    """

    key: PipelineKey
    """Pipeline key for identification."""
    resampler_key: ResamplerKey
    """Key to the shared resampler state."""
    encoder_key: EncoderKey
    """Key to the shared encoder state."""
    buffer: bytearray = field(default_factory=bytearray)
    """Resampled PCM buffer awaiting encoding."""
    buffer_start_timestamp_us: int | None = None
    """Timestamp of the first sample in buffer."""


class PipelineManager:
    """
    Manages encoding pipelines for push-based streaming.

    Uses two-level keying to share resamplers and encoders:
    - Resamplers are shared when source format and target PCM params match
    - Encoders are shared when target format (including codec) matches

    Pipelines are identified by (channel_id, source_format, target_format).
    """

    def __init__(self) -> None:
        """Create a new PipelineManager."""
        self._pipelines: dict[PipelineKey, _PipelineState] = {}
        self._resamplers: dict[ResamplerKey, _ResamplerState] = {}
        self._encoders: dict[EncoderKey, _EncoderState] = {}

    def add_pipeline(
        self,
        *,
        channel_id: UUID,
        source_format: AudioFormat,
        target_format: AudioFormat,
    ) -> PipelineKey:
        """
        Add or get an encoding pipeline.

        If a pipeline with the same (channel_id, source_format, target_format)
        already exists, returns the existing key (deduplication).
        Resamplers and encoders are shared across pipelines when possible.

        Args:
            channel_id: The channel this pipeline encodes from.
            source_format: Source audio format (input PCM).
            target_format: Target audio format (output).

        Returns:
            PipelineKey identifying this pipeline.
        """
        key = PipelineKey(
            channel_id=channel_id,
            source_format=source_format,
            target_format=target_format,
        )

        if key in self._pipelines:
            return key

        # Create the pipeline with shared resampler and encoder
        self._pipelines[key] = self._create_pipeline_state(key)
        return key

    def _create_pipeline_state(self, key: PipelineKey) -> _PipelineState:
        """Create internal pipeline state, reusing shared resampler and encoder."""
        source_format = key.source_format
        target_format = key.target_format

        # Create or get shared resampler
        resampler_key = ResamplerKey(
            channel_id=key.channel_id,
            source_format=source_format,
            target_sample_rate=target_format.sample_rate,
            target_channels=target_format.channels,
            target_bit_depth=target_format.bit_depth,
        )
        if resampler_key not in self._resamplers:
            self._resamplers[resampler_key] = self._create_resampler_state(
                resampler_key, source_format, target_format
            )

        # Create or get encoder for this stream + codec.
        # Note: encoders must not be shared across channels.
        encoder_key = EncoderKey(resampler_key=resampler_key, codec=target_format.codec)
        if encoder_key not in self._encoders:
            resampler_state = self._resamplers[resampler_key]
            self._encoders[encoder_key] = self._create_encoder_state(
                encoder_key,
                target_format,
                input_audio_layout=resampler_state.target_layout,
                input_audio_format=resampler_state.target_av_format,
            )

        return _PipelineState(
            key=key,
            resampler_key=resampler_key,
            encoder_key=encoder_key,
        )

    def _create_resampler_state(
        self,
        key: ResamplerKey,
        source_format: AudioFormat,
        target_format: AudioFormat,
    ) -> _ResamplerState:
        """Create a new resampler state."""
        av = _get_av()

        # Get source format params
        _source_bytes_per_sample, source_av_format, source_layout = _resolve_audio_format(
            source_format
        )

        # Get target format params
        target_bytes_per_sample, target_av_format, target_layout = _resolve_audio_format(
            target_format
        )

        # Create resampler
        resampler = av.AudioResampler(
            format=target_av_format,
            layout=target_layout,
            rate=target_format.sample_rate,
        )

        return _ResamplerState(
            key=key,
            resampler=resampler,
            source_av_format=source_av_format,
            source_av_layout=source_layout,
            target_av_format=target_av_format,
            target_layout=target_layout,
            target_frame_stride=target_bytes_per_sample * target_format.channels,
        )

    def _create_encoder_state(
        self,
        key: EncoderKey,
        target_format: AudioFormat,
        *,
        input_audio_layout: str,
        input_audio_format: str,
    ) -> _EncoderState:
        """Create a new encoder state."""
        encoder, codec_header, chunk_samples = self._build_encoder(
            target_format,
            input_audio_layout=input_audio_layout,
            input_audio_format=input_audio_format,
        )

        return _EncoderState(
            key=key,
            encoder=encoder,
            codec_header=codec_header,
            chunk_samples=chunk_samples,
        )

    def _build_encoder(
        self,
        audio_format: AudioFormat,
        *,
        input_audio_layout: str,
        input_audio_format: str,
    ) -> tuple[av.AudioCodecContext | None, bytes | None, int]:
        """Create and configure an encoder for the target audio format."""
        if audio_format.codec == AudioCodec.PCM:
            samples_per_chunk = int(audio_format.sample_rate * 0.025)
            return None, None, samples_per_chunk

        av = _get_av()
        codec = "libopus" if audio_format.codec == AudioCodec.OPUS else audio_format.codec.value

        encoder = av.AudioCodecContext.create(codec, "w")
        encoder.sample_rate = audio_format.sample_rate
        encoder.layout = input_audio_layout
        encoder.format = input_audio_format
        if audio_format.codec == AudioCodec.FLAC:
            encoder.options = {"compression_level": "5"}

        with av.logging.Capture():
            encoder.open()

        header = bytes(encoder.extradata) if encoder.extradata else b""
        if audio_format.codec == AudioCodec.FLAC and header:
            # For FLAC, construct proper FLAC stream header
            # See https://datatracker.ietf.org/doc/rfc9639/ Section 8.1
            header = b"fLaC\x80" + len(header).to_bytes(3, "big") + header

        # Calculate samples per chunk
        if audio_format.codec == AudioCodec.FLAC:
            samples_per_chunk = int(audio_format.sample_rate * 0.025)
        elif encoder.frame_size and encoder.frame_size > 0:
            samples_per_chunk = int(encoder.frame_size)
        else:
            msg = f"Codec {audio_format.codec.value} encoder has invalid frame_size"
            raise ValueError(f"{msg}: {encoder.frame_size}")

        return encoder, header if header else None, samples_per_chunk

    def has_pipeline(self, key: PipelineKey) -> bool:
        """Check if a pipeline exists."""
        return key in self._pipelines

    def get_codec_header(self, key: PipelineKey) -> bytes | None:
        """
        Get the codec header for a pipeline.

        Args:
            key: Pipeline key.

        Returns:
            Codec header bytes, or None for PCM pipelines.
        """
        if key not in self._pipelines:
            return None
        pipeline = self._pipelines[key]
        encoder_state = self._encoders.get(pipeline.encoder_key)
        if encoder_state is None:
            return None
        return encoder_state.codec_header

    def get_codec_header_b64(self, key: PipelineKey) -> str | None:
        """
        Get the codec header as base64 string.

        Args:
            key: Pipeline key.

        Returns:
            Base64-encoded codec header, or None for PCM pipelines.
        """
        header = self.get_codec_header(key)
        if header is None:
            return None
        return base64.b64encode(header).decode()

    def remove_pipeline(self, key: PipelineKey) -> None:
        """
        Remove a pipeline.

        Note: Shared resampler and encoder states are not removed since
        they may be used by other pipelines. They will be cleared on reset().

        Args:
            key: Pipeline key to remove.
        """
        self._pipelines.pop(key, None)

    def reset(self) -> None:
        """Clear all pipelines, resamplers, and encoders."""
        self._pipelines.clear()
        self._resamplers.clear()
        self._encoders.clear()

    def process(
        self,
        prepared_by_channel: dict[UUID, tuple[bytes, AudioFormat, int]],
        pipeline_keys: set[PipelineKey],
    ) -> dict[PipelineKey, list[EncodedChunk]]:
        """
        Process prepared PCM through requested pipelines (sync version).

        Args:
            prepared_by_channel: Dict of channel_id -> (pcm_bytes, audio_format,
                start_timestamp_us).
            pipeline_keys: Set of pipeline keys to process.

        Returns:
            Dict of pipeline_key -> list of EncodedChunks produced.
        """
        result: dict[PipelineKey, list[EncodedChunk]] = {}

        # Group pipelines by shared resampler key so we only resample once per key.
        pipelines_by_resampler: dict[ResamplerKey, list[_PipelineState]] = {}
        for key in pipeline_keys:
            pipeline = self._pipelines.get(key)
            if pipeline is None:
                continue
            pipelines_by_resampler.setdefault(pipeline.resampler_key, []).append(pipeline)

        # Resample once per resampler key, then fan-out to pipeline buffers.
        for resampler_key, pipelines in pipelines_by_resampler.items():
            channel_id = resampler_key.channel_id
            if channel_id not in prepared_by_channel:
                continue

            pcm_data, source_format, input_start_timestamp_us = prepared_by_channel[channel_id]
            resampler_state = self._resamplers[resampler_key]
            if resampler_state.pending_timestamp_us is None:
                resampler_state.pending_timestamp_us = input_start_timestamp_us

            resampled_pcm = self._resample_to_pcm_bytes(resampler_state, pcm_data, source_format)
            if not resampled_pcm:
                continue

            resampled_start_ts = resampler_state.pending_timestamp_us
            assert resampled_start_ts is not None

            sample_count = len(resampled_pcm) // resampler_state.target_frame_stride
            duration_us = int(sample_count * 1_000_000 / resampler_key.target_sample_rate)
            resampler_state.pending_timestamp_us += duration_us

            for pipeline in pipelines:
                if not pipeline.buffer and pipeline.buffer_start_timestamp_us is None:
                    pipeline.buffer_start_timestamp_us = resampled_start_ts
                pipeline.buffer.extend(resampled_pcm)

        # Drain each requested pipeline buffer into encoded chunks.
        for key in pipeline_keys:
            pipeline = self._pipelines.get(key)
            if pipeline is None:
                continue
            chunks = self._drain_pipeline_buffer(pipeline)
            result[key] = chunks

        return result

    async def process_async(
        self,
        prepared_by_channel: dict[UUID, tuple[bytes, AudioFormat, int]],
        pipeline_keys: set[PipelineKey],
    ) -> dict[PipelineKey, list[EncodedChunk]]:
        """
        Process prepared PCM through requested pipelines (async version).

        Runs encoding in a thread pool to avoid blocking the event loop.
        Uses a shared ThreadPoolExecutor for CPU-bound encoding work.

        Args:
            prepared_by_channel: Dict of channel_id -> (pcm_bytes, audio_format,
                start_timestamp_us).
            pipeline_keys: Set of pipeline keys to process.

        Returns:
            Dict of pipeline_key -> list of EncodedChunks produced.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _get_encoder_pool(),
            self.process,
            prepared_by_channel,
            pipeline_keys,
        )

    def _resample_to_pcm_bytes(
        self,
        resampler_state: _ResamplerState,
        pcm_data: bytes,
        source_format: AudioFormat,
    ) -> bytes:
        """Resample PCM data to the target PCM format for a resampler key."""
        av = _get_av()

        # Calculate sample count from input
        bytes_per_sample = source_format.bit_depth // 8
        frame_stride = bytes_per_sample * source_format.channels
        sample_count = len(pcm_data) // frame_stride

        if sample_count == 0:
            return b""

        # Create input frame
        frame = av.AudioFrame(
            format=resampler_state.source_av_format,
            layout=resampler_state.source_av_layout,
            samples=sample_count,
        )
        frame.sample_rate = source_format.sample_rate
        frame.planes[0].update(pcm_data)

        # Resample
        out_frames = resampler_state.resampler.resample(frame)
        out_pcm = bytearray()
        for out_frame in out_frames:
            expected = resampler_state.target_frame_stride * out_frame.samples
            pcm_bytes = bytes(out_frame.planes[0])[:expected]
            out_pcm.extend(pcm_bytes)

        return bytes(out_pcm)

    def _drain_pipeline_buffer(self, pipeline: _PipelineState) -> list[EncodedChunk]:
        """Drain the pipeline buffer into encoded chunks using shared encoder."""
        av = _get_av()
        chunks: list[EncodedChunk] = []
        target_format = pipeline.key.target_format

        # Get shared resampler and encoder states
        resampler_state = self._resamplers[pipeline.resampler_key]
        encoder_state = self._encoders[pipeline.encoder_key]

        frame_stride = resampler_state.target_frame_stride
        chunk_samples = encoder_state.chunk_samples

        while len(pipeline.buffer) >= frame_stride * chunk_samples:
            if pipeline.buffer_start_timestamp_us is None:
                raise RuntimeError("Pipeline buffer has data without a timestamp")
            chunk_size = chunk_samples * frame_stride
            chunk_pcm = bytes(pipeline.buffer[:chunk_size])
            del pipeline.buffer[:chunk_size]
            chunk_timestamp_us = pipeline.buffer_start_timestamp_us

            if encoder_state.encoder is None:
                # PCM path: output directly
                duration_us = int(chunk_samples * 1_000_000 / target_format.sample_rate)
                chunks.append(
                    EncodedChunk(
                        timestamp_us=chunk_timestamp_us,
                        data=chunk_pcm,
                        byte_count=len(chunk_pcm),
                        sample_count=chunk_samples,
                        duration_us=duration_us,
                    )
                )
                pipeline.buffer_start_timestamp_us += duration_us
            else:
                # Encoder path: encode and emit packets
                frame = av.AudioFrame(
                    format=resampler_state.target_av_format,
                    layout=resampler_state.target_layout,
                    samples=chunk_samples,
                )
                frame.sample_rate = target_format.sample_rate
                frame.planes[0].update(chunk_pcm)
                packets = encoder_state.encoder.encode(frame)

                for packet in packets:
                    if not packet.duration or packet.duration <= 0:
                        raise ValueError(f"Invalid packet duration: {packet.duration!r}")
                    duration_us = int(packet.duration * 1_000_000 / target_format.sample_rate)
                    chunks.append(
                        EncodedChunk(
                            timestamp_us=chunk_timestamp_us,
                            data=bytes(packet),
                            byte_count=len(bytes(packet)),
                            sample_count=packet.duration,
                            duration_us=duration_us,
                        )
                    )
                    chunk_timestamp_us += duration_us
                pipeline.buffer_start_timestamp_us = chunk_timestamp_us

        if not pipeline.buffer:
            pipeline.buffer_start_timestamp_us = None

        return chunks
