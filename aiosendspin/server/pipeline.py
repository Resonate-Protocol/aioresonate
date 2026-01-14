"""Pipeline management for parallel audio encoding."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from aiosendspin.models import AudioCodec
from aiosendspin.server.stream import AudioFormat, _get_av, _resolve_audio_format

if TYPE_CHECKING:
    from uuid import UUID

    import av


class PipelineKey(NamedTuple):
    """Unique key for a pipeline: (channel_id, source_format, target_format)."""

    channel_id: UUID
    source_format: AudioFormat
    target_format: AudioFormat


@dataclass(frozen=True)
class EncodedChunk:
    """Encoded audio chunk ready for delivery (no timestamps assigned yet)."""

    data: bytes
    """Encoded audio data."""
    byte_count: int
    """Size of data in bytes."""
    sample_count: int
    """Number of samples in this chunk."""
    duration_us: int
    """Duration in microseconds."""


@dataclass
class _PipelineState:
    """Internal state for a single encoding pipeline."""

    key: PipelineKey
    """Pipeline key for identification."""
    resampler: av.AudioResampler
    """PyAV audio resampler."""
    encoder: av.AudioCodecContext | None
    """PyAV encoder (None for PCM)."""
    codec_header: bytes | None
    """Codec header bytes (e.g., FLAC streaminfo)."""
    chunk_samples: int
    """Number of samples per output chunk."""
    target_frame_stride: int
    """Bytes per frame in target format."""
    target_av_format: str
    """PyAV format string for target."""
    target_layout: str
    """PyAV channel layout for target."""
    source_av_format: str
    """PyAV format string for source."""
    source_av_layout: str
    """PyAV channel layout for source."""
    buffer: bytearray = field(default_factory=bytearray)
    """Resampled PCM buffer awaiting encoding."""


class PipelineManager:
    """
    Manages encoding pipelines for push-based streaming.

    Each pipeline encodes audio from a source format to a target format.
    Pipelines are identified by (channel_id, source_format, target_format).
    """

    def __init__(self) -> None:
        """Create a new PipelineManager."""
        self._pipelines: dict[PipelineKey, _PipelineState] = {}

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

        # Create the pipeline
        self._pipelines[key] = self._create_pipeline_state(key)
        return key

    def _create_pipeline_state(self, key: PipelineKey) -> _PipelineState:
        """Create internal pipeline state with resampler and encoder."""
        av = _get_av()
        source_format = key.source_format
        target_format = key.target_format

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

        # Create encoder
        encoder, codec_header, chunk_samples = self._build_encoder(
            target_format,
            input_audio_layout=target_layout,
            input_audio_format=target_av_format,
        )

        return _PipelineState(
            key=key,
            resampler=resampler,
            encoder=encoder,
            codec_header=codec_header,
            chunk_samples=chunk_samples,
            target_frame_stride=target_bytes_per_sample * target_format.channels,
            target_av_format=target_av_format,
            target_layout=target_layout,
            source_av_format=source_av_format,
            source_av_layout=source_layout,
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
        return self._pipelines[key].codec_header

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

        Args:
            key: Pipeline key to remove.
        """
        self._pipelines.pop(key, None)

    def reset(self) -> None:
        """Clear all pipelines."""
        self._pipelines.clear()

    def process(
        self,
        prepared_by_channel: dict[UUID, tuple[bytes, AudioFormat]],
        pipeline_keys: set[PipelineKey],
    ) -> dict[PipelineKey, list[EncodedChunk]]:
        """
        Process prepared PCM through requested pipelines.

        Args:
            prepared_by_channel: Dict of channel_id -> (pcm_bytes, audio_format).
            pipeline_keys: Set of pipeline keys to process.

        Returns:
            Dict of pipeline_key -> list of EncodedChunks produced.
        """
        result: dict[PipelineKey, list[EncodedChunk]] = {}

        for key in pipeline_keys:
            if key not in self._pipelines:
                continue

            channel_id = key.channel_id
            if channel_id not in prepared_by_channel:
                continue

            pcm_data, source_format = prepared_by_channel[channel_id]
            pipeline = self._pipelines[key]

            # Process PCM through this pipeline
            chunks = self._process_pipeline(pipeline, pcm_data, source_format)
            result[key] = chunks

        return result

    def _process_pipeline(
        self,
        pipeline: _PipelineState,
        pcm_data: bytes,
        source_format: AudioFormat,
    ) -> list[EncodedChunk]:
        """Process PCM data through a single pipeline."""
        av = _get_av()
        chunks: list[EncodedChunk] = []

        # Calculate sample count from input
        bytes_per_sample = source_format.bit_depth // 8
        frame_stride = bytes_per_sample * source_format.channels
        sample_count = len(pcm_data) // frame_stride

        if sample_count == 0:
            return chunks

        # Create input frame
        frame = av.AudioFrame(
            format=pipeline.source_av_format,
            layout=pipeline.source_av_layout,
            samples=sample_count,
        )
        frame.sample_rate = source_format.sample_rate
        frame.planes[0].update(pcm_data)

        # Resample
        out_frames = pipeline.resampler.resample(frame)
        for out_frame in out_frames:
            expected = pipeline.target_frame_stride * out_frame.samples
            pcm_bytes = bytes(out_frame.planes[0])[:expected]
            pipeline.buffer.extend(pcm_bytes)

        # Drain buffer into chunks
        chunks.extend(self._drain_pipeline_buffer(pipeline))

        return chunks

    def _drain_pipeline_buffer(self, pipeline: _PipelineState) -> list[EncodedChunk]:
        """Drain the pipeline buffer into encoded chunks."""
        av = _get_av()
        chunks: list[EncodedChunk] = []
        target_format = pipeline.key.target_format

        frame_stride = pipeline.target_frame_stride
        while len(pipeline.buffer) >= frame_stride * pipeline.chunk_samples:
            chunk_size = pipeline.chunk_samples * frame_stride
            chunk_pcm = bytes(pipeline.buffer[:chunk_size])
            del pipeline.buffer[:chunk_size]

            if pipeline.encoder is None:
                # PCM path: output directly
                duration_us = int(pipeline.chunk_samples * 1_000_000 / target_format.sample_rate)
                chunks.append(
                    EncodedChunk(
                        data=chunk_pcm,
                        byte_count=len(chunk_pcm),
                        sample_count=pipeline.chunk_samples,
                        duration_us=duration_us,
                    )
                )
            else:
                # Encoder path: encode and emit packets
                frame = av.AudioFrame(
                    format=pipeline.target_av_format,
                    layout=pipeline.target_layout,
                    samples=pipeline.chunk_samples,
                )
                frame.sample_rate = target_format.sample_rate
                frame.planes[0].update(chunk_pcm)
                packets = pipeline.encoder.encode(frame)

                for packet in packets:
                    if not packet.duration or packet.duration <= 0:
                        raise ValueError(f"Invalid packet duration: {packet.duration!r}")
                    duration_us = int(packet.duration * 1_000_000 / target_format.sample_rate)
                    chunks.append(
                        EncodedChunk(
                            data=bytes(packet),
                            byte_count=len(bytes(packet)),
                            sample_count=packet.duration,
                            duration_us=duration_us,
                        )
                    )

        return chunks
