"""High-level streaming pipeline primitives."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections import defaultdict, deque
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, cast

import av
from av.logging import Capture

from aioresonate.models import BinaryMessageType, pack_binary_header_raw
from aioresonate.models.player import StreamStartPlayer

logger = logging.getLogger(__name__)


class AudioCodec(Enum):
    """Supported audio codecs."""

    PCM = "pcm"
    FLAC = "flac"
    OPUS = "opus"


@dataclass(frozen=True)
class AudioFormat:
    """Audio format of a stream."""

    sample_rate: int
    """Sample rate in Hz (e.g., 44100, 48000)."""
    bit_depth: int
    """Bit depth in bits per sample (16 or 24)."""
    channels: int
    """Number of audio channels (1 for mono, 2 for stereo)."""
    codec: AudioCodec = AudioCodec.PCM
    """Audio codec of the stream."""


class BufferedChunk(NamedTuple):
    """Represents compressed audio bytes scheduled for playback."""

    end_time_us: int
    """Absolute timestamp when these bytes should be fully consumed."""
    byte_count: int
    """Compressed byte count occupying the device buffer."""


class BufferTracker:
    """
    Track buffered compressed audio for a client and apply backpressure when needed.

    This class monitors the amount of compressed audio data buffered on a client device
    and ensures the server doesn't exceed the client's buffer capacity by applying
    backpressure when necessary.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        client_id: str,
        capacity_bytes: int,
    ) -> None:
        """
        Initialize the buffer tracker for a client.

        Args:
            loop: The event loop for timing calculations.
            logger: Logger instance for this tracker.
            client_id: Identifier for the client being tracked.
            capacity_bytes: Maximum buffer capacity in bytes reported by the client.
        """
        self._loop = loop
        self.client_id = client_id
        self.capacity_bytes = capacity_bytes
        self.buffered_chunks: deque[BufferedChunk] = deque()
        self.buffered_bytes = 0
        self.max_usage_bytes = 0

    def prune_consumed(self, now_us: int | None = None) -> int:
        """Drop finished chunks and return the timestamp used for the calculation."""
        if now_us is None:
            now_us = int(self._loop.time() * 1_000_000)
        while self.buffered_chunks and self.buffered_chunks[0].end_time_us <= now_us:
            self.buffered_bytes -= self.buffered_chunks.popleft().byte_count
        self.buffered_bytes = max(self.buffered_bytes, 0)
        return now_us

    async def wait_for_capacity(self, bytes_needed: int, expected_end_us: int) -> None:
        """Block until the device buffer can accept bytes_needed more bytes."""
        if bytes_needed <= 0:
            return
        if bytes_needed >= self.capacity_bytes:
            # TODO: raise exception instead?
            logger.warning(
                "Chunk size %s exceeds reported buffer capacity %s for client %s",
                bytes_needed,
                self.capacity_bytes,
                self.client_id,
            )
            return

        while True:
            now_us = self.prune_consumed()
            projected_usage = self.buffered_bytes + bytes_needed
            if projected_usage <= self.capacity_bytes:
                # Returning here keeps the producer running because we are below capacity.
                return

            sleep_target_us = (
                self.buffered_chunks[0].end_time_us if self.buffered_chunks else expected_end_us
            )
            sleep_us = sleep_target_us - now_us
            if sleep_us <= 0:
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(sleep_us / 1_000_000)

    def register(self, end_time_us: int, byte_count: int) -> None:
        """Record bytes added to the buffer finishing at end_time_us."""
        if byte_count <= 0:
            return
        self.buffered_chunks.append(BufferedChunk(end_time_us, byte_count))
        self.buffered_bytes += byte_count
        self.max_usage_bytes = max(self.max_usage_bytes, self.buffered_bytes)


def build_encoder_for_format(
    audio_format: AudioFormat,
    *,
    input_audio_layout: str,
    input_audio_format: str,
) -> tuple[av.AudioCodecContext | None, str | None, int]:
    """Create and configure an encoder for the target audio format."""
    if audio_format.codec == AudioCodec.PCM:
        samples_per_chunk = int(audio_format.sample_rate * 0.025)
        return None, None, samples_per_chunk

    encoder = cast(
        "av.AudioCodecContext", av.AudioCodecContext.create(audio_format.codec.value, "w")
    )
    encoder.sample_rate = audio_format.sample_rate
    encoder.layout = input_audio_layout
    encoder.format = input_audio_format
    if audio_format.codec == AudioCodec.FLAC:
        encoder.options = {"compression_level": "5"}

    with Capture() as logs:
        encoder.open()
    if logger is not None:
        for log in logs:
            logger.debug("Opening AudioCodecContext log from av: %s", log)

    header = bytes(encoder.extradata) if encoder.extradata else b""
    if audio_format.codec == AudioCodec.FLAC and header:
        # For FLAC, we need to construct a proper FLAC stream header ourselves
        # since ffmpeg only provides the StreamInfo metadata block in extradata:
        # See https://datatracker.ietf.org/doc/rfc9639/ Section 8.1

        # FLAC stream signature (4 bytes): "fLaC"
        # Metadata block header (4 bytes):
        # - Bit 0: last metadata block (1 since we only have one)
        # - Bits 1-7: block type (0 for StreamInfo)
        # - Next 3 bytes: block length of the next metadata block in bytes
        # StreamInfo block (34 bytes): as provided by ffmpeg
        header = b"fLaC\x80" + len(header).to_bytes(3, "big") + header

    codec_header_b64 = base64.b64encode(header).decode()
    samples_per_chunk = (
        int(encoder.frame_size) if encoder.frame_size else int(audio_format.sample_rate * 0.025)
    )
    return encoder, codec_header_b64, samples_per_chunk


@dataclass
class Channel:
    """Represents a PCM-producing audio channel."""

    name: str
    audio_format: AudioFormat
    source: AsyncGenerator[bytes, None]


@dataclass(frozen=True)
class ChannelSpec:
    """Expanded channel metadata used by the Streamer."""

    name: str
    audio_format: AudioFormat
    bytes_per_sample: int
    frame_stride: int
    av_format: str
    av_layout: str


@dataclass
class ClientStreamConfig:
    """Configuration for delivering audio to a player."""

    client_id: str
    target_format: AudioFormat
    buffer_capacity_bytes: int
    send: Callable[[bytes], None] | None = None


@dataclass
class PreparedChunkState:
    """Prepared chunk shared between all subscribers of a pipeline."""

    payload: bytes
    start_time_us: int
    end_time_us: int
    sample_count: int
    byte_count: int
    refcount: int


@dataclass
class PipelineState:
    """Holds state for a distinct channel/format/chunk-size pipeline."""

    channel: ChannelSpec
    target_format: AudioFormat
    target_bytes_per_sample: int
    target_frame_stride: int
    target_av_format: str
    target_layout: str
    chunk_samples: int
    resampler: av.AudioResampler
    encoder: av.AudioCodecContext | None
    codec_header_b64: str | None
    buffer: bytearray = field(default_factory=bytearray)
    prepared: deque[PreparedChunkState] = field(default_factory=deque)
    subscribers: list[str] = field(default_factory=list)
    samples_produced: int = 0
    samples_enqueued: int = 0
    samples_encoded: int = 0
    flushed: bool = False


@dataclass
class PlayerState:
    """Tracks delivery state for a single player."""

    config: ClientStreamConfig
    pipeline_key: tuple[str, AudioFormat]
    queue: deque[PreparedChunkState] = field(default_factory=deque)
    buffer_tracker: BufferTracker | None = None


class MediaStream:
    """Container for a single audio stream with its format."""

    def __init__(
        self,
        *,
        main_channel: AsyncGenerator[bytes, None],
        audio_format: AudioFormat,
        supports_seek: bool = False,
    ) -> None:
        """Initialise the media stream with audio source and format."""
        self._source = main_channel
        self._audio_format = audio_format
        self.supports_seek = supports_seek

    @property
    def source(self) -> AsyncGenerator[bytes, None]:
        """Return the audio source generator."""
        return self._source

    @property
    def audio_format(self) -> AudioFormat:
        """Return the audio format of the stream."""
        return self._audio_format


class Streamer:
    """Adapts incoming channel data to player-specific formats."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        play_start_time_us: int,
    ) -> None:
        """Create a streamer bound to the event loop and playback start time."""
        self._loop = loop
        self._play_start_time_us = play_start_time_us
        self._channels: dict[str, ChannelSpec] = {}
        self._pipelines: dict[tuple[str, AudioFormat], PipelineState] = {}
        self._pipelines_by_channel: dict[str, list[PipelineState]] = defaultdict(list)
        self._players: dict[str, PlayerState] = {}
        self._last_chunk_end_us: int | None = None

    def configure(
        self,
        *,
        channels: Mapping[str, AudioFormat],
        clients: Iterable[ClientStreamConfig],
    ) -> dict[str, StreamStartPlayer]:
        """Configure pipelines for the provided clients and channels."""
        self._channels.clear()
        self._pipelines.clear()
        self._pipelines_by_channel.clear()
        self._players.clear()
        self._last_chunk_end_us = None

        for name, audio_format in channels.items():
            bytes_per_sample, av_format, av_layout = _resolve_audio_format(audio_format)
            self._channels[name] = ChannelSpec(
                name=name,
                audio_format=audio_format,
                bytes_per_sample=bytes_per_sample,
                frame_stride=bytes_per_sample * audio_format.channels,
                av_format=av_format,
                av_layout=av_layout,
            )

        start_payloads: dict[str, StreamStartPlayer] = {}

        for client_cfg in clients:
            if client_cfg.send is None:
                raise ValueError(f"Client {client_cfg.client_id} missing send callback")

            channel_name = next(iter(self._channels.keys()))
            if channel_name not in self._channels:
                raise KeyError(
                    f"Unknown channel {channel_name!r} requested by {client_cfg.client_id}"
                )

            pipeline_key = (
                channel_name,
                client_cfg.target_format,
            )
            pipeline = self._pipelines.get(pipeline_key)
            if pipeline is None:
                channel_spec = self._channels[channel_name]
                (
                    target_bytes_per_sample,
                    target_av_format,
                    target_layout,
                ) = _resolve_audio_format(client_cfg.target_format)

                resampler = av.AudioResampler(
                    format=target_av_format,
                    layout=target_layout,
                    rate=client_cfg.target_format.sample_rate,
                )
                encoder, codec_header_b64, chunk_samples = build_encoder_for_format(
                    client_cfg.target_format,
                    input_audio_layout=target_layout,
                    input_audio_format=target_av_format,
                )
                pipeline = PipelineState(
                    channel=channel_spec,
                    target_format=client_cfg.target_format,
                    target_bytes_per_sample=target_bytes_per_sample,
                    target_frame_stride=target_bytes_per_sample * client_cfg.target_format.channels,
                    target_av_format=target_av_format,
                    target_layout=target_layout,
                    chunk_samples=chunk_samples,
                    resampler=resampler,
                    encoder=encoder,
                    codec_header_b64=codec_header_b64,
                )
                self._pipelines[pipeline_key] = pipeline
                self._pipelines_by_channel[channel_spec.name].append(pipeline)

            pipeline.subscribers.append(client_cfg.client_id)

            buffer_tracker = BufferTracker(
                loop=self._loop,
                client_id=client_cfg.client_id,
                capacity_bytes=client_cfg.buffer_capacity_bytes,
            )
            player_state = PlayerState(
                config=client_cfg,
                pipeline_key=pipeline_key,
                buffer_tracker=buffer_tracker,
            )
            self._players[client_cfg.client_id] = player_state

            start_payloads[client_cfg.client_id] = StreamStartPlayer(
                codec=client_cfg.target_format.codec.value,
                sample_rate=client_cfg.target_format.sample_rate,
                channels=client_cfg.target_format.channels,
                bit_depth=client_cfg.target_format.bit_depth,
                codec_header=pipeline.codec_header_b64,
            )

        return start_payloads

    def prepare(self, channel_payloads: Mapping[str, bytes]) -> None:
        """Ingest raw PCM data for each channel and prepare per-player chunks."""
        for name, payload in channel_payloads.items():
            pipeline_list = self._pipelines_by_channel.get(name)
            if not pipeline_list:
                continue
            channel_spec = self._channels[name]
            if len(payload) % channel_spec.frame_stride:
                raise ValueError(f"Payload for channel {name!r} must be aligned to whole samples")
            sample_count = len(payload) // channel_spec.frame_stride
            if sample_count == 0:
                continue
            for pipeline in pipeline_list:
                self._process_pipeline_payload(
                    pipeline,
                    payload,
                    sample_count,
                )

    async def send(self) -> bool:
        """Send all ready chunks to clients, applying buffer backpressure."""
        pending = False
        for player_state in self._players.values():
            tracker = player_state.buffer_tracker
            if tracker is None:
                continue
            queue = player_state.queue
            while queue:
                chunk = queue[0]
                await tracker.wait_for_capacity(chunk.byte_count, chunk.end_time_us)
                header = pack_binary_header_raw(
                    BinaryMessageType.AUDIO_CHUNK.value, chunk.start_time_us
                )
                send_fn = player_state.config.send
                if send_fn is None:
                    raise RuntimeError(
                        f"Player {player_state.config.client_id} missing send callback"
                    )
                send_fn(header + chunk.payload)
                tracker.register(chunk.end_time_us, chunk.byte_count)
                queue.popleft()
                chunk.refcount -= 1
                pipeline = self._pipelines[player_state.pipeline_key]
                if chunk.refcount == 0 and pipeline.prepared and pipeline.prepared[0] is chunk:
                    pipeline.prepared.popleft()
            if queue:
                pending = True
        return not pending

    def flush(self) -> None:
        """Flush all pipelines, preparing any buffered data for sending."""
        for pipeline in self._pipelines.values():
            if pipeline.flushed:
                continue
            if pipeline.buffer:
                self._drain_pipeline_buffer(pipeline, force_flush=True)
            if pipeline.encoder is not None:
                packets = pipeline.encoder.encode(None)
                for packet in packets:
                    self._handle_encoded_packet(pipeline, packet)
            pipeline.flushed = True

    def reset(self) -> None:
        """Reset state, releasing encoders and resamplers."""
        for pipeline in self._pipelines.values():
            pipeline.encoder = None
        self._channels.clear()
        self._pipelines.clear()
        self._pipelines_by_channel.clear()
        self._players.clear()

    def _process_pipeline_payload(
        self,
        pipeline: PipelineState,
        payload: bytes,
        sample_count: int,
    ) -> None:
        frame = av.AudioFrame(
            format=pipeline.channel.av_format,
            layout=pipeline.channel.av_layout,
            samples=sample_count,
        )
        frame.sample_rate = pipeline.channel.audio_format.sample_rate
        frame.planes[0].update(payload)
        out_frames = pipeline.resampler.resample(frame)
        for out_frame in out_frames:
            expected = pipeline.target_frame_stride * out_frame.samples
            pcm_bytes = bytes(out_frame.planes[0])[:expected]
            pipeline.buffer.extend(pcm_bytes)
        self._drain_pipeline_buffer(pipeline, force_flush=False)

    def _drain_pipeline_buffer(
        self,
        pipeline: PipelineState,
        *,
        force_flush: bool,
    ) -> None:
        if not pipeline.subscribers:
            pipeline.buffer.clear()
            return

        frame_stride = pipeline.target_frame_stride
        while len(pipeline.buffer) >= frame_stride:
            available_samples = len(pipeline.buffer) // frame_stride
            if not force_flush and available_samples < pipeline.chunk_samples:
                break
            sample_count = (
                available_samples if force_flush else min(available_samples, pipeline.chunk_samples)
            )
            chunk_size = sample_count * frame_stride
            chunk = bytes(pipeline.buffer[:chunk_size])
            del pipeline.buffer[:chunk_size]
            if pipeline.encoder is None:
                self._publish_chunk(pipeline, chunk, sample_count)
            else:
                self._encode_and_publish(pipeline, chunk, sample_count)

    def _encode_and_publish(
        self,
        pipeline: PipelineState,
        chunk: bytes,
        sample_count: int,
    ) -> None:
        if pipeline.encoder is None:
            raise RuntimeError("Encoder not configured for this pipeline")
        frame = av.AudioFrame(
            format=pipeline.target_av_format,
            layout=pipeline.target_layout,
            samples=sample_count,
        )
        frame.sample_rate = pipeline.target_format.sample_rate
        frame.planes[0].update(chunk)
        pipeline.samples_enqueued += sample_count
        packets = pipeline.encoder.encode(frame)
        for packet in packets:
            self._handle_encoded_packet(pipeline, packet)

    def _handle_encoded_packet(self, pipeline: PipelineState, packet: av.Packet) -> None:
        available = max(pipeline.samples_enqueued - pipeline.samples_encoded, 0)
        packet_samples = packet.duration or available or pipeline.chunk_samples
        payload = bytes(packet)
        self._publish_chunk(pipeline, payload, packet_samples)
        pipeline.samples_encoded += packet_samples

    def _publish_chunk(
        self,
        pipeline: PipelineState,
        payload: bytes,
        sample_count: int,
    ) -> None:
        if not pipeline.subscribers or sample_count <= 0:
            return
        start_samples = pipeline.samples_produced
        start_us = self._play_start_time_us + int(
            start_samples * 1_000_000 / pipeline.target_format.sample_rate
        )
        end_us = self._play_start_time_us + int(
            (start_samples + sample_count) * 1_000_000 / pipeline.target_format.sample_rate
        )
        chunk = PreparedChunkState(
            payload=payload,
            start_time_us=start_us,
            end_time_us=end_us,
            sample_count=sample_count,
            byte_count=len(payload),
            refcount=len(pipeline.subscribers),
        )
        pipeline.prepared.append(chunk)
        pipeline.samples_produced += sample_count
        self._last_chunk_end_us = end_us

        for client_id in pipeline.subscribers:
            player_state = self._players[client_id]
            player_state.queue.append(chunk)

    @property
    def last_chunk_end_time_us(self) -> int | None:
        """Return the end timestamp of the most recently prepared chunk."""
        return self._last_chunk_end_us


def _resolve_audio_format(audio_format: AudioFormat) -> tuple[int, str, str]:
    """Resolve helper data for an audio format."""
    if audio_format.bit_depth == 16:
        bytes_per_sample = 2
        av_format = "s16"
    elif audio_format.bit_depth == 24:
        bytes_per_sample = 3
        av_format = "s24"
    else:
        raise ValueError("Only 16-bit and 24-bit PCM are supported")

    if audio_format.channels == 1:
        layout = "mono"
    elif audio_format.channels == 2:
        layout = "stereo"
    else:
        raise ValueError("Only mono and stereo layouts are supported")

    return bytes_per_sample, av_format, layout


__all__ = [
    "AudioCodec",
    "AudioFormat",
    "Channel",
    "ChannelSpec",
    "ClientStreamConfig",
    "MediaStream",
    "Streamer",
]
