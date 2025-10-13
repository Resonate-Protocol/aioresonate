"""High-level streaming pipeline primitives."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections import deque
from collections.abc import AsyncGenerator, Callable, Iterable
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


@dataclass
class SourceChunk:
    """Raw PCM chunk received from the source."""

    payload: bytes
    """Raw PCM audio data."""
    start_time_us: int
    """Absolute timestamp when this chunk starts playing."""
    end_time_us: int
    """Absolute timestamp when this chunk finishes playing."""
    sample_count: int
    """Number of audio samples in this chunk."""


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

    def has_capacity_now(self, bytes_needed: int) -> bool:
        """Check if buffer can accept bytes_needed without waiting.

        This is a non-blocking version of wait_for_capacity that returns immediately.

        Args:
            bytes_needed: Number of bytes to check capacity for.

        Returns:
            True if the buffer has capacity for bytes_needed, False otherwise.
        """
        if bytes_needed <= 0:
            return True
        if bytes_needed >= self.capacity_bytes:
            # Chunk exceeds capacity, but allow it through
            logger.warning(
                "Chunk size %s exceeds reported buffer capacity %s for client %s",
                bytes_needed,
                self.capacity_bytes,
                self.client_id,
            )
            return True

        self.prune_consumed()
        projected_usage = self.buffered_bytes + bytes_needed
        return projected_usage <= self.capacity_bytes

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

    codec = "libopus" if audio_format.codec == AudioCodec.OPUS else audio_format.codec.value

    encoder = cast("av.AudioCodecContext", av.AudioCodecContext.create(codec, "w"))
    encoder.sample_rate = audio_format.sample_rate
    encoder.layout = input_audio_layout
    encoder.format = input_audio_format
    if audio_format.codec == AudioCodec.FLAC:
        encoder.options = {"compression_level": "5"}

    with Capture() as logs:
        encoder.open()
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

    # Calculate samples per chunk
    if audio_format.codec == AudioCodec.FLAC:
        # FLAC: Use 25ms chunks regardless of encoder frame_size
        samples_per_chunk = int(audio_format.sample_rate * 0.025)
    elif encoder.frame_size and encoder.frame_size > 0:
        # Use recommended frame size for other codecs (e.g., OPUS)
        samples_per_chunk = int(encoder.frame_size)
    else:
        raise ValueError(
            f"Codec {audio_format.codec.value} encoder has invalid frame_size: {encoder.frame_size}"
        )
    return encoder, codec_header_b64, samples_per_chunk


@dataclass(frozen=True)
class ChannelSpec:
    """Source audio format with computed PyAV parameters for processing."""

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
    send: Callable[[bytes], None]


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
    source_read_position: int = 0


@dataclass
class PlayerState:
    """Tracks delivery state for a single player."""

    config: ClientStreamConfig
    pipeline_key: AudioFormat
    queue: deque[PreparedChunkState] = field(default_factory=deque)
    buffer_tracker: BufferTracker | None = None
    join_time_us: int | None = None


class MediaStream:
    """Container for a single audio stream with its format."""

    def __init__(
        self,
        *,
        source: AsyncGenerator[bytes, None],
        audio_format: AudioFormat,
    ) -> None:
        """Initialise the media stream with audio source and format."""
        self._source = source
        self._audio_format = audio_format

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

    _loop: asyncio.AbstractEventLoop
    """Event loop used for time calculations and task scheduling."""
    _play_start_time_us: int
    """Absolute timestamp in microseconds when playback should start."""
    _pipelines: dict[AudioFormat, PipelineState]
    """Mapping of target_format to pipeline state."""
    _players: dict[str, PlayerState]
    """Mapping of client IDs to their player delivery state."""
    _last_chunk_end_us: int | None
    """End timestamp of the most recently prepared chunk, None if no chunks prepared yet."""
    _channel: ChannelSpec | None = None
    """The default audio channel."""
    _source_buffer: deque[SourceChunk]
    """Buffer of raw PCM chunks that are scheduled for playback but not yet finished playing."""
    _source_samples_produced: int
    """Total number of samples added to the source buffer."""
    _source_buffer_target_duration_us: int
    """Target duration for source buffer in microseconds."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        play_start_time_us: int,
    ) -> None:
        """Create a streamer bound to the event loop and playback start time."""
        self._loop = loop
        self._play_start_time_us = play_start_time_us
        self._pipelines: dict[AudioFormat, PipelineState] = {}
        self._players: dict[str, PlayerState] = {}
        self._last_chunk_end_us: int | None = None
        self._source_buffer: deque[SourceChunk] = deque()
        self._source_samples_produced = 0
        self._source_buffer_target_duration_us = 5_000_000  # 5 seconds

    def configure(
        self,
        *,
        audio_format: AudioFormat,
        clients: Iterable[ClientStreamConfig],
    ) -> dict[str, StreamStartPlayer]:
        """Configure or reconfigure pipelines for the provided clients.

        This method sets up or updates the streaming pipelines for all clients.

        Args:
            audio_format: The source audio format.
            clients: Configuration for each client/player.

        Returns:
            Dictionary mapping client IDs to their StreamStartPlayer messages.
        """
        # Update channel spec if audio format changed
        bytes_per_sample, av_format, av_layout = _resolve_audio_format(audio_format)
        self._channel = ChannelSpec(
            audio_format=audio_format,
            bytes_per_sample=bytes_per_sample,
            frame_stride=bytes_per_sample * audio_format.channels,
            av_format=av_format,
            av_layout=av_layout,
        )

        # Clear subscriber lists to rebuild them
        for existing_pipeline in self._pipelines.values():
            existing_pipeline.subscribers.clear()

        # Build new player and subscription configuration
        new_players: dict[str, PlayerState] = {}
        start_payloads: dict[str, StreamStartPlayer] = {}

        for client_cfg in clients:
            pipeline_key = client_cfg.target_format
            pipeline: PipelineState | None = self._pipelines.get(pipeline_key)
            if pipeline is None:
                # Create new pipeline for this format
                if self._channel is None:
                    raise RuntimeError("Streamer channel not initialized")
                channel_spec = self._channel
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

            pipeline.subscribers.append(client_cfg.client_id)

            old_player = self._players.get(client_cfg.client_id)

            # Reuse existing player if format unchanged
            if old_player and old_player.pipeline_key == pipeline_key:
                old_player.config = client_cfg
                new_players[client_cfg.client_id] = old_player
                continue

            # Format changed - clean up old queue refcounts
            if old_player and old_player.pipeline_key != pipeline_key:
                for chunk in old_player.queue:
                    chunk.refcount -= 1
                old_player.queue.clear()

            # Create new player or reconfigure existing one
            buffer_tracker = (
                old_player.buffer_tracker
                if old_player
                else BufferTracker(
                    loop=self._loop,
                    client_id=client_cfg.client_id,
                    capacity_bytes=client_cfg.buffer_capacity_bytes,
                )
            )

            # Calculate join time for new/reconfigured player
            join_time_us = int(self._loop.time() * 1_000_000)

            player_state = PlayerState(
                config=client_cfg,
                pipeline_key=pipeline_key,
                buffer_tracker=buffer_tracker,
                join_time_us=join_time_us,
            )

            # Queue future chunks for new/reconfigured player
            for chunk in pipeline.prepared:
                if chunk.start_time_us >= join_time_us:
                    player_state.queue.append(chunk)
                    chunk.refcount += 1

            # Check for and log any gaps in coverage
            self._catch_up_player(player_state, join_time_us)

            new_players[client_cfg.client_id] = player_state

            start_payloads[client_cfg.client_id] = StreamStartPlayer(
                codec=client_cfg.target_format.codec.value,
                sample_rate=client_cfg.target_format.sample_rate,
                channels=client_cfg.target_format.channels,
                bit_depth=client_cfg.target_format.bit_depth,
                codec_header=pipeline.codec_header_b64,
            )

        # Remove pipelines with no subscribers
        pipelines_to_remove = [
            key for key, pipeline in self._pipelines.items() if not pipeline.subscribers
        ]
        for key in pipelines_to_remove:
            pipeline = self._pipelines.pop(key)
            if pipeline.encoder:
                pipeline.encoder = None

        # Replace players dict
        self._players = new_players

        return start_payloads

    def prepare(self, chunk: bytes) -> bool:
        """Buffer raw PCM data and process through pipelines.

        Args:
            chunk: Raw PCM audio data to buffer.

        Returns:
            True if more data is wanted (buffer duration < target), False otherwise.
        """
        if self._channel is None:
            raise RuntimeError("Streamer not configured")
        if len(chunk) % self._channel.frame_stride:
            raise ValueError("Chunk must be aligned to whole samples")
        sample_count = len(chunk) // self._channel.frame_stride
        if sample_count == 0:
            return True

        # Calculate timestamps for this chunk
        start_samples = self._source_samples_produced
        start_us = self._play_start_time_us + int(
            start_samples * 1_000_000 / self._channel.audio_format.sample_rate
        )
        end_us = self._play_start_time_us + int(
            (start_samples + sample_count) * 1_000_000 / self._channel.audio_format.sample_rate
        )

        # Create and buffer the source chunk
        source_chunk = SourceChunk(
            payload=chunk,
            start_time_us=start_us,
            end_time_us=end_us,
            sample_count=sample_count,
        )
        self._source_buffer.append(source_chunk)
        self._source_samples_produced += sample_count

        # Process the buffered data through all pipelines
        for pipeline in self._pipelines.values():
            self._process_pipeline_from_source(pipeline)

        # Calculate current buffer duration
        if self._source_buffer:
            buffer_duration_us = (
                self._source_buffer[-1].end_time_us - self._source_buffer[0].start_time_us
            )
            return buffer_duration_us < self._source_buffer_target_duration_us

        return True

    async def send(self) -> None:
        """Send prepared chunks to clients until all queues are empty.

        This method performs two stages in a loop:
        1. Send prepared chunks to players (waits for earliest blocked player)
        2. Prune old data

        Continues until all player queues are empty.
        """
        while True:
            # Stage 1: Deliver to players with smart waiting
            earliest_blocked_player = None
            earliest_blocked_chunk = None
            earliest_end_time_us = None

            for player_state in self._players.values():
                tracker = player_state.buffer_tracker
                if tracker is None:
                    continue
                queue = player_state.queue

                # Send as much as we can without blocking
                while queue:
                    chunk = queue[0]

                    # Skip chunks that are too close to playback or already in the past
                    now_us = int(self._loop.time() * 1_000_000)
                    min_send_margin_us = 100_000  # 100ms for network + client processing
                    if chunk.start_time_us < now_us + min_send_margin_us:
                        # Chunk is stale, skip it without sending
                        logger.debug(
                            "Skipping stale chunk for %s (starts in %d us)",
                            player_state.config.client_id,
                            chunk.start_time_us - now_us,
                        )
                        queue.popleft()
                        chunk.refcount -= 1
                        pipeline = self._pipelines[player_state.pipeline_key]
                        if (
                            chunk.refcount == 0
                            and pipeline.prepared
                            and pipeline.prepared[0] is chunk
                        ):
                            pipeline.prepared.popleft()
                        continue

                    # Check if we can send without waiting
                    if not tracker.has_capacity_now(chunk.byte_count):
                        # This player is blocked - track if this chunk is earliest
                        if earliest_end_time_us is None or chunk.end_time_us < earliest_end_time_us:
                            earliest_blocked_player = player_state
                            earliest_blocked_chunk = chunk
                            earliest_end_time_us = chunk.end_time_us
                        break

                    # We have capacity - send immediately
                    header = pack_binary_header_raw(
                        BinaryMessageType.AUDIO_CHUNK.value, chunk.start_time_us
                    )
                    player_state.config.send(header + chunk.payload)
                    tracker.register(chunk.end_time_us, chunk.byte_count)
                    queue.popleft()
                    chunk.refcount -= 1
                    pipeline = self._pipelines[player_state.pipeline_key]
                    # Remove from prepared immediately when all active players have consumed it
                    if chunk.refcount == 0 and pipeline.prepared and pipeline.prepared[0] is chunk:
                        pipeline.prepared.popleft()

            # If any player is blocked, wait for the one with earliest chunk
            if earliest_blocked_player is not None and earliest_blocked_chunk is not None:
                tracker = earliest_blocked_player.buffer_tracker
                if tracker is not None:
                    await tracker.wait_for_capacity(
                        earliest_blocked_chunk.byte_count, earliest_blocked_chunk.end_time_us
                    )
                continue  # More work to do, loop again

            # Stage 2: Cleanup
            self._prune_old_data()

            # Check if there's still work pending (any non-empty queues)
            if not any(player_state.queue for player_state in self._players.values()):
                break  # All queues empty, we're done

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
        self._channel = None
        self._pipelines.clear()
        self._players.clear()

    def _prune_old_data(self) -> None:
        """Prune old source chunks to free memory.

        Removes source chunks that have finished playing (end_time_us <= now).
        Prepared chunks are managed separately by refcount in send().
        """
        # Prune source buffer based on playback time
        now_us = int(self._loop.time() * 1_000_000)
        chunks_removed = 0
        while self._source_buffer and self._source_buffer[0].end_time_us <= now_us:
            self._source_buffer.popleft()
            chunks_removed += 1

        # Update pipeline read positions to account for removed chunks
        if chunks_removed > 0:
            for pipeline in self._pipelines.values():
                pipeline.source_read_position = max(
                    0, pipeline.source_read_position - chunks_removed
                )

    def _catch_up_player(self, player_state: PlayerState, join_time_us: int) -> None:
        """Check for gaps in player coverage and log warnings.

        When a late joiner arrives, prepared chunks may have been removed after all
        active players consumed them. This method checks for gaps and logs warnings.
        Full catch-up with re-processing can be added in a future enhancement.

        Args:
            player_state: The late joining player.
            join_time_us: Timestamp when the player joined.
        """
        # Check if there's a gap between join time and first queued chunk
        if player_state.queue:
            first_chunk_start = player_state.queue[0].start_time_us
            gap_us = first_chunk_start - join_time_us
            if gap_us > 100_000:  # Only log for gaps >100ms
                logger.info(
                    "Late joiner %s: %.1f ms gap between join and first chunk",
                    player_state.config.client_id,
                    gap_us / 1000,
                )
        elif self._source_buffer:
            # No queued chunks, check against source buffer
            earliest_source_us = self._source_buffer[0].start_time_us
            gap_us = earliest_source_us - join_time_us
            if gap_us > 0:
                logger.info(
                    "Late joiner %s: %.1f ms gap, no prepared chunks available",
                    player_state.config.client_id,
                    gap_us / 1000,
                )

    def _process_pipeline_from_source(self, pipeline: PipelineState) -> bool:
        """Process available source chunks through this pipeline.

        Args:
            pipeline: The pipeline to process.

        Returns:
            True if any work was done, False otherwise.
        """
        if not pipeline.subscribers:
            return False

        any_work = False
        # Process all available source chunks that haven't been processed yet
        while pipeline.source_read_position < len(self._source_buffer):
            source_chunk = self._source_buffer[pipeline.source_read_position]
            self._process_pipeline_payload(
                pipeline,
                source_chunk.payload,
                source_chunk.sample_count,
            )
            pipeline.source_read_position += 1
            any_work = True

        return any_work

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

        # Determine packet sample count
        # For OPUS: Always use chunk_samples for consistent timing
        # OPUS packet.duration can be unreliable or in wrong timebase
        if pipeline.target_format.codec == AudioCodec.OPUS:
            packet_samples = pipeline.chunk_samples
        elif packet.duration and packet.duration > 0:
            # Trust packet duration when provided by encoder (for FLAC, etc.)
            packet_samples = packet.duration
        else:
            # Fallback: use chunk_samples (NOT available, which can be huge for buffered encoders)
            packet_samples = pipeline.chunk_samples
            logger.warning(
                "Codec %s: packet.duration is %s, using chunk_samples=%d (available=%d)",
                pipeline.target_format.codec.value,
                packet.duration,
                pipeline.chunk_samples,
                available,
            )

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

        # Debug OPUS timestamps
        if pipeline.target_format.codec == AudioCodec.OPUS and pipeline.samples_produced < 10000:
            logger.warning(
                "OPUS chunk #%d: start_samples=%d, sample_count=%d, "
                "sample_rate=%d, start_us=%d, duration_us=%d",
                len(pipeline.prepared),
                start_samples,
                sample_count,
                pipeline.target_format.sample_rate,
                start_us - self._play_start_time_us,
                end_us - start_us,
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
    "ChannelSpec",
    "ClientStreamConfig",
    "MediaStream",
    "Streamer",
]
