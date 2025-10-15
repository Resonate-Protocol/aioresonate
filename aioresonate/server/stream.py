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

    pcm_data: bytes
    """Raw PCM audio data."""
    start_time_us: int
    """Absolute timestamp when this chunk starts playing."""
    end_time_us: int
    """Absolute timestamp when this chunk finishes playing."""
    sample_count: int
    """Number of audio samples in this chunk."""


class BufferedChunk(NamedTuple):
    """Buffered chunk metadata tracked by BufferTracker for backpressure control."""

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
            client_id: Identifier for the client being tracked.
            capacity_bytes: Maximum buffer capacity in bytes reported by the client.
        """
        self._loop = loop
        self.client_id = client_id
        self.capacity_bytes = capacity_bytes
        self.buffered_chunks: deque[BufferedChunk] = deque()
        self.buffered_bytes = 0

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

    async def wait_for_capacity(self, bytes_needed: int) -> None:
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

            sleep_target_us = self.buffered_chunks[0].end_time_us
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
class SourceAudioSpec:
    """Source audio format with computed PyAV parameters for processing."""

    audio_format: AudioFormat
    """Source audio format."""
    bytes_per_sample: int
    """Bytes per sample (derived from bit depth)."""
    frame_stride: int
    """Bytes per frame (bytes_per_sample * channels)."""
    av_format: str
    """PyAV format string (e.g., 's16', 's24')."""
    av_layout: str
    """PyAV channel layout (e.g., 'mono', 'stereo')."""


@dataclass
class ClientStreamConfig:
    """Configuration for delivering audio to a player."""

    client_id: str
    """Unique client identifier."""
    target_format: AudioFormat
    """Target audio format for this client."""
    buffer_capacity_bytes: int
    """Client's buffer capacity in bytes."""
    send: Callable[[bytes], None]
    """Function to send data to client."""


@dataclass
class PreparedChunkState:
    """Prepared chunk shared between all subscribers of a pipeline."""

    data: bytes
    """Prepared/encoded audio data."""
    start_time_us: int
    """Chunk playback start time in microseconds."""
    end_time_us: int
    """Chunk playback end time in microseconds."""
    sample_count: int
    """Number of samples in this chunk."""
    byte_count: int
    """Size of chunk data in bytes."""
    refcount: int
    """Number of subscribers using this chunk."""


@dataclass
class PipelineState:
    """Holds state for a distinct channel/format/chunk-size pipeline."""

    channel: SourceAudioSpec
    """Source audio specification."""
    target_format: AudioFormat
    """Target output format."""
    target_frame_stride: int
    """Target bytes per frame."""
    target_av_format: str
    """Target PyAV format string."""
    target_layout: str
    """Target PyAV channel layout."""
    chunk_samples: int
    """Target samples per chunk."""
    resampler: av.AudioResampler
    """PyAV audio resampler."""
    encoder: av.AudioCodecContext | None
    """PyAV encoder (None for PCM)."""
    codec_header_b64: str | None
    """Base64 encoded codec header."""
    buffer: bytearray = field(default_factory=bytearray)
    """Resampled PCM buffer awaiting encoding."""
    prepared: deque[PreparedChunkState] = field(default_factory=deque)
    """Prepared chunks ready for delivery."""
    subscribers: list[str] = field(default_factory=list)
    """Client IDs subscribed to this pipeline."""
    samples_produced: int = 0
    """Total samples published from this pipeline."""
    flushed: bool = False
    """Whether pipeline has been flushed."""
    source_read_position: int = 0
    """Position in source buffer for this pipeline."""
    next_chunk_start_us: int | None = None
    """Next output chunk start timestamp, initialized from first source chunk."""


@dataclass
class PlayerState:
    """Tracks delivery state for a single player."""

    config: ClientStreamConfig
    """Client streaming configuration."""
    audio_format: AudioFormat
    """Format key for pipeline lookup."""
    queue: deque[PreparedChunkState] = field(default_factory=deque)
    """Chunks queued for delivery."""
    buffer_tracker: BufferTracker | None = None
    """Tracks client buffer state."""
    join_time_us: int | None = None
    """When player joined in microseconds."""
    needs_catchup: bool = False
    """Whether player needs catch-up processing."""


class MediaStream:
    """Container for a single audio stream with its format."""

    _source: AsyncGenerator[bytes, None]
    """Audio source generator yielding PCM bytes."""
    _audio_format: AudioFormat
    """Audio format of the stream."""

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
    _last_chunk_end_us: int | None = None
    """End timestamp of the most recently prepared chunk, None if no chunks prepared yet."""
    _channel: SourceAudioSpec | None = None
    """The source audio specification."""
    _source_buffer: deque[SourceChunk]
    """Buffer of raw PCM chunks that are scheduled for playback but not yet finished playing."""
    _source_samples_produced: int = 0
    """Total number of samples added to the source buffer (used for timestamp calculation)."""
    _source_buffer_target_duration_us: int = 5_000_000
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
        self._pipelines = {}
        self._players = {}
        self._source_buffer = deque()

    def _cleanup_consumed_chunks(self, pipeline: PipelineState) -> None:
        """Clean up consumed chunks from a pipeline.

        Note: consumed chunks (refcount == 0) can only appear as a contiguous block
        at the front since chunks are consumed in FIFO order.

        Args:
            pipeline: The pipeline to clean up consumed chunks from.
        """
        while pipeline.prepared and pipeline.prepared[0].refcount == 0:
            pipeline.prepared.popleft()

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
        # Update source audio spec if audio format changed
        bytes_per_sample, av_format, av_layout = _resolve_audio_format(audio_format)
        self._channel = SourceAudioSpec(
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
            audio_format = client_cfg.target_format
            pipeline: PipelineState | None = self._pipelines.get(audio_format)
            if pipeline is None:
                # Create new pipeline for this format
                source_spec = self._channel
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
                    channel=source_spec,
                    target_format=client_cfg.target_format,
                    target_frame_stride=target_bytes_per_sample * client_cfg.target_format.channels,
                    target_av_format=target_av_format,
                    target_layout=target_layout,
                    chunk_samples=chunk_samples,
                    resampler=resampler,
                    encoder=encoder,
                    codec_header_b64=codec_header_b64,
                )
                self._pipelines[audio_format] = pipeline

            pipeline.subscribers.append(client_cfg.client_id)

            old_player = self._players.get(client_cfg.client_id)

            # Reuse existing player if format unchanged
            if old_player and old_player.audio_format == audio_format:
                old_player.config = client_cfg
                new_players[client_cfg.client_id] = old_player
                continue

            # Format changed - clean up old queue refcounts
            if old_player and old_player.audio_format != audio_format:
                for chunk in old_player.queue:
                    chunk.refcount -= 1
                old_player.queue.clear()
                # Clean up consumed chunks from old pipeline
                if old_pipeline := self._pipelines.get(old_player.audio_format):
                    self._cleanup_consumed_chunks(old_pipeline)

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
                audio_format=audio_format,
                buffer_tracker=buffer_tracker,
                join_time_us=join_time_us,
            )

            # Queue future chunks for new/reconfigured player
            for chunk in pipeline.prepared:
                if chunk.start_time_us >= join_time_us:
                    player_state.queue.append(chunk)
                    chunk.refcount += 1

            # Mark if player needs catch-up (actual sending happens in send())
            # player_state.needs_catchup = self._check_needs_catchup(player_state, join_time_us)  # noqa: E501, ERA001
            # TODO: fix and re-enable catchup logic
            player_state.needs_catchup = False  # disable catchup for now

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

        # Clean up refcounts for players being removed
        for old_client_id, old_player in self._players.items():
            if old_client_id not in new_players:
                for chunk in old_player.queue:
                    chunk.refcount -= 1
                old_player.queue.clear()
                # Clean up consumed chunks from old pipeline
                if old_pipeline := self._pipelines.get(old_player.audio_format):
                    self._cleanup_consumed_chunks(old_pipeline)

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

        # Check if this chunk would be stale (arriving too late to stream)
        now_us = int(self._loop.time() * 1_000_000)
        min_send_margin_us = 1_000_000  # 1s margin for network + client processing
        if start_us < now_us + min_send_margin_us:
            # Adjust timing globally (this modifies _play_start_time_us and all existing chunks)
            self._adjust_timing_for_stale_chunk(now_us, start_us)
            # Recalculate timestamps after adjustment
            start_us = self._play_start_time_us + int(
                start_samples * 1_000_000 / self._channel.audio_format.sample_rate
            )
            end_us = self._play_start_time_us + int(
                (start_samples + sample_count) * 1_000_000 / self._channel.audio_format.sample_rate
            )

        # Create and buffer the source chunk
        source_chunk = SourceChunk(
            pcm_data=chunk,
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

    def _adjust_timing_for_stale_chunk(self, now_us: int, chunk_start_us: int) -> None:
        """Adjust timing when a stale chunk is detected.

        Args:
            now_us: Current time in microseconds.
            chunk_start_us: Start time of the stale chunk.
        """
        min_send_margin_us = 1_000_000  # Must match constant in prepare()
        target_buffer_us = self._source_buffer_target_duration_us  # 5 seconds for web players

        # Calculate current buffer depth (from now to end of buffer)
        current_buffer_us = 0
        if self._source_buffer:
            # Buffer depth is from now to the end of the last buffered chunk
            last_chunk_end = self._source_buffer[-1].end_time_us
            current_buffer_us = max(0, last_chunk_end - now_us)

        # Calculate minimum adjustment needed to give this chunk proper headroom
        headroom_shortfall_us = (now_us + min_send_margin_us) - chunk_start_us

        # Determine total adjustment based on buffer status
        if current_buffer_us >= target_buffer_us:
            # We already have enough buffer, just ensure headroom
            timing_adjustment_us = headroom_shortfall_us
            logger.warning(
                "Adjusting timing: chunk needs %.3fs more headroom, "
                "already have %.3fs buffer (adjusting %.3fs)",
                headroom_shortfall_us / 1_000_000,
                current_buffer_us / 1_000_000,
                timing_adjustment_us / 1_000_000,
            )
        else:
            # Need to build buffer to target level
            buffer_shortfall_us = target_buffer_us - current_buffer_us
            # Use the larger of headroom need and buffer need
            timing_adjustment_us = max(headroom_shortfall_us, buffer_shortfall_us)
            logger.warning(
                "Adjusting timing: chunk needs %.3fs headroom, have %.3fs buffer, "
                "target %.3fs buffer (adjusting %.3fs)",
                headroom_shortfall_us / 1_000_000,
                current_buffer_us / 1_000_000,
                target_buffer_us / 1_000_000,
                timing_adjustment_us / 1_000_000,
            )

        # Adjust timing forward
        self._play_start_time_us += timing_adjustment_us

        # Update source buffer chunk timestamps
        for source_chunk in self._source_buffer:
            source_chunk.start_time_us += timing_adjustment_us
            source_chunk.end_time_us += timing_adjustment_us

        # Update pipeline timestamps and prepared chunks
        for pipeline in self._pipelines.values():
            if pipeline.next_chunk_start_us is not None:
                pipeline.next_chunk_start_us += timing_adjustment_us
            # Update timestamps of already-prepared chunks to prevent cascading adjustments
            for prepared_chunk in pipeline.prepared:
                prepared_chunk.start_time_us += timing_adjustment_us
                prepared_chunk.end_time_us += timing_adjustment_us

    async def send(self) -> None:
        """Send prepared audio to all clients.

        This method performs four stages in a loop:
        1. Perform catch-up for late joiners (if needed)
        2. Send chunks to players with backpressure control
        3. Prune old data
        4. Check exit conditions and apply source buffer backpressure

        Continues until all pending audio has been delivered and source buffer is below target.
        """
        while True:
            # Stage 1: Perform catch-up for players that need it
            for player_state in self._players.values():
                if player_state.needs_catchup:
                    self._perform_catchup(player_state)

            # Stage 2: Send chunks to players with backpressure control
            now_us = int(self._loop.time() * 1_000_000)
            min_send_margin_us = 100_000  # 100ms for network + client processing
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
                    if chunk.start_time_us < now_us + min_send_margin_us:
                        # Chunk is stale, skip it without sending
                        logger.debug(
                            "Skipping stale chunk for %s (starts in %d us)",
                            player_state.config.client_id,
                            chunk.start_time_us - now_us,
                        )
                        self._dequeue_chunk(player_state, chunk)
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
                    player_state.config.send(header + chunk.data)
                    tracker.register(chunk.end_time_us, chunk.byte_count)
                    self._dequeue_chunk(player_state, chunk)

            # If any player is blocked, wait for the one with earliest chunk
            if earliest_blocked_player is not None and earliest_blocked_chunk is not None:
                tracker = earliest_blocked_player.buffer_tracker
                if tracker is not None:
                    await tracker.wait_for_capacity(earliest_blocked_chunk.byte_count)
                continue  # More work to do, loop again

            # Stage 3: Cleanup
            self._prune_old_data()

            # Stage 4: Check exit conditions and apply source buffer backpressure
            has_client_work = any(
                player_state.queue or player_state.needs_catchup
                for player_state in self._players.values()
            )

            # Check source buffer status
            source_buffer_ok = True
            if self._source_buffer:
                buffer_duration_us = (
                    self._source_buffer[-1].end_time_us - self._source_buffer[0].start_time_us
                )
                source_buffer_ok = buffer_duration_us <= self._source_buffer_target_duration_us

            # Exit when both conditions met
            if not has_client_work and source_buffer_ok:
                break

            # If client work pending, continue immediately
            if has_client_work:
                continue

            # Otherwise wait for source buffer to drain
            now_us = int(self._loop.time() * 1_000_000)
            sleep_until_us = self._source_buffer[0].end_time_us
            sleep_duration_s = (sleep_until_us - now_us) / 1_000_000
            if sleep_duration_s > 0:
                await asyncio.sleep(sleep_duration_s)

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
                    # Skip packets with invalid duration from encoder flush
                    if not packet.duration or packet.duration <= 0:
                        continue
                    # Calculate timestamps for each flushed packet from its duration
                    start_us, end_us = self._calculate_chunk_timestamps(pipeline, packet.duration)
                    self._handle_encoded_packet(pipeline, packet, start_us, end_us)
                    # Advance next_chunk_start_us for each flushed packet
                    pipeline.next_chunk_start_us = end_us
            pipeline.flushed = True

    def reset(self) -> None:
        """Reset state, releasing encoders and resamplers."""
        for pipeline in self._pipelines.values():
            pipeline.encoder = None
        self._channel = None
        self._pipelines.clear()
        self._players.clear()

    def _dequeue_chunk(self, player_state: PlayerState, chunk: PreparedChunkState) -> None:
        """Remove chunk from player queue and clean up pipeline if fully consumed."""
        player_state.queue.popleft()
        chunk.refcount -= 1
        pipeline = self._pipelines[player_state.audio_format]
        if chunk.refcount == 0 and pipeline.prepared and pipeline.prepared[0] is chunk:
            pipeline.prepared.popleft()

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

    def _check_needs_catchup(self, player_state: PlayerState, join_time_us: int) -> bool:
        """Check if player needs catch-up processing.

        Args:
            player_state: The player to check.
            join_time_us: Timestamp when the player joined.

        Returns:
            True if player has a gap that needs catch-up, False otherwise.
        """
        if not self._source_buffer:
            return False

        # Determine if there's a gap that can be filled from source buffer
        first_queued_start_us = player_state.queue[0].start_time_us if player_state.queue else None

        # Check if any source chunks exist in the gap range
        for source_chunk in self._source_buffer:
            # Skip chunks before join time
            if source_chunk.end_time_us <= join_time_us:
                continue
            # Stop when we reach prepared chunks
            if first_queued_start_us and source_chunk.start_time_us >= first_queued_start_us:
                break
            # Found at least one chunk in the gap
            return True

        return False

    def _perform_catchup(self, player_state: PlayerState) -> None:
        """Process and queue missing chunks for late joiners from source buffer.

        When a late joiner arrives, this reprocesses source chunks to fill the gap
        between join_time and the first queued chunk. Chunks are added to the player's
        queue and will be sent by the normal send loop with proper backpressure.

        Args:
            player_state: The late joining player.
        """
        if not self._source_buffer or player_state.join_time_us is None or self._channel is None:
            return

        join_time_us = player_state.join_time_us
        pipeline = self._pipelines[player_state.audio_format]

        # Determine the coverage range we need to fill
        first_queued_start_us = player_state.queue[0].start_time_us if player_state.queue else None

        # Find source chunks that cover the gap
        catchup_sources = []
        for source_chunk in self._source_buffer:
            # Skip chunks before join time
            if source_chunk.end_time_us <= join_time_us:
                continue
            # Stop when we reach prepared chunks
            if first_queued_start_us and source_chunk.start_time_us >= first_queued_start_us:
                break
            catchup_sources.append(source_chunk)

        if not catchup_sources:
            player_state.needs_catchup = False
            return

        gap_duration_ms = (
            catchup_sources[-1].end_time_us - catchup_sources[0].start_time_us
        ) / 1000
        logger.info(
            "Catching up %s: processing %.1f ms from source buffer",
            player_state.config.client_id,
            gap_duration_ms,
        )

        # Process catch-up chunks and get PreparedChunkState objects
        catchup_chunks = self._process_and_send_catchup(
            pipeline=pipeline,
            player_state=player_state,
            source_chunks=catchup_sources,
            first_queued_start_us=first_queued_start_us,
        )

        # Prepend catch-up chunks to player queue (they'll be sent by send loop)
        if catchup_chunks:
            player_state.queue = deque(catchup_chunks + list(player_state.queue))
            logger.info(
                "Catch-up complete for %s: queued %d chunks for delivery",
                player_state.config.client_id,
                len(catchup_chunks),
            )
        else:
            logger.info("Catch-up for %s: no chunks to queue", player_state.config.client_id)

        # Mark catch-up as complete
        player_state.needs_catchup = False

    def _process_and_send_catchup(  # noqa: PLR0915
        self,
        *,
        pipeline: PipelineState,
        player_state: PlayerState,
        source_chunks: list[SourceChunk],
        first_queued_start_us: int | None,
    ) -> list[PreparedChunkState]:
        """Process source chunks and create catch-up chunks for queueing.

        Creates temporary resampler/encoder to avoid corrupting shared pipeline state.
        Uses sample-based timestamp calculation to align perfectly with prepared chunks.

        Args:
            pipeline: Pipeline config to use for processing.
            player_state: Player to send chunks to.
            source_chunks: Source chunks to process.
            first_queued_start_us: Start time of first queued prepared chunk (for alignment).

        Returns:
            List of PreparedChunkState objects to prepend to player queue.
        """
        if not source_chunks or self._channel is None:
            return []

        # Store processed chunks with their sample counts
        processed_chunks: list[tuple[bytes, int]] = []

        # Create temporary resampler (always needed)
        temp_resampler = av.AudioResampler(
            format=pipeline.target_av_format,
            layout=pipeline.target_layout,
            rate=pipeline.target_format.sample_rate,
        )

        # Create temporary encoder if needed
        temp_encoder: av.AudioCodecContext | None = None
        if pipeline.encoder is not None:
            codec = (
                "libopus"
                if pipeline.target_format.codec == AudioCodec.OPUS
                else pipeline.target_format.codec.value
            )
            temp_encoder = cast("av.AudioCodecContext", av.AudioCodecContext.create(codec, "w"))
            temp_encoder.sample_rate = pipeline.target_format.sample_rate
            temp_encoder.layout = pipeline.target_layout
            temp_encoder.format = pipeline.target_av_format
            if pipeline.target_format.codec == AudioCodec.FLAC:
                temp_encoder.options = {"compression_level": "5"}
            with Capture():
                temp_encoder.open()

        # PHASE 1: Process all source chunks and collect output chunks
        temp_buffer = bytearray()

        for source_chunk in source_chunks:
            # Resample
            frame = av.AudioFrame(
                format=self._channel.av_format,
                layout=self._channel.av_layout,
                samples=source_chunk.sample_count,
            )
            frame.sample_rate = self._channel.audio_format.sample_rate
            frame.planes[0].update(source_chunk.pcm_data)
            out_frames = temp_resampler.resample(frame)

            for out_frame in out_frames:
                expected = pipeline.target_frame_stride * out_frame.samples
                pcm_bytes = bytes(out_frame.planes[0])[:expected]
                temp_buffer.extend(pcm_bytes)

            # Drain buffer and collect chunks (don't send yet)
            self._collect_catchup_chunks(
                temp_buffer=temp_buffer,
                temp_encoder=temp_encoder,
                pipeline=pipeline,
                processed_chunks=processed_chunks,
                force_flush=False,
            )

        # Final flush
        if temp_buffer:
            self._collect_catchup_chunks(
                temp_buffer=temp_buffer,
                temp_encoder=temp_encoder,
                pipeline=pipeline,
                processed_chunks=processed_chunks,
                force_flush=True,
            )

        # Flush encoder if used
        if temp_encoder is not None:
            packets = temp_encoder.encode(None)
            for packet in packets:
                if not packet.duration or packet.duration <= 0:
                    raise ValueError(f"Invalid packet duration: {packet.duration!r}")
                chunk_data = bytes(packet)
                processed_chunks.append((chunk_data, packet.duration))

        # PHASE 2: Calculate timestamps using sample-based math (like prepared chunks)
        # Work backwards from first_queued_start_us to ensure perfect alignment
        total_samples = sum(sample_count for _, sample_count in processed_chunks)
        target_rate = pipeline.target_format.sample_rate

        if first_queued_start_us:
            # Work backwards from first queued chunk
            actual_duration_us = int(total_samples * 1_000_000 / target_rate)
            catchup_start_time_us = first_queued_start_us - actual_duration_us

            # CRITICAL: Ensure catch-up doesn't start before join time
            # If it would, skip chunks from the beginning to align with join time
            if player_state.join_time_us and catchup_start_time_us < player_state.join_time_us:
                # Calculate how many samples to skip
                skip_duration_us = player_state.join_time_us - catchup_start_time_us
                skip_samples = int(skip_duration_us * target_rate / 1_000_000)

                logger.debug(
                    "Catch-up would start %d us before join time, "
                    "skipping first %d samples (%.1f ms)",
                    player_state.join_time_us - catchup_start_time_us,
                    skip_samples,
                    skip_duration_us / 1000,
                )

                # Skip entire chunks until we've skipped enough samples
                samples_to_skip = skip_samples
                chunks_to_skip = []

                for i, (_payload, sample_count) in enumerate(processed_chunks):
                    if samples_to_skip >= sample_count:
                        # Skip entire chunk
                        samples_to_skip -= sample_count
                        chunks_to_skip.append(i)
                    else:
                        # Partial skip would require splitting chunk - stop here
                        break

                # Remove chunks we're skipping
                for i in reversed(chunks_to_skip):
                    processed_chunks.pop(i)

                # Recalculate after skipping
                total_samples = sum(sample_count for _, sample_count in processed_chunks)
                actual_duration_us = int(total_samples * 1_000_000 / target_rate)
                catchup_start_time_us = first_queued_start_us - actual_duration_us

                # Ensure start time is not before join time (due to rounding)
                catchup_start_time_us = max(catchup_start_time_us, player_state.join_time_us)

            logger.debug(
                "Catch-up aligned: %d samples = %.1f ms, "
                "starting at offset +%.1f ms, ending at offset +%.1f ms",
                total_samples,
                actual_duration_us / 1000,
                (catchup_start_time_us - self._play_start_time_us) / 1000,
                (first_queued_start_us - self._play_start_time_us) / 1000,
            )
        else:
            # No queued chunks - start from join time
            catchup_start_time_us = player_state.join_time_us or source_chunks[0].start_time_us
            logger.debug(
                "Catch-up with no prepared chunks: %d samples starting at offset +%.1f ms",
                total_samples,
                (catchup_start_time_us - self._play_start_time_us) / 1000,
            )

        # PHASE 3: Create PreparedChunkState objects for queueing
        catchup_chunks: list[PreparedChunkState] = []
        samples_sent = 0

        for chunk_data, sample_count in processed_chunks:
            # Calculate timestamps from sample position (same method as prepared chunks)
            start_us = catchup_start_time_us + int(samples_sent * 1_000_000 / target_rate)
            end_us = catchup_start_time_us + int(
                (samples_sent + sample_count) * 1_000_000 / target_rate
            )

            # Create chunk with refcount=1 (not shared, player-specific)
            chunk = PreparedChunkState(
                data=chunk_data,
                start_time_us=start_us,
                end_time_us=end_us,
                sample_count=sample_count,
                byte_count=len(chunk_data),
                refcount=1,  # Not shared - only for this player
            )
            catchup_chunks.append(chunk)
            samples_sent += sample_count

        return catchup_chunks

    def _collect_catchup_chunks(
        self,
        *,
        temp_buffer: bytearray,
        temp_encoder: av.AudioCodecContext | None,
        pipeline: PipelineState,
        processed_chunks: list[tuple[bytes, int]],
        force_flush: bool,
    ) -> None:
        """Drain temporary buffer and collect chunks (without sending).

        Args:
            temp_buffer: Temporary buffer to drain.
            temp_encoder: Temporary encoder (or None for PCM).
            pipeline: Pipeline config.
            processed_chunks: List to append (chunk_data, sample_count) tuples to.
            force_flush: Whether to flush all remaining samples.
        """
        frame_stride = pipeline.target_frame_stride

        while len(temp_buffer) >= frame_stride:
            available_samples = len(temp_buffer) // frame_stride
            if not force_flush and available_samples < pipeline.chunk_samples:
                break

            # Extract data to fit sample count
            sample_count = pipeline.chunk_samples
            if force_flush and available_samples < pipeline.chunk_samples:
                # Pad incomplete chunk with zeros to reach full chunk_samples
                audio_data_bytes = available_samples * frame_stride
                padding_bytes = (sample_count - available_samples) * frame_stride
                chunk = bytes(temp_buffer[:audio_data_bytes]) + bytes(padding_bytes)
                del temp_buffer[:audio_data_bytes]
            else:
                chunk_size = sample_count * frame_stride
                chunk = bytes(temp_buffer[:chunk_size])
                del temp_buffer[:chunk_size]

            if temp_encoder is None:
                # PCM - collect directly
                processed_chunks.append((chunk, sample_count))
            else:
                # Encode then collect
                frame = av.AudioFrame(
                    format=pipeline.target_av_format,
                    layout=pipeline.target_layout,
                    samples=sample_count,
                )
                frame.sample_rate = pipeline.target_format.sample_rate
                frame.planes[0].update(chunk)
                packets = temp_encoder.encode(frame)

                for packet in packets:
                    if not packet.duration or packet.duration <= 0:
                        raise ValueError(f"Invalid packet duration: {packet.duration!r}")
                    chunk_data = bytes(packet)
                    processed_chunks.append((chunk_data, packet.duration))

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
            self._process_source_pcm(
                pipeline,
                source_chunk,
            )
            pipeline.source_read_position += 1
            any_work = True

        return any_work

    def _process_source_pcm(
        self,
        pipeline: PipelineState,
        source_chunk: SourceChunk,
    ) -> None:
        """Process source PCM data through the pipeline's resampler.

        Args:
            pipeline: The pipeline to process through.
            source_chunk: The source PCM chunk to process.
        """
        # Initialize next_chunk_start_us from first source chunk
        if pipeline.next_chunk_start_us is None and not pipeline.buffer:
            pipeline.next_chunk_start_us = source_chunk.start_time_us

        frame = av.AudioFrame(
            format=pipeline.channel.av_format,
            layout=pipeline.channel.av_layout,
            samples=source_chunk.sample_count,
        )
        frame.sample_rate = pipeline.channel.audio_format.sample_rate
        frame.planes[0].update(source_chunk.pcm_data)
        out_frames = pipeline.resampler.resample(frame)
        for out_frame in out_frames:
            expected = pipeline.target_frame_stride * out_frame.samples
            pcm_bytes = bytes(out_frame.planes[0])[:expected]
            pipeline.buffer.extend(pcm_bytes)
        self._drain_pipeline_buffer(pipeline, force_flush=False)

    def _calculate_chunk_timestamps(
        self,
        pipeline: PipelineState,
        sample_count: int,
    ) -> tuple[int, int]:
        """Calculate start and end timestamps for a chunk.

        Uses the pipeline's next_chunk_start_us to maintain alignment with source timestamps.

        Args:
            pipeline: The pipeline producing the chunk.
            sample_count: Number of samples in the chunk.

        Returns:
            Tuple of (start_us, end_us) timestamps.
        """
        if pipeline.next_chunk_start_us is None:
            raise RuntimeError("Pipeline next_chunk_start_us not initialized")

        start_us = pipeline.next_chunk_start_us
        duration_us = int(sample_count * 1_000_000 / pipeline.target_format.sample_rate)
        end_us = start_us + duration_us
        return start_us, end_us

    def _drain_pipeline_buffer(
        self,
        pipeline: PipelineState,
        *,
        force_flush: bool,
    ) -> None:
        """Drain the pipeline buffer by creating and publishing chunks.

        Extracts complete chunks from the pipeline buffer and either publishes them
        directly (for PCM) or encodes them first (for compressed codecs).
        Calculates timestamps based on the pipeline's current sample position.

        Args:
            pipeline: The pipeline whose buffer to drain.
            force_flush: If True, publish all available samples even if less than chunk_samples.
        """
        if not pipeline.subscribers:
            pipeline.buffer.clear()
            return

        frame_stride = pipeline.target_frame_stride
        while len(pipeline.buffer) >= frame_stride:
            available_samples = len(pipeline.buffer) // frame_stride
            if not force_flush and available_samples < pipeline.chunk_samples:
                break

            # Extract data to fit sample count
            sample_count = pipeline.chunk_samples
            if force_flush and available_samples < pipeline.chunk_samples:
                # Pad incomplete chunk with zeros to reach full chunk_samples
                audio_data_bytes = available_samples * frame_stride
                padding_bytes = (sample_count - available_samples) * frame_stride
                chunk = bytes(pipeline.buffer[:audio_data_bytes]) + bytes(padding_bytes)
                del pipeline.buffer[:audio_data_bytes]
            else:
                chunk_size = sample_count * frame_stride
                chunk = bytes(pipeline.buffer[:chunk_size])
                del pipeline.buffer[:chunk_size]

            if pipeline.encoder is None:
                # PCM path: calculate timestamps from input sample count
                start_us, end_us = self._calculate_chunk_timestamps(pipeline, sample_count)
                self._publish_chunk(pipeline, chunk, sample_count, start_us, end_us)
                # Advance next_chunk_start_us for the next chunk
                pipeline.next_chunk_start_us = end_us
            else:
                # Encoder path: let encoder calculate timestamps from output packets
                self._encode_and_publish(pipeline, chunk, sample_count)

    def _encode_and_publish(
        self,
        pipeline: PipelineState,
        chunk: bytes,
        sample_count: int,
    ) -> None:
        """Encode a PCM chunk and publish the resulting packets.

        The encoder may buffer input and produce 0, 1, or multiple output packets.
        Timestamps are calculated from each output packet's duration.

        Args:
            pipeline: The pipeline containing the encoder.
            chunk: Raw PCM audio data to encode.
            sample_count: Number of samples in the chunk.
        """
        if pipeline.encoder is None:
            raise RuntimeError("Encoder not configured for this pipeline")
        frame = av.AudioFrame(
            format=pipeline.target_av_format,
            layout=pipeline.target_layout,
            samples=sample_count,
        )
        frame.sample_rate = pipeline.target_format.sample_rate
        frame.planes[0].update(chunk)
        packets = pipeline.encoder.encode(frame)

        # Encoder may produce 0 or more packets
        for packet in packets:
            if not packet.duration or packet.duration <= 0:
                raise ValueError(f"Invalid packet duration: {packet.duration!r}")
            # Calculate timestamps from output packet duration
            start_us, end_us = self._calculate_chunk_timestamps(pipeline, packet.duration)
            self._handle_encoded_packet(pipeline, packet, start_us, end_us)
            # Advance next_chunk_start_us for each packet produced
            pipeline.next_chunk_start_us = end_us

    def _handle_encoded_packet(
        self,
        pipeline: PipelineState,
        packet: av.Packet,
        start_us: int,
        end_us: int,
    ) -> None:
        """Handle an encoded packet by publishing it as a chunk.

        Args:
            pipeline: The pipeline that produced the packet.
            packet: The encoded audio packet from the encoder.
            start_us: Start timestamp in microseconds.
            end_us: End timestamp in microseconds.
        """
        assert packet.duration is not None  # For type checking
        chunk_data = bytes(packet)
        self._publish_chunk(pipeline, chunk_data, packet.duration, start_us, end_us)

    def _publish_chunk(
        self,
        pipeline: PipelineState,
        audio_data: bytes,
        sample_count: int,
        start_us: int,
        end_us: int,
    ) -> None:
        """Create a PreparedChunkState and queue it for all subscribers.

        Queues the chunk for delivery to all clients subscribed to this pipeline.

        Args:
            pipeline: The pipeline publishing the chunk.
            audio_data: The encoded or PCM audio data.
            sample_count: Number of samples in the chunk.
            start_us: Start timestamp in microseconds.
            end_us: End timestamp in microseconds.
        """
        if not pipeline.subscribers or sample_count <= 0:
            return

        chunk = PreparedChunkState(
            data=audio_data,
            start_time_us=start_us,
            end_time_us=end_us,
            sample_count=sample_count,
            byte_count=len(audio_data),
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


__all__ = [
    "AudioCodec",
    "AudioFormat",
    "ClientStreamConfig",
    "MediaStream",
    "SourceAudioSpec",
    "Streamer",
]
