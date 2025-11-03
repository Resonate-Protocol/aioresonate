"""High-level streaming pipeline primitives."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections import deque
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, cast
from uuid import UUID, uuid4

import av
from av.logging import Capture

from aioresonate.models import BinaryMessageType, pack_binary_header_raw
from aioresonate.models.player import StreamStartPlayer

logger = logging.getLogger(__name__)

# Universal main channel ID for the primary audio source.
# Used as the canonical source for visualization and as a fallback when
# player_channel() returns None.
MAIN_CHANNEL_ID: UUID = UUID("00000000-0000-0000-0000-000000000000")


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

    def time_for_capacity(self, bytes_needed: int) -> int:
        """Calculate time in microseconds until the buffer can accept bytes_needed more bytes."""
        if bytes_needed <= 0:
            return 0
        if bytes_needed >= self.capacity_bytes:
            # TODO: raise exception instead?
            logger.warning(
                "Chunk size %s exceeds reported buffer capacity %s for client %s",
                bytes_needed,
                self.capacity_bytes,
                self.client_id,
            )
            return 0

        # Prune consumed chunks once at the start
        now_us = self.prune_consumed()
        time_needed_us = 0

        # Simulate state without modifying it
        virtual_buffered_bytes = self.buffered_bytes
        virtual_chunk_idx = 0

        while True:
            projected_usage = virtual_buffered_bytes + bytes_needed
            if projected_usage <= self.capacity_bytes:
                # Returning here keeps the producer running because we are below capacity.
                return time_needed_us

            # Check if we have chunks left to virtually consume
            if virtual_chunk_idx >= len(self.buffered_chunks):
                # No more chunks to consume, but still over capacity
                return time_needed_us

            # Virtually advance time to when the next chunk finishes
            chunk = self.buffered_chunks[virtual_chunk_idx]
            sleep_target_us = chunk.end_time_us
            sleep_us = sleep_target_us - now_us
            time_needed_us += max(sleep_us, 0)

            # Virtually consume the chunk
            now_us = sleep_target_us
            virtual_buffered_bytes -= chunk.byte_count
            virtual_chunk_idx += 1

    async def wait_for_capacity(self, bytes_needed: int) -> None:
        """Block until the device buffer can accept bytes_needed more bytes."""
        sleep_time_us = self.time_for_capacity(bytes_needed)
        if sleep_time_us > 0:
            await asyncio.sleep(sleep_time_us / 1_000_000)

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
class AudioFormatParams:
    """Audio format parameters with computed PyAV values for processing."""

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

    source_format_params: AudioFormatParams
    """Source audio format parameters."""
    channel_id: UUID
    """Channel this pipeline consumes from."""
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
    """Read position in this pipeline's source channel buffer."""
    next_chunk_start_us: int | None = None
    """Next output chunk start timestamp, initialized from first source chunk."""


@dataclass
class ChannelState:
    """State for a single time-synchronized audio channel."""

    source_format_params: AudioFormatParams
    """Audio format parameters for this channel."""
    source_buffer: deque[SourceChunk] = field(default_factory=deque)
    """Buffer of raw PCM chunks scheduled for playback."""
    samples_produced: int = 0
    """Total samples added to this channel's buffer."""


@dataclass
class PlayerState:
    """Tracks delivery state for a single player."""

    config: ClientStreamConfig
    """Client streaming configuration."""
    audio_format: AudioFormat
    """Format key for pipeline lookup."""
    channel_id: UUID
    """Channel this player consumes from."""
    queue: deque[PreparedChunkState] = field(default_factory=deque)
    """Chunks queued for delivery."""
    buffer_tracker: BufferTracker | None = None
    """Tracks client buffer state."""
    join_time_us: int | None = None
    """When player joined in microseconds."""
    needs_catchup: bool = False
    """Whether player needs catch-up processing."""


class MediaStream:
    """
    Container for audio stream with optional per-device DSP support.

    Provides a main audio source used for visualization and playback. Optionally,
    implementations can override player_channel() to provide device-specific channels
    for individual DSP processing chains. If player_channel returns None, the main
    channel is used as fallback.
    """

    _main_channel_source: AsyncGenerator[bytes, None]
    """
    Main audio source generator yielding PCM bytes.

    Used for visualization, and as fallback when player_channel() returns None.
    """
    _main_channel_format: AudioFormat
    """Audio format of the main_channel()."""

    def __init__(
        self,
        *,
        main_channel_source: AsyncGenerator[bytes, None],
        main_channel_format: AudioFormat,
    ) -> None:
        """Initialise the media stream with audio source and format for main_channel()."""
        self._main_channel_source = main_channel_source
        self._main_channel_format = main_channel_format

    @property
    def main_channel(self) -> tuple[AsyncGenerator[bytes, None], AudioFormat]:
        """Return the main audio source generator and its audio format."""
        return self._main_channel_source, self._main_channel_format

    async def player_channel(
        self,
        player_id: str,
        preferred_format: AudioFormat | None = None,
        position_us: int = 0,
    ) -> tuple[AsyncGenerator[bytes, None], AudioFormat, int] | None:
        """
        Get a player-specific audio channel (time-synchronized with main channel).

        Args:
            player_id: Identifier for the player requesting the channel.
            preferred_format: The player's preferred native format.
            position_us: Position in microseconds relative to main_channel start.

        Returns:
            Tuple of (generator, format, actual_position_us) or None for fallback.
            The actual_position_us may differ from requested position_us if the
            implementation can only provide channels at specific boundaries.
        """
        _ = (player_id, preferred_format, position_us)
        return None


class Streamer:
    """Adapts incoming channel data to player-specific formats."""

    _loop: asyncio.AbstractEventLoop
    """Event loop used for time calculations and task scheduling."""
    _play_start_time_us: int
    """Absolute timestamp in microseconds when playback should start."""
    _channels: dict[UUID, ChannelState]
    """Mapping of channel IDs to their state."""
    _pipelines: dict[tuple[UUID, AudioFormat], PipelineState]
    """Mapping of (channel_id, target_format) to pipeline state."""
    _players: dict[str, PlayerState]
    """Mapping of client IDs to their player delivery state."""
    _last_chunk_end_us: int | None = None
    """End timestamp of the most recently prepared chunk, None if no chunks prepared yet."""
    _source_buffer_target_duration_us: int = 5_000_000
    """Target duration for source buffer in microseconds."""
    _min_send_margin_us: int = 1_000_000
    """Minimum time margin before playback for stale chunk detection (1 second)."""
    _skip_stale_check_once: bool = False
    """Skip stale check for one iteration after reconfigure to avoid false positives."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        play_start_time_us: int,
    ) -> None:
        """Create a streamer bound to the event loop and playback start time."""
        self._loop = loop
        self._play_start_time_us = play_start_time_us
        self._channels = {}
        self._pipelines = {}
        self._players = {}

    def _cleanup_consumed_chunks(self, pipeline: PipelineState) -> None:
        """Clean up consumed chunks from a pipeline.

        Note: consumed chunks (refcount == 0) can only appear as a contiguous block
        at the front since chunks are consumed in FIFO order.

        Args:
            pipeline: The pipeline to clean up consumed chunks from.
        """
        while pipeline.prepared and pipeline.prepared[0].refcount == 0:
            pipeline.prepared.popleft()

    async def configure(
        self,
        all_player_configs: list[ClientStreamConfig],
        ## TODO: refactor so new_player_ids is not needed
        new_player_ids: set[str],
        media_stream: MediaStream,
    ) -> tuple[dict[str, StreamStartPlayer], dict[UUID, AsyncGenerator[bytes, None]]]:
        """Configure or reconfigure pipelines for the provided players.

        Resolves topology (which players get which channels) by querying MediaStream
        for player-specific channels and calculating synchronization offsets.

        Args:
            all_player_configs: List of ClientStreamConfig for all players (existing and new).
            new_player_ids: Set of client IDs that are newly joining.
            media_stream: Media stream providing audio sources and player channels.

        Returns:
            Tuple of (start_payloads, new_channel_sources):
            - start_payloads: Dict mapping client IDs to StreamStartPlayer messages
            - new_channel_sources: Dict mapping channel IDs to source generators for new channels
        """
        # Build topology: determine which players are new, query channels
        channel_formats: dict[UUID, AudioFormat] = {MAIN_CHANNEL_ID: media_stream.main_channel[1]}
        new_channel_sources: dict[UUID, AsyncGenerator[bytes, None]] = {
            MAIN_CHANNEL_ID: media_stream.main_channel[0]
        }
        player_channel_assignments: dict[str, UUID] = {}
        channel_initial_samples: dict[UUID, int] = {MAIN_CHANNEL_ID: 0}

        # Build set of player IDs in the new configuration
        new_config_player_ids = {cfg.client_id for cfg in all_player_configs}

        # Preserve existing channel assignments and add their formats to channel_formats
        # Only preserve for players that are still in the new configuration
        for player_id, player_state in self._players.items():
            if player_id not in new_config_player_ids:
                continue  # Skip players being removed
            player_channel_assignments[player_id] = player_state.channel_id
            # Preserve existing channel formats to prevent them from being removed
            channel_id = player_state.channel_id
            if channel_id not in channel_formats and channel_id in self._channels:
                channel_state = self._channels[channel_id]
                channel_formats[channel_id] = channel_state.source_format_params.audio_format

        # Query channels for new players
        for player_id in new_player_ids:
            # Find the config for this player
            player_config = next((c for c in all_player_configs if c.client_id == player_id), None)
            if player_config is None:
                logger.warning("Config not found for new player %s", player_id)
                # Assign to main channel
                player_channel_assignments[player_id] = MAIN_CHANNEL_ID
                continue

            # Try to get player-specific channel with error handling
            try:
                player_channel_result = await media_stream.player_channel(
                    player_id=player_id,
                    preferred_format=player_config.target_format,
                    position_us=int(self._loop.time() * 1_000_000) - self._play_start_time_us,
                )
            except Exception:
                logger.exception(
                    "Failed to query player_channel for %s, falling back to main channel",
                    player_id,
                )
                player_channel_result = None

            if player_channel_result is not None:
                source, channel_format, actual_pos_us = player_channel_result
                channel_id = uuid4()

                # Add new channel
                channel_formats[channel_id] = channel_format
                new_channel_sources[channel_id] = source
                player_channel_assignments[player_id] = channel_id

                # Calculate and store position offset
                initial_samples = round(actual_pos_us * channel_format.sample_rate / 1_000_000)
                channel_initial_samples[channel_id] = initial_samples

                # Calculate when the first chunk from this channel will play
                first_chunk_start_us = self._play_start_time_us + int(
                    initial_samples * 1_000_000 / channel_format.sample_rate
                )
                now_us = int(self._loop.time() * 1_000_000)
                delay_s = (first_chunk_start_us - now_us) / 1_000_000

                logger.info(
                    "Player %s assigned to dedicated channel with offset %d us (%.3f s). "
                    "First chunk will play in %.3f seconds from now.",
                    player_id,
                    actual_pos_us,
                    actual_pos_us / 1_000_000,
                    delay_s,
                )
            else:
                # Fallback to main channel
                player_channel_assignments[player_id] = MAIN_CHANNEL_ID
                logger.info("Player %s assigned to main channel", player_id)

        # Apply topology to internal state
        start_payloads = self._apply_topology(
            channel_formats=channel_formats,
            channel_initial_samples=channel_initial_samples,
            clients=all_player_configs,
            player_channel_assignments=player_channel_assignments,
        )

        return start_payloads, new_channel_sources

    ## merge this into configure
    def _apply_topology(  # noqa: PLR0915
        self,
        *,
        channel_formats: dict[UUID, AudioFormat],
        channel_initial_samples: Mapping[UUID, int] | None = None,
        clients: Iterable[ClientStreamConfig],
        player_channel_assignments: dict[str, UUID],
    ) -> dict[str, StreamStartPlayer]:
        """Apply resolved topology to internal state.

        Args:
            channel_formats: Mapping of channel_id to source audio format.
            channel_initial_samples: Optional mapping of channel_id to the sample index
                (relative to play_start) already covered when the channel was created.
            clients: Configuration for each client/player.
            player_channel_assignments: Mapping of client_id to channel_id.

        Returns:
            Dictionary mapping client IDs to their StreamStartPlayer messages.
        """
        # Update or create channel states
        for channel_id, audio_format in channel_formats.items():
            if channel_id not in self._channels:
                # New channel - create it
                bytes_per_sample, av_format, av_layout = _resolve_audio_format(audio_format)
                self._channels[channel_id] = ChannelState(
                    source_format_params=AudioFormatParams(
                        audio_format=audio_format,
                        bytes_per_sample=bytes_per_sample,
                        frame_stride=bytes_per_sample * audio_format.channels,
                        av_format=av_format,
                        av_layout=av_layout,
                    ),
                )
                # Only set initial samples for newly created channels
                if channel_initial_samples and channel_id in channel_initial_samples:
                    initial_sample_count = channel_initial_samples[channel_id]
                    self._channels[channel_id].samples_produced = initial_sample_count

        # Remove channels that are no longer needed
        channels_to_remove = set(self._channels) - set(channel_formats)
        for channel_id in channels_to_remove:
            self._channels.pop(channel_id)

        # Clear subscriber lists to rebuild them
        for existing_pipeline in self._pipelines.values():
            existing_pipeline.subscribers.clear()

        # Build new player and subscription configuration
        new_players: dict[str, PlayerState] = {}
        start_payloads: dict[str, StreamStartPlayer] = {}

        for client_cfg in clients:
            channel_id = player_channel_assignments[client_cfg.client_id]
            audio_format = client_cfg.target_format
            pipeline_key = (channel_id, audio_format)
            pipeline: PipelineState | None = self._pipelines.get(pipeline_key)
            if pipeline is None:
                # Create new pipeline for this channel/format
                channel_state = self._channels[channel_id]
                source_format_params = channel_state.source_format_params

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
                    source_format_params=source_format_params,
                    channel_id=channel_id,
                    target_format=client_cfg.target_format,
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
                old_pipeline_key = (old_player.channel_id, old_player.audio_format)
                if old_pipeline := self._pipelines.get(old_pipeline_key):
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

            # Find synchronized join point based on SOURCE CHANNEL
            # Use the earliest prepared chunk across ALL pipelines on this channel
            # This ensures late joiners start from the beginning of available audio,
            # not from where existing players' queues currently are (which would skip
            # already-sent chunks that are still playing on those clients)
            sync_point_start_time_us: int | None = None
            channel_id = player_channel_assignments[client_cfg.client_id]

            # Check all pipelines consuming from this channel for their earliest prepared chunk
            for (pipe_channel_id, _), pipe in self._pipelines.items():
                if pipe_channel_id == channel_id and pipe.prepared:
                    chunk_start = pipe.prepared[0].start_time_us
                    if sync_point_start_time_us is None or chunk_start < sync_point_start_time_us:
                        sync_point_start_time_us = chunk_start

            # Fallback to play_start_time_us if no prepared chunks exist yet (initial startup)
            # This ensures chunks timestamped relative to play_start_time_us will be queued
            if sync_point_start_time_us is None:
                sync_point_start_time_us = self._play_start_time_us

            player_state = PlayerState(
                config=client_cfg,
                audio_format=audio_format,
                channel_id=channel_id,
                buffer_tracker=buffer_tracker,
                join_time_us=sync_point_start_time_us,
            )

            # Queue chunks starting from the sync point
            for chunk in pipeline.prepared:
                if chunk.start_time_us >= sync_point_start_time_us:
                    player_state.queue.append(chunk)
                    chunk.refcount += 1

            # No catchup needed since we're starting at the sync point
            player_state.needs_catchup = False

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
                pipeline_key = (old_player.channel_id, old_player.audio_format)
                if old_pipeline := self._pipelines.get(pipeline_key):
                    self._cleanup_consumed_chunks(old_pipeline)

        # Replace players dict
        self._players = new_players

        # Skip stale check on next send() iteration to avoid false positives
        # when newly joined players have chunks with past timestamps
        self._skip_stale_check_once = True

        return start_payloads

    def channel_wait_time_us(self, channel_id: UUID, now_us: int) -> int | None:
        """Calculate time in microseconds until a channel needs data.

        Args:
            channel_id: ID of the channel to check.
            now_us: Current time in microseconds.

        Returns:
            Wait time in microseconds (0 if immediate), or None if channel has no buffer.
        """
        channel_state = self._channels.get(channel_id)
        assert channel_state is not None
        if not channel_state.source_buffer:
            return None

        buffer_end = channel_state.source_buffer[-1].end_time_us
        # Calculate when buffer will drop below target duration from now
        target_time_us = buffer_end - self._source_buffer_target_duration_us
        return max(0, target_time_us - now_us)

    def channel_needs_data(self, channel_id: UUID) -> bool:
        """Check if a channel's buffer is below the target duration from now.

        Args:
            channel_id: ID of the channel to check.

        Returns:
            True if buffer depth from now < target, False otherwise.
        """
        channel_state = self._channels.get(channel_id)
        assert channel_state is not None
        if not channel_state.source_buffer:
            return True

        # Measure buffer depth from now (not from buffer start)
        now_us = int(self._loop.time() * 1_000_000)
        wait_time_us = self.channel_wait_time_us(channel_id, now_us)
        return wait_time_us is None or wait_time_us == 0

    def prepare(
        self, channel_id: UUID, chunk: bytes, *, during_initial_buffering: bool = False
    ) -> None:
        """Buffer raw PCM data and process through pipelines.

        Args:
            channel_id: ID of the channel this chunk belongs to.
            chunk: Raw PCM audio data to buffer.
            during_initial_buffering: True when filling initial buffer on startup,
                which skips building full 5-second buffer during timing adjustments.
        """
        channel_state = self._channels[channel_id]
        if len(chunk) % channel_state.source_format_params.frame_stride:
            raise ValueError("Chunk must be aligned to whole samples")
        sample_count = len(chunk) // channel_state.source_format_params.frame_stride
        if sample_count == 0:
            return

        # Calculate timestamps for this chunk
        start_samples = channel_state.samples_produced

        # Check and adjust for stale chunks (skip during initial buffering)
        if not during_initial_buffering:
            start_us, end_us = self._check_and_adjust_for_stale_chunk(
                channel_id, channel_state, start_samples, sample_count
            )
        else:
            # During initial buffering, just calculate timestamps without stale detection
            start_us = self._play_start_time_us + int(
                start_samples
                * 1_000_000
                / channel_state.source_format_params.audio_format.sample_rate
            )
            end_us = self._play_start_time_us + int(
                (start_samples + sample_count)
                * 1_000_000
                / channel_state.source_format_params.audio_format.sample_rate
            )

        # Create and buffer the source chunk
        source_chunk = SourceChunk(
            pcm_data=chunk,
            start_time_us=start_us,
            end_time_us=end_us,
            sample_count=sample_count,
        )
        channel_state.source_buffer.append(source_chunk)
        channel_state.samples_produced += sample_count

        # Process through pipelines that consume from this channel
        for pipeline in self._pipelines.values():
            if pipeline.channel_id == channel_id:
                self._process_pipeline_from_source(pipeline, channel_state)

    def _check_and_adjust_for_stale_chunk(
        self,
        channel_id: UUID,
        channel_state: ChannelState,
        start_samples: int,
        sample_count: int,
    ) -> tuple[int, int]:
        """Check if the next chunk would be stale and adjust timing if needed.

        Args:
            channel_id: ID of the channel.
            channel_state: The channel state for this chunk.
            start_samples: Sample position where the chunk starts.
            sample_count: Number of samples in the chunk.

        Returns:
            Tuple of (start_us, end_us) timestamps after any adjustments.
        """
        # Calculate initial timestamps
        start_us = self._play_start_time_us + int(
            start_samples * 1_000_000 / channel_state.source_format_params.audio_format.sample_rate
        )

        # Check if this chunk would be stale
        now_us = int(self._loop.time() * 1_000_000)

        # Only apply global timing adjustments for the main channel
        # Dedicated player channels with offsets should not trigger global adjustments
        # as this would desynchronize all channels. Stale chunks from offset channels
        # will be naturally skipped by send().
        if start_us < now_us + self._min_send_margin_us and channel_id == MAIN_CHANNEL_ID:
            # Adjust timing globally
            self._adjust_timing_for_stale_chunk(channel_id, now_us, start_us)
            # Recalculate timestamps after adjustment
            start_us = self._play_start_time_us + int(
                start_samples
                * 1_000_000
                / channel_state.source_format_params.audio_format.sample_rate
            )

        end_us = self._play_start_time_us + int(
            (start_samples + sample_count)
            * 1_000_000
            / channel_state.source_format_params.audio_format.sample_rate
        )

        return start_us, end_us

    def _adjust_timing_for_stale_chunk(
        self, channel_id: UUID, now_us: int, chunk_start_us: int
    ) -> None:
        """Adjust timing when a stale chunk is detected.

        Args:
            channel_id: ID of the channel.
            now_us: Current time in microseconds.
            chunk_start_us: Start time of the stale chunk.
        """
        channel_state = self._channels[channel_id]
        target_buffer_us = self._source_buffer_target_duration_us

        # Calculate current buffer depth (from now to end of buffer)
        current_buffer_us = 0
        if channel_state.source_buffer:
            # Buffer depth is from now to the end of the last buffered chunk
            last_chunk_end = channel_state.source_buffer[-1].end_time_us
            current_buffer_us = max(0, last_chunk_end - now_us)

        # Calculate minimum adjustment needed to give this chunk proper headroom
        headroom_shortfall_us = (now_us + self._min_send_margin_us) - chunk_start_us

        # Determine total adjustment based on buffer status
        if current_buffer_us >= target_buffer_us:
            # We already have enough buffer, just ensure headroom
            timing_adjustment_us = headroom_shortfall_us
            logger.debug(
                "Adjusting timing globally from channel %s: needs %.3fs headroom, "
                "have %.3fs buffer (adjusting %.3fs)",
                channel_id,
                headroom_shortfall_us / 1_000_000,
                current_buffer_us / 1_000_000,
                timing_adjustment_us / 1_000_000,
            )
        else:
            # Need to build buffer to target level
            buffer_shortfall_us = target_buffer_us - current_buffer_us
            # Use the larger of headroom need and buffer need
            timing_adjustment_us = max(headroom_shortfall_us, buffer_shortfall_us)
            logger.debug(
                "Adjusting timing globally from channel %s: needs %.3fs headroom, "
                "have %.3fs buffer, target %.3fs (adjusting %.3fs)",
                channel_id,
                headroom_shortfall_us / 1_000_000,
                current_buffer_us / 1_000_000,
                target_buffer_us / 1_000_000,
                timing_adjustment_us / 1_000_000,
            )

        # Adjust timing forward
        self._play_start_time_us += timing_adjustment_us

        # Update source buffer chunk timestamps
        for ch_state in self._channels.values():
            for source_chunk in ch_state.source_buffer:
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

    def _validate_group_sync(self) -> None:
        """Validate that all players on the same channel have synchronized timestamps.

        This is a safeguard to detect synchronization issues early. All players
        sharing the same source channel must have chunks with matching timestamps
        at the front of their queues (within tolerance), even if they're on
        different pipelines (different target formats).
        """
        # Group players by source channel
        channel_players: dict[UUID, list[PlayerState]] = {}
        for player_state in self._players.values():
            channel_id = player_state.channel_id
            if channel_id not in channel_players:
                channel_players[channel_id] = []
            channel_players[channel_id].append(player_state)

        # Check each channel's players for timestamp sync
        for channel_id, players in channel_players.items():
            if len(players) <= 1:
                # Single player, no sync issues possible
                continue

            # Get the timestamps at the front of each player's queue
            front_timestamps: list[int | None] = []
            for player in players:
                front_ts = player.queue[0].start_time_us if player.queue else None
                front_timestamps.append(front_ts)

            # Check if all players have matching timestamps (within small tolerance)
            # Players on different pipelines might have slightly different timestamps
            # due to resampling, but should be very close (within 1ms)
            first_ts = front_timestamps[0]
            if first_ts is not None:
                for ts in front_timestamps[1:]:
                    if ts is None:
                        continue
                    # Allow 1ms tolerance for resampling differences
                    if abs(ts - first_ts) > 1000:
                        # Desync detected!
                        player_ids = [p.config.client_id for p in players]
                        chunk_times = [
                            f"{ts}us" if ts is not None else "empty" for ts in front_timestamps
                        ]
                        logger.error(
                            "SYNC ERROR: Players on channel %s are desynchronized! "
                            "Players: %s, Front chunk timestamps: %s",
                            channel_id,
                            player_ids,
                            chunk_times,
                        )
                        # Raise assertion error to catch sync violations
                        msg = (
                            f"Group sync violation: players {player_ids} on channel "
                            f"{channel_id} have timestamps differing by >1ms: {chunk_times}"
                        )
                        raise AssertionError(msg)

    def _adjust_timing_for_stale_chunk_all_channels(self, now_us: int, chunk_start_us: int) -> None:
        """Adjust timing globally when a stale chunk is detected (for all channels).

        This method adjusts play_start_time_us forward to prevent chunk skipping,
        ensuring all players stay synchronized even when timing adjustments are needed.
        Unlike _adjust_timing_for_stale_chunk, this applies to ALL channels.

        Args:
            now_us: Current time in microseconds.
            chunk_start_us: Start time of the stale chunk.
        """
        target_buffer_us = self._source_buffer_target_duration_us

        # Calculate adjustment needed across all channels
        max_buffer_us = 0
        min_buffer_us = None
        for channel_state in self._channels.values():
            if channel_state.source_buffer:
                last_chunk_end = channel_state.source_buffer[-1].end_time_us
                current_buffer_us = max(0, last_chunk_end - now_us)
                max_buffer_us = max(max_buffer_us, current_buffer_us)
                if min_buffer_us is None or current_buffer_us < min_buffer_us:
                    min_buffer_us = current_buffer_us

        # Calculate minimum adjustment needed to give this chunk proper headroom
        headroom_shortfall_us = (now_us + self._min_send_margin_us) - chunk_start_us

        # Determine total adjustment based on buffer status
        if min_buffer_us is not None and min_buffer_us >= target_buffer_us:
            # We already have enough buffer, just ensure headroom
            timing_adjustment_us = headroom_shortfall_us
            logger.debug(
                "Adjusting timing globally: needs %.3fs headroom, "
                "have %.3fs min buffer (adjusting %.3fs)",
                headroom_shortfall_us / 1_000_000,
                min_buffer_us / 1_000_000,
                timing_adjustment_us / 1_000_000,
            )
        else:
            # Need to build buffer to target level
            current_buffer_us = min_buffer_us if min_buffer_us is not None else 0
            buffer_shortfall_us = target_buffer_us - current_buffer_us
            # Use the larger of headroom need and buffer need
            timing_adjustment_us = max(headroom_shortfall_us, buffer_shortfall_us)
            logger.debug(
                "Adjusting timing globally: needs %.3fs headroom, "
                "have %.3fs min buffer, target %.3fs (adjusting %.3fs)",
                headroom_shortfall_us / 1_000_000,
                current_buffer_us / 1_000_000,
                target_buffer_us / 1_000_000,
                timing_adjustment_us / 1_000_000,
            )

        # Adjust timing forward
        self._play_start_time_us += timing_adjustment_us

        # Update source buffer chunk timestamps for all channels
        for ch_state in self._channels.values():
            for source_chunk in ch_state.source_buffer:
                source_chunk.start_time_us += timing_adjustment_us
                source_chunk.end_time_us += timing_adjustment_us

        # Update pipeline timestamps and prepared chunks for all pipelines
        for pipeline in self._pipelines.values():
            if pipeline.next_chunk_start_us is not None:
                pipeline.next_chunk_start_us += timing_adjustment_us
            # Update timestamps of already-prepared chunks to prevent cascading adjustments
            for prepared_chunk in pipeline.prepared:
                prepared_chunk.start_time_us += timing_adjustment_us
                prepared_chunk.end_time_us += timing_adjustment_us

    async def send(self) -> None:  # noqa: PLR0915, PLR0912, C901
        """Send prepared audio to all clients with perfect group synchronization.

        This method performs stages in a loop:
        1. Perform catch-up for late joiners (if needed)
        2. Check for stale chunks and adjust timing globally if needed (prevents skipping)
        3. Send chunks to players with backpressure control (per-player throughput)
        4. Validate synchronization (debug safeguard)
        5. Prune old data
        6. Check exit conditions and apply source buffer backpressure

        Global timing adjustments prevent chunk skipping, ensuring all players
        receive identical audio content for perfect synchronization. Players can
        have different queue depths (due to buffer capacity differences), but all
        receive the same chunks with the same timestamps.

        Continues until all pending audio has been delivered and source buffer is below target.
        """
        while True:
            logger.debug("Streamer send loop iteration started")
            # Stage 1: Perform catch-up for players that need it
            for player_state in self._players.values():
                if player_state.needs_catchup:
                    self._perform_catchup(player_state)

            # Stage 2: Check for stale chunks on MAIN_CHANNEL only
            # Skip this check for one iteration after reconfigure to avoid false positives
            # when newly joined players have chunks with past timestamps
            if self._skip_stale_check_once:
                self._skip_stale_check_once = False
                logger.debug("Skipping stale check after reconfiguration")
            else:
                # Dedicated player channels can have arbitrary timestamps from player_channel()
                now_us = int(self._loop.time() * 1_000_000)
                min_send_margin_us = 100_000  # 100ms for network + client processing

                # Find the earliest chunk on MAIN_CHANNEL only
                earliest_main_chunk_start: int | None = None
                for player_state in self._players.values():
                    if player_state.channel_id == MAIN_CHANNEL_ID and player_state.queue:
                        chunk = player_state.queue[0]
                        if (
                            earliest_main_chunk_start is None
                            or chunk.start_time_us < earliest_main_chunk_start
                        ):
                            earliest_main_chunk_start = chunk.start_time_us

                # If main channel chunk is stale, adjust timing globally
                if (
                    earliest_main_chunk_start is not None
                    and earliest_main_chunk_start < now_us + min_send_margin_us
                ):
                    logger.warning(
                        "Main channel chunk is stale (starts at %d us, now is %d us). "
                        "Adjusting timing globally.",
                        earliest_main_chunk_start,
                        now_us,
                    )
                    self._adjust_timing_for_stale_chunk_all_channels(
                        now_us, earliest_main_chunk_start
                    )
                    # After adjustment, continue to next iteration with updated timing
                    continue

            # Stage 3: Send chunks to players with backpressure control
            earliest_blocked_player: PlayerState | None = None
            earliest_blocked_chunk_size = 0
            smallest_wait_time = 0

            for player_state in self._players.values():
                tracker = player_state.buffer_tracker
                if tracker is None:
                    continue
                queue = player_state.queue

                # Send as many chunks as possible for this player
                while queue:
                    chunk = queue[0]

                    # No per-player stale skipping - timing adjustment prevents this

                    # Check if we can send without waiting
                    if requested_wait := tracker.time_for_capacity(chunk.byte_count):
                        # This player is blocked - track if this chunk is earliest
                        if smallest_wait_time == 0 or requested_wait < smallest_wait_time:
                            earliest_blocked_player = player_state
                            earliest_blocked_chunk_size = chunk.byte_count
                            smallest_wait_time = requested_wait
                        break

                    # We have capacity - send immediately
                    header = pack_binary_header_raw(
                        BinaryMessageType.AUDIO_CHUNK.value, chunk.start_time_us
                    )
                    player_state.config.send(header + chunk.data)
                    tracker.register(chunk.end_time_us, chunk.byte_count)
                    self._dequeue_chunk(player_state, chunk)

            # Stage 3b: Handle backpressure - compare client buffer wait vs source buffer urgency
            if smallest_wait_time > 0:
                # Calculate when source buffers will need refilling using helper method
                now_us_for_scheduling = int(self._loop.time() * 1_000_000)
                source_buffer_wait_us = None

                for channel_id in self._channels:
                    wait_us = self.channel_wait_time_us(channel_id, now_us_for_scheduling)
                    if wait_us is None:
                        # Channel has no buffer - needs immediate refilling
                        source_buffer_wait_us = None
                        break
                    if source_buffer_wait_us is None or wait_us < source_buffer_wait_us:
                        source_buffer_wait_us = wait_us

                # Choose the more urgent wait
                # If source_buffer_wait_us is None, it means at least one channel is empty
                if source_buffer_wait_us is None or source_buffer_wait_us < smallest_wait_time:
                    # Source buffer is more urgent - wait for it, then exit to refill
                    if source_buffer_wait_us and source_buffer_wait_us > 0:
                        sleep_duration_s = source_buffer_wait_us / 1_000_000
                        logger.debug(
                            "Source buffer more urgent (%.3fs vs client %.3fs), "
                            "waiting then exiting to refill",
                            sleep_duration_s,
                            smallest_wait_time / 1_000_000,
                        )
                        await asyncio.sleep(sleep_duration_s)
                    else:
                        logger.debug("Source buffer needs immediate refilling, exiting")
                    break  # Exit to refill source buffer

                # Client buffer is more urgent - wait for it, then continue sending
                assert earliest_blocked_player is not None
                tracker = earliest_blocked_player.buffer_tracker
                if tracker is not None:
                    logger.debug(
                        "Client buffer more urgent (%.3fs vs source %.3fs), "
                        "waiting for client capacity",
                        smallest_wait_time / 1_000_000,
                        source_buffer_wait_us / 1_000_000,
                    )
                    await tracker.wait_for_capacity(earliest_blocked_chunk_size)
                continue  # More work to do, loop again

            # Stage 4: Validate synchronization (debug safeguard)
            self._validate_group_sync()

            # Stage 5: Cleanup
            self._prune_old_data()

            # Stage 6: Check exit conditions and apply source buffer backpressure
            has_client_work = any(
                player_state.queue or player_state.needs_catchup
                for player_state in self._players.values()
            )

            # Check source buffer status - exit send() when buffers need more data
            # so that prepare() can be called to fill them
            any_channel_needs_data = any(
                self.channel_needs_data(channel_id) for channel_id in self._channels
            )

            if has_client_work:
                # If client work pending, continue immediately
                continue
            if any_channel_needs_data:
                # Exit when both conditions met: no client work pending in send() AND
                # at least one buffer needs data
                break

            # Stage 6b: Wait for source buffer to drain below target
            # Calculate when source buffers will need refilling using helper method
            now_us = int(self._loop.time() * 1_000_000)
            source_buffer_wait_us = None

            for channel_id in self._channels:
                wait_us = self.channel_wait_time_us(channel_id, now_us)
                if wait_us is None:
                    # Channel has no buffer - needs immediate refilling
                    source_buffer_wait_us = None
                    break
                if source_buffer_wait_us is None or wait_us < source_buffer_wait_us:
                    source_buffer_wait_us = wait_us

            # If no source buffer wait calculated or immediate need, exit to refill
            if source_buffer_wait_us is None:
                # At least one channel has no buffer
                # Wait for first chunk to be consumed to avoid busy loop
                earliest_chunk_end_us = None
                for channel_state in self._channels.values():
                    if channel_state.source_buffer:
                        end_us = channel_state.source_buffer[0].end_time_us
                        if earliest_chunk_end_us is None or end_us < earliest_chunk_end_us:
                            earliest_chunk_end_us = end_us
                if earliest_chunk_end_us and earliest_chunk_end_us > now_us:
                    sleep_duration_s = (earliest_chunk_end_us - now_us) / 1_000_000
                    logger.debug(
                        "No buffer on at least one channel, waiting %.3fs for first chunk",
                        sleep_duration_s,
                    )
                    await asyncio.sleep(sleep_duration_s)
                else:
                    logger.debug("Source buffer needs immediate refilling, exiting send()")
                break

            if source_buffer_wait_us == 0:
                # Source buffer needs immediate attention - exit to refill
                logger.debug("Source buffer needs immediate refilling, exiting send()")
                break

            sleep_duration_s = source_buffer_wait_us / 1_000_000
            logger.debug("Waiting %.3fs for source buffer to drain below target", sleep_duration_s)
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
        self._channels.clear()
        self._pipelines.clear()
        self._players.clear()

    def _dequeue_chunk(self, player_state: PlayerState, chunk: PreparedChunkState) -> None:
        """Remove chunk from player queue and clean up pipeline if fully consumed."""
        player_state.queue.popleft()
        chunk.refcount -= 1
        pipeline = self._pipelines[(player_state.channel_id, player_state.audio_format)]
        if chunk.refcount == 0 and pipeline.prepared and pipeline.prepared[0] is chunk:
            pipeline.prepared.popleft()

    def _prune_old_data(self) -> None:
        """Prune old source chunks to free memory.

        Removes source chunks that have finished playing (end_time_us <= now).
        Prepared chunks are managed separately by refcount in send().
        """
        # Prune source buffer based on playback time
        now_us = int(self._loop.time() * 1_000_000)

        # Prune each channel's buffer independently
        for channel_id, channel_state in self._channels.items():
            chunks_removed = 0
            while (
                channel_state.source_buffer and channel_state.source_buffer[0].end_time_us <= now_us
            ):
                channel_state.source_buffer.popleft()
                chunks_removed += 1

            # Update pipeline read positions for pipelines consuming from this channel
            if chunks_removed > 0:
                for pipeline in self._pipelines.values():
                    if pipeline.channel_id == channel_id:
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
        channel_state = self._channels.get(player_state.channel_id)
        if not channel_state or not channel_state.source_buffer:
            return False

        # Determine if there's a gap that can be filled from source buffer
        first_queued_start_us = player_state.queue[0].start_time_us if player_state.queue else None

        # Check if any source chunks exist in the gap range
        for source_chunk in channel_state.source_buffer:
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
        channel_state = self._channels.get(player_state.channel_id)
        if (
            not channel_state
            or not channel_state.source_buffer
            or player_state.join_time_us is None
        ):
            return

        join_time_us = player_state.join_time_us
        pipeline = self._pipelines[(player_state.channel_id, player_state.audio_format)]

        # Determine the coverage range we need to fill
        first_queued_start_us = player_state.queue[0].start_time_us if player_state.queue else None

        # Find source chunks that cover the gap
        catchup_sources = []
        for source_chunk in channel_state.source_buffer:
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
            channel_state=channel_state,
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
        channel_state: ChannelState,
        player_state: PlayerState,
        source_chunks: list[SourceChunk],
        first_queued_start_us: int | None,
    ) -> list[PreparedChunkState]:
        """Process source chunks and create catch-up chunks for queueing.

        Creates temporary resampler/encoder to avoid corrupting shared pipeline state.
        Uses sample-based timestamp calculation to align perfectly with prepared chunks.

        Args:
            pipeline: Pipeline config to use for processing.
            channel_state: Channel state for source format data.
            player_state: Player to send chunks to.
            source_chunks: Source chunks to process.
            first_queued_start_us: Start time of first queued prepared chunk (for alignment).

        Returns:
            List of PreparedChunkState objects to prepend to player queue.
        """
        if not source_chunks:
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
                format=channel_state.source_format_params.av_format,
                layout=channel_state.source_format_params.av_layout,
                samples=source_chunk.sample_count,
            )
            frame.sample_rate = channel_state.source_format_params.audio_format.sample_rate
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

    def _process_pipeline_from_source(
        self, pipeline: PipelineState, channel_state: ChannelState
    ) -> bool:
        """Process available source chunks through this pipeline.

        Args:
            pipeline: The pipeline to process.
            channel_state: The channel state to read from.

        Returns:
            True if any work was done, False otherwise.
        """
        if not pipeline.subscribers:
            return False

        any_work = False
        # Process all available source chunks that haven't been processed yet
        while pipeline.source_read_position < len(channel_state.source_buffer):
            source_chunk = channel_state.source_buffer[pipeline.source_read_position]
            self._process_source_pcm(
                pipeline,
                channel_state,
                source_chunk,
            )
            pipeline.source_read_position += 1
            any_work = True

        return any_work

    def _process_source_pcm(
        self,
        pipeline: PipelineState,
        channel_state: ChannelState,
        source_chunk: SourceChunk,
    ) -> None:
        """Process source PCM data through the pipeline's resampler.

        Args:
            pipeline: The pipeline to process through.
            channel_state: The channel state for source format data.
            source_chunk: The source PCM chunk to process.
        """
        # Initialize next_chunk_start_us from first source chunk
        if pipeline.next_chunk_start_us is None and not pipeline.buffer:
            pipeline.next_chunk_start_us = source_chunk.start_time_us

        frame = av.AudioFrame(
            format=channel_state.source_format_params.av_format,
            layout=channel_state.source_format_params.av_layout,
            samples=source_chunk.sample_count,
        )
        frame.sample_rate = channel_state.source_format_params.audio_format.sample_rate
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

    def get_channel_ids(self) -> set[UUID]:
        """Get the set of active channel IDs.

        Returns:
            Set of currently active channel IDs.
        """
        return set(self._channels.keys())

    def get_player_ids(self) -> set[str]:
        """Get the set of active player IDs.

        Returns:
            Set of currently active player client IDs.
        """
        return set(self._players.keys())

    @property
    def last_chunk_end_time_us(self) -> int | None:
        """Return the end timestamp of the most recently prepared chunk."""
        return self._last_chunk_end_us


__all__ = [
    "MAIN_CHANNEL_ID",
    "AudioCodec",
    "AudioFormat",
    "AudioFormatParams",
    "ClientStreamConfig",
    "MediaStream",
    "Streamer",
]
