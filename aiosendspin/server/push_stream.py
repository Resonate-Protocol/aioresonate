"""Push-based audio streaming API."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from aiosendspin.models.types import Roles
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.pipeline import EncodedChunk, PipelineKey, PipelineManager

if TYPE_CHECKING:
    from aiosendspin.server.audio import AudioFormat
    from aiosendspin.server.channels import ChannelRouter
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.clock import Clock
    from aiosendspin.server.group import SendspinGroup

_LOGGER = logging.getLogger(__name__)

# Default initial delay before first audio plays (microseconds)
DEFAULT_INITIAL_DELAY_US = 250_000  # 250ms

# Minimum lead time (from now) for sending catch-up audio to late joiners.
# This must be lower than DEFAULT_INITIAL_DELAY_US, otherwise a steady-state low-latency
# stream may have no chunks whose *start* timestamp is >= now + DEFAULT_INITIAL_DELAY_US.
LATE_JOINER_MIN_LEAD_US = 100_000  # 100ms

# How long to keep encoding/caching a removed player's format. This enables quick
# group/ungroup/regroup to stay synced without having to schedule audio very close to "now".
PIPELINE_KEEPALIVE_US = 2_000_000  # 2s

# Default cache window for late joiner chunks (microseconds)
DEFAULT_CACHE_WINDOW_US = 10_000_000  # 10 seconds

# Amount of PCM history to include when building a first-time catch-up pipeline.
# Some resamplers/encoders may require a small warm-up window before emitting
# stable output packets. We encode this pre-roll but only *send* chunks that
# start far enough in the future (see LATE_JOINER_MIN_LEAD_US).
CATCHUP_PREROLL_US = 500_000  # 0.5s


@dataclass(frozen=True)
class CachedChunk:
    """Cached chunk for late joiner catch-up."""

    timestamp_us: int
    """Start timestamp for this chunk."""
    duration_us: int
    """Duration of this chunk in microseconds."""
    payload: bytes
    """Encoded audio payload bytes (without binary header)."""
    byte_count: int
    """Size of encoded audio data (without header)."""


@dataclass(frozen=True)
class _PcmCacheItem:
    start_timestamp_us: int
    duration_us: int
    pcm: bytes
    audio_format: AudioFormat


class StreamStoppedError(Exception):
    """Raised when trying to commit audio on a stopped stream."""


class PushStream:
    """
    Push-based audio streaming API.

    This class provides a push-based interface for streaming audio to players.
    Audio is prepared via prepare_audio(), then committed and sent via commit_audio().
    Backpressure is handled via wait_for_buffer_space() and timeline shifting.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        clock: Clock,
        group: SendspinGroup,
        channel_router: ChannelRouter,
    ) -> None:
        """
        Create a new PushStream.

        Args:
            loop: Event loop for timing and async operations.
            clock: Time source used for timestamping.
            group: Group this stream belongs to.
            channel_router: Router for channel assignments.
        """
        self._loop = loop
        self._clock = clock
        self._group = group
        self._channel_router = channel_router
        self._is_stopped = False
        # Pending audio per channel: channel_id -> (pcm_bytes, audio_format)
        self._channel_buffers: dict[UUID, tuple[bytes, AudioFormat]] = {}
        # Per-channel timing: channel_id -> next_chunk_start_us
        self._channel_timing: dict[UUID, int] = {}
        # Pipeline manager for encoding
        self._pipeline_manager = PipelineManager()
        # Late joiner cache: pipeline_key -> list of cached chunks
        self._chunk_cache: dict[PipelineKey, list[CachedChunk]] = {}
        # Raw PCM cache per channel for late-join re-encoding.
        self._pcm_cache: dict[UUID, list[_PcmCacheItem]] = {}
        self._last_catchup_pipeline_key: PipelineKey | None = None
        # Keepalive pipelines for recently removed players: (channel_id, target_format) -> expiry_us
        self._keepalive_pipelines: dict[tuple[UUID, AudioFormat], int] = {}

    @property
    def is_stopped(self) -> bool:
        """Whether this stream has been stopped."""
        return self._is_stopped

    def _get_group_players(self) -> list[SendspinClient]:
        """Get all connected player clients in this stream's group."""
        return [
            c
            for c in self._group.clients
            if c.is_connected and c.player_role is not None and c.check_role(Roles.PLAYER)
        ]

    def has_pending_audio(self) -> bool:
        """Return True if there is pending audio to commit."""
        return len(self._channel_buffers) > 0

    def get_pending_audio(self) -> dict[UUID, tuple[bytes, AudioFormat]]:
        """Return the pending audio buffers (for testing/inspection)."""
        return self._channel_buffers

    def prepare_audio(
        self,
        pcm: bytes,
        audio_format: AudioFormat,
        *,
        channel_id: UUID = MAIN_CHANNEL,
    ) -> None:
        """
        Prepare PCM audio for the next commit.

        This is a synchronous method that stores the PCM data for encoding
        during commit_audio(). Calling twice for the same channel replaces
        the previous data (does not append).

        Args:
            pcm: Raw PCM audio data.
            audio_format: Format of the PCM data.
            channel_id: Channel to prepare audio for (default: MAIN_CHANNEL).
        """
        self._channel_buffers[channel_id] = (pcm, audio_format)

    async def commit_audio(self) -> int:
        """
        Encode and send all prepared audio to players.

        This is an asynchronous method that:
        1. Encodes prepared PCM for each required format
        2. Applies backpressure (timeline shift if needed)
        3. Assigns timestamps to encoded chunks
        4. Sends chunks to connected players

        Returns:
            The earliest play_start_us timestamp across all channels.

        Raises:
            StreamStoppedError: If the stream has been stopped.
        """
        # Check if stopped
        if self._is_stopped:
            raise StreamStoppedError("Cannot commit audio on a stopped stream")

        # If no pending audio, return earliest channel timing (or initialize)
        if not self._channel_buffers:
            now_us = self._clock.now_us()
            if not self._channel_timing:
                # Initialize MAIN_CHANNEL timing if nothing exists
                self._channel_timing[MAIN_CHANNEL] = now_us + DEFAULT_INITIAL_DELAY_US
            return min(self._channel_timing.values())

        # Drain channel buffers
        prepared = dict(self._channel_buffers)
        self._channel_buffers.clear()

        # Calculate duration for each channel and warn on misalignment
        durations_us = self._calculate_channel_durations(prepared)
        self._warn_duration_misalignment(durations_us)

        # Initialize timing for new channels
        now_us = self._clock.now_us()
        for channel_id in prepared:
            if channel_id not in self._channel_timing:
                self._channel_timing[channel_id] = now_us + DEFAULT_INITIAL_DELAY_US

        # Calculate approximate byte count for backpressure (use total of all channels)
        total_bytes = sum(len(pcm) for pcm, _ in prepared.values())

        # Apply backpressure: query connected players for wait time
        max_wait_us = self._calculate_backpressure(total_bytes)
        if max_wait_us > 0:
            # Shift all channel timings equally for group synchronization
            for channel_id in self._channel_timing:
                self._channel_timing[channel_id] += max_wait_us

        # If audio production stalls (e.g., the upstream source blocks), the scheduled
        # play timeline can drift into the past. Rebase the timeline so new audio is
        # always scheduled with at least the default initial delay from "now".
        min_timing_us = min(self._channel_timing.values())
        target_min_us = now_us + DEFAULT_INITIAL_DELAY_US
        if min_timing_us < target_min_us:
            shift_us = target_min_us - min_timing_us
            for channel_id in self._channel_timing:
                self._channel_timing[channel_id] += shift_us

        # Capture play_start_us for each channel before advancing
        channel_play_start: dict[UUID, int] = {}
        for channel_id in prepared:
            channel_play_start[channel_id] = self._channel_timing[channel_id]

        # Advance each channel's timing by its duration
        for channel_id, duration_us in durations_us.items():
            self._channel_timing[channel_id] += duration_us

        # Determine required pipelines and encode.
        #
        # NOTE: Avoid running PyAV encoding in a background thread here. In practice,
        # PyAV/FFmpeg interactions can be sensitive to thread usage in some environments
        # and may hang under tests. Phase 3 can reintroduce parallelism in a targeted,
        # well-tested way (encode-only fan-out), rather than offloading the full pipeline.
        pipeline_keys = self._get_required_pipeline_keys(prepared)
        consumer_keys: set[PipelineKey] = set()
        for player in self._get_group_players():
            channel_id = self._channel_router.get_channel(player.client_id)
            if channel_id not in prepared:
                continue
            _, source_format = prepared[channel_id]
            target_format = self._get_player_target_format(player, source_format)
            consumer_keys.add(
                PipelineKey(
                    channel_id=channel_id,
                    source_format=source_format,
                    target_format=target_format,
                )
            )

        keepalive_keys: set[PipelineKey] = set()
        for channel_id, target_format in self._keepalive_pipelines:
            if channel_id not in prepared:
                continue
            _, source_format = prepared[channel_id]
            keepalive_keys.add(
                PipelineKey(
                    channel_id=channel_id,
                    source_format=source_format,
                    target_format=target_format,
                )
            )
        prepared_with_timestamps: dict[UUID, tuple[bytes, AudioFormat, int]] = {
            channel_id: (pcm, fmt, channel_play_start[channel_id])
            for channel_id, (pcm, fmt) in prepared.items()
        }
        encoded_results = self._pipeline_manager.process(prepared_with_timestamps, pipeline_keys)

        # Store raw PCM for possible late-join catch-up in unique formats.
        self._store_pcm_cache(
            prepared, channel_play_start=channel_play_start, durations_us=durations_us
        )

        # Send chunks to players using per-channel timestamps
        self._send_chunks_to_players(prepared, encoded_results)

        # Cache chunks for late joiners (once per pipeline, not per player).
        #
        # Cache for:
        # - Any pipeline actively consumed by a connected player
        # - Keepalive pipelines for recently removed players (so quick regroup can catch up)
        self._cache_encoded_results(encoded_results, consumer_keys | keepalive_keys)

        # Resync any dropped non-blocking players
        self._resync_dropped_players()

        # Prune old chunks from cache
        self._prune_chunk_cache()

        # Return earliest play_start_us
        return min(channel_play_start.values())

    def _calculate_backpressure(self, byte_count: int) -> int:
        """
        Calculate backpressure delay based on blocking player buffer capacity.

        Only blocking players contribute to backpressure. Non-blocking players
        are skipped and will be dropped/resynced if they fall behind.

        Args:
            byte_count: Approximate bytes being sent to players.

        Returns:
            Maximum wait time in microseconds across all blocking players.
        """
        max_wait_us = 0
        for player in self._get_group_players():
            # Skip non-blocking players - they don't affect group timing
            if not player.blocking:
                continue
            if player.buffer_tracker is not None:
                wait_us = player.buffer_tracker.time_until_capacity(byte_count)
                if wait_us > 0 and _LOGGER.isEnabledFor(logging.DEBUG):
                    _LOGGER.debug(
                        "Backpressure from %s: wait_us=%s buffered=%s capacity=%s",
                        player.client_id,
                        wait_us,
                        player.buffer_tracker.buffered_bytes,
                        player.buffer_tracker.capacity_bytes,
                    )
                max_wait_us = max(max_wait_us, wait_us)
        return max_wait_us

    def _calculate_channel_durations(
        self,
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
    ) -> dict[UUID, int]:
        """Calculate duration in microseconds for each prepared channel."""
        durations: dict[UUID, int] = {}
        for channel_id, (pcm, fmt) in prepared.items():
            bytes_per_sample = fmt.bit_depth // 8
            frame_stride = bytes_per_sample * fmt.channels
            sample_count = len(pcm) // frame_stride
            duration_us = int(sample_count * 1_000_000 / fmt.sample_rate)
            durations[channel_id] = duration_us
        return durations

    def _prune_pcm_cache(self) -> None:
        """Drop PCM cache blocks that are fully in the past."""
        now_us = self._clock.now_us()
        for channel_id in list(self._pcm_cache.keys()):
            self._pcm_cache[channel_id] = [
                item
                for item in self._pcm_cache[channel_id]
                if item.start_timestamp_us + item.duration_us >= now_us
            ]
            if not self._pcm_cache[channel_id]:
                del self._pcm_cache[channel_id]

    def _store_pcm_cache(
        self,
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
        *,
        channel_play_start: dict[UUID, int],
        durations_us: dict[UUID, int],
    ) -> None:
        """Store committed PCM for potential late-join re-encoding."""
        now_us = self._clock.now_us()
        max_ts_us = now_us + DEFAULT_CACHE_WINDOW_US
        for channel_id, (pcm, fmt) in prepared.items():
            start_us = channel_play_start[channel_id]
            if start_us > max_ts_us:
                continue
            self._pcm_cache.setdefault(channel_id, []).append(
                _PcmCacheItem(
                    start_timestamp_us=start_us,
                    duration_us=durations_us[channel_id],
                    pcm=pcm,
                    audio_format=fmt,
                )
            )
        self._prune_pcm_cache()

    def _warn_duration_misalignment(self, durations_us: dict[UUID, int]) -> None:
        """Log a warning if channel durations differ significantly."""
        if len(durations_us) <= 1:
            return

        values = list(durations_us.values())
        min_dur = min(values)
        max_dur = max(values)

        # Warn if durations differ by more than 5ms
        tolerance_us = 5000
        if max_dur - min_dur > tolerance_us:
            _LOGGER.warning(
                "Channel durations differ by %dus (tolerance: %dus)",
                max_dur - min_dur,
                tolerance_us,
            )

    def _get_player_target_format(
        self,
        player: SendspinClient,
        source_format: AudioFormat,
    ) -> AudioFormat:
        """
        Get the target format for a player.

        Uses player's preferred_format if set, otherwise uses source format
        (no encoding - direct PCM passthrough).
        """
        if player.preferred_format is not None:
            return player.preferred_format
        # Default to source format (no resampling/encoding needed)
        return source_format

    def _get_required_pipeline_keys(
        self,
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
    ) -> set[PipelineKey]:
        """
        Determine which pipelines are needed for connected players.

        For each connected player, determines their channel and target format,
        then ensures a pipeline exists for that combination.

        Returns:
            Set of pipeline keys needed for this commit.
        """
        pipeline_keys: set[PipelineKey] = set()

        for player in self._get_group_players():
            # Get player's assigned channel
            channel_id = self._channel_router.get_channel(player.client_id)

            # Skip if we don't have audio for this channel
            if channel_id not in prepared:
                continue

            _, source_format = prepared[channel_id]
            target_format = self._get_player_target_format(player, source_format)

            # Add or get pipeline
            key = self._pipeline_manager.add_pipeline(
                channel_id=channel_id,
                source_format=source_format,
                target_format=target_format,
            )
            pipeline_keys.add(key)

        # Keep recently-removed player formats warm for a short window so quick regrouping can
        # catch up with continuous cached audio, rather than starting late and drifting.
        now_us = self._clock.now_us()
        for (channel_id, target_format), expiry_us in list(self._keepalive_pipelines.items()):
            if expiry_us <= now_us:
                del self._keepalive_pipelines[(channel_id, target_format)]
                continue
            if channel_id not in prepared:
                continue
            _, source_format = prepared[channel_id]
            key = self._pipeline_manager.add_pipeline(
                channel_id=channel_id,
                source_format=source_format,
                target_format=target_format,
            )
            pipeline_keys.add(key)

        return pipeline_keys

    def _send_chunks_to_players(
        self,
        prepared: dict[UUID, tuple[bytes, AudioFormat]],
        encoded_results: dict[PipelineKey, list[EncodedChunk]],
    ) -> None:
        """
        Send encoded chunks to connected players via their PlayerRole.

        For each connected player:
        1. Determine their channel and target format
        2. Find the correct pipeline
        3. Send each chunk via PlayerRole (handles stream/start, packing, buffer tracking)
        4. Cache chunks for late joiners

        Args:
            prepared: Prepared audio per channel.
            encoded_results: Encoded chunks per pipeline.
        """
        for player in self._get_group_players():
            # Skip if no PlayerRole (shouldn't happen for connected players)
            if player.player_role is None:
                continue

            # Get player's assigned channel
            channel_id = self._channel_router.get_channel(player.client_id)

            # Skip if we don't have audio for this channel
            if channel_id not in prepared:
                continue

            _, source_format = prepared[channel_id]
            target_format = self._get_player_target_format(player, source_format)

            # Find the pipeline key for this player
            key = PipelineKey(
                channel_id=channel_id,
                source_format=source_format,
                target_format=target_format,
            )

            # Skip if no encoded chunks for this pipeline
            if key not in encoded_results:
                continue

            chunks = encoded_results[key]
            if not chunks:
                continue

            # Get codec header for this pipeline
            codec_header_b64 = self._pipeline_manager.get_codec_header_b64(key)

            # Send each chunk via PlayerRole (handles stream/start automatically)
            for chunk in chunks:
                # Send via PlayerRole - handles stream/start, packing, buffer tracking
                player.player_role.send_audio(
                    chunk=chunk,
                    timestamp_us=chunk.timestamp_us,
                    audio_format=target_format,
                    codec_header_b64=codec_header_b64,
                )

    def _cache_encoded_results(
        self,
        encoded_results: dict[PipelineKey, list[EncodedChunk]],
        cache_keys: set[PipelineKey],
    ) -> None:
        """Cache encoded chunks for late joiner catch-up."""
        for key in cache_keys:
            chunks = encoded_results.get(key)
            if not chunks:
                continue
            self._chunk_cache.setdefault(key, []).extend(
                CachedChunk(
                    timestamp_us=chunk.timestamp_us,
                    duration_us=chunk.duration_us,
                    payload=chunk.data,
                    byte_count=chunk.byte_count,
                )
                for chunk in chunks
            )

    def _resync_dropped_players(self) -> None:
        """
        Resync any non-blocking players that were dropped during send.

        For each player needing resync:
        1. Call resync() which sends stream/clear and resets state
        2. Player will receive fresh stream/start on next commit
        """
        for player in self._get_group_players():
            if player.player_role is None:
                continue

            send_state = player.player_role.get_send_state()
            if not send_state.needs_resync:
                continue

            # Avoid sending control messages while the connection is still congested.
            # stream/clear is a JSON/control message and must not be dropped; if the
            # queue is full, SendspinClient will disconnect. Instead, wait until the
            # queue drains below the high-water mark.
            if player.connection is None or player.connection.queue_high_water(threshold=0.5):
                if _LOGGER.isEnabledFor(logging.DEBUG):
                    qsize, qmax = player.queue_status()
                    _LOGGER.debug(
                        "Resync deferred for %s: queue=%s/%s dropped_commits=%s",
                        player.client_id,
                        qsize,
                        qmax,
                        send_state.dropped_commits,
                    )
                continue

            # Perform resync (sends stream/clear, resets state)
            # Next audio send will automatically include stream/start
            if _LOGGER.isEnabledFor(logging.INFO):
                _LOGGER.info(
                    "Resyncing player %s after dropped audio: dropped_commits=%s",
                    player.client_id,
                    send_state.dropped_commits,
                )
            player.player_role.resync()

    async def wait_for_buffer_space(self) -> None:
        """
        Wait until there is buffer space available on blocking players.

        This is useful for throttling audio production to match
        player consumption rates. Uses an estimated chunk size
        to determine buffer capacity needs. Non-blocking players
        are skipped (they don't affect group timing).
        """
        # Estimate chunk size: 25ms of 48kHz stereo 16-bit PCM = 4800 bytes
        # This is a reasonable default for typical audio streaming
        estimated_chunk_bytes = 4800

        max_wait_us = 0
        for player in self._get_group_players():
            # Skip non-blocking players
            if not player.blocking:
                continue
            if player.buffer_tracker is not None:
                wait_us = player.buffer_tracker.time_until_capacity(estimated_chunk_bytes)
                max_wait_us = max(max_wait_us, wait_us)

        if max_wait_us > 0:
            # Convert microseconds to seconds for asyncio.sleep
            await asyncio.sleep(max_wait_us / 1_000_000)

    def has_cached_chunks(self) -> bool:
        """Return True if there are cached chunks for late joiners."""
        return any(len(chunks) > 0 for chunks in self._chunk_cache.values())

    def _get_player_by_id(self, player_id: str) -> SendspinClient | None:
        for client in self._group.clients:
            if client.client_id == player_id:
                return client
        return None

    def get_catchup_chunks(self, player_id: str) -> list[CachedChunk]:
        """
        Get cached chunks for a player's channel and format.

        Returns only chunks with timestamps >= now (future playback).

        Args:
            player_id: Player ID to get chunks for.

        Returns:
            List of cached chunks for the player's channel, sorted by timestamp.
        """
        # Get player's channel
        channel_id = self._channel_router.get_channel(player_id)

        player = self._get_player_by_id(player_id)
        if player is None:
            return []

        # Get current time and enforce a minimum lead time for late joiners.
        #
        # Late joiners need enough time to receive stream/start, buffer audio, and
        # schedule playback accurately. Sending chunks with timestamps too close to
        # "now" can lead to dropped chunks and audible gaps.
        now_us = self._clock.now_us()
        min_timestamp_us = now_us + LATE_JOINER_MIN_LEAD_US

        # Find matching pipeline keys for this channel
        result: list[CachedChunk] = []
        for key, chunks in self._chunk_cache.items():
            if key.channel_id != channel_id:
                continue

            # Check if format matches player's preference
            if player.preferred_format is not None and key.target_format != player.preferred_format:
                continue

            # Filter to chunks far enough in the future for reliable scheduling
            result.extend(chunk for chunk in chunks if chunk.timestamp_us >= min_timestamp_us)

        # Sort by timestamp
        result.sort(key=lambda c: c.timestamp_us)
        return result

    def _prune_chunk_cache(self) -> None:
        """Remove old chunks from the cache."""
        now_us = self._clock.now_us()

        for key in list(self._chunk_cache.keys()):
            # Filter out chunks older than now
            self._chunk_cache[key] = [
                chunk for chunk in self._chunk_cache[key] if chunk.timestamp_us >= now_us
            ]
            # Remove empty lists
            if not self._chunk_cache[key]:
                del self._chunk_cache[key]

        # Drop expired keepalive entries as well.
        for keepalive_key, expiry_us in list(self._keepalive_pipelines.items()):
            if expiry_us <= now_us:
                del self._keepalive_pipelines[keepalive_key]

        self._prune_pcm_cache()

    def on_player_leave(self, player_id: str, *, target_format: AudioFormat | None = None) -> None:
        """Keep the player's format pipeline warm briefly for seamless quick rejoin."""
        if target_format is None:
            player = self._get_player_by_id(player_id)
            if player is None:
                return
            target_format = player.preferred_format
        if target_format is None:
            return
        channel_id = self._channel_router.get_channel(player_id)
        self._keepalive_pipelines[(channel_id, target_format)] = (
            self._clock.now_us() + PIPELINE_KEEPALIVE_US
        )

    def on_player_join(self, player_id: str) -> None:
        """
        Handle a player joining (late joiner catch-up).

        Sends stream/start and cached chunks to the player via PlayerRole.

        Args:
            player_id: Player ID that joined.
        """
        player = self._get_player_by_id(player_id)
        if player is None or not player.is_connected or player.player_role is None:
            return

        # Get cached chunks for this player
        cached_chunks = self.get_catchup_chunks(player_id)
        if not cached_chunks:
            cached_chunks = self._build_encoded_catchup_from_pcm_cache(player_id)
            catchup_key = self._last_catchup_pipeline_key
            if cached_chunks and catchup_key is not None:
                self._chunk_cache.setdefault(catchup_key, []).extend(cached_chunks)

        if not cached_chunks:
            return

        # Get player's channel and format for stream/start
        channel_id = self._channel_router.get_channel(player_id)

        # Find a matching pipeline key for stream/start
        target_format = player.preferred_format
        pipeline_key: PipelineKey | None = None
        for key in self._chunk_cache:
            if key.channel_id == channel_id and (
                target_format is None or key.target_format == target_format
            ):
                pipeline_key = key
                target_format = key.target_format
                break

        if pipeline_key is None or target_format is None:
            return

        # Get codec header for this pipeline
        codec_header_b64 = self._pipeline_manager.get_codec_header_b64(pipeline_key)

        if _LOGGER.isEnabledFor(logging.DEBUG):
            first_ts = cached_chunks[0].timestamp_us
            last_ts = cached_chunks[-1].timestamp_us
            _LOGGER.debug(
                "Late join catch-up for %s: chunks=%s ts_range=%s..%s format=%s",
                player_id,
                len(cached_chunks),
                first_ts,
                last_ts,
                target_format,
            )

        # Send stream/start via PlayerRole
        player.player_role.send_stream_start(target_format, codec_header_b64)

        # Send cached chunks via PlayerRole
        for chunk in cached_chunks:
            player.player_role.send_cached_chunk(
                payload=chunk.payload,
                timestamp_us=chunk.timestamp_us,
                duration_us=chunk.duration_us,
                byte_count=chunk.byte_count,
            )

    def _build_encoded_catchup_from_pcm_cache(self, player_id: str) -> list[CachedChunk]:
        """Encode catch-up chunks from cached PCM when no encoded cache exists yet."""
        self._last_catchup_pipeline_key = None
        player = self._get_player_by_id(player_id)
        if player is None:
            return []
        channel_id = self._channel_router.get_channel(player_id)
        target_format = player.preferred_format
        if target_format is None:
            return []

        now_us = self._clock.now_us()
        send_min_ts_us = now_us + LATE_JOINER_MIN_LEAD_US
        max_ts_us = now_us + DEFAULT_CACHE_WINDOW_US

        prime_min_ts_us = send_min_ts_us - CATCHUP_PREROLL_US
        pcm_items = [
            item
            for item in self._pcm_cache.get(channel_id, [])
            if prime_min_ts_us <= item.start_timestamp_us <= max_ts_us
        ]
        if not pcm_items:
            return []

        pcm_items.sort(key=lambda i: i.start_timestamp_us)
        out: list[CachedChunk] = []
        for item in pcm_items:
            key = self._pipeline_manager.add_pipeline(
                channel_id=channel_id,
                source_format=item.audio_format,
                target_format=target_format,
            )
            prepared_with_timestamps: dict[UUID, tuple[bytes, AudioFormat, int]] = {
                channel_id: (item.pcm, item.audio_format, item.start_timestamp_us)
            }
            encoded = self._pipeline_manager.process(prepared_with_timestamps, {key})
            out.extend(
                CachedChunk(
                    timestamp_us=chunk.timestamp_us,
                    duration_us=chunk.duration_us,
                    payload=chunk.data,
                    byte_count=chunk.byte_count,
                )
                for chunk in encoded.get(key, [])
            )
            self._last_catchup_pipeline_key = key

        out.sort(key=lambda c: c.timestamp_us)
        return [chunk for chunk in out if chunk.timestamp_us >= send_min_ts_us]

    def on_format_request(self, player_id: str, new_format: AudioFormat) -> bool:
        """
        Handle a format change request from a player.

        Validates the format against the player's supported formats,
        updates the preferred format, and notifies PlayerRole to send
        new stream/start on next audio.

        Args:
            player_id: Player ID requesting the format change.
            new_format: The requested audio format.

        Returns:
            True if format change was accepted, False if invalid or player not found.
        """
        player = self._get_player_by_id(player_id)
        if player is None or not player.is_connected:
            return False

        # Get supported formats from client info
        player_support = player.info.player_support
        if player_support is None:
            return False
        supported = player_support.supported_formats

        # Validate format against supported formats
        is_supported = any(
            fmt.codec == new_format.codec
            and fmt.sample_rate == new_format.sample_rate
            and fmt.channels == new_format.channels
            and fmt.bit_depth == new_format.bit_depth
            for fmt in supported
        )

        if not is_supported:
            return False

        # Update preferred format
        player.preferred_format = new_format

        # Notify PlayerRole of format change (triggers new stream/start on next audio)
        if player.player_role is not None:
            player.player_role.on_format_change(new_format)

        return True

    def stop(self) -> None:
        """
        Stop the stream.

        After calling stop(), commit_audio() will raise StreamStoppedError.
        Sends stream/end message to connected players via their PlayerRole.
        """
        self._is_stopped = True

        # Send stream/end via PlayerRole (handles buffer tracker reset)
        for player in self._get_group_players():
            if player.player_role is not None:
                player.player_role.end_stream()

    def clear(self) -> None:
        """
        Clear all pending audio and reset timing.

        This is used for seek operations where buffered audio is discarded.
        Sends stream/clear to connected players via their PlayerRole.
        """
        # Clear pending audio
        self._channel_buffers.clear()

        # Reset per-channel timing
        self._channel_timing.clear()

        # Clear chunk cache
        self._chunk_cache.clear()

        # Reset encoding pipelines to drop any buffered resampler/encoder state.
        self._pipeline_manager.reset()

        # Send stream/clear via PlayerRole (handles buffer tracker reset)
        for player in self._get_group_players():
            if player.player_role is not None:
                player.player_role.clear_stream()
