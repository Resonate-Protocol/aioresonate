"""Push-based audio streaming API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from aiosendspin.models import BinaryMessageType, pack_binary_header_raw
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.pipeline import EncodedChunk, PipelineKey, PipelineManager

if TYPE_CHECKING:
    from aiosendspin.server.channels import ChannelRouter
    from aiosendspin.server.player_state import PlayerRecord, PlayerRegistry
    from aiosendspin.server.stream import AudioFormat

# Default initial delay before first audio plays (microseconds)
DEFAULT_INITIAL_DELAY_US = 250_000  # 250ms

# Default cache window for late joiner chunks (microseconds)
DEFAULT_CACHE_WINDOW_US = 10_000_000  # 10 seconds


@dataclass(frozen=True)
class CachedChunk:
    """Cached chunk for late joiner catch-up."""

    timestamp_us: int
    """Start timestamp for this chunk."""
    data: bytes
    """Encoded audio data (with binary header)."""
    byte_count: int
    """Size of encoded audio data (without header)."""


class StreamStoppedError(Exception):
    """Raised when trying to commit audio on a stopped stream."""


class DurationMismatchError(Exception):
    """Raised when prepared channels have mismatched durations."""


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
        group_id: str,
        player_registry: PlayerRegistry,
        channel_router: ChannelRouter,
    ) -> None:
        """
        Create a new PushStream.

        Args:
            loop: Event loop for timing and async operations.
            group_id: ID of the group this stream belongs to.
            player_registry: Registry for player state management.
            channel_router: Router for channel assignments.
        """
        self._loop = loop
        self._group_id = group_id
        self._player_registry = player_registry
        self._channel_router = channel_router
        self._is_stopped = False
        # Pending audio per channel: channel_id -> (pcm_bytes, audio_format)
        self._channel_buffers: dict[UUID, tuple[bytes, AudioFormat]] = {}
        # Timing state
        self._next_chunk_start_us: int | None = None  # Initialized on first commit
        # Pipeline manager for encoding
        self._pipeline_manager = PipelineManager()
        # Late joiner cache: pipeline_key -> list of cached chunks
        self._chunk_cache: dict[PipelineKey, list[CachedChunk]] = {}

    @property
    def is_stopped(self) -> bool:
        """Whether this stream has been stopped."""
        return self._is_stopped

    def _get_group_players(self) -> list[PlayerRecord]:
        """Get all connected players in this stream's group."""
        return [p for p in self._player_registry.get_in_group(self._group_id) if p.is_connected]

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
            The play_start_us timestamp for this commit.

        Raises:
            StreamStoppedError: If the stream has been stopped.
            DurationMismatchError: If prepared channels have mismatched durations.
        """
        # Check if stopped
        if self._is_stopped:
            raise StreamStoppedError("Cannot commit audio on a stopped stream")

        # If no pending audio, return current timing
        if not self._channel_buffers:
            if self._next_chunk_start_us is None:
                # Initialize timing even with no audio
                now_us = int(self._loop.time() * 1_000_000)
                self._next_chunk_start_us = now_us + DEFAULT_INITIAL_DELAY_US
            return self._next_chunk_start_us

        # Drain channel buffers and validate duration alignment
        prepared = dict(self._channel_buffers)
        self._channel_buffers.clear()

        # Calculate duration for each channel and validate alignment
        durations_us = self._calculate_channel_durations(prepared)
        self._validate_duration_alignment(durations_us)

        # Initialize timing on first commit
        if self._next_chunk_start_us is None:
            now_us = int(self._loop.time() * 1_000_000)
            self._next_chunk_start_us = now_us + DEFAULT_INITIAL_DELAY_US

        # Calculate approximate byte count for backpressure (use total of all channels)
        total_bytes = sum(len(pcm) for pcm, _ in prepared.values())

        # Apply backpressure: query connected players for wait time
        max_wait_us = self._calculate_backpressure(total_bytes)
        if max_wait_us > 0:
            self._next_chunk_start_us += max_wait_us

        # Get the play_start_us for this commit
        play_start_us = self._next_chunk_start_us

        # Advance timing by the audio duration (use first channel's duration)
        if durations_us:
            duration_us = next(iter(durations_us.values()))
            self._next_chunk_start_us += duration_us

        # Determine required pipelines and encode
        pipeline_keys = self._get_required_pipeline_keys(prepared)
        encoded_results = self._pipeline_manager.process(prepared, pipeline_keys)

        # Send chunks to players
        self._send_chunks_to_players(play_start_us, prepared, encoded_results)

        # Resync any dropped non-blocking players
        self._resync_dropped_players()

        # Prune old chunks from cache
        self._prune_chunk_cache()

        return play_start_us

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
            if hasattr(player, "buffer_tracker") and player.buffer_tracker:
                wait_us = player.buffer_tracker.time_until_capacity(byte_count)
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

    def _validate_duration_alignment(self, durations_us: dict[UUID, int]) -> None:
        """Validate that all channels have approximately the same duration."""
        if len(durations_us) <= 1:
            return

        values = list(durations_us.values())
        min_dur = min(values)
        max_dur = max(values)

        # Allow up to 5ms (5000 us) difference for rounding
        tolerance_us = 5000
        if max_dur - min_dur > tolerance_us:
            raise DurationMismatchError(
                f"Channel durations differ by {max_dur - min_dur}us (max allowed: {tolerance_us}us)"
            )

    def _get_player_target_format(
        self,
        player: PlayerRecord,
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

        return pipeline_keys

    def _send_chunks_to_players(
        self,
        play_start_us: int,
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
            chunk_start_us = play_start_us
            for chunk in chunks:
                # Send via PlayerRole - handles stream/start, packing, buffer tracking
                player.player_role.send_audio(
                    chunk=chunk,
                    timestamp_us=chunk_start_us,
                    audio_format=target_format,
                    codec_header_b64=codec_header_b64,
                )

                # Cache chunk for late joiners (need to pack data for cache)
                header = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, chunk_start_us)
                packed_data = header + chunk.data
                cached = CachedChunk(
                    timestamp_us=chunk_start_us,
                    data=packed_data,
                    byte_count=chunk.byte_count,
                )
                if key not in self._chunk_cache:
                    self._chunk_cache[key] = []
                self._chunk_cache[key].append(cached)

                # Advance to next chunk
                chunk_start_us = chunk_start_us + chunk.duration_us

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

            # Perform resync (sends stream/clear, resets state)
            # Next audio send will automatically include stream/start
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

        # Get player record for format selection
        player = self._player_registry.get(player_id)
        if player is None:
            return []

        # Get current time
        now_us = int(self._loop.time() * 1_000_000)

        # Find matching pipeline keys for this channel
        result: list[CachedChunk] = []
        for key, chunks in self._chunk_cache.items():
            if key.channel_id != channel_id:
                continue

            # Check if format matches player's preference
            if player.preferred_format is not None and key.target_format != player.preferred_format:
                continue

            # Filter to future chunks only
            result.extend(chunk for chunk in chunks if chunk.timestamp_us >= now_us)

        # Sort by timestamp
        result.sort(key=lambda c: c.timestamp_us)
        return result

    def _prune_chunk_cache(self) -> None:
        """Remove old chunks from the cache."""
        now_us = int(self._loop.time() * 1_000_000)

        for key in list(self._chunk_cache.keys()):
            # Filter out chunks older than now
            self._chunk_cache[key] = [
                chunk for chunk in self._chunk_cache[key] if chunk.timestamp_us >= now_us
            ]
            # Remove empty lists
            if not self._chunk_cache[key]:
                del self._chunk_cache[key]

    def on_player_join(self, player_id: str) -> None:
        """
        Handle a player joining (late joiner catch-up).

        Sends stream/start and cached chunks to the player via PlayerRole.

        Args:
            player_id: Player ID that joined.
        """
        player = self._player_registry.get(player_id)
        if player is None or player.player_role is None:
            return

        # Get cached chunks for this player
        cached_chunks = self.get_catchup_chunks(player_id)
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

        # Send stream/start via PlayerRole
        player.player_role.send_stream_start(target_format, codec_header_b64)

        # Send cached chunks via PlayerRole
        for chunk in cached_chunks:
            player.player_role.send_cached_chunk(
                packed_data=chunk.data,
                timestamp_us=chunk.timestamp_us,
                byte_count=chunk.byte_count,
            )

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
        player = self._player_registry.get(player_id)
        if player is None or player.connection is None:
            return False

        # Get supported formats from client info
        player_support = player.connection.info.player_support
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

        # Reset timing
        self._next_chunk_start_us = None

        # Clear chunk cache
        self._chunk_cache.clear()

        # Send stream/clear via PlayerRole (handles buffer tracker reset)
        for player in self._get_group_players():
            if player.player_role is not None:
                player.player_role.clear_stream()
