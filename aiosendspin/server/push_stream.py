"""Push-based audio streaming API."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import UUID

from aiosendspin.models import BinaryMessageType, pack_binary_header_raw
from aiosendspin.models.core import (
    StreamClearMessage,
    StreamClearPayload,
    StreamEndMessage,
    StreamEndPayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.player import StreamStartPlayer
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.pipeline import EncodedChunk, PipelineKey, PipelineManager

if TYPE_CHECKING:
    from aiosendspin.server.channels import ChannelRouter
    from aiosendspin.server.player_state import PlayerRecord, PlayerRegistry
    from aiosendspin.server.stream import AudioFormat

# Default initial delay before first audio plays (microseconds)
DEFAULT_INITIAL_DELAY_US = 100_000  # 100ms


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
        player_registry: PlayerRegistry,
        channel_router: ChannelRouter,
    ) -> None:
        """
        Create a new PushStream.

        Args:
            loop: Event loop for timing and async operations.
            player_registry: Registry for player state management.
            channel_router: Router for channel assignments.
        """
        self._loop = loop
        self._player_registry = player_registry
        self._channel_router = channel_router
        self._is_stopped = False
        # Pending audio per channel: channel_id -> (pcm_bytes, audio_format)
        self._channel_buffers: dict[UUID, tuple[bytes, AudioFormat]] = {}
        # Timing state
        self._next_chunk_start_us: int | None = None  # Initialized on first commit
        # Pipeline manager for encoding
        self._pipeline_manager = PipelineManager()
        # Track players that have received stream/start (by client_id)
        self._player_started: set[str] = set()

    @property
    def is_stopped(self) -> bool:
        """Whether this stream has been stopped."""
        return self._is_stopped

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

        return play_start_us

    def _calculate_backpressure(self, byte_count: int) -> int:
        """
        Calculate backpressure delay based on player buffer capacity.

        Args:
            byte_count: Approximate bytes being sent to players.

        Returns:
            Maximum wait time in microseconds across all connected players.
        """
        max_wait_us = 0
        for player in self._player_registry.get_connected():
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

        for player in self._player_registry.get_connected():
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
        Send encoded chunks to connected players.

        For each connected player:
        1. Determine their channel and target format
        2. Find the correct pipeline
        3. Send each chunk with appropriate timestamp
        4. Register chunks with buffer tracker
        """
        for player in self._player_registry.get_connected():
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

            # Send stream/start if this is first audio for this player
            if player.client_id not in self._player_started:
                self._send_stream_start(player, target_format, key)
                self._player_started.add(player.client_id)

            # Send each chunk with contiguous timestamps
            chunk_start_us = play_start_us
            for chunk in chunks:
                chunk_end_us = chunk_start_us + chunk.duration_us

                # Pack binary header and send
                header = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, chunk_start_us)
                if player.connection is not None:
                    player.connection.send_message(header + chunk.data)

                # Register with buffer tracker
                if player.buffer_tracker is not None:
                    player.buffer_tracker.register(chunk_end_us, chunk.byte_count)

                # Advance to next chunk
                chunk_start_us = chunk_end_us

    def _send_stream_start(
        self,
        player: PlayerRecord,
        target_format: AudioFormat,
        pipeline_key: PipelineKey,
    ) -> None:
        """
        Send stream/start message to a player.

        Args:
            player: Player to send to.
            target_format: Audio format for this player.
            pipeline_key: Pipeline key (for codec header lookup).
        """
        if player.connection is None:
            return

        # Get codec header if applicable (e.g., FLAC)
        codec_header_b64 = self._pipeline_manager.get_codec_header_b64(pipeline_key)

        # Create stream/start message
        stream_start = StreamStartMessage(
            payload=StreamStartPayload(
                player=StreamStartPlayer(
                    codec=target_format.codec,
                    sample_rate=target_format.sample_rate,
                    channels=target_format.channels,
                    bit_depth=target_format.bit_depth,
                    codec_header=codec_header_b64,
                )
            )
        )

        player.connection.send_message(stream_start)

    async def wait_for_buffer_space(self) -> None:
        """
        Wait until there is buffer space available on players.

        This is useful for throttling audio production to match
        player consumption rates. Uses an estimated chunk size
        to determine buffer capacity needs.
        """
        # Estimate chunk size: 25ms of 48kHz stereo 16-bit PCM = 4800 bytes
        # This is a reasonable default for typical audio streaming
        estimated_chunk_bytes = 4800

        max_wait_us = 0
        for player in self._player_registry.get_connected():
            if player.buffer_tracker is not None:
                wait_us = player.buffer_tracker.time_until_capacity(estimated_chunk_bytes)
                max_wait_us = max(max_wait_us, wait_us)

        if max_wait_us > 0:
            # Convert microseconds to seconds for asyncio.sleep
            await asyncio.sleep(max_wait_us / 1_000_000)

    def stop(self) -> None:
        """
        Stop the stream.

        After calling stop(), commit_audio() will raise StreamStoppedError.
        Sends stream/end message to connected players.
        """
        self._is_stopped = True

        # Send stream/end to connected players
        stream_end = StreamEndMessage(payload=StreamEndPayload())
        for player in self._player_registry.get_connected():
            if player.connection is not None:
                player.connection.send_message(stream_end)

    def clear(self) -> None:
        """
        Clear all pending audio and reset timing.

        This is used for seek operations where buffered audio is discarded.
        Sends stream/clear to connected players and resets their buffer trackers.
        """
        # Clear pending audio
        self._channel_buffers.clear()

        # Reset timing
        self._next_chunk_start_us = None

        # Reset player_started set (stream/start will be re-sent)
        self._player_started.clear()

        # Send stream/clear and reset buffer trackers for connected players
        stream_clear = StreamClearMessage(payload=StreamClearPayload())
        for player in self._player_registry.get_connected():
            if player.connection is not None:
                player.connection.send_message(stream_clear)
            if player.buffer_tracker is not None:
                player.buffer_tracker.reset()
