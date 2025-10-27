"""Audio playback for the Resonate CLI."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import sounddevice
from sounddevice import CallbackFlags

from aioresonate.client import PCMFormat

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _QueuedChunk:
    """Represents a queued audio chunk with timing information."""

    server_timestamp_us: int
    """Server timestamp when this chunk should start playing."""
    audio_data: bytes
    """Raw PCM audio bytes."""


class AudioPlayer:
    """
    Audio player for the Resonate CLI with time synchronization support.

    This player accepts audio chunks with server timestamps and uses time
    synchronization functions to schedule playback at the exact loop time when
    it should occur. This ensures multiple clients start playback in sync.

    Attributes:
        _loop: The asyncio event loop used for scheduling.
        _compute_play_time: Function that converts server timestamps to loop time,
            accounting for clock drift and offset.
        _compute_server_time: Inverse function that converts loop time to server
            timestamps, used to calculate synchronized startup targets.
    """

    _loop: asyncio.AbstractEventLoop
    _compute_play_time: Callable[[int], int]
    _compute_server_time: Callable[[int], int]

    _MIN_CHUNKS_TO_START: Final[int] = 16
    """Minimum chunks buffered before starting playback to absorb network jitter."""
    _MIN_CHUNKS_TO_MAINTAIN: Final[int] = 8
    """Minimum chunks to maintain during playback to avoid underruns."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        compute_play_time: Callable[[int], int],
        compute_server_time: Callable[[int], int],
    ) -> None:
        """
        Initialize the audio player.

        Args:
            loop: The asyncio event loop to use for scheduling.
            compute_play_time: Function that converts server timestamps to loop time.
                Accounts for clock drift and offset.
            compute_server_time: Inverse function that converts loop time to server
                timestamps, used to calculate synchronized startup targets.
        """
        self._loop = loop
        self._compute_play_time = compute_play_time
        self._compute_server_time = compute_server_time
        self._format: PCMFormat | None = None
        self._queue: asyncio.Queue[_QueuedChunk] = asyncio.Queue()
        self._stream: sounddevice.RawOutputStream | None = None
        self._closed = False
        self._stream_started = False
        self._first_real_chunk = True  # Flag to initialize timing from first chunk

        # Partial chunk tracking (to avoid discarding partial chunks)
        self._current_chunk: _QueuedChunk | None = None
        self._current_chunk_offset = 0

        # Track expected next chunk timestamp for intelligent gap/overlap handling
        self._expected_next_timestamp: int | None = None

        # Underrun tracking
        self._underrun_count = 0
        self._last_buffer_warning_us = 0

        # Track queued audio duration instead of just item count
        self._queued_duration_us = 0

        # Phase 1: Timestamp-aware playback
        self._current_server_timestamp_us: int | None = None

        # Synchronized startup: target loop time when first chunk should play
        self._target_play_time_us: int | None = None
        self._startup_task: asyncio.Task[None] | None = None

    def set_format(self, pcm_format: PCMFormat) -> None:
        """Configure the audio output format."""
        self._format = pcm_format
        self._close_stream()

        # Reset state on format change
        self._stream_started = False
        self._first_real_chunk = True

        dtype = "int16"
        # Use callback-based output stream with larger blocksize to reduce callback frequency
        # Larger blocksize = fewer callbacks requesting data = more time to buffer chunks
        # Default blocksize is device-dependent; use a larger fixed size for stability
        blocksize = 4096  # ~92ms at 44.1kHz - gives plenty of time for network chunks to arrive
        self._stream = sounddevice.RawOutputStream(
            samplerate=pcm_format.sample_rate,
            channels=pcm_format.channels,
            dtype=dtype,
            blocksize=blocksize,
            callback=self._audio_callback,
            latency="high",  # Use high latency to give more buffering time
        )

    async def stop(self) -> None:
        """Stop playback and release resources."""
        self._closed = True
        self._close_stream()

    def clear(self) -> None:
        """Drop all queued audio chunks."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - race condition guard
                break
        # Reset playback state
        self._first_real_chunk = True
        self._current_chunk = None
        self._current_chunk_offset = 0
        self._expected_next_timestamp = None
        self._underrun_count = 0
        self._queued_duration_us = 0
        self._current_server_timestamp_us = None
        self._target_play_time_us = None
        # Cancel startup task if pending
        if self._startup_task is not None and not self._startup_task.done():
            self._startup_task.cancel()

    def _audio_callback(
        self,
        outdata: memoryview,
        frames: int,
        time: sounddevice.CallbackTimeInfo,  # noqa: ARG002
        status: CallbackFlags,
    ) -> None:
        """
        Audio callback invoked by sounddevice when output buffer needs filling.

        Implements Phase 1 of time-synchronization: timestamp-aware playback where
        chunks are consumed in order and the current server timestamp is advanced as
        chunks are played. This ensures all clients playing the same stream stay in sync.

        Args:
            outdata: Output buffer to fill with audio data.
            frames: Number of frames requested.
            time: Timing information (unused - Phase 1 uses server timestamps instead).
            status: Status flags (underrun, overflow, etc.).
        """
        if status:
            logger.debug("Audio callback status: %s", status)

        assert self._format is not None

        self._fill_audio_buffer(outdata, frames)

    def _get_next_chunk_if_needed(self) -> bool:
        """Pull next chunk from queue if current is exhausted. Return True if chunk available."""
        if self._current_chunk is not None:
            return True

        queue_size = self._queue.qsize()
        buffer_ms = self._queued_duration_us / 1_000

        if buffer_ms < 100:
            logger.error(
                "Critical low buffer: %.1f ms (min 100ms). Queue: %d chunks",
                buffer_ms,
                queue_size,
            )

        if self._queue.empty():
            self._underrun_count += 1
            logger.error("Buffer underrun #%d", self._underrun_count)
            return False

        # Get next chunk from queue
        chunk = self._queue.get_nowait()

        # Initialize current server timestamp from the first chunk's actual timestamp
        if self._current_server_timestamp_us is None:
            self._current_server_timestamp_us = chunk.server_timestamp_us
            logger.info(
                "Initialized playback from first chunk at server time %d us",
                self._current_server_timestamp_us,
            )
        else:
            # Log if chunk timestamp is far from where we expect it
            chunk_delta_us = chunk.server_timestamp_us - self._current_server_timestamp_us
            if abs(chunk_delta_us) > 10_000:  # More than 10ms off
                logger.warning(
                    "Chunk mismatch: expected %d, got %d, delta %d us",
                    self._current_server_timestamp_us,
                    chunk.server_timestamp_us,
                    chunk_delta_us,
                )

        self._current_chunk = chunk
        self._current_chunk_offset = 0
        return True

    def _fill_audio_buffer(
        self,
        outdata: memoryview,
        frames: int,
    ) -> None:
        """
        Fill output buffer with audio data from queue, using server timestamps.

        Phase 1: Timestamp-aware playback - consume chunks in order and track
        position using server timestamps. This ensures synchronization across
        multiple clients when chunks have matching server timestamps.
        """
        assert self._format is not None

        bytes_needed = frames * self._format.frame_size
        output_buffer = memoryview(outdata).cast("B")
        bytes_written = 0

        try:
            while bytes_written < bytes_needed:
                # Get next chunk if needed
                if not self._get_next_chunk_if_needed():
                    # No chunks available - fill rest with silence
                    silence_bytes = bytes_needed - bytes_written
                    if silence_bytes > 0:
                        silence_data = b"\x00" * silence_bytes
                        output_buffer[bytes_written : bytes_written + silence_bytes] = memoryview(
                            silence_data
                        )
                    break

                # At this point, _current_chunk is guaranteed to be set
                assert self._current_chunk is not None
                # Copy from current chunk starting at offset
                chunk_data = self._current_chunk.audio_data
                remaining_in_chunk = len(chunk_data) - self._current_chunk_offset
                bytes_to_copy = min(remaining_in_chunk, bytes_needed - bytes_written)

                # Use memoryview for both sides to ensure compatible assignment
                output_buffer[bytes_written : bytes_written + bytes_to_copy] = memoryview(
                    chunk_data
                )[self._current_chunk_offset : self._current_chunk_offset + bytes_to_copy]
                bytes_written += bytes_to_copy
                self._current_chunk_offset += bytes_to_copy

                # If chunk is exhausted, advance timestamp
                if self._current_chunk_offset >= len(chunk_data):
                    chunk_frames = len(chunk_data) // self._format.frame_size
                    chunk_duration_us = (chunk_frames * 1_000_000) // self._format.sample_rate
                    self._queued_duration_us = max(0, self._queued_duration_us - chunk_duration_us)
                    if self._current_server_timestamp_us is not None:
                        self._current_server_timestamp_us += chunk_duration_us
                    self._current_chunk = None
                    self._current_chunk_offset = 0

        except Exception:  # pragma: no cover - error handling
            logger.exception("Error in audio callback")
            # Fill rest with silence on error
            if bytes_written < bytes_needed:
                silence_bytes = bytes_needed - bytes_written
                silence_data = b"\x00" * silence_bytes
                output_buffer[bytes_written : bytes_written + silence_bytes] = memoryview(
                    silence_data
                )
            # Reset partial chunk state on error
            self._current_chunk = None
            self._current_chunk_offset = 0

    def submit(self, server_timestamp_us: int, payload: bytes) -> None:
        """
        Queue an audio payload for playback, intelligently handling gaps and overlaps.

        Fills gaps with silence and trims overlaps to ensure a continuous stream.

        Args:
            server_timestamp_us: Server timestamp when this audio should play.
            payload: Raw PCM audio bytes.
        """
        if self._format is None:
            logger.debug("Audio format missing; dropping audio chunk")
            return
        if self._format.frame_size == 0:
            return
        if len(payload) % self._format.frame_size != 0:
            logger.warning(
                "Dropping audio chunk with invalid size: %s bytes (frame size %s)",
                len(payload),
                self._format.frame_size,
            )
            return

        # Calculate chunk duration in microseconds
        chunk_frames = len(payload) // self._format.frame_size
        chunk_duration_us = (chunk_frames * 1_000_000) // self._format.sample_rate

        # Initialize expected next timestamp and target play time on first chunk
        if self._expected_next_timestamp is None:
            self._expected_next_timestamp = server_timestamp_us

            # Calculate synchronized startup time: "now + 500ms buffer" on server
            # This ensures all clients start at the exact same absolute moment
            current_loop_time_us = int(self._loop.time() * 1_000_000)
            current_server_time_us = self._compute_server_time(current_loop_time_us)
            buffer_delay_us = 500_000  # 500ms buffer for network jitter
            target_server_time_us = current_server_time_us + buffer_delay_us

            # Convert target server time back to our loop time
            self._target_play_time_us = self._compute_play_time(target_server_time_us)

            logger.info(
                "First chunk: calculated target_play_time=%d us (loop time), "
                "target_server_time=%d us, current_server_time=%d us, buffer_delay=%d us",
                self._target_play_time_us,
                target_server_time_us,
                current_server_time_us,
                buffer_delay_us,
            )

            # Launch background task to start playback at the exact target time
            if self._startup_task is None or self._startup_task.done():
                self._startup_task = asyncio.create_task(self._scheduled_start())
        # Handle gap: insert silence to fill the gap
        elif server_timestamp_us > self._expected_next_timestamp:
            gap_us = server_timestamp_us - self._expected_next_timestamp
            gap_frames = (gap_us * self._format.sample_rate) // 1_000_000
            silence_bytes = gap_frames * self._format.frame_size
            silence = b"\x00" * silence_bytes
            self._queue.put_nowait(
                _QueuedChunk(
                    server_timestamp_us=self._expected_next_timestamp,
                    audio_data=silence,
                )
            )
            logger.info("Gap detected: %d us, inserted %d bytes of silence", gap_us, silence_bytes)
            self._expected_next_timestamp = server_timestamp_us

        # Handle overlap: trim the start of the chunk
        elif server_timestamp_us < self._expected_next_timestamp:
            overlap_us = self._expected_next_timestamp - server_timestamp_us
            overlap_frames = (overlap_us * self._format.sample_rate) // 1_000_000
            trim_bytes = overlap_frames * self._format.frame_size
            if trim_bytes < len(payload):
                payload = payload[trim_bytes:]
                server_timestamp_us = self._expected_next_timestamp
                logger.info("Overlap detected: %d us, trimmed %d bytes", overlap_us, trim_bytes)
            else:
                # Entire chunk is overlap, skip it
                logger.info("Overlap detected: %d us, skipped entire chunk", overlap_us)
                return

        # Queue the chunk
        if len(payload) > 0:
            chunk = _QueuedChunk(
                server_timestamp_us=server_timestamp_us,
                audio_data=payload,
            )
            self._queue.put_nowait(chunk)
            # Track duration of queued audio
            self._queued_duration_us += chunk_duration_us

        # Update expected position for next chunk
        self._expected_next_timestamp = server_timestamp_us + chunk_duration_us

    async def _scheduled_start(self) -> None:
        """
        Wait for the target play time, then start playback when ready.

        This ensures all clients start playback at the exact same moment,
        synchronized via the Kalman filter's target time calculation.
        """
        assert self._target_play_time_us is not None

        # Wait until we reach the target play time (10ms polling precision)
        while True:
            try:
                current_loop_time_us = int(self._loop.time() * 1_000_000)
                if current_loop_time_us >= self._target_play_time_us:
                    break
                await asyncio.sleep(0.01)  # Check every 10ms
            except asyncio.CancelledError:  # pragma: no cover - cleanup on stop
                return

        # We've reached or passed target time
        # Decide minimum buffer based on how late we are
        current_loop_time_us = int(self._loop.time() * 1_000_000)
        time_margin_us = current_loop_time_us - self._target_play_time_us

        # If we're already >100ms late, start with whatever buffer we have
        # Otherwise wait for full 500ms buffer
        min_buffer_us = 500_000 if time_margin_us < 100_000 else 0

        # Wait until we have enough buffer
        while self._queued_duration_us < min_buffer_us:
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:  # pragma: no cover - cleanup on stop
                return

        # Time to start the stream!
        if self._stream is not None and not self._stream_started:
            self._stream.start()
            self._stream_started = True
            current_loop_time_us = int(self._loop.time() * 1_000_000)
            logger.info(
                "Stream started at target: loop_time=%d us, target=%d us, "
                "buffer=%.1fs (offset=%d us)",
                current_loop_time_us,
                self._target_play_time_us,
                self._queued_duration_us / 1_000_000,
                current_loop_time_us - self._target_play_time_us,
            )

    def _close_stream(self) -> None:
        """Close the audio output stream."""
        stream = self._stream
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # pragma: no cover - backend failure
                logger.exception("Failed to close audio output stream")
        self._stream = None
