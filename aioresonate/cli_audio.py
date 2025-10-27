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

    This player accepts audio chunks with server timestamps and dynamically
    computes playback times using a time synchronization function. This allows
    for accurate synchronization even when the time base changes during playback.

    Attributes:
        _loop: The asyncio event loop used for scheduling.
        _compute_client_time: Function that converts server timestamps to client
            timestamps (monotonic loop time), accounting for clock drift, offset,
            and static delay.
        _compute_server_time: Function that converts client timestamps (monotonic
            loop time) to server timestamps (inverse of _compute_client_time).
    """

    _loop: asyncio.AbstractEventLoop
    _compute_client_time: Callable[[int], int]
    _compute_server_time: Callable[[int], int]

    _MIN_CHUNKS_TO_START: Final[int] = 16
    """Minimum chunks buffered before starting playback to absorb network jitter."""
    _MIN_CHUNKS_TO_MAINTAIN: Final[int] = 8
    """Minimum chunks to maintain during playback to avoid underruns."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        compute_client_time: Callable[[int], int],
        compute_server_time: Callable[[int], int],
    ) -> None:
        """
        Initialize the audio player.

        Args:
            loop: The asyncio event loop to use for scheduling.
            compute_client_time: Function that converts server timestamps to client
                timestamps (monotonic loop time), accounting for clock drift, offset,
                and static delay.
            compute_server_time: Function that converts client timestamps (monotonic
                loop time) to server timestamps (inverse of compute_client_time).
        """
        self._loop = loop
        self._compute_client_time = compute_client_time
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
            latency=1.0,  # 1 second latency to allow buffering
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
            except asyncio.QueueEmpty:
                break
        # Reset playback state
        self._first_real_chunk = True
        self._current_chunk = None
        self._current_chunk_offset = 0
        self._expected_next_timestamp = None
        self._underrun_count = 0
        self._queued_duration_us = 0

    def _audio_callback(
        self,
        outdata: memoryview,
        frames: int,
        _time: sounddevice.CallbackTimeInfo,
        status: CallbackFlags,
    ) -> None:
        """
        Audio callback invoked by sounddevice when output buffer needs filling.

        Args:
            outdata: Output buffer to fill with audio data.
            frames: Number of frames requested.
            _time: Timing information (unused, kept for sounddevice API).
            status: Status flags (underrun, overflow, etc.).
        """
        if status:
            logger.debug("Audio callback status: %s", status)

        assert self._format is not None
        bytes_needed = frames * self._format.frame_size
        output_buffer = memoryview(outdata).cast("B")
        bytes_written = 0

        try:
            while bytes_written < bytes_needed:
                # Get current chunk or pull new one
                if self._current_chunk is None:
                    queue_size = self._queue.qsize()
                    buffer_ms = self._queued_duration_us / 1_000

                    # Check for buffer depletion - warn if less than 100ms of audio remains
                    if buffer_ms < 100:
                        logger.error(
                            "Critical low buffer: %.1f ms (min 100ms) - underflow imminent. "
                            "Queue: %d chunks",
                            buffer_ms,
                            queue_size,
                        )

                    if self._queue.empty():
                        # No chunks available - fill with silence (underrun)
                        self._underrun_count += 1
                        silence_bytes = bytes_needed - bytes_written
                        output_buffer[bytes_written : bytes_written + silence_bytes] = (
                            b"\x00" * silence_bytes
                        )
                        logger.error(
                            "Buffer underrun #%d: filling %d bytes with silence",
                            self._underrun_count,
                            silence_bytes,
                        )
                        break

                    # Get next chunk from queue (already continuous from submit())
                    chunk = self._queue.get_nowait()
                    self._current_chunk = chunk
                    self._current_chunk_offset = 0

                # Copy from current chunk starting at offset
                chunk_data = self._current_chunk.audio_data
                remaining_in_chunk = len(chunk_data) - self._current_chunk_offset
                bytes_to_copy = min(remaining_in_chunk, bytes_needed - bytes_written)

                output_buffer[bytes_written : bytes_written + bytes_to_copy] = chunk_data[
                    self._current_chunk_offset : self._current_chunk_offset + bytes_to_copy
                ]
                bytes_written += bytes_to_copy
                self._current_chunk_offset += bytes_to_copy

                # If chunk is exhausted, clear it
                if self._current_chunk_offset >= len(chunk_data):
                    # Chunk consumed - update duration tracking
                    chunk_frames = len(chunk_data) // self._format.frame_size
                    chunk_duration_us = (chunk_frames * 1_000_000) // self._format.sample_rate
                    self._queued_duration_us = max(0, self._queued_duration_us - chunk_duration_us)
                    self._current_chunk = None
                    self._current_chunk_offset = 0

        except Exception:
            logger.exception("Error in audio callback")
            # Fill rest with silence on error
            if bytes_written < bytes_needed:
                silence_bytes = bytes_needed - bytes_written
                output_buffer[bytes_written : bytes_written + silence_bytes] = (
                    b"\x00" * silence_bytes
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

        # Initialize expected next timestamp on first chunk
        if self._expected_next_timestamp is None:
            self._expected_next_timestamp = server_timestamp_us
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

        # Start stream when we have enough audio duration buffered (not just item count)
        # Need at least 500ms of audio to survive network jitter
        min_duration_us = 500_000  # 500ms of audio
        queue_size = self._queue.qsize()
        if self._queued_duration_us >= min_duration_us and not self._stream_started:
            if self._stream is not None:
                self._stream.start()
                self._stream_started = True
                logger.info(
                    "Stream started with %d chunks, %.1f seconds of audio buffered",
                    queue_size,
                    self._queued_duration_us / 1_000_000,
                )
        elif not self._stream_started:
            logger.debug(
                "Buffering... %.1f/%.1f seconds of audio",
                self._queued_duration_us / 1_000_000,
                min_duration_us / 1_000_000,
            )

    def _close_stream(self) -> None:
        """Close the audio output stream."""
        stream = self._stream
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.exception("Failed to close audio output stream")
        self._stream = None
