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

    _PREPLAY_MARGIN_US: Final[int] = 1_000
    """Microseconds before target play time to wake up for final scheduling."""

    # Drift correction thresholds (from ESP32 implementation)
    _HARD_SYNC_THRESHOLD_US: Final[int] = 50_000  # 50ms - aggressive correction
    _SOFT_SYNC_THRESHOLD_US: Final[int] = 10_000  # 10ms - gentle correction

    _loop: asyncio.AbstractEventLoop
    _compute_client_time: Callable[[int], int]
    _compute_server_time: Callable[[int], int]

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
        self._stream: sounddevice.RawStream | None = None
        self._closed = False
        self._stream_started = False
        self._first_real_chunk = True  # Flag to initialize timing from first chunk

        # Partial chunk tracking (to avoid discarding partial chunks)
        self._current_chunk: _QueuedChunk | None = None
        self._current_chunk_offset = 0

        # Track current playback position in server timeline
        self._playback_position_server_us: int | None = None

    def set_format(self, pcm_format: PCMFormat) -> None:
        """Configure the audio output format."""
        self._format = pcm_format
        self._close_stream()

        # Reset state on format change
        self._stream_started = False
        self._first_real_chunk = True

        dtype = "int16"
        # Use callback-based stream for true timing feedback
        # Note: Don't start yet - wait for audio to be buffered
        self._stream = sounddevice.RawStream(
            samplerate=pcm_format.sample_rate,
            channels=pcm_format.channels,
            dtype=dtype,
            callback=self._audio_callback,
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
        self._playback_position_server_us = None

    def _audio_callback(
        self,
        indata: memoryview,  # noqa: ARG002 - unused but required by signature
        outdata: memoryview,
        frames: int,
        time: sounddevice.CallbackTimeInfo,
        status: CallbackFlags,
    ) -> None:
        """
        Audio callback invoked by sounddevice when output buffer needs filling.

        This implements closed-loop feedback control similar to ESP32's speaker callback.

        Args:
            indata: Input data (unused for output-only stream).
            outdata: Output buffer to fill with audio data.
            frames: Number of frames requested.
            time: Timing information including outputBufferDacTime.
            status: Status flags (underrun, overflow, etc.).
        """
        if status:
            logger.debug("Audio callback status: %s", status)

        assert self._format is not None

        # Get DAC time - when this audio will actually play
        dac_time_s = time.outputBufferDacTime
        dac_time_us = int(dac_time_s * 1_000_000)

        # Convert to server timestamp - what should be playing at this DAC time
        target_server_time_us = self._compute_server_time(dac_time_us)

        # Fill the output buffer
        # Drift correction is handled in _check_and_prepare_chunk()
        self._fill_audio_buffer(outdata, frames, target_server_time_us)

    def _check_and_prepare_chunk(
        self, chunk: _QueuedChunk, target_server_time_us: int
    ) -> tuple[bool, bool]:
        """
        Check if chunk matches playback position and apply drift correction.

        Returns:
            (should_play, should_wait) tuple:
            - should_play: True if chunk should be played now
            - should_wait: True if chunk is in future (fill silence and keep it)
        """
        assert self._format is not None

        # Initialize playback position to what DAC wants RIGHT NOW
        # This syncs us to the CURRENT Kalman mapping, not the old one from when chunk was created
        if self._playback_position_server_us is None:
            self._playback_position_server_us = target_server_time_us
            logger.info(
                "Initialized playback position to %d us (DAC target, chunk was %d us, delta=%d us)",
                self._playback_position_server_us,
                chunk.server_timestamp_us,
                chunk.server_timestamp_us - target_server_time_us,
            )

        # Check if this chunk matches our playback position
        chunk_delta_us = chunk.server_timestamp_us - self._playback_position_server_us

        # Measure drift: compare playback position to target (what DAC wants)
        drift_us = self._playback_position_server_us - target_server_time_us
        if abs(drift_us) > 5000:  # Log drift > 5ms
            logger.debug(
                "Playback drift: %d us (position=%d, target=%d)",
                drift_us,
                self._playback_position_server_us,
                target_server_time_us,
            )

        # Apply drift correction by adjusting playback position
        if abs(drift_us) > self._HARD_SYNC_THRESHOLD_US:  # Hard sync at >50ms
            logger.info("Hard sync: adjusting position by %d us", -drift_us)
            self._playback_position_server_us = target_server_time_us
            chunk_delta_us = chunk.server_timestamp_us - self._playback_position_server_us
        elif abs(drift_us) > self._SOFT_SYNC_THRESHOLD_US:  # Gentle correction at >10ms
            # Nudge position by 10% of error
            correction_us = int(drift_us * 0.1)
            self._playback_position_server_us -= correction_us

        # Drop chunks that are too old (>100ms behind)
        if chunk_delta_us < -100_000:
            logger.debug("Dropping old chunk: %d us behind position", -chunk_delta_us)
            return (False, False)

        # If chunk is in the future, wait for it
        if chunk_delta_us > 50_000:
            logger.debug("Chunk is %d us in future, waiting", chunk_delta_us)
            return (False, True)

        # Chunk is ready to play
        return (True, False)

    def _fill_audio_buffer(
        self,
        outdata: memoryview,
        frames: int,
        target_server_time_us: int,
    ) -> None:
        """Fill output buffer with audio data, using timestamp-based chunk selection."""
        assert self._format is not None
        bytes_needed = frames * self._format.frame_size
        output_buffer = memoryview(outdata).cast("B")
        bytes_written = 0

        try:
            while bytes_written < bytes_needed:
                # Get current chunk or pull new one
                if self._current_chunk is None:
                    if self._queue.empty():
                        # No chunks available - fill with silence
                        silence_bytes = bytes_needed - bytes_written
                        output_buffer[bytes_written : bytes_written + silence_bytes] = (
                            b"\x00" * silence_bytes
                        )
                        logger.debug("Buffer underrun: no chunks available")
                        break

                    # Get next chunk from queue
                    chunk = self._queue.get_nowait()

                    # Check if chunk is ready to play
                    should_play, should_wait = self._check_and_prepare_chunk(
                        chunk, target_server_time_us
                    )

                    if should_wait:
                        # Chunk is in future, keep it and fill silence
                        self._current_chunk = chunk
                        self._current_chunk_offset = 0
                        silence_bytes = bytes_needed - bytes_written
                        output_buffer[bytes_written : bytes_written + silence_bytes] = (
                            b"\x00" * silence_bytes
                        )
                        break

                    if not should_play:
                        # Chunk was dropped (too old), get next one
                        continue

                    # Chunk is ready to play
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

                # If chunk is exhausted, clear it and update playback position
                if self._current_chunk_offset >= len(chunk_data):
                    # Update playback position based on chunk duration
                    if self._playback_position_server_us is not None:
                        chunk_frames = len(chunk_data) // self._format.frame_size
                        chunk_duration_us = int(
                            (chunk_frames * 1_000_000) / self._format.sample_rate
                        )
                        self._playback_position_server_us += chunk_duration_us

                    self._current_chunk = None
                    self._current_chunk_offset = 0

        except Exception:  # pragma: no cover - error handling
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
        Queue an audio payload for playback.

        The callback will pull chunks from the queue when needed.

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

        # Queue chunk for playback (chunks arrive in order)
        chunk = _QueuedChunk(
            server_timestamp_us=server_timestamp_us,
            audio_data=payload,
        )
        self._queue.put_nowait(chunk)

        # Start stream immediately when we have enough chunks buffered
        # The callback will initialize position to current DAC time for perfect sync
        if not self._stream_started and self._queue.qsize() >= 3 and self._stream is not None:
            self._stream.start()
            self._stream_started = True
            logger.info("Stream started with %d chunks buffered", self._queue.qsize())

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
