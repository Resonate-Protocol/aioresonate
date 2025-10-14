"""Audio playback for the Resonate CLI."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Protocol, cast

try:
    import sounddevice as _sounddevice
except ImportError:  # pragma: no cover - optional dependency
    _sounddevice = None

from aioresonate.client import PCMFormat

logger = logging.getLogger(__name__)


class _AudioStreamProto(Protocol):
    """Subset of methods used from the sounddevice RawOutputStream."""

    def start(self) -> None: ...

    def write(self, data: bytes) -> Any: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class _QueuedChunk:
    """Represents a queued audio chunk with timing information."""

    server_timestamp_us: int
    """Original server timestamp for this chunk."""
    audio_data: bytes
    """Raw PCM audio bytes."""
    counter: int
    """Sequence counter for priority queue ordering."""


class AudioPlayer:
    """
    Audio player for the Resonate CLI with time synchronization support.

    This player accepts audio chunks with server timestamps and dynamically
    computes playback times using a time synchronization function. This allows
    for accurate synchronization even when the time base changes during playback.
    """

    _PREPLAY_MARGIN_US: Final[int] = 1_000
    """Microseconds before target play time to wake up for final scheduling."""

    # Drift correction thresholds (from ESP32 implementation)
    _HARD_SYNC_THRESHOLD_US: Final[int] = 50_000  # 50ms - aggressive correction
    _SOFT_SYNC_THRESHOLD_US: Final[int] = 10_000  # 10ms - gentle correction

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
                timestamps, accounting for clock drift and offset.
            compute_server_time: Function that converts client timestamps to server
                timestamps (inverse of compute_client_time).
        """
        self._loop = loop
        self._compute_client_time = compute_client_time
        self._compute_server_time = compute_server_time
        self._format: PCMFormat | None = None
        self._queue: asyncio.PriorityQueue[tuple[int, int, _QueuedChunk]] = asyncio.PriorityQueue()
        self._counter = 0
        self._stream: _AudioStreamProto | None = None
        self._closed = False
        self._audio_available = _sounddevice is not None
        self._output_latency_us = 0

        # Drift tracking for feedback control
        self._last_expected_server_time_us: int | None = None
        self._drift_error_us = 0
        self._correction_frames = 0.0
        self._callback_lock = threading.Lock()
        self._stream_started = False
        self._first_real_chunk = True  # Flag to initialize timing from first chunk

        # Partial chunk tracking (to avoid discarding partial chunks)
        self._current_chunk: _QueuedChunk | None = None
        self._current_chunk_offset = 0

        if not self._audio_available:
            logger.warning("sounddevice is not installed. Audio playback will be disabled.")

    @property
    def audio_available(self) -> bool:
        """Return True if an audio backend is available."""
        return self._audio_available

    def set_format(self, pcm_format: PCMFormat) -> None:
        """Configure the audio output format."""
        self._format = pcm_format
        if not self._audio_available:
            return
        self._close_stream()

        # Reset drift tracking on format change
        self._last_expected_server_time_us = None
        self._drift_error_us = 0
        self._correction_frames = 0.0
        self._stream_started = False
        self._first_real_chunk = True

        assert _sounddevice is not None  # for mypy
        dtype = "int16"
        try:
            # Use callback-based stream for true timing feedback
            # Note: Don't start yet - wait for audio to be buffered
            stream = _sounddevice.RawStream(
                samplerate=pcm_format.sample_rate,
                channels=pcm_format.channels,
                dtype=dtype,
                callback=self._audio_callback,
            )

            # Get output latency for logging (no longer used for compensation)
            # RawStream returns (input_latency, output_latency) tuple
            latency = stream.latency
            latency_s = latency[1] if isinstance(latency, tuple) else latency
            self._output_latency_us = int(latency_s * 1_000_000)
            logger.info("Audio output latency: %.1f ms (managed by callback)", latency_s * 1000)

            self._stream = cast("_AudioStreamProto", stream)
        except Exception:  # pragma: no cover - backend failure
            logger.exception("Failed to open audio output stream")
            self._stream = None
            self._audio_available = False
            self._output_latency_us = 0

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
        # Reset drift tracking
        self._last_expected_server_time_us = None
        self._drift_error_us = 0
        self._correction_frames = 0.0
        self._first_real_chunk = True
        # Reset partial chunk tracking
        self._current_chunk = None
        self._current_chunk_offset = 0

    def _audio_callback(
        self,
        indata: Any,  # noqa: ARG002 - unused but required by signature
        outdata: Any,
        frames: int,
        time: Any,
        status: Any,
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

        # Measure and correct drift
        correction_frames = self._measure_and_correct_drift(frames, target_server_time_us)

        # Fill the output buffer
        self._fill_audio_buffer(outdata, frames, target_server_time_us, correction_frames)

    def _measure_and_correct_drift(self, frames: int, target_server_time_us: int) -> int:
        """
        Measure drift and compute correction frames needed.

        Returns:
            Number of frames to correct (positive = insert silence, negative = skip audio).
        """
        assert self._format is not None

        # Don't measure drift until we've started playing real audio
        if self._first_real_chunk:
            return 0

        # Measure drift if we have a previous reference
        if self._last_expected_server_time_us is not None:
            # Calculate how much time passed in the audio stream
            frame_duration_us = int((frames * 1_000_000) / self._format.sample_rate)
            expected_server_time_us = self._last_expected_server_time_us + frame_duration_us

            # Drift = how far off we are from expected
            self._drift_error_us = target_server_time_us - expected_server_time_us

            if abs(self._drift_error_us) > 1000:  # Log if > 1ms
                logger.debug("Drift error: %d us", self._drift_error_us)

        # Update expected time for next callback (will be overridden by first chunk)
        frame_duration_us = int((frames * 1_000_000) / self._format.sample_rate)
        self._last_expected_server_time_us = target_server_time_us + frame_duration_us

        # TODO: Re-enable drift correction after basic playback works
        # For now, just measure but don't correct
        return 0

    def _get_next_chunk(
        self,
        target_server_time_us: int,  # noqa: ARG002 - unused temporarily
        frames_to_skip: int,
    ) -> int:
        """
        Pull next chunk from queue and prepare it for playback.

        Returns:
            Updated frames_to_skip count after processing this chunk.
        """
        assert self._format is not None

        if self._queue.empty():
            return frames_to_skip

        _priority, _counter, chunk = self._queue.get_nowait()

        # Initialize timing from first real chunk
        if self._first_real_chunk:
            self._first_real_chunk = False
            self._last_expected_server_time_us = chunk.server_timestamp_us
            logger.debug("Initialized timing from first chunk at %d us", chunk.server_timestamp_us)

        # TODO: Re-enable lateness check after basic playback works
        # For now, just play all chunks in order

        # Don't apply corrections for now
        self._current_chunk_offset = 0
        self._current_chunk = chunk
        return 0

    def _fill_audio_buffer(
        self,
        outdata: Any,
        frames: int,
        target_server_time_us: int,
        correction_frames: int,  # noqa: ARG002 - unused temporarily
    ) -> None:
        """Fill output buffer with audio data, applying drift correction."""
        assert self._format is not None
        bytes_needed = frames * self._format.frame_size
        output_buffer = memoryview(outdata).cast("B")
        bytes_written = 0

        try:
            while bytes_written < bytes_needed:
                # Get current chunk or pull new one
                if self._current_chunk is None:
                    if self._queue.empty():
                        # Underrun - fill rest with silence
                        silence_bytes = bytes_needed - bytes_written
                        output_buffer[bytes_written : bytes_written + silence_bytes] = (
                            b"\x00" * silence_bytes
                        )
                        if not self._first_real_chunk:
                            logger.debug(
                                "Audio underrun: filled %d bytes with silence", silence_bytes
                            )
                        break

                    # Pull next chunk (ignoring corrections for now)
                    self._get_next_chunk(target_server_time_us, 0)
                    if self._current_chunk is None:
                        continue  # Shouldn't happen, but handle gracefully

                # Copy from current chunk starting at offset
                chunk_data = self._current_chunk.audio_data
                remaining_in_chunk = len(chunk_data) - self._current_chunk_offset
                bytes_to_copy = min(remaining_in_chunk, bytes_needed - bytes_written)

                output_buffer[bytes_written : bytes_written + bytes_to_copy] = chunk_data[
                    self._current_chunk_offset : self._current_chunk_offset + bytes_to_copy
                ]
                bytes_written += bytes_to_copy
                self._current_chunk_offset += bytes_to_copy

                # If chunk is exhausted, clear it so we pull next one
                if self._current_chunk_offset >= len(chunk_data):
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

        # Store server timestamp for priority queue ordering
        self._counter += 1
        chunk = _QueuedChunk(
            server_timestamp_us=server_timestamp_us,
            audio_data=payload,
            counter=self._counter,
        )
        # Use server timestamp for priority queue ordering
        self._queue.put_nowait((server_timestamp_us, self._counter, chunk))

        # Start stream after buffering minimum audio (reduces initial underruns)
        if not self._stream_started and self._queue.qsize() >= 3 and self._stream is not None:
            self._stream.start()
            self._stream_started = True
            logger.debug("Stream started after buffering %d chunks", self._queue.qsize())

    def _close_stream(self) -> None:
        """Close the audio output stream."""
        stream = self._stream
        if stream is not None and self._audio_available:
            try:
                stream.stop()
                stream.close()
            except Exception:  # pragma: no cover - backend failure
                logger.exception("Failed to close audio output stream")
        self._stream = None
