"""Audio playback for the Resonate CLI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
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

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        compute_client_time: Callable[[int], int],
    ) -> None:
        """
        Initialize the audio player.

        Args:
            loop: The asyncio event loop to use for scheduling.
            compute_client_time: Function that converts server timestamps to client
                timestamps, accounting for clock drift and offset.
        """
        self._loop = loop
        self._compute_client_time = compute_client_time
        self._format: PCMFormat | None = None
        self._queue: asyncio.PriorityQueue[tuple[int, int, _QueuedChunk]] = asyncio.PriorityQueue()
        self._counter = 0
        self._task: asyncio.Task[None] | None = None
        self._stream: _AudioStreamProto | None = None
        self._closed = False
        self._audio_available = _sounddevice is not None
        self._output_latency_us = 0
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
        assert _sounddevice is not None  # for mypy
        dtype = "int16"
        try:
            stream = _sounddevice.RawOutputStream(
                samplerate=pcm_format.sample_rate,
                channels=pcm_format.channels,
                dtype=dtype,
                blocksize=0,
            )
            stream.start()

            # Get output latency to compensate for sounddevice buffering
            latency_s = stream.latency
            self._output_latency_us = int(latency_s * 1_000_000)
            logger.info("Audio output latency: %.1f ms", latency_s * 1000)

            self._stream = cast("_AudioStreamProto", stream)
        except Exception:  # pragma: no cover - backend failure
            logger.exception("Failed to open audio output stream")
            self._stream = None
            self._audio_available = False
            self._output_latency_us = 0

    def start(self) -> None:
        """Start the background playback task."""
        if self._task is None:
            self._task = self._loop.create_task(self._run())

    async def stop(self) -> None:
        """Stop playback and release resources."""
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._close_stream()

    def clear(self) -> None:
        """Drop all queued audio chunks."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - race condition guard
                break

    def submit(self, server_timestamp_us: int, payload: bytes) -> None:
        """
        Queue an audio payload for playback.

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

        # Store server timestamp so we can recompute client time later
        self._counter += 1
        chunk = _QueuedChunk(
            server_timestamp_us=server_timestamp_us,
            audio_data=payload,
            counter=self._counter,
        )
        # Use server timestamp for priority queue ordering
        self._queue.put_nowait((server_timestamp_us, self._counter, chunk))
        self.start()

    async def _write_audio(self, payload: bytes) -> None:
        """Write audio data to the output stream."""
        if not self._audio_available:
            return
        stream = self._stream
        if stream is None:
            return
        await self._loop.run_in_executor(None, stream.write, payload)

    async def _run(self) -> None:
        """Process queued audio chunks in the main playback loop."""
        try:
            while True:
                _priority, _counter, chunk = await self._queue.get()
                if self._closed:
                    break

                # Wait until play time, continuously recomputing from server timestamp
                await self._wait_until_server_time(chunk.server_timestamp_us)
                await self._write_audio(chunk.audio_data)
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            pass
        finally:
            self._close_stream()

    async def _wait_until_server_time(self, server_timestamp_us: int) -> None:
        """
        Wait until a server timestamp should play, continuously recomputing.

        This continuously recomputes the client play time from the server timestamp
        during the wait loop, tracking time filter updates in real-time to prevent drift.

        Accounts for output latency: we wait until (play_time - latency) so that
        by the time the audio actually plays (after buffering), it's at the right time.
        """
        while True:
            # Continuously recompute play time with current time filter state
            play_at_us = self._compute_client_time(server_timestamp_us)

            # Compensate for output latency - submit earlier so it plays at the right time
            write_at_us = play_at_us - self._output_latency_us

            now_us = self._now_us()
            delta = write_at_us - now_us

            # Drop chunks that are too late
            if delta < -200_000:
                logger.debug("Dropping stale audio chunk (late by %d us)", -delta)
                return

            # If we're close enough, write immediately
            if delta <= self._PREPLAY_MARGIN_US:
                break

            # Sleep briefly, then recompute (tracking time filter changes)
            sleep_time = max((delta - self._PREPLAY_MARGIN_US) / 1_000_000, 0.0)
            await asyncio.sleep(min(sleep_time, 0.1))

    def _now_us(self) -> int:
        """Get current time in microseconds."""
        return int(self._loop.time() * 1_000_000)

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
