"""Audio playback helpers for the Resonate client."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Final, Protocol, cast

try:
    import sounddevice as _sounddevice
except ImportError:  # pragma: no cover - optional dependency
    _sounddevice = None

logger = logging.getLogger(__name__)


class _AudioStreamProto(Protocol):
    """Subset of methods used from the sounddevice RawOutputStream."""

    def start(self) -> None: ...

    def write(self, data: bytes) -> Any: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class PCMFormat:
    """PCM audio format description."""

    sample_rate: int
    channels: int
    bit_depth: int

    def __post_init__(self) -> None:
        """Validate the provided PCM audio format."""
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels not in (1, 2):
            raise ValueError("channels must be 1 or 2 for the CLI player")
        if self.bit_depth != 16:
            raise ValueError("Only 16-bit PCM is supported by the CLI player")

    @property
    def frame_size(self) -> int:
        """Return bytes per PCM frame."""
        return self.channels * 2  # 16-bit -> 2 bytes per sample


class AudioPlayer:
    """Minimal audio player that schedules PCM chunks for playback."""

    _PREPLAY_MARGIN_US: Final[int] = 1_000

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        """Initialise a player bound to the provided event loop."""
        self._loop = loop
        self._format: PCMFormat | None = None
        self._queue: asyncio.PriorityQueue[tuple[int, int, bytes]] = asyncio.PriorityQueue()
        self._counter = 0
        self._task: asyncio.Task[None] | None = None
        self._stream: _AudioStreamProto | None = None
        self._closed = False
        self._audio_available = _sounddevice is not None
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
            self._stream = cast("_AudioStreamProto", stream)
        except Exception:  # pragma: no cover - backend failure
            logger.exception("Failed to open audio output stream")
            self._stream = None
            self._audio_available = False

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

    def submit(self, play_at_us: int, payload: bytes) -> None:
        """Queue an audio payload for playback."""
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
        self._counter += 1
        self._queue.put_nowait((play_at_us, self._counter, payload))
        self.start()

    async def _write_audio(self, payload: bytes) -> None:
        if not self._audio_available:
            return
        stream = self._stream
        if stream is None:
            return
        await self._loop.run_in_executor(None, stream.write, payload)

    async def _run(self) -> None:
        try:
            while True:
                play_at_us, _counter, payload = await self._queue.get()
                if self._closed:
                    break
                await self._wait_until(play_at_us)
                await self._write_audio(payload)
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            pass
        finally:
            self._close_stream()

    async def _wait_until(self, play_at_us: int) -> None:
        while True:
            now_us = self._now_us()
            delta = play_at_us - now_us
            if delta <= self._PREPLAY_MARGIN_US:
                break
            sleep_time = max((delta - self._PREPLAY_MARGIN_US) / 1_000_000, 0.0)
            await asyncio.sleep(min(sleep_time, 0.1))

    def _now_us(self) -> int:
        return int(self._loop.time() * 1_000_000)

    def _close_stream(self) -> None:
        stream = self._stream
        if stream is not None and self._audio_available:
            try:
                stream.stop()
                stream.close()
            except Exception:  # pragma: no cover - backend failure
                logger.exception("Failed to close audio output stream")
        self._stream = None
