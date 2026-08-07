"""Single-consumer async stream of decoded source PCM."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from aiosendspin.audio.format import AudioFormat

logger = logging.getLogger(__name__)

# Bound memory when a consumer falls behind.
DEFAULT_QUEUE_MAXLEN = 512
DEFAULT_MAX_BUFFERED_SECONDS = 10


class SourceStream:
    """Consume decoded source PCM from ``SourceStreamStartedEvent.handle``.

    The source role creates this handle. Iteration yields PCM with server timestamps.
    """

    def __init__(self, audio_format: AudioFormat, *, maxlen: int = DEFAULT_QUEUE_MAXLEN) -> None:
        """Create the role-owned queue for decoded ``audio_format`` chunks."""
        self._audio_format = audio_format
        self._queue: deque[tuple[bytes, int]] = deque()
        self._maxlen = maxlen
        bytes_per_second = (
            audio_format.sample_rate * audio_format.channels * (audio_format.bit_depth // 8)
        )
        self._max_buffered_bytes = bytes_per_second * DEFAULT_MAX_BUFFERED_SECONDS
        self._buffered_bytes = 0
        self._event = asyncio.Event()
        self._ended = False
        self._dropped_warned = False

    @property
    def audio_format(self) -> AudioFormat:
        """Native PCM format of the chunks produced by this stream."""
        return self._audio_format

    def _push(self, pcm: bytes, timestamp_us: int) -> None:
        """Append a decoded chunk, dropping the oldest if the buffer is full."""
        if not pcm:
            return
        self._queue.append((pcm, timestamp_us))
        self._buffered_bytes += len(pcm)
        # The newest chunk always stays, even when it alone exceeds the byte budget.
        while len(self._queue) > 1 and (
            len(self._queue) > self._maxlen or self._buffered_bytes > self._max_buffered_bytes
        ):
            dropped, _ = self._queue.popleft()
            self._buffered_bytes -= len(dropped)
            if not self._dropped_warned:
                logger.warning(
                    "Source stream consumer is not keeping up; dropping oldest decoded audio"
                )
                self._dropped_warned = True
        self._event.set()

    def _end(self) -> None:
        """End iteration after buffered chunks drain."""
        self._ended = True
        self._event.set()

    def __aiter__(self) -> AsyncIterator[tuple[bytes, int]]:
        """Return self as the single-consumer iterator."""
        return self

    async def __anext__(self) -> tuple[bytes, int]:
        """Return the next decoded chunk, waiting for one if the buffer is empty."""
        while True:
            if self._queue:
                pcm, timestamp_us = self._queue.popleft()
                self._buffered_bytes -= len(pcm)
                return pcm, timestamp_us
            if self._ended:
                raise StopAsyncIteration
            self._event.clear()
            await self._event.wait()
