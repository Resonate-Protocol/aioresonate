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

# Bounded so a slow or absent consumer cannot grow memory without limit. Live capture
# never fills it when the host keeps up. On overflow the oldest chunk is dropped.
DEFAULT_QUEUE_MAXLEN = 512


class SourceStream:
    """Async iterator yielding decoded ``(pcm, timestamp_us)`` chunks for one input stream.

    Backed by a drop-oldest bounded deque for a single consumer. The iterator
    ends once the producer closes the stream and the buffer is drained.
    """

    def __init__(self, audio_format: AudioFormat, *, maxlen: int = DEFAULT_QUEUE_MAXLEN) -> None:
        """Create a stream carrying decoded audio at ``audio_format``."""
        self._audio_format = audio_format
        self._queue: deque[tuple[bytes, int]] = deque(maxlen=maxlen)
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
        if len(self._queue) == self._queue.maxlen and not self._dropped_warned:
            logger.warning(
                "Source stream consumer is not keeping up; dropping oldest decoded audio"
            )
            self._dropped_warned = True
        self._queue.append((pcm, timestamp_us))
        self._event.set()

    def _end(self) -> None:
        """Mark the stream finished so the iterator stops after draining buffered chunks."""
        self._ended = True
        self._event.set()

    def __aiter__(self) -> AsyncIterator[tuple[bytes, int]]:
        """Return self as the single-consumer iterator."""
        return self

    async def __anext__(self) -> tuple[bytes, int]:
        """Return the next decoded chunk, waiting for one if the buffer is empty."""
        while True:
            if self._queue:
                return self._queue.popleft()
            if self._ended:
                raise StopAsyncIteration
            self._event.clear()
            await self._event.wait()
