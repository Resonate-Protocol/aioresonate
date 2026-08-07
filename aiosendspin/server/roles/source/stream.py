"""Capture stream handed to the host application.

``SourceAudioStream`` is the hand-off point between the library and the host.
The group role pushes captured frames in; the host iterates the stream and gets
decoded PCM with the capture timestamp of each chunk, and routes that audio
through its own pipeline (mix, resample, feed a ``PushStream``, record, ...).

The library deliberately does not decide what happens to the audio: a source
feeding the group it is a member of is one policy among several, and that policy
belongs to the host.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from aiosendspin.server.audio import AudioFormat
    from aiosendspin.server.roles.source.decoder import SourceDecoder

logger = logging.getLogger(__name__)

# Captured frames buffered before the oldest is dropped. A source sends at most
# 150 ms per chunk, so this is several seconds of slack for a host that briefly
# stops consuming.
DEFAULT_QUEUE_MAXSIZE = 128


@dataclass(frozen=True, slots=True)
class SourceAudioChunk:
    """A decoded chunk of captured audio."""

    pcm: bytes
    """Interleaved PCM in the stream's ``audio_format``."""

    timestamp_us: int
    """Server-clock time at which the first sample in this chunk was captured.

    This is the capture time, not a playback time: it is in the past by however
    long capture, encoding, and network transport took. A host that schedules
    this audio for playback must add its own lead time on top.
    """

    duration_us: int
    """Playback duration of this chunk in microseconds."""


@dataclass(frozen=True, slots=True)
class _CapturedFrame:
    """An encoded frame awaiting decode."""

    timestamp_us: int
    data: bytes


class SourceAudioStream:
    """Async iterator over one source client's decoded capture stream.

    Iteration ends when the source ends its stream, leaves the group, or the
    group is torn down. Decoding happens while iterating, so a host that stops
    consuming stops paying for decode; captured frames queue up to
    ``maxsize`` and the oldest are dropped after that.

    Intended for a single consumer.
    """

    def __init__(
        self,
        *,
        client_id: str,
        decoder: SourceDecoder,
        maxsize: int = DEFAULT_QUEUE_MAXSIZE,
    ) -> None:
        """Create a stream fed by the given decoder."""
        self._client_id = client_id
        self._decoder = decoder
        self._maxsize = maxsize
        # One slot beyond the frame budget is reserved for the end sentinel, so
        # closing a full stream never has to discard captured audio to fit it.
        self._queue: asyncio.Queue[_CapturedFrame | None] = asyncio.Queue(maxsize=maxsize + 1)
        self._pending: deque[SourceAudioChunk] = deque()
        self._ended = False
        self._dropped_frames = 0

    @property
    def client_id(self) -> str:
        """Identifier of the source client feeding this stream."""
        return self._client_id

    @property
    def audio_format(self) -> AudioFormat:
        """PCM format of the chunks this stream yields."""
        return self._decoder.audio_format

    @property
    def dropped_frames(self) -> int:
        """Captured frames dropped because the consumer fell behind."""
        return self._dropped_frames

    # --- Fed by SourceGroupRole ---

    def push(self, timestamp_us: int, data: bytes) -> None:
        """Queue a captured frame for decode. Called by the library, not the host."""
        if self._ended:
            return
        if self._queue.qsize() >= self._maxsize:
            # Prefer live audio over stale audio: drop the oldest frame.
            self._drop_oldest()
        self._queue.put_nowait(_CapturedFrame(timestamp_us=timestamp_us, data=data))

    def end(self) -> None:
        """End the stream so the consumer's iteration stops. Called by the library."""
        if self._ended:
            return
        self._ended = True
        self._queue.put_nowait(None)

    def _drop_oldest(self) -> None:
        try:
            self._queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover - only with a concurrent consumer
            return
        self._dropped_frames += 1
        if self._dropped_frames % 100 == 1:
            logger.warning(
                "Source %s: consumer is behind, dropped %d captured frame(s)",
                self._client_id,
                self._dropped_frames,
            )

    # --- Consumed by the host ---

    def __aiter__(self) -> Self:
        """Return this stream as its own async iterator."""
        return self

    async def __anext__(self) -> SourceAudioChunk:
        """Return the next decoded chunk, waiting for capture if needed."""
        while not self._pending:
            frame = await self._queue.get()
            if frame is None:
                raise StopAsyncIteration
            self._decode_into_pending(frame)
        return self._pending.popleft()

    def _decode_into_pending(self, frame: _CapturedFrame) -> None:
        """Decode one captured frame, anchoring the result on its capture time."""
        try:
            chunks = self._decoder.decode(frame.data)
        except Exception:
            logger.exception("Source %s: failed to decode captured frame", self._client_id)
            return

        # Every frame re-anchors on its own capture timestamp rather than
        # extrapolating from the first one: source timestamps legitimately drift
        # and jump (ADC clock variance, filter re-estimation), and a consumer can
        # only detect gaps if it sees the reported time per chunk. Chunks decoded
        # from the same frame are offset by their own duration.
        offset_us = 0
        for pcm in chunks:
            if not pcm:
                continue
            duration_us = self._decoder.duration_us(pcm)
            self._pending.append(
                SourceAudioChunk(
                    pcm=pcm,
                    timestamp_us=frame.timestamp_us + offset_us,
                    duration_us=duration_us,
                )
            )
            offset_us += duration_us
