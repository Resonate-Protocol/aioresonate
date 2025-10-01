"""Streaming helpers."""

import asyncio
import logging
from collections import deque
from typing import NamedTuple


def build_flac_stream_header(extradata: bytes) -> bytes:
    """Return a complete FLAC stream header for encoder extradata."""
    if not extradata:
        return extradata
    return b"fLaC\x80" + len(extradata).to_bytes(3, "big") + extradata


def _samples_to_microseconds(sample_count: int, sample_rate: int) -> int:
    """Convert a number of samples to microseconds using sample_rate."""
    return int(sample_count * 1_000_000 / sample_rate)


class BufferedChunk(NamedTuple):
    """Represents compressed audio bytes scheduled for playback."""

    end_time_us: int
    """Absolute timestamp when these bytes should be fully consumed."""
    byte_count: int
    """Compressed byte count occupying the device buffer."""


class BufferTracker:
    """
    Track buffered compressed audio for a client and apply backpressure when needed.

    This class monitors the amount of compressed audio data buffered on a client device
    and ensures the server doesn't exceed the client's buffer capacity by applying
    backpressure when necessary.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        logger: logging.Logger,
        client_id: str,
        capacity_bytes: int,
    ) -> None:
        """
        Initialize the buffer tracker for a client.

        Args:
            loop: The event loop for timing calculations.
            logger: Logger instance for this tracker.
            client_id: Identifier for the client being tracked.
            capacity_bytes: Maximum buffer capacity in bytes reported by the client.
        """
        self._loop = loop
        self._logger = logger
        self.client_id = client_id
        self.capacity_bytes = capacity_bytes
        self.buffered_chunks: deque[BufferedChunk] = deque()
        self.buffered_bytes = 0
        self.max_usage_bytes = 0

    def prune_consumed(self, now_us: int | None = None) -> int:
        """Drop finished chunks and return the timestamp used for the calculation."""
        if now_us is None:
            now_us = int(self._loop.time() * 1_000_000)
        while self.buffered_chunks and self.buffered_chunks[0].end_time_us <= now_us:
            self.buffered_bytes -= self.buffered_chunks.popleft().byte_count
        self.buffered_bytes = max(self.buffered_bytes, 0)
        return now_us

    async def wait_for_capacity(self, bytes_needed: int, expected_end_us: int) -> None:
        """Block until the device buffer can accept bytes_needed more bytes."""
        if bytes_needed <= 0:
            return
        if bytes_needed >= self.capacity_bytes:
            self._logger.warning(
                "Chunk size %s exceeds reported buffer capacity %s for client %s",
                bytes_needed,
                self.capacity_bytes,
                self.client_id,
            )
            return

        while True:
            now_us = self.prune_consumed()
            projected_usage = self.buffered_bytes + bytes_needed
            if projected_usage <= self.capacity_bytes:
                # Returning here keeps the producer running because we are below capacity.
                return

            sleep_target_us = (
                self.buffered_chunks[0].end_time_us if self.buffered_chunks else expected_end_us
            )
            sleep_us = sleep_target_us - now_us
            if sleep_us <= 0:
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(sleep_us / 1_000_000)

    def register(self, end_time_us: int, byte_count: int) -> None:
        """Record bytes added to the buffer finishing at end_time_us."""
        if byte_count <= 0:
            return
        self.buffered_chunks.append(BufferedChunk(end_time_us, byte_count))
        self.buffered_bytes += byte_count
        self.max_usage_bytes = max(self.max_usage_bytes, self.buffered_bytes)
