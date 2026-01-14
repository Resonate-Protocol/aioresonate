"""Audio types and buffer tracking utilities."""

from __future__ import annotations

import asyncio
import logging
import types
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

from aiosendspin.models import AudioCodec

logger = logging.getLogger(__name__)


def _get_av() -> types.ModuleType:
    """Lazy import of av module to avoid slow startup."""
    import av as _av  # noqa: PLC0415

    return _av


@dataclass(frozen=True)
class AudioFormat:
    """Audio format of a stream."""

    sample_rate: int
    """Sample rate in Hz (e.g., 44100, 48000)."""
    bit_depth: int
    """Bit depth in bits per sample (16 or 24)."""
    channels: int
    """Number of audio channels (1 for mono, 2 for stereo)."""
    codec: AudioCodec = AudioCodec.PCM
    """Audio codec of the stream."""


class BufferedChunk(NamedTuple):
    """Buffered chunk metadata tracked by BufferTracker for backpressure control."""

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
        client_id: str,
        capacity_bytes: int,
    ) -> None:
        """
        Initialize the buffer tracker for a client.

        Args:
            loop: The event loop for timing calculations.
            client_id: Identifier for the client being tracked.
            capacity_bytes: Maximum buffer capacity in bytes reported by the client.
        """
        self._loop = loop
        self.client_id = client_id
        self.capacity_bytes = capacity_bytes
        self.buffered_chunks: deque[BufferedChunk] = deque()
        self.buffered_bytes = 0

    def prune_consumed(self, now_us: int | None = None) -> int:
        """Drop finished chunks and return the timestamp used for the calculation."""
        if now_us is None:
            now_us = int(self._loop.time() * 1_000_000)
        while self.buffered_chunks and self.buffered_chunks[0].end_time_us <= now_us:
            self.buffered_bytes -= self.buffered_chunks.popleft().byte_count
        self.buffered_bytes = max(self.buffered_bytes, 0)
        return now_us

    def has_capacity_now(self, bytes_needed: int) -> bool:
        """
        Check if buffer can accept bytes_needed without waiting.

        This is a non-blocking version of wait_for_capacity that returns immediately.

        Args:
            bytes_needed: Number of bytes to check capacity for.

        Returns:
            True if the buffer has capacity for bytes_needed, False otherwise.
        """
        if bytes_needed <= 0:
            return True
        if bytes_needed >= self.capacity_bytes:
            # Chunk exceeds capacity, but allow it through
            logger.warning(
                "Chunk size %s exceeds reported buffer capacity %s for client %s",
                bytes_needed,
                self.capacity_bytes,
                self.client_id,
            )
            return True

        self.prune_consumed()
        projected_usage = self.buffered_bytes + bytes_needed
        return projected_usage <= self.capacity_bytes

    def time_until_capacity(self, bytes_needed: int) -> int:
        """
        Calculate time in microseconds until the buffer can accept bytes_needed more bytes.

        Returns 0 if bytes_needed <= 0 (immediate capacity) or bytes_needed >= capacity_bytes
        (chunk exceeds capacity but is allowed through anyway).
        """
        if bytes_needed <= 0:
            return 0
        if bytes_needed >= self.capacity_bytes:
            # TODO: raise exception instead?
            logger.warning(
                "Chunk size %s exceeds reported buffer capacity %s for client %s",
                bytes_needed,
                self.capacity_bytes,
                self.client_id,
            )
            return 0

        # Prune consumed chunks once at the start
        cursor_time_us = self.prune_consumed()
        time_needed_us = 0

        # Simulate state without modifying it to find when capacity is available
        virtual_buffered_bytes = self.buffered_bytes
        cursor_index = 0

        while cursor_index < len(self.buffered_chunks):
            projected_usage = virtual_buffered_bytes + bytes_needed
            if projected_usage <= self.capacity_bytes:
                # We have enough capacity at this point
                break

            chunk = self.buffered_chunks[cursor_index]
            cursor_end_time_us = chunk.end_time_us
            time_needed_us += max(cursor_end_time_us - cursor_time_us, 0)

            # Advance cursor to the next chunk
            cursor_index += 1
            cursor_time_us = cursor_end_time_us
            virtual_buffered_bytes -= chunk.byte_count
        return time_needed_us

    async def wait_for_capacity(self, bytes_needed: int) -> None:
        """Block until the device buffer can accept bytes_needed more bytes."""
        if sleep_time_us := self.time_until_capacity(bytes_needed):
            await asyncio.sleep(sleep_time_us / 1_000_000)

    def register(self, end_time_us: int, byte_count: int) -> None:
        """Record bytes added to the buffer finishing at end_time_us."""
        if byte_count <= 0:
            return
        self.buffered_chunks.append(BufferedChunk(end_time_us, byte_count))
        self.buffered_bytes += byte_count

    def reset(self) -> None:
        """Clear all tracked chunks and reset buffered bytes to zero."""
        self.buffered_chunks.clear()
        self.buffered_bytes = 0


def _resolve_audio_format(audio_format: AudioFormat) -> tuple[int, str, str]:
    """Resolve helper data for an audio format."""
    if audio_format.bit_depth == 16:
        bytes_per_sample = 2
        av_format = "s16"
    elif audio_format.bit_depth == 24:
        bytes_per_sample = 3
        av_format = "s24"
    else:
        raise ValueError("Only 16-bit and 24-bit PCM are supported")

    if audio_format.channels == 1:
        layout = "mono"
    elif audio_format.channels == 2:
        layout = "stereo"
    else:
        raise ValueError("Only mono and stereo layouts are supported")

    return bytes_per_sample, av_format, layout


__all__ = [
    "AudioCodec",
    "AudioFormat",
    "BufferTracker",
]
