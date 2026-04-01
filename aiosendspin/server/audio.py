"""Audio types and buffer tracking utilities."""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import types
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aiosendspin.server.clock import Clock


def _get_av() -> types.ModuleType:
    """Lazy import of av module to avoid slow startup."""
    return importlib.import_module("av")


_numpy_unavailable = False


def _get_numpy() -> types.ModuleType | None:
    """Lazy import numpy to optimize s32->s24 conversion when available."""
    global _numpy_unavailable  # noqa: PLW0603
    if _numpy_unavailable:
        return None
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        _numpy_unavailable = True
        return None
    return np  # type: ignore[no-any-return,unused-ignore]


@dataclass(frozen=True)
class AudioFormat:
    """PCM audio format descriptor.

    This describes the raw PCM audio parameters without specifying an encoding codec.
    The codec is determined by the transformer (e.g., FlacEncoder, PcmPassthrough).
    """

    sample_rate: int
    """Sample rate in Hz (e.g., 44100, 48000)."""
    bit_depth: int
    """Bit depth in bits per sample (16, 24, or 32)."""
    channels: int
    """Number of audio channels (1 for mono, 2 for stereo)."""
    sample_type: Literal["int", "float"] = "int"
    """PCM sample type. Use ``float`` to represent 32-bit floating-point PCM input."""

    def resolve_av_format(self) -> tuple[int, str, str, int]:
        """Resolve helper data for this audio format.

        Returns:
            A tuple of (wire_bytes_per_sample, av_format, layout, av_bytes_per_sample) where:
            - wire_bytes_per_sample: Number of bytes per audio sample on the wire
            - av_format: PyAV sample format string ("s16", "s32", or "flt")
            - layout: Channel layout string ("mono" or "stereo")
            - av_bytes_per_sample: Number of bytes per sample produced/consumed by PyAV

        Raises:
            ValueError: If bit_depth/channels/sample_type combination is unsupported.
        """
        if self.sample_type not in ("int", "float"):
            raise ValueError("sample_type must be 'int' or 'float'")

        if self.sample_type == "float":
            if self.bit_depth != 32:
                raise ValueError("Only 32-bit float PCM is supported")
            wire_bytes_per_sample = 4
            av_format = "flt"
            av_bytes_per_sample = 4
        elif self.bit_depth == 16:
            wire_bytes_per_sample = 2
            av_format = "s16"
            av_bytes_per_sample = 2
        elif self.bit_depth == 24:
            # PyAV does not support packed s24 sample format; use s32 and convert if needed.
            wire_bytes_per_sample = 3
            av_format = "s32"
            av_bytes_per_sample = 4
        elif self.bit_depth == 32:
            wire_bytes_per_sample = 4
            av_format = "s32"
            av_bytes_per_sample = 4
        else:
            raise ValueError("Only 16-bit, 24-bit, and 32-bit PCM are supported")

        if self.channels == 1:
            layout = "mono"
        elif self.channels == 2:
            layout = "stereo"
        else:
            raise ValueError("Only mono and stereo layouts are supported")

        return wire_bytes_per_sample, av_format, layout, av_bytes_per_sample


class BufferedChunk(NamedTuple):
    """Buffered chunk metadata tracked by BufferTracker for backpressure control."""

    end_time_us: int
    """Absolute timestamp when these bytes should be fully consumed."""
    byte_count: int
    """Compressed byte count occupying the device buffer."""
    duration_us: int
    """Duration of audio in microseconds (independent of compression)."""


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
        clock: Clock,
        client_id: str,
        capacity_bytes: int,
        max_duration_us: int = 0,
    ) -> None:
        """
        Initialize the buffer tracker for a client.

        Args:
            clock: Time source used for timing calculations.
            client_id: Identifier for the client being tracked.
            capacity_bytes: Maximum buffer capacity in bytes reported by the client.
            max_duration_us: Maximum buffer duration in microseconds. If 0, duration
                is not tracked and has_duration_capacity() always returns True.
        """
        self._clock = clock
        self.client_id = client_id
        self.capacity_bytes = capacity_bytes
        self.max_duration_us = max_duration_us
        self.buffered_chunks: deque[BufferedChunk] = deque()
        self.buffered_bytes = 0
        self.buffered_duration_us = 0
        self.blocked_until_us: int | None = None

    def prune_consumed(self, now_us: int | None = None) -> int:
        """Drop finished chunks and return the timestamp used for the calculation."""
        if now_us is None:
            now_us = self._clock.now_us()
        while self.buffered_chunks and self.buffered_chunks[0].end_time_us <= now_us:
            chunk = self.buffered_chunks.popleft()
            self.buffered_bytes -= chunk.byte_count
            self.buffered_duration_us -= chunk.duration_us
        self.buffered_bytes = max(self.buffered_bytes, 0)
        self.buffered_duration_us = max(self.buffered_duration_us, 0)
        return now_us

    def buffered_horizon_us(self, now_us: int | None = None) -> int:
        """Return buffer horizon from now until the furthest scheduled end time."""
        now_us = self.prune_consumed(now_us)
        if not self.buffered_chunks:
            return 0
        return max(self.buffered_chunks[-1].end_time_us - now_us, 0)

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

    def has_duration_capacity(self, duration_needed_us: int = 0) -> bool:
        """
        Check if buffer can accept duration_needed_us without exceeding max_duration_us.

        This is independent of byte-based capacity. If max_duration_us is 0 (not configured),
        this always returns True.

        Args:
            duration_needed_us: Duration in microseconds to check capacity for.

        Returns:
            True if the buffer has capacity for duration_needed_us, False otherwise.
        """
        if self.max_duration_us == 0:
            # Duration tracking not configured
            return True
        if duration_needed_us <= 0:
            return True

        self.prune_consumed()
        projected_duration = self.buffered_duration_us + duration_needed_us
        return projected_duration <= self.max_duration_us

    def time_until_duration_capacity(self, duration_needed_us: int = 0) -> int:
        """
        Calculate time in microseconds until the buffer can accept duration_needed_us more.

        Since audio drains at 1x real time, the wait time equals the excess duration.
        Returns 0 if max_duration_us is 0 (not configured) or if there's already capacity.

        Args:
            duration_needed_us: Duration in microseconds to check capacity for.

        Returns:
            Time in microseconds to wait, or 0 if capacity is immediately available.
        """
        if self.max_duration_us == 0:
            return 0
        if duration_needed_us <= 0:
            return 0

        self.prune_consumed()
        projected_duration = self.buffered_duration_us + duration_needed_us
        if projected_duration <= self.max_duration_us:
            return 0

        # Wait for the excess duration to drain (audio plays at 1x real time)
        return projected_duration - self.max_duration_us

    def time_until_end_time_capacity(self, end_time_us: int) -> int:
        """
        Calculate wait time until the buffer horizon can extend to end_time_us.

        This preserves effective playback headroom based on the furthest buffered end time,
        which is more accurate than summing durations when chunks are intentionally shifted
        on the timeline (for example, player static delay).
        """
        if self.max_duration_us == 0:
            return 0

        now_us = self.prune_consumed()
        if end_time_us <= now_us:
            return 0

        latest_end_us = now_us
        if self.buffered_chunks:
            # Chunks are appended in timestamp order, so the last entry is the furthest.
            latest_end_us = max(now_us, self.buffered_chunks[-1].end_time_us)

        projected_end_us = max(latest_end_us, end_time_us)
        projected_horizon_us = projected_end_us - now_us
        if projected_horizon_us <= self.max_duration_us:
            return 0
        return projected_horizon_us - self.max_duration_us

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

    def time_until_ready(
        self,
        bytes_needed: int,
        duration_needed_us: int,
        *,
        end_time_us: int | None = None,
    ) -> int:
        """
        Calculate time until buffer can accept both bytes and duration.

        Combines byte-based and duration-based backpressure into a single wait time.
        Returns the maximum of both wait times.

        Args:
            bytes_needed: Number of bytes to check capacity for.
            duration_needed_us: Duration in microseconds to check capacity for.
            end_time_us: Absolute end timestamp for horizon-based duration gating.

        Returns:
            Time in microseconds to wait, or 0 if ready immediately.
        """
        byte_wait = self.time_until_capacity(bytes_needed)
        if end_time_us is not None:
            duration_wait = self.time_until_end_time_capacity(end_time_us)
        else:
            duration_wait = self.time_until_duration_capacity(duration_needed_us)
        return max(byte_wait, duration_wait)

    # TODO: if unused delete
    async def wait_for_capacity(self, bytes_needed: int) -> None:
        """Block until the device buffer can accept bytes_needed more bytes."""
        if sleep_time_us := self.time_until_capacity(bytes_needed):
            await asyncio.sleep(sleep_time_us / 1_000_000)

    def register(self, end_time_us: int, byte_count: int, duration_us: int = 0) -> None:
        """Record bytes added to the buffer finishing at end_time_us.

        Args:
            end_time_us: Absolute timestamp when these bytes should be fully consumed.
            byte_count: Compressed byte count occupying the device buffer.
            duration_us: Duration of audio in microseconds (for duration-based tracking).
        """
        if byte_count <= 0:
            return
        self.buffered_chunks.append(BufferedChunk(end_time_us, byte_count, duration_us))
        self.buffered_bytes += byte_count
        self.buffered_duration_us += duration_us

    def reset(self) -> None:
        """Clear all tracked chunks and reset counters to zero."""
        self.buffered_chunks.clear()
        self.buffered_bytes = 0
        self.buffered_duration_us = 0
        self.blocked_until_us = None

    def set_send_blocked(self, delay_us: int) -> None:
        """Block sending for delay_us microseconds from now.

        Used to give clients time to process stream/start before receiving audio.

        Args:
            delay_us: Delay in microseconds before sending is allowed.
        """
        self.blocked_until_us = self._clock.now_us() + delay_us

    def time_until_unblocked(self) -> int:
        """Return microseconds until sending is unblocked, or 0 if not blocked.

        Returns:
            Time in microseconds to wait, or 0 if sending is allowed.
        """
        if self.blocked_until_us is None:
            return 0
        now_us = self._clock.now_us()
        if now_us >= self.blocked_until_us:
            self.blocked_until_us = None
            return 0
        return self.blocked_until_us - now_us


def _convert_s32_to_s24(data: bytes) -> bytes:
    """Convert 32-bit PCM samples to packed 24-bit samples."""
    if len(data) % 4:
        raise ValueError("s32 PCM buffer length must be a multiple of 4 bytes")

    if np := _get_numpy():
        if sys.byteorder == "little":
            arr = np.frombuffer(data, dtype="<i4")
            return bytes(arr.view(np.uint8).reshape(-1, 4)[:, 1:4].tobytes())
        arr = np.frombuffer(data, dtype=">i4")
        return bytes(arr.view(np.uint8).reshape(-1, 4)[:, 0:3].tobytes())

    if sys.byteorder == "little":
        return b"".join(data[i + 1 : i + 4] for i in range(0, len(data), 4))
    return b"".join(data[i : i + 3] for i in range(0, len(data), 4))


__all__ = [
    "AudioFormat",
    "BufferTracker",
    "_convert_s32_to_s24",
]
