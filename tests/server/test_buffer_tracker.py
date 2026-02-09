"""Tests for BufferTracker duration tracking."""

from __future__ import annotations

from aiosendspin.server.audio import BufferTracker


class _FakeClock:
    """Fake clock for testing."""

    def __init__(self, now_us: int = 0) -> None:
        self._now_us = now_us

    def now_us(self) -> int:
        return self._now_us

    def set_now(self, now_us: int) -> None:
        self._now_us = now_us


def test_buffer_tracker_tracks_duration() -> None:
    """BufferTracker should track duration when registering chunks."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,  # 1 second
    )

    # Register a chunk with duration
    tracker.register(end_time_us=100_000, byte_count=1000, duration_us=100_000)

    assert tracker.buffered_bytes == 1000
    assert tracker.buffered_duration_us == 100_000


def test_buffer_tracker_prune_removes_duration() -> None:
    """prune_consumed() should remove duration from consumed chunks."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,
    )

    tracker.register(end_time_us=100_000, byte_count=1000, duration_us=100_000)
    tracker.register(end_time_us=200_000, byte_count=1000, duration_us=100_000)

    assert tracker.buffered_duration_us == 200_000

    # Advance time past first chunk
    clock.set_now(150_000)
    tracker.prune_consumed()

    assert tracker.buffered_bytes == 1000
    assert tracker.buffered_duration_us == 100_000


def test_has_duration_capacity_when_not_configured() -> None:
    """has_duration_capacity() returns True when max_duration_us is 0."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        # max_duration_us defaults to 0
    )

    # Should always return True when duration tracking not configured
    assert tracker.has_duration_capacity(1_000_000_000) is True


def test_has_duration_capacity_with_space() -> None:
    """has_duration_capacity() returns True when buffer has space."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,  # 1 second max
    )

    tracker.register(end_time_us=500_000, byte_count=1000, duration_us=500_000)

    # Has space for another 400ms
    assert tracker.has_duration_capacity(400_000) is True


def test_has_duration_capacity_full() -> None:
    """has_duration_capacity() returns False when buffer is full."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,  # 1 second max
    )

    tracker.register(end_time_us=800_000, byte_count=1000, duration_us=800_000)

    # No space for another 300ms (800ms + 300ms > 1000ms)
    assert tracker.has_duration_capacity(300_000) is False


def test_reset_clears_duration() -> None:
    """reset() should clear buffered_duration_us."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,
    )

    tracker.register(end_time_us=100_000, byte_count=1000, duration_us=100_000)
    tracker.reset()

    assert tracker.buffered_bytes == 0
    assert tracker.buffered_duration_us == 0


def test_buffered_chunk_includes_duration() -> None:
    """BufferedChunk should store duration_us."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
    )

    tracker.register(end_time_us=100_000, byte_count=1000, duration_us=50_000)

    chunk = tracker.buffered_chunks[0]
    assert chunk.end_time_us == 100_000
    assert chunk.byte_count == 1000
    assert chunk.duration_us == 50_000


def test_time_until_duration_capacity_when_not_configured() -> None:
    """time_until_duration_capacity() returns 0 when max_duration_us is 0."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        # max_duration_us defaults to 0
    )

    # Should return 0 when duration tracking not configured
    assert tracker.time_until_duration_capacity(1_000_000) == 0


def test_time_until_duration_capacity_with_space() -> None:
    """time_until_duration_capacity() returns 0 when buffer has space."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,  # 1 second max
    )

    tracker.register(end_time_us=500_000, byte_count=1000, duration_us=500_000)

    # Has space for another 400ms, no wait needed
    assert tracker.time_until_duration_capacity(400_000) == 0


def test_time_until_duration_capacity_returns_excess() -> None:
    """time_until_duration_capacity() returns excess duration when full."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,  # 1 second max
    )

    tracker.register(end_time_us=800_000, byte_count=1000, duration_us=800_000)

    # Need 300ms more, but only 200ms capacity → wait 100ms
    # (800ms + 300ms) - 1000ms = 100ms
    assert tracker.time_until_duration_capacity(300_000) == 100_000


def test_time_until_duration_capacity_prunes_first() -> None:
    """time_until_duration_capacity() prunes consumed chunks before checking."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
        max_duration_us=1_000_000,  # 1 second max
    )

    tracker.register(end_time_us=500_000, byte_count=1000, duration_us=500_000)
    tracker.register(end_time_us=1_000_000, byte_count=1000, duration_us=500_000)

    # Buffer is full (1000ms), but advance time to consume first chunk
    clock.set_now(600_000)

    # Now only 500ms buffered, should have space for 400ms
    assert tracker.time_until_duration_capacity(400_000) == 0


def test_buffered_horizon_us_tracks_furthest_end_from_now() -> None:
    """buffered_horizon_us() should report furthest scheduled end minus now."""
    clock = _FakeClock(now_us=0)
    tracker = BufferTracker(
        clock=clock,
        client_id="test",
        capacity_bytes=10000,
    )

    tracker.register(end_time_us=200_000, byte_count=1000, duration_us=100_000)
    tracker.register(end_time_us=500_000, byte_count=1000, duration_us=100_000)

    assert tracker.buffered_horizon_us() == 500_000

    clock.set_now(250_000)
    # First chunk is pruned; horizon is from now to the second chunk's end.
    assert tracker.buffered_horizon_us() == 250_000
