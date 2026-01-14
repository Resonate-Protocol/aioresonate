"""Tests for BufferTracker backpressure tracking."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.server.stream import BufferTracker


class TestBufferTrackerCapacity:
    """Tests for capacity tracking behavior."""

    def test_empty_tracker_has_capacity_for_any_reasonable_chunk(
        self, mock_loop: MagicMock
    ) -> None:
        """Empty tracker should have capacity for any chunk smaller than buffer size."""
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100_000,
        )
        assert tracker.has_capacity_now(50_000) is True
        assert tracker.has_capacity_now(1) is True
        assert tracker.has_capacity_now(99_999) is True

    def test_has_capacity_returns_true_for_zero_bytes(self, mock_loop: MagicMock) -> None:
        """Zero bytes should always have capacity."""
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100_000,
        )
        assert tracker.has_capacity_now(0) is True

    def test_has_capacity_returns_true_for_oversized_chunk(self, mock_loop: MagicMock) -> None:
        """Oversized chunks are allowed through with a warning."""
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100,
        )
        # Chunk larger than capacity should still return True (allowed through)
        assert tracker.has_capacity_now(200) is True


class TestBufferTrackerRegister:
    """Tests for register() behavior."""

    def test_register_tracks_buffered_bytes(self, mock_loop: MagicMock) -> None:
        """Registered bytes should be tracked in buffered_bytes."""
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100_000,
        )
        tracker.register(end_time_us=1_000_000, byte_count=10_000)
        assert tracker.buffered_bytes == 10_000

    def test_register_multiple_chunks_accumulates_bytes(self, mock_loop: MagicMock) -> None:
        """Multiple registrations should accumulate."""
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100_000,
        )
        tracker.register(end_time_us=1_000_000, byte_count=10_000)
        tracker.register(end_time_us=2_000_000, byte_count=15_000)
        tracker.register(end_time_us=3_000_000, byte_count=5_000)
        assert tracker.buffered_bytes == 30_000

    def test_register_zero_bytes_is_ignored(self, mock_loop: MagicMock) -> None:
        """Registering zero bytes should have no effect."""
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100_000,
        )
        tracker.register(end_time_us=1_000_000, byte_count=0)
        assert tracker.buffered_bytes == 0
        assert len(tracker.buffered_chunks) == 0


class TestBufferTrackerPrune:
    """Tests for prune_consumed() behavior."""

    def test_prune_removes_chunks_past_end_time(self, mock_loop: MagicMock) -> None:
        """Chunks with end_time <= now should be removed."""
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100_000,
        )
        # Register chunks ending at 1s, 2s, 3s
        tracker.register(end_time_us=1_000_000, byte_count=10_000)
        tracker.register(end_time_us=2_000_000, byte_count=10_000)
        tracker.register(end_time_us=3_000_000, byte_count=10_000)
        assert tracker.buffered_bytes == 30_000

        # Prune at 1.5s - should remove first chunk
        tracker.prune_consumed(now_us=1_500_000)
        assert tracker.buffered_bytes == 20_000
        assert len(tracker.buffered_chunks) == 2

    def test_prune_uses_loop_time_when_now_not_provided(self, mock_loop: MagicMock) -> None:
        """When now_us is None, should use loop.time()."""
        mock_loop.time.return_value = 1.5  # 1.5 seconds = 1_500_000 us
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100_000,
        )
        tracker.register(end_time_us=1_000_000, byte_count=10_000)
        tracker.register(end_time_us=2_000_000, byte_count=10_000)

        tracker.prune_consumed()
        assert tracker.buffered_bytes == 10_000

    def test_prune_removes_all_past_chunks(self, mock_loop: MagicMock) -> None:
        """Should remove all chunks when time advances past all of them."""
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100_000,
        )
        tracker.register(end_time_us=1_000_000, byte_count=10_000)
        tracker.register(end_time_us=2_000_000, byte_count=10_000)

        tracker.prune_consumed(now_us=5_000_000)
        assert tracker.buffered_bytes == 0
        assert len(tracker.buffered_chunks) == 0


class TestBufferTrackerTimeUntilCapacity:
    """Tests for time_until_capacity() behavior."""

    def test_returns_zero_when_has_capacity(self, mock_loop: MagicMock) -> None:
        """Should return 0 when buffer has room."""
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100_000,
        )
        assert tracker.time_until_capacity(50_000) == 0

    def test_returns_zero_for_zero_bytes(self, mock_loop: MagicMock) -> None:
        """Should return 0 for zero bytes needed."""
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100_000,
        )
        tracker.register(end_time_us=1_000_000, byte_count=100_000)
        assert tracker.time_until_capacity(0) == 0

    def test_returns_zero_for_oversized_chunk(self, mock_loop: MagicMock) -> None:
        """Oversized chunks should return 0 (allowed through)."""
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100,
        )
        assert tracker.time_until_capacity(200) == 0

    def test_calculates_wait_time_when_buffer_full(self, mock_loop: MagicMock) -> None:
        """Should calculate time until capacity is available."""
        mock_loop.time.return_value = 0.0
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100_000,
        )
        # Fill buffer to capacity with chunk ending at 1s
        tracker.register(end_time_us=1_000_000, byte_count=100_000)

        # Need to wait 1s for this chunk to be consumed
        wait_time = tracker.time_until_capacity(10_000)
        assert wait_time == 1_000_000  # 1 second in microseconds

    def test_calculates_partial_wait_time(self, mock_loop: MagicMock) -> None:
        """Should calculate time based on when space becomes available."""
        mock_loop.time.return_value = 0.0
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100_000,
        )
        # Register two chunks: 60k ending at 1s, 30k ending at 2s
        tracker.register(end_time_us=1_000_000, byte_count=60_000)
        tracker.register(end_time_us=2_000_000, byte_count=30_000)
        assert tracker.buffered_bytes == 90_000

        # Need 15k more - after first chunk plays (at 1s), we have 40k space
        wait_time = tracker.time_until_capacity(15_000)
        assert wait_time == 1_000_000  # Wait for first chunk to finish


class TestBufferTrackerReset:
    """Tests for reset() behavior."""

    def test_reset_clears_all_chunks(self, mock_loop: MagicMock) -> None:
        """Reset should clear all tracked chunks."""
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100_000,
        )
        tracker.register(end_time_us=1_000_000, byte_count=10_000)
        tracker.register(end_time_us=2_000_000, byte_count=10_000)

        tracker.reset()

        assert tracker.buffered_bytes == 0
        assert len(tracker.buffered_chunks) == 0

    def test_reset_allows_fresh_registrations(self, mock_loop: MagicMock) -> None:
        """After reset, should work normally for new registrations."""
        tracker = BufferTracker(
            loop=mock_loop,
            client_id="test-client",
            capacity_bytes=100_000,
        )
        tracker.register(end_time_us=1_000_000, byte_count=50_000)
        tracker.reset()
        tracker.register(end_time_us=2_000_000, byte_count=25_000)

        assert tracker.buffered_bytes == 25_000
        assert len(tracker.buffered_chunks) == 1
