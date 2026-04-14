"""Tests for clock implementations."""

from itertools import pairwise

from aiosendspin.server.clock import RawMonotonicClock


def test_raw_monotonic_clock_returns_int() -> None:
    """RawMonotonicClock.now_us() returns an int."""
    clock = RawMonotonicClock()
    assert isinstance(clock.now_us(), int)


def test_raw_monotonic_clock_is_non_decreasing() -> None:
    """Successive reads of RawMonotonicClock are non-decreasing."""
    clock = RawMonotonicClock()
    samples = [clock.now_us() for _ in range(1000)]
    for a, b in pairwise(samples):
        assert b >= a
