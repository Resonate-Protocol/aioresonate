"""Clock abstractions for deterministic server timing."""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import Protocol

if sys.platform == "linux" and hasattr(time, "CLOCK_MONOTONIC_RAW"):
    _RAW_CLOCK_ID = time.CLOCK_MONOTONIC_RAW

    def _raw_now_us() -> int:
        return time.clock_gettime_ns(_RAW_CLOCK_ID) // 1_000
else:

    def _raw_now_us() -> int:
        return time.monotonic_ns() // 1_000


class Clock(Protocol):
    """Time source used for timestamping in microseconds."""

    def now_us(self) -> int:
        """Return current time in microseconds (monotonic)."""


@dataclass(frozen=True, slots=True)
class RawMonotonicClock:
    """Clock backed by CLOCK_MONOTONIC_RAW on Linux, monotonic elsewhere.

    Unlike CLOCK_MONOTONIC, CLOCK_MONOTONIC_RAW is not slewed by NTP/adjtime,
    so elapsed-microsecond measurements reflect hardware ticks. This matters
    for audio sync: NTP slewing poisons the client-side time filter.
    """

    def now_us(self) -> int:
        """Return current raw monotonic time in microseconds."""
        return _raw_now_us()


@dataclass(frozen=True, slots=True)
class LoopClock:
    """Clock implementation backed by an asyncio event loop."""

    loop: asyncio.AbstractEventLoop

    def now_us(self) -> int:
        """Return current loop time in microseconds."""
        return int(self.loop.time() * 1_000_000)


@dataclass(slots=True)
class ManualClock:
    """Manually-advanced clock used in tests."""

    now_us_value: int = 0

    def now_us(self) -> int:
        """Return current manual time in microseconds."""
        return self.now_us_value

    def advance_us(self, delta_us: int) -> None:
        """Advance the clock by delta_us microseconds."""
        self.now_us_value += delta_us
