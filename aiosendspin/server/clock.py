"""Clock abstractions for deterministic server timing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol


class Clock(Protocol):
    """Time source used for timestamping in microseconds."""

    def now_us(self) -> int:
        """Return current time in microseconds (monotonic)."""


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
