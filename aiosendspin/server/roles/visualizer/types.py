"""Shared visualizer role protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aiosendspin.models.visualizer import BeatAvailability, BeatTiming


@runtime_checkable
class VisualizerRoleProtocol(Protocol):
    """Protocol for visualizer role implementations.

    Defines the hooks that VisualizerGroupRole invokes on subscribed roles
    to drive the beat schedule. Roles that do not negotiate the "beat" type
    may treat both methods as no-ops.
    """

    @property
    def role_id(self) -> str:
        """Return the versioned role identifier."""
        ...

    @property
    def wants_beats(self) -> bool:
        """True if the client negotiated `beat` and beats are not unavailable."""
        ...

    def append_beats(self, beats: list[BeatTiming]) -> None:
        """Extend the pending beat schedule for this role."""
        ...

    def clear_beats(self) -> None:
        """Drop any pending beat schedule for this role."""
        ...

    def set_beat_availability(self, availability: BeatAvailability) -> None:
        """Declare whether beats will arrive for the current source."""
        ...
