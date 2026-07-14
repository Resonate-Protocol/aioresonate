"""Tests for Metadata.equals progress-drift handling."""

from __future__ import annotations

from aiosendspin.server.roles.metadata.state import Metadata


def test_equals_treats_paused_progress_as_frozen() -> None:
    """While paused (speed 0), unchanged progress across time stays equal (no phantom drift)."""
    first = Metadata(track_progress=42_000, playback_speed=0, timestamp_us=0)
    later = Metadata(track_progress=42_000, playback_speed=0, timestamp_us=2_000_000)

    assert first.equals(later)


def test_equals_detects_drift_at_normal_speed() -> None:
    """At 1x, progress that did not advance with elapsed time is not equal."""
    first = Metadata(track_progress=42_000, playback_speed=1000, timestamp_us=0)
    later = Metadata(track_progress=42_000, playback_speed=1000, timestamp_us=2_000_000)

    assert not first.equals(later)
