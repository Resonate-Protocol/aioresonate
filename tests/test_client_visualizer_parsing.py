"""Tests for strict client-side visualizer payload parsing."""

from __future__ import annotations

import struct

from aiosendspin.client.client import SendspinClient


def _build_payload(
    *,
    frame_count: int,
    timestamp_us: int = 1_000_000,
    loudness: int = 1000,
    f_peak: int = 2000,
) -> bytes:
    payload = bytearray()
    payload.append(frame_count)
    for _ in range(frame_count):
        payload.extend(struct.pack(">q", timestamp_us))
        payload.extend(struct.pack(">H", loudness))
        payload.extend(struct.pack(">H", f_peak))
    return bytes(payload)


def test_parse_visualization_frames_strict_layout() -> None:
    """Parses strict payload format [frame_count, frames...]."""
    payload = _build_payload(frame_count=1)
    frames = SendspinClient._parse_visualization_frames(  # noqa: SLF001
        payload, ["loudness", "f_peak"], 0
    )

    assert len(frames) == 1
    assert frames[0].timestamp_us == 1_000_000
    assert frames[0].loudness == 1000
    assert frames[0].f_peak == 2000


def test_parse_visualization_frames_rejects_legacy_layout_with_leading_type() -> None:
    """Legacy payload [16, frame_count, ...] is rejected in strict mode."""
    strict = _build_payload(frame_count=1)
    legacy = bytes([16]) + strict

    frames = SendspinClient._parse_visualization_frames(  # noqa: SLF001
        legacy, ["loudness", "f_peak"], 0
    )
    assert frames == []


def test_parse_visualization_frames_rejects_incomplete_payload() -> None:
    """Incomplete frame payload is rejected."""
    payload = _build_payload(frame_count=1)[:-1]
    frames = SendspinClient._parse_visualization_frames(  # noqa: SLF001
        payload, ["loudness", "f_peak"], 0
    )
    assert frames == []


def test_parse_visualization_frames_zero_frames_is_empty() -> None:
    """Zero frame payload parses as empty list."""
    frames = SendspinClient._parse_visualization_frames(b"\x00", ["loudness", "f_peak"], 0)  # noqa: SLF001
    assert frames == []
