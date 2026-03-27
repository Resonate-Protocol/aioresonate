"""Shared test helpers for visualizer role tests."""

from __future__ import annotations

import math
import struct


def sine_pcm_16bit(*, sample_rate: int, channels: int, hz: float, duration_s: float) -> bytes:
    """Generate a sine wave as little-endian signed 16-bit PCM."""
    sample_count = int(sample_rate * duration_s)
    frame_bytes = bytearray()
    for i in range(sample_count):
        value = int(32767 * math.sin((2.0 * math.pi * hz * i) / sample_rate))
        packed = struct.pack("<h", value)
        for _ in range(channels):
            frame_bytes.extend(packed)
    return bytes(frame_bytes)
