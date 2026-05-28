"""Tests for strict client-side visualizer payload parsing (v1 wire)."""

from __future__ import annotations

import struct

import pytest

from aiosendspin.client.client import SendspinClient
from aiosendspin.models.types import BinaryMessageType, Roles
from aiosendspin.models.visualizer import (
    ClientHelloVisualizerSpectrum,
    ClientHelloVisualizerSupport,
    StreamStartVisualizer,
    VisualizerFrame,
)


def _basic_config(
    *,
    types: tuple[str, ...] = ("loudness",),
    n_disp_bins: int = 8,
) -> StreamStartVisualizer:
    spectrum: ClientHelloVisualizerSpectrum | None = None
    if "spectrum" in types:
        spectrum = ClientHelloVisualizerSpectrum(
            n_disp_bins=n_disp_bins, scale="lin", f_min=20, f_max=16_000
        )
    # The parser only consults `types` + spectrum metadata to validate frame
    # widths, so the rest of the config can stay at defaults.
    return StreamStartVisualizer(types=types, rate_max=30, spectrum=spectrum)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# loudness (msg 16)
# ---------------------------------------------------------------------------


def test_parse_loudness_frame() -> None:
    """Parse loudness frame."""
    payload = struct.pack(">q", 1_234_000) + struct.pack(">H", 12345)
    cfg = _basic_config()
    frame = SendspinClient._parse_visualization_frame(  # noqa: SLF001
        BinaryMessageType.VISUALIZATION_LOUDNESS, payload, cfg
    )
    assert frame is not None
    assert frame.timestamp_us == 1_234_000
    assert frame.loudness == 12345


def test_parse_loudness_frame_rejects_wrong_length() -> None:
    """Parse loudness frame rejects wrong length."""
    cfg = _basic_config()
    bad = struct.pack(">q", 1) + b"\x00"  # only 1 byte of data instead of 2
    assert (
        SendspinClient._parse_visualization_frame(  # noqa: SLF001
            BinaryMessageType.VISUALIZATION_LOUDNESS, bad, cfg
        )
        is None
    )


# ---------------------------------------------------------------------------
# f_peak (msg 18)
# ---------------------------------------------------------------------------


def test_parse_f_peak_frame() -> None:
    """Parse f peak frame."""
    payload = struct.pack(">q", 100) + struct.pack(">HH", 1024, 0x4000)
    cfg = _basic_config(types=("f_peak",))
    frame = SendspinClient._parse_visualization_frame(  # noqa: SLF001
        BinaryMessageType.VISUALIZATION_F_PEAK, payload, cfg
    )
    assert frame is not None
    assert frame.f_peak_freq == 1024
    assert frame.f_peak_amp == 0x4000


def test_parse_f_peak_rejects_wrong_length() -> None:
    """Parse f peak rejects wrong length."""
    cfg = _basic_config(types=("f_peak",))
    bad = struct.pack(">q", 1) + struct.pack(">H", 100)  # missing amp
    assert (
        SendspinClient._parse_visualization_frame(  # noqa: SLF001
            BinaryMessageType.VISUALIZATION_F_PEAK, bad, cfg
        )
        is None
    )


# ---------------------------------------------------------------------------
# spectrum (msg 19)
# ---------------------------------------------------------------------------


def test_parse_spectrum_frame() -> None:
    """Parse spectrum frame."""
    bins = list(range(8))
    payload = struct.pack(">q", 42) + struct.pack(">8H", *bins)
    cfg = _basic_config(types=("spectrum",), n_disp_bins=8)
    frame = SendspinClient._parse_visualization_frame(  # noqa: SLF001
        BinaryMessageType.VISUALIZATION_SPECTRUM, payload, cfg
    )
    assert frame is not None
    assert frame.spectrum == bins


def test_parse_spectrum_rejects_wrong_bin_count() -> None:
    """Parse spectrum rejects wrong bin count."""
    payload = struct.pack(">q", 0) + struct.pack(">4H", 1, 2, 3, 4)
    cfg = _basic_config(types=("spectrum",), n_disp_bins=8)
    assert (
        SendspinClient._parse_visualization_frame(  # noqa: SLF001
            BinaryMessageType.VISUALIZATION_SPECTRUM, payload, cfg
        )
        is None
    )


# ---------------------------------------------------------------------------
# peak (msg 20)
# ---------------------------------------------------------------------------


def test_parse_peak_frame() -> None:
    """Parse peak frame."""
    payload = struct.pack(">q", 99) + bytes([0xC8])
    cfg = _basic_config(types=("peak",))
    frame = SendspinClient._parse_visualization_frame(  # noqa: SLF001
        BinaryMessageType.VISUALIZATION_PEAK, payload, cfg
    )
    assert frame is not None
    assert frame.peak_strength == 0xC8


# ---------------------------------------------------------------------------
# pitch (msg 21)
# ---------------------------------------------------------------------------


def test_parse_pitch_frame() -> None:
    """Parse pitch frame."""
    # A4 = MIDI 69 → 0x4500. Confidence 200.
    payload = struct.pack(">q", 1) + struct.pack(">H", 0x4500) + bytes([200])
    cfg = _basic_config(types=("pitch",))
    frame = SendspinClient._parse_visualization_frame(  # noqa: SLF001
        BinaryMessageType.VISUALIZATION_PITCH, payload, cfg
    )
    assert frame is not None
    assert frame.pitch_midi_q88 == 0x4500
    assert frame.pitch_confidence == 200


def test_parse_pitch_rejects_wrong_length() -> None:
    """Parse pitch rejects wrong length."""
    cfg = _basic_config(types=("pitch",))
    bad = struct.pack(">q", 1) + struct.pack(">H", 0x4500)  # missing confidence byte
    assert (
        SendspinClient._parse_visualization_frame(  # noqa: SLF001
            BinaryMessageType.VISUALIZATION_PITCH, bad, cfg
        )
        is None
    )


# ---------------------------------------------------------------------------
# beat (msg 17) — delivered through the visualizer callback
# ---------------------------------------------------------------------------


def _client_with_visualizer_callback() -> tuple[SendspinClient, list[VisualizerFrame]]:
    support = ClientHelloVisualizerSupport(
        types=["loudness", "beat"], buffer_capacity=65536, rate_max=30
    )
    client = SendspinClient(
        client_id="x",
        client_name="x",
        roles=[Roles.VISUALIZER],
        visualizer_support=support,
    )
    received: list[VisualizerFrame] = []

    def _cb(frames: list[VisualizerFrame]) -> None:
        received.extend(frames)

    client.add_visualizer_listener(_cb)
    return client, received


@pytest.mark.asyncio
async def test_handle_beat_dispatches_downbeat_frame() -> None:
    """Downbeat byte sets is_downbeat=True on the dispatched frame."""
    client, received = _client_with_visualizer_callback()
    body = struct.pack(">q", 100) + bytes([0b0000_0001])
    client._handle_visualization_beat(body)  # noqa: SLF001
    assert len(received) == 1
    assert received[0].timestamp_us == 100
    assert received[0].is_downbeat is True


@pytest.mark.asyncio
async def test_handle_beat_dispatches_regular_frame() -> None:
    """Flags=0 dispatches is_downbeat=False."""
    client, received = _client_with_visualizer_callback()
    body = struct.pack(">q", 200) + bytes([0])
    client._handle_visualization_beat(body)  # noqa: SLF001
    assert len(received) == 1
    assert received[0].timestamp_us == 200
    assert received[0].is_downbeat is False


@pytest.mark.asyncio
async def test_handle_beat_rejects_wrong_length() -> None:
    """Body without the trailing flag byte is dropped silently."""
    client, received = _client_with_visualizer_callback()
    client._handle_visualization_beat(struct.pack(">q", 100))  # noqa: SLF001
    assert received == []


@pytest.mark.asyncio
async def test_handle_beat_rejects_empty_payload() -> None:
    """Empty body is dropped silently."""
    client, received = _client_with_visualizer_callback()
    client._handle_visualization_beat(b"")  # noqa: SLF001
    assert received == []


# ---------------------------------------------------------------------------
# Truncated header
# ---------------------------------------------------------------------------


def test_parse_visualization_frame_rejects_truncated_header() -> None:
    """Parse visualization frame rejects truncated header."""
    cfg = _basic_config()
    assert (
        SendspinClient._parse_visualization_frame(  # noqa: SLF001
            BinaryMessageType.VISUALIZATION_LOUDNESS, b"\x00\x01", cfg
        )
        is None
    )
