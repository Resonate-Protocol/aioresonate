"""Tests for VisualizerV1Role draft visualizer implementation."""

from __future__ import annotations

import math
import struct
from unittest.mock import MagicMock

import pytest

from aiosendspin.models.core import StreamClearMessage, StreamEndMessage, StreamStartMessage
from aiosendspin.models.types import BinaryMessageType
from aiosendspin.models.visualizer import (
    ClientHelloVisualizerSpectrum,
    ClientHelloVisualizerSupport,
)
from aiosendspin.server.roles.base import AudioChunk
from aiosendspin.server.roles.visualizer.v1 import VisualizerV1Role


def _make_client_stub() -> MagicMock:
    """Create a mock client for testing."""
    client = MagicMock()
    client.client_id = "client-1"
    client.group = MagicMock()
    client.group.group_role.return_value = None
    client.info = MagicMock()
    client.info.visualizer_support = {
        "types": ["loudness", "f_peak", "spectrum"],
        "buffer_capacity": 65536,
        "batch_max": 8,
        "spectrum": {
            "n_disp_bins": 8,
            "scale": "lin",
            "f_min": 20,
            "f_max": 16_000,
            "rate_max": 60,
        },
    }
    client.send_role_message = MagicMock()
    client.send_binary = MagicMock()
    client._logger = MagicMock()  # noqa: SLF001
    client.connection = MagicMock()
    return client


def _sine_pcm_16bit(*, sample_rate: int, channels: int, hz: float, duration_s: float) -> bytes:
    sample_count = int(sample_rate * duration_s)
    frame_bytes = bytearray()
    for i in range(sample_count):
        value = int(32767 * math.sin((2.0 * math.pi * hz * i) / sample_rate))
        packed = struct.pack("<h", value)
        for _ in range(channels):
            frame_bytes.extend(packed)
    return bytes(frame_bytes)


def test_visualizer_role_has_role_id() -> None:
    """VisualizerV1Role has role_id of 'visualizer@_draft_r1'."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    assert role.role_id == "visualizer@_draft_r1"


def test_visualizer_role_has_role_family() -> None:
    """VisualizerV1Role has role_family of 'visualizer'."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    assert role.role_family == "visualizer"


def test_visualizer_role_requires_client() -> None:
    """VisualizerV1Role raises ValueError if no client provided."""
    with pytest.raises(ValueError, match="requires a client"):
        VisualizerV1Role(client=None)


def test_visualizer_role_on_connect_subscribes_to_group_role() -> None:
    """on_connect() subscribes to VisualizerGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = VisualizerV1Role(client=client)
    role.on_connect()

    client.group.group_role.assert_called_with("visualizer")
    group_role.subscribe.assert_called_once_with(role)


def test_visualizer_role_on_stream_start_sends_stream_start() -> None:
    """on_stream_start() sends stream/start with negotiated config."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    role.on_connect()

    role.on_stream_start()

    client.send_role_message.assert_called()
    _family, message = client.send_role_message.call_args.args
    assert isinstance(message, StreamStartMessage)
    assert message.payload.visualizer is not None
    assert list(message.payload.visualizer.types) == ["loudness", "f_peak", "spectrum"]
    assert message.payload.visualizer.batch_max == 8


def test_visualizer_role_on_connect_does_not_send_stream_start() -> None:
    """on_connect() initializes config but does not send stream/start."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)

    role.on_connect()

    # No stream/start should be sent until on_stream_start is called.
    for call in client.send_role_message.call_args_list:
        assert not isinstance(call.args[1], StreamStartMessage)


def test_visualizer_role_accepts_legacy_support_object() -> None:
    """Legacy visualizer support shape still initializes draft role."""
    client = _make_client_stub()
    client.info.visualizer_support = ClientHelloVisualizerSupport(buffer_capacity=65536)
    role = VisualizerV1Role(client=client)
    role.on_connect()

    role.on_stream_start()

    client.send_role_message.assert_called()
    _family, message = client.send_role_message.call_args.args
    assert isinstance(message, StreamStartMessage)
    assert message.payload.visualizer is not None
    assert list(message.payload.visualizer.types) == ["loudness", "f_peak"]
    assert message.payload.visualizer.batch_max == 8


def test_visualizer_role_preserves_draft_support_fields() -> None:
    """Draft support fields survive model parsing and are used in stream/start."""
    client = _make_client_stub()
    client.info.visualizer_support = ClientHelloVisualizerSupport(
        types=["loudness", "f_peak", "spectrum"],
        buffer_capacity=65536,
        batch_max=6,
        spectrum=ClientHelloVisualizerSpectrum(
            n_disp_bins=16,
            scale="lin",
            f_min=40,
            f_max=14000,
            rate_max=30,
        ),
    )
    role = VisualizerV1Role(client=client)
    role.on_connect()

    role.on_stream_start()

    client.send_role_message.assert_called()
    _family, message = client.send_role_message.call_args.args
    assert isinstance(message, StreamStartMessage)
    assert message.payload.visualizer is not None
    assert list(message.payload.visualizer.types) == ["loudness", "f_peak", "spectrum"]
    assert message.payload.visualizer.batch_max == 6
    assert message.payload.visualizer.spectrum is not None
    assert message.payload.visualizer.spectrum.n_disp_bins == 16


def test_visualizer_role_on_disconnect_unsubscribes_from_group_role() -> None:
    """on_disconnect() unsubscribes from VisualizerGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = VisualizerV1Role(client=client)
    role.on_connect()
    role.on_disconnect()

    group_role.unsubscribe.assert_called_once_with(role)


def test_visualizer_role_has_stream_requirements() -> None:
    """Visualizer role declares stream requirements."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    assert role.get_stream_requirements() is not None


def test_visualizer_role_has_audio_requirements() -> None:
    """Visualizer role consumes audio for feature extraction."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    req = role.get_audio_requirements()
    assert req is not None
    assert req.sample_rate == 48_000
    assert req.bit_depth == 16
    assert req.channels == 2


def test_visualizer_role_emits_binary_visualization_frame() -> None:
    """on_audio_chunk() sends binary type 16 frame."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    role.on_connect()

    pcm = _sine_pcm_16bit(sample_rate=48_000, channels=2, hz=1000.0, duration_s=0.025)
    role.on_audio_chunk(
        AudioChunk(
            data=pcm,
            timestamp_us=1_000_000,
            duration_us=25_000,
            byte_count=len(pcm),
        )
    )

    client.send_binary.assert_called()
    kwargs = client.send_binary.call_args.kwargs
    payload = client.send_binary.call_args.args[0]

    assert kwargs["role_family"] == "visualizer"
    assert kwargs["message_type"] == BinaryMessageType.VISUALIZATION_DATA.value
    assert kwargs["timestamp_us"] == 1_000_000

    assert payload[0] == BinaryMessageType.VISUALIZATION_DATA.value
    assert payload[1] == 1  # frame count
    assert struct.unpack(">q", payload[2:10])[0] == 1_000_000


def test_visualizer_role_on_stream_clear_sends_clear_message() -> None:
    """on_stream_clear() sends stream/clear for visualizer role."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    role.on_connect()
    role.on_stream_clear()

    _family, message = client.send_role_message.call_args.args
    assert isinstance(message, StreamClearMessage)
    assert message.payload.roles == ["visualizer"]


def test_visualizer_role_on_stream_end_sends_end_message() -> None:
    """on_stream_end() sends stream/end for visualizer role."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    role.on_connect()
    role.on_stream_end()

    _family, message = client.send_role_message.call_args.args
    assert isinstance(message, StreamEndMessage)
    assert message.payload.roles == ["visualizer"]


def test_visualizer_role_stream_start_is_resent_after_stream_end() -> None:
    """A new stream/start must be sent again after stream/end."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    role.on_connect()

    role.on_stream_start()
    role.on_stream_end()
    role.on_stream_start()

    stream_start_calls = [
        call
        for call in client.send_role_message.call_args_list
        if isinstance(call.args[1], StreamStartMessage)
    ]
    assert len(stream_start_calls) == 2
