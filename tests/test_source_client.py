"""Tests for the client SDK source role support."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from aiosendspin.client import SendspinClient
from aiosendspin.models import BINARY_HEADER_SIZE, unpack_binary_header
from aiosendspin.models.core import ServerCommandPayload
from aiosendspin.models.source import (
    ClientHelloSourceSupport,
    ClientStreamStartSource,
    SourceCommandPayload,
    SourceSupportedFormat,
)
from aiosendspin.models.types import AudioCodec, Roles, SourceCommand, SourceSignal


def _source_support() -> ClientHelloSourceSupport:
    return ClientHelloSourceSupport(
        supported_formats=[
            SourceSupportedFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16)
        ]
    )


def _make_source_client() -> SendspinClient:
    client = SendspinClient(
        client_id="src-1",
        client_name="Source",
        roles=[Roles.SOURCE],
        source_support=_source_support(),
    )
    mock_ws = MagicMock()
    mock_ws.closed = False
    client._ws = mock_ws  # noqa: SLF001
    client._connected = True  # noqa: SLF001
    return client


async def test_source_support_required_for_source_role() -> None:
    """Constructing a source client without support raises."""
    with pytest.raises(ValueError, match="source_support is required"):
        SendspinClient(client_id="x", client_name="x", roles=[Roles.SOURCE])


async def test_client_hello_advertises_source_support() -> None:
    """client/hello advertises the source role and its support object."""
    client = _make_source_client()
    hello = client._build_client_hello()  # noqa: SLF001
    assert "source@v1" in hello.payload.supported_roles
    assert hello.payload.source_support is not None
    assert "source@v1_support" in hello.to_json()


async def test_send_client_stream_start_and_end() -> None:
    """client_stream/start and client_stream/end are sent as JSON control messages."""
    client = _make_source_client()
    sent: list[str] = []

    async def _capture(payload: str) -> None:
        sent.append(payload)

    client._send_message = _capture  # noqa: SLF001

    await client.send_client_stream_start(
        ClientStreamStartSource(codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16)
    )
    await client.send_client_stream_end()

    assert json.loads(sent[0])["type"] == "client_stream/start"
    assert json.loads(sent[0])["payload"]["source"]["codec"] == "pcm"
    assert json.loads(sent[1])["type"] == "client_stream/end"


async def test_send_source_state_reports_signal() -> None:
    """send_source_state emits client/state with the source signal."""
    client = _make_source_client()
    sent: list[str] = []

    async def _capture(payload: str) -> None:
        sent.append(payload)

    client._send_message = _capture  # noqa: SLF001
    await client.send_source_state(signal=SourceSignal.PRESENT)

    msg = json.loads(sent[0])
    assert msg["type"] == "client/state"
    assert msg["payload"]["source"]["signal"] == "present"


async def test_send_source_audio_chunk_packs_server_timestamp() -> None:
    """Audio chunks are stamped in server time (no static delay) with binary type 12."""
    client = _make_source_client()
    time_filter = MagicMock()
    time_filter.is_synchronized = True
    time_filter.compute_server_time = lambda t: t + 1000
    client._time_filter = time_filter  # noqa: SLF001

    sent: list[bytes] = []

    async def _capture(data: bytes) -> None:
        sent.append(data)

    client._send_binary = _capture  # noqa: SLF001

    ok = await client.send_source_audio_chunk(b"pcmdata", capture_timestamp_us=5000)
    assert ok is True
    header = unpack_binary_header(sent[0])
    assert header.message_type == 12
    assert header.timestamp_us == 6000  # 5000 + filter offset, no static delay
    assert sent[0][BINARY_HEADER_SIZE:] == b"pcmdata"


async def test_send_source_audio_chunk_skipped_when_not_synchronized() -> None:
    """No audio is sent until time sync has converged (spec requirement)."""
    client = _make_source_client()
    time_filter = MagicMock()
    time_filter.is_synchronized = False
    client._time_filter = time_filter  # noqa: SLF001

    sent: list[bytes] = []

    async def _capture(data: bytes) -> None:
        sent.append(data)

    client._send_binary = _capture  # noqa: SLF001

    ok = await client.send_source_audio_chunk(b"pcmdata", capture_timestamp_us=5000)
    assert ok is False
    assert sent == []


async def test_add_source_command_listener_fires() -> None:
    """A source start/stop command from the server reaches the listener."""
    client = _make_source_client()
    received: list[SourceCommandPayload] = []
    client.add_source_command_listener(received.append)

    client._handle_server_command(  # noqa: SLF001
        ServerCommandPayload(source=SourceCommandPayload(command=SourceCommand.START))
    )

    assert len(received) == 1
    assert received[0].command == SourceCommand.START
