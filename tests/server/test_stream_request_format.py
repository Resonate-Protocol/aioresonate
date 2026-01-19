"""Tests for stream/request-format behavior in the presence of an active PushStream."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.models.core import StreamRequestFormatPayload, StreamStartMessage
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    StreamRequestFormatPlayer,
    SupportedAudioFormat,
)
from aiosendspin.models.types import AudioCodec, Roles
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.group import SendspinGroup


class _FakeConnection:
    def __init__(self) -> None:
        self.sent: list[object] = []

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:
        self.sent.append(message)

    def try_send_binary(self, data: bytes) -> bool:  # noqa: ARG002
        return True

    def queue_high_water(self, threshold: float = 0.8) -> bool:  # noqa: ARG002
        return False


@pytest.fixture
def mock_loop() -> MagicMock:
    """Mock event loop for deterministic timestamps."""
    loop = MagicMock()
    loop.time.return_value = 1000.0
    return loop


@pytest.fixture
def mock_server(mock_loop: MagicMock) -> MagicMock:
    """Mock server."""
    server = MagicMock()
    server.loop = mock_loop
    return server


def _make_player_client(
    server: MagicMock,
    client_id: str,
) -> tuple[SendspinClient, _FakeConnection]:
    client = SendspinClient(server, client_id=client_id)
    SendspinGroup(server, client)

    conn = _FakeConnection()
    hello = MagicMock()
    hello.client_id = client_id
    hello.name = client_id
    hello.player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, sample_rate=48000, bit_depth=16, channels=2),
            SupportedAudioFormat(
                codec=AudioCodec.FLAC,
                sample_rate=48000,
                bit_depth=16,
                channels=2,
            ),
        ],
        buffer_capacity=100_000,
        supported_commands=[],
    )
    hello.artwork_support = None
    hello.visualizer_support = None

    client.attach_connection(conn, client_info=hello, active_roles=[Roles.PLAYER.value])
    client.mark_connected()
    return client, conn


@pytest.mark.asyncio
async def test_player_format_request_does_not_send_stream_start_when_stream_active(
    mock_server: MagicMock,
) -> None:
    """
    When a PushStream is active, the server must not send stream/start(new) immediately.

    Doing so is unsafe because old-format audio may still be in flight; binary audio chunks
    do not self-describe their codec, so the client could interpret old bytes as new format.
    """
    owner = MagicMock()
    owner.client_id = "owner"
    owner.name = "owner"
    owner.check_role.return_value = False
    owner.group = MagicMock()
    owner.group.stop = AsyncMock()
    owner.player = None

    group = SendspinGroup(mock_server, owner)
    group.start_stream()

    client, conn = _make_player_client(mock_server, "p1")

    request = StreamRequestFormatPayload(
        player=StreamRequestFormatPlayer(
            codec=AudioCodec.FLAC, sample_rate=48000, channels=2, bit_depth=16
        )
    )
    await group.handle_stream_format_request(client, request)

    # No immediate stream/start should be sent while streaming is active.
    assert not any(isinstance(msg, StreamStartMessage) for msg in conn.sent)


@pytest.mark.asyncio
async def test_player_format_request_sends_stream_start_when_no_stream_active(
    mock_server: MagicMock,
) -> None:
    """When no PushStream is active, stream/request-format should be acked with stream/start."""
    owner = MagicMock()
    owner.client_id = "owner"
    owner.name = "owner"
    owner.check_role.return_value = False
    owner.group = MagicMock()
    owner.group.stop = AsyncMock()
    owner.player = None

    group = SendspinGroup(mock_server, owner)

    client, conn = _make_player_client(mock_server, "p1")
    request = StreamRequestFormatPayload(
        player=StreamRequestFormatPlayer(
            codec=AudioCodec.FLAC, sample_rate=48000, channels=2, bit_depth=16
        )
    )
    await group.handle_stream_format_request(client, request)

    assert any(isinstance(msg, StreamStartMessage) for msg in conn.sent)
