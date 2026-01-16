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
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.player_state import PlayerRegistry


@pytest.fixture
def mock_loop() -> MagicMock:
    """Mock event loop for deterministic timestamps."""
    loop = MagicMock()
    loop.time.return_value = 1000.0
    return loop


@pytest.fixture
def mock_server(mock_loop: MagicMock) -> MagicMock:
    """Mock server with a PlayerRegistry."""
    server = MagicMock()
    server.loop = mock_loop
    server.player_registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
    return server


def _make_player_client(client_id: str) -> MagicMock:
    client = MagicMock()
    client.client_id = client_id
    client.name = client_id
    client.check_role.side_effect = lambda role: role == Roles.PLAYER
    client.send_message = MagicMock()
    client.disconnect = AsyncMock()

    client.info = MagicMock()
    client.info.player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, sample_rate=48000, bit_depth=16, channels=2),
            SupportedAudioFormat(
                codec=AudioCodec.FLAC, sample_rate=48000, bit_depth=16, channels=2
            ),
        ],
        buffer_capacity=100_000,
        supported_commands=[],
    )
    return client


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

    client = _make_player_client("p1")

    # Register connection-independent player state so PushStream can apply the request
    record = mock_server.player_registry.get_or_create(client.client_id)
    record.connection = client
    record.player_role = MagicMock()

    request = StreamRequestFormatPayload(
        player=StreamRequestFormatPlayer(
            codec=AudioCodec.FLAC, sample_rate=48000, channels=2, bit_depth=16
        )
    )
    await group.handle_stream_format_request(client, request)

    # No immediate stream/start should be sent while streaming is active.
    assert not any(
        isinstance(call.args[0], StreamStartMessage) for call in client.send_message.call_args_list
    )


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

    client = _make_player_client("p1")
    request = StreamRequestFormatPayload(
        player=StreamRequestFormatPlayer(
            codec=AudioCodec.FLAC, sample_rate=48000, channels=2, bit_depth=16
        )
    )
    await group.handle_stream_format_request(client, request)

    assert any(
        isinstance(call.args[0], StreamStartMessage) for call in client.send_message.call_args_list
    )
