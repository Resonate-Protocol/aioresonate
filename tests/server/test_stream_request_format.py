"""Tests for stream/request-format behavior in the presence of an active PushStream."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.models.core import (
    StreamClearMessage,
    StreamRequestFormatPayload,
    StreamStartMessage,
)
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    StreamRequestFormatPlayer,
    SupportedAudioFormat,
)
from aiosendspin.models.types import AudioCodec, Roles
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.roles.player.v1 import PlayerV1Role


class _FakeConnection:
    def __init__(self) -> None:
        self.sent: list[object] = []

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:
        self.sent.append(message)

    def try_send_binary(
        self,
        data: bytes,  # noqa: ARG002
        *,
        role_family: str,  # noqa: ARG002
        timestamp_us: int,  # noqa: ARG002
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,  # noqa: ARG002
        buffer_byte_count: int | None = None,  # noqa: ARG002
    ) -> bool:
        return True


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
    server.clock = LoopClock(mock_loop)
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


def test_player_format_request_defers_stream_start_when_stream_active(
    mock_server: MagicMock,
) -> None:
    """When a PushStream is active, stream/start is deferred (via _pending_stream_start).

    No immediate stream/start or stream/clear is sent. The requirements
    are rebuilt with the new format.
    """
    owner = MagicMock()
    owner.client_id = "owner"
    owner.name = "owner"
    owner.check_role.return_value = False
    owner.group = MagicMock()
    owner.group.stop = AsyncMock()

    group = SendspinGroup(mock_server, owner)
    group.start_stream()

    client, conn = _make_player_client(mock_server, "p1")

    request = StreamRequestFormatPayload(
        player=StreamRequestFormatPlayer(
            codec=AudioCodec.FLAC, sample_rate=48000, channels=2, bit_depth=16
        )
    )

    for role in client.active_roles:
        role.on_stream_request_format(request, stream_active=group.has_active_stream)

    # No immediate stream/start or stream/clear should be sent.
    assert not any(isinstance(msg, StreamStartMessage) for msg in conn.sent)
    assert not any(isinstance(msg, StreamClearMessage) for msg in conn.sent)

    # _pending_stream_start should be set (deferred until first audio chunk).
    player_role = client.role("player@v1")
    assert isinstance(player_role, PlayerV1Role)
    assert player_role._pending_stream_start is True  # noqa: SLF001

    # AudioRequirements should be rebuilt with the new format.
    req = player_role.get_audio_requirements()
    assert req is not None
    assert req.sample_rate == 48000
    assert req.bit_depth == 16
    assert req.channels == 2


def test_player_format_request_defers_stream_start_when_no_stream_active(
    mock_server: MagicMock,
) -> None:
    """When no PushStream is active, stream/start is also deferred via _pending_stream_start."""
    owner = MagicMock()
    owner.client_id = "owner"
    owner.name = "owner"
    owner.check_role.return_value = False
    owner.group = MagicMock()
    owner.group.stop = AsyncMock()

    group = SendspinGroup(mock_server, owner)

    client, conn = _make_player_client(mock_server, "p1")
    request = StreamRequestFormatPayload(
        player=StreamRequestFormatPlayer(
            codec=AudioCodec.FLAC, sample_rate=48000, channels=2, bit_depth=16
        )
    )

    for role in client.active_roles:
        role.on_stream_request_format(request, stream_active=group.has_active_stream)

    # No immediate stream/start (deferred until first audio chunk).
    assert not any(isinstance(msg, StreamStartMessage) for msg in conn.sent)

    # _pending_stream_start should be set.
    player_role = client.role("player@v1")
    assert isinstance(player_role, PlayerV1Role)
    assert player_role._pending_stream_start is True  # noqa: SLF001
