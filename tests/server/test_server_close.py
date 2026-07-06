"""Tests for SendspinServer.close teardown robustness."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.models.core import ClientHelloPayload
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.noise.keys import Identity
from aiosendspin.noise.trust_store import InMemoryServerPairingStore
from aiosendspin.server import SendspinServer


def _make_server() -> SendspinServer:
    loop = asyncio.get_running_loop()
    client_session = MagicMock()
    client_session.closed = True
    client_session.close = AsyncMock()
    return SendspinServer(
        loop=loop,
        identity=Identity.generate(),
        server_name="server",
        client_session=client_session,
        pairing_store=InMemoryServerPairingStore(),
    )


class _DummyConnection:
    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:  # noqa: ARG002
        return

    def send_role_message(self, role: str, message: object) -> None:  # noqa: ARG002
        return

    def send_binary(self, data: bytes, **kwargs: object) -> bool:  # noqa: ARG002
        return True


def _player_hello(client_id: str) -> ClientHelloPayload:
    return ClientHelloPayload(
        client_id=client_id,
        name=client_id,
        version=1,
        supported_roles=[Roles.PLAYER.value],
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16
                )
            ],
            buffer_capacity=100_000,
            supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
        ),
    )


@pytest.mark.asyncio
async def test_close_tolerates_unprepared_pending_connection() -> None:
    """A pending connection not past wsock.prepare() must not abort close()."""
    server = _make_server()
    conn = MagicMock()
    conn.websocket_connection.closed = False
    conn.websocket_connection.close = AsyncMock(side_effect=RuntimeError("Call .prepare() first"))
    server._pending_connections.add(conn)  # noqa: SLF001

    await server.close()  # must not raise


@pytest.mark.asyncio
async def test_close_disarms_registry_timers() -> None:
    """close() cancels reclaim, external-registration, and per-client cleanup timers."""
    server = _make_server()
    server._schedule_reclaim_timeout("a", 100.0)  # noqa: SLF001
    server._schedule_external_registration_timeout("b", 100.0)  # noqa: SLF001

    client = server.get_or_create_client("c")
    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("c"),
        negotiated_roles=[Roles.PLAYER.value],
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()
    client.detach_connection(goodbye_reason=None)
    assert client._cleanup_handle is not None  # noqa: SLF001

    await server.close()

    assert not server._reclaim_timeouts  # noqa: SLF001
    assert not server._external_registration_timeouts  # noqa: SLF001
    assert client._cleanup_handle is None  # noqa: SLF001
