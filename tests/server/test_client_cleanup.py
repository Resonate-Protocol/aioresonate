"""Tests for client cleanup on disconnect."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import pytest

from aiosendspin.models.core import ClientHelloPayload
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, GoodbyeReason, PlayerCommand, Roles
from aiosendspin.server import client as client_module
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.group import SendspinGroup


@dataclass
class _MockServer:
    """Mock server with remove_client tracking."""

    loop: asyncio.AbstractEventLoop
    clock: LoopClock
    id: str = "srv"
    name: str = "server"
    remove_client: AsyncMock = field(default_factory=AsyncMock)


class _DummyConnection:
    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:  # noqa: ARG002
        return

    def send_binary(
        self,
        data: bytes,  # noqa: ARG002
        *,
        role: str,  # noqa: ARG002
        timestamp_us: int,  # noqa: ARG002
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,  # noqa: ARG002
        buffer_byte_count: int | None = None,  # noqa: ARG002
        duration_us: int | None = None,  # noqa: ARG002
    ) -> bool:
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
                    codec=AudioCodec.PCM,
                    channels=2,
                    sample_rate=48000,
                    bit_depth=16,
                )
            ],
            buffer_capacity=100_000,
            supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
        ),
    )


@pytest.fixture
async def mock_server() -> _MockServer:
    """Create a mock server with remove_client tracking."""
    loop = asyncio.get_running_loop()
    return _MockServer(loop=loop, clock=LoopClock(loop))


@pytest.fixture(autouse=True)
def _fast_cleanup_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "CLIENT_CLEANUP_DELAY", 0.2)


@pytest.fixture
async def client(mock_server: _MockServer) -> SendspinClient:
    """Create a connected client attached to the mock server."""
    client = SendspinClient(mock_server, client_id="player-1")
    SendspinGroup(mock_server, client)
    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        GoodbyeReason.SHUTDOWN,
        GoodbyeReason.USER_REQUEST,
    ],
)
async def test_immediate_cleanup_on_explicit_disconnect(
    mock_server: _MockServer, client: SendspinClient, reason: GoodbyeReason
) -> None:
    """Client is removed from registry immediately on SHUTDOWN/USER_REQUEST."""
    client.detach_connection(reason)

    # Allow scheduled callback and resulting task to run
    # (call_soon schedules _do_cleanup, which creates a task)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    mock_server.remove_client.assert_awaited_once_with("player-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        GoodbyeReason.RESTART,
        GoodbyeReason.ANOTHER_SERVER,  # Delayed for multi-server reclaim support
        None,  # Ungraceful disconnect
    ],
)
async def test_delayed_cleanup_on_reconnectable_disconnect(
    mock_server: _MockServer, client: SendspinClient, reason: GoodbyeReason | None
) -> None:
    """Client cleanup is delayed for RESTART, ANOTHER_SERVER, and ungraceful disconnects."""
    client.detach_connection(reason)

    # Allow immediate callbacks to run
    await asyncio.sleep(0)

    # Should not be cleaned up yet
    mock_server.remove_client.assert_not_awaited()

    # Wait for the delayed cleanup
    await asyncio.sleep(client_module.CLIENT_CLEANUP_DELAY + 0.1)

    mock_server.remove_client.assert_awaited_once_with("player-1")


@pytest.mark.asyncio
async def test_cleanup_cancelled_on_reconnect(
    mock_server: _MockServer, client: SendspinClient
) -> None:
    """Pending cleanup is cancelled if client reconnects."""
    client.detach_connection(GoodbyeReason.RESTART)

    # Wait some time but not until cleanup fires
    await asyncio.sleep(client_module.CLIENT_CLEANUP_DELAY / 2)
    mock_server.remove_client.assert_not_awaited()

    # Client reconnects
    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    # Wait past the original cleanup time
    await asyncio.sleep(client_module.CLIENT_CLEANUP_DELAY)

    # Should not have been cleaned up
    mock_server.remove_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_skipped_if_reconnected_before_callback(
    mock_server: _MockServer, client: SendspinClient
) -> None:
    """Cleanup callback is a no-op if client reconnected (double-check via _connected flag)."""
    client.detach_connection(GoodbyeReason.SHUTDOWN)

    # Reconnect immediately (before call_soon callback runs)
    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    # Allow callbacks to run
    await asyncio.sleep(0)

    # Cleanup should have been cancelled by attach_connection
    mock_server.remove_client.assert_not_awaited()
