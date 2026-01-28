"""Tests for persistent SendspinClient device state across reconnects."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from aiosendspin.models.core import ClientHelloPayload
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, GoodbyeReason, PlayerCommand, Roles
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.roles.player.v1 import PlayerPersistentState


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: LoopClock
    id: str = "srv"
    name: str = "server"


class _DummyConnection:
    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:  # noqa: ARG002
        return

    def try_send_binary(
        self,
        data: bytes,  # noqa: ARG002
        *,
        role_family: str,  # noqa: ARG002
        buffer_end_time_us: int | None = None,  # noqa: ARG002
        buffer_byte_count: int | None = None,  # noqa: ARG002
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


@pytest.mark.asyncio
async def test_goodbye_disconnect_resets_buffer_tracker() -> None:
    """client/goodbye disconnect resets BufferTracker immediately."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = SendspinClient(server, client_id="player-1")
    SendspinGroup(server, client)

    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    state = client.get_role_state("player", PlayerPersistentState)
    assert state is not None
    assert state.buffer_tracker is not None
    state.buffer_tracker.register(end_time_us=1_000_000, byte_count=1234)
    assert state.buffer_tracker.buffered_bytes == 1234

    client.detach_connection(GoodbyeReason.USER_REQUEST)
    assert state.buffer_tracker.buffered_bytes == 0


@pytest.mark.asyncio
async def test_ungraceful_disconnect_delays_buffer_tracker_reset() -> None:
    """Ungraceful disconnect delays BufferTracker reset to tolerate brief blips."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = SendspinClient(server, client_id="player-1")
    SendspinGroup(server, client)

    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    state = client.get_role_state("player", PlayerPersistentState)
    assert state is not None
    assert state.buffer_tracker is not None
    state.buffer_tracker.register(end_time_us=1_000_000, byte_count=1234)
    client.detach_connection(None)

    assert state.buffer_tracker.buffered_bytes == 1234
    await asyncio.sleep(2.2)
    assert state.buffer_tracker.buffered_bytes == 0


@pytest.mark.asyncio
async def test_reconnect_cancels_scheduled_buffer_tracker_reset() -> None:
    """Reconnect cancels the pending delayed BufferTracker reset."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = SendspinClient(server, client_id="player-1")
    SendspinGroup(server, client)

    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    state = client.get_role_state("player", PlayerPersistentState)
    assert state is not None
    assert state.buffer_tracker is not None
    state.buffer_tracker.register(end_time_us=1_000_000, byte_count=1234)
    client.detach_connection(None)

    await asyncio.sleep(1.0)
    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    await asyncio.sleep(1.5)
    assert state.buffer_tracker.buffered_bytes == 1234
