"""The connection dispatch forwards client availability to the client state machine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.models.core import (
    ClientHelloMessage,
    ClientHelloPayload,
    ClientStateMessage,
    ClientStatePayload,
)
from aiosendspin.models.management import ManagementResultMessage, ManagementResultPayload
from aiosendspin.models.player import PlayerStatePayload
from aiosendspin.models.types import ManagementResult
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.connection import SendspinConnection


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: Any
    id: str = "srv"
    name: str = "server"


@pytest.mark.asyncio
async def test_available_false_drives_external_source_transition() -> None:
    """A new client's `available: false` must trigger the external-source transition."""
    loop = asyncio.get_running_loop()
    conn = SendspinConnection(
        _DummyServer(loop=loop, clock=LoopClock(loop)), wsock_client=MagicMock()
    )

    client = MagicMock()
    client.available = True
    client.handle_availability_change = AsyncMock()
    client.active_roles = []
    conn._client = client  # noqa: SLF001
    conn._initial_state_received = True  # noqa: SLF001

    await conn._handle_message(  # noqa: SLF001
        ClientStateMessage(payload=ClientStatePayload(available=False)), timestamp_us=0
    )

    client.handle_availability_change.assert_awaited_once_with(available=False)


def _conn_with_client() -> tuple[SendspinConnection, MagicMock]:
    loop = asyncio.get_running_loop()
    conn = SendspinConnection(
        _DummyServer(loop=loop, clock=LoopClock(loop)), wsock_client=MagicMock()
    )
    client = MagicMock()
    conn._client = client  # noqa: SLF001
    return conn, client


@pytest.mark.asyncio
async def test_second_client_hello_is_flagged() -> None:
    """A client/hello after the hello exchange is flagged as non-compliant."""
    conn, client = _conn_with_client()
    await conn._handle_message(  # noqa: SLF001
        ClientHelloMessage(payload=ClientHelloPayload(name="c", supported_roles=[])),
        timestamp_us=0,
    )
    client.flag_noncompliance.assert_called_once()


@pytest.mark.asyncio
async def test_unsolicited_management_result_is_flagged() -> None:
    """A management/result with no request in flight is flagged as non-compliant."""
    conn, client = _conn_with_client()
    conn._management_waiter = None  # noqa: SLF001
    await conn._handle_message(  # noqa: SLF001
        ManagementResultMessage(payload=ManagementResultPayload(result=ManagementResult.OK)),
        timestamp_us=0,
    )
    client.flag_noncompliance.assert_called_once()


def _player_role_mock() -> MagicMock:
    role = MagicMock()
    role.role_family = "player"
    return role


@pytest.mark.asyncio
async def test_initial_state_deviations_flags_incomplete_player_state() -> None:
    """An empty initial client/state for a player is flagged as incomplete."""
    conn, client = _conn_with_client()
    client.active_roles = [_player_role_mock()]
    reasons = conn._initial_state_deviations(ClientStatePayload())  # noqa: SLF001
    assert any("available" in r for r in reasons)
    assert any("player" in r for r in reasons)


@pytest.mark.asyncio
async def test_initial_state_deviations_accepts_complete_player_state() -> None:
    """A complete initial client/state for a player yields no deviations."""
    conn, client = _conn_with_client()
    client.active_roles = [_player_role_mock()]
    payload = ClientStatePayload(
        available=True,
        player=PlayerStatePayload(static_delay_ms=0, required_lead_time_ms=100, min_buffer_ms=200),
    )
    assert conn._initial_state_deviations(payload) == []  # noqa: SLF001
