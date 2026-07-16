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
from aiosendspin.server.compliance import ClientComplianceError
from aiosendspin.server.connection import SendspinConnection


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: Any
    id: str = "srv"
    name: str = "server"

    def on_client_first_connect(self, client_id: str) -> None:
        """No-op: the dispatch tests don't exercise first-connect side effects."""


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


def _role_mock(deviations: list[str]) -> MagicMock:
    role = MagicMock()
    role.initial_state_deviations.return_value = deviations
    return role


@pytest.mark.asyncio
async def test_initial_state_flags_missing_available_and_role_reasons() -> None:
    """Missing `available` plus each active role's own deviations are each flagged."""
    conn, client = _conn_with_client()
    client.active_roles = [_role_mock(["role-specific problem"])]
    conn._flag_initial_state_deviations(ClientStatePayload())  # noqa: SLF001
    flagged = [call.args[0] for call in client.flag_noncompliance.call_args_list]
    assert any("available" in r for r in flagged)
    assert any("role-specific problem" in r for r in flagged)


@pytest.mark.asyncio
async def test_initial_state_complete_state_is_not_flagged() -> None:
    """A complete initial client/state with compliant roles is not flagged."""
    conn, client = _conn_with_client()
    client.active_roles = [_role_mock([])]
    conn._flag_initial_state_deviations(ClientStatePayload(available=True))  # noqa: SLF001
    client.flag_noncompliance.assert_not_called()


@pytest.mark.asyncio
async def test_missing_initial_state_rejects_when_flagged() -> None:
    """A never-sent initial state hard-disconnects when the flag raises."""
    conn, client = _conn_with_client()
    client.flag_noncompliance.side_effect = ClientComplianceError("nope")
    conn.disconnect = AsyncMock()  # type: ignore[method-assign]
    conn._initial_state_timeout_callback()  # noqa: SLF001
    await asyncio.sleep(0)
    conn.disconnect.assert_awaited_once_with(retry_connection=False)
    client.mark_connected.assert_not_called()


@pytest.mark.asyncio
async def test_missing_initial_state_marks_connected_when_lenient() -> None:
    """A never-sent initial state is tolerated: the client is marked connected."""
    conn, client = _conn_with_client()
    conn._initial_state_timeout_callback()  # noqa: SLF001
    assert conn._initial_state_received is True  # noqa: SLF001
    client.mark_connected.assert_called_once()


def _role(family: str) -> MagicMock:
    role = MagicMock()
    role.role_family = family
    return role


@pytest.mark.asyncio
async def test_client_state_player_object_for_inactive_role_is_flagged() -> None:
    """A player state object with no active player role is flagged."""
    conn, client = _conn_with_client()
    client.active_roles = [_role("controller")]
    conn._initial_state_received = True  # noqa: SLF001
    client.available = None
    await conn._handle_message(  # noqa: SLF001
        ClientStateMessage(payload=ClientStatePayload(player=PlayerStatePayload())), timestamp_us=0
    )
    flagged = [call.args[0] for call in client.flag_noncompliance.call_args_list]
    assert any("player" in r and "inactive role" in r for r in flagged)


@pytest.mark.asyncio
async def test_client_state_player_object_for_active_role_is_not_flagged() -> None:
    """A player state object with an active player role is not flagged."""
    conn, client = _conn_with_client()
    client.active_roles = [_role("player")]
    conn._initial_state_received = True  # noqa: SLF001
    client.available = None
    await conn._handle_message(  # noqa: SLF001
        ClientStateMessage(payload=ClientStatePayload(player=PlayerStatePayload())), timestamp_us=0
    )
    client.flag_noncompliance.assert_not_called()
