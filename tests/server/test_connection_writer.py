"""Tests for SendspinConnection writer task behavior."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Never
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.models.core import ServerTimeMessage, ServerTimePayload
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.connection import SendspinConnection


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: LoopClock
    id: str = "srv"
    name: str = "server"

    def get_or_create_client(self, client_id: str) -> Never:
        raise AssertionError(f"unexpected get_or_create_client({client_id}) in this test")


@pytest.mark.asyncio
async def test_server_initiated_connection_starts_writer_task() -> None:
    """Server-initiated connections must start a writer task so enqueued messages are sent."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    wsock = MagicMock()
    wsock.closed = False
    wsock.send_str = AsyncMock()
    wsock.send_bytes = AsyncMock()

    conn = SendspinConnection(server, wsock_client=wsock)
    await conn._setup_connection()  # noqa: SLF001
    assert conn._writer_task is not None  # noqa: SLF001

    conn.send_message(
        ServerTimeMessage(
            payload=ServerTimePayload(
                client_transmitted=1,
                server_received=2,
                server_transmitted=3,
            )
        )
    )

    for _ in range(50):
        if wsock.send_str.called:
            break
        await asyncio.sleep(0)

    assert wsock.send_str.call_count == 1

    await conn.disconnect(retry_connection=False)
