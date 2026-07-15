"""A malformed client/hello is rejected instead of raising out of the connection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.server.clock import LoopClock
from aiosendspin.server.connection import SendspinConnection


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: Any
    id: str = "srv"
    name: str = "server"


@pytest.mark.asyncio
async def test_malformed_client_hello_rejects_without_raising() -> None:
    """Undeserializable hello text disconnects and returns False, it does not raise."""
    loop = asyncio.get_running_loop()
    conn = SendspinConnection(
        _DummyServer(loop=loop, clock=LoopClock(loop)), wsock_client=MagicMock()
    )
    conn.disconnect = AsyncMock()  # type: ignore[method-assign]

    assert await conn._ingest_client_hello("{not json") is False  # noqa: SLF001

    conn.disconnect.assert_awaited_once_with(retry_connection=False)
