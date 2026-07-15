"""disconnect() cancels an in-progress pairing task."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

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
async def test_disconnect_cancels_pairing_task() -> None:
    """A pairing task parked on operator input is cancelled when the connection drops."""
    loop = asyncio.get_running_loop()
    conn = SendspinConnection(
        _DummyServer(loop=loop, clock=LoopClock(loop)), wsock_client=MagicMock()
    )
    parked = loop.create_task(asyncio.Event().wait())
    conn._pairing_task = parked  # noqa: SLF001

    await conn.disconnect(retry_connection=False)

    assert parked.cancelled()
