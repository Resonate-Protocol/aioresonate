"""Tests for SendspinConnection._send_message locking."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.types import Roles

from .conftest import make_sdk_client


@pytest.mark.asyncio
async def test_send_message_raises_when_ws_nulled_while_awaiting_lock() -> None:
    """Disconnecting mid-send raises RuntimeError, not AttributeError on a None ws."""
    client = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER])
    conn = SendspinConnection(client)
    conn._ws = MagicMock(send_str=AsyncMock())  # noqa: SLF001

    await conn._send_lock.acquire()  # noqa: SLF001
    send = asyncio.ensure_future(conn._send_message("payload"))  # noqa: SLF001
    await asyncio.sleep(0)  # let the send block on the lock

    conn._ws = None  # noqa: SLF001  # disconnect() nulls the socket
    conn._send_lock.release()  # noqa: SLF001

    with pytest.raises(RuntimeError, match="not connected"):
        await send
