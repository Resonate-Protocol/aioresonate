"""Tests for PlayerRole stream lifecycle message payloads."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from aiosendspin.models.core import StreamClearMessage, StreamEndMessage
from aiosendspin.server.player_state import PlayerRecord
from aiosendspin.server.roles import PlayerRole


def test_player_role_stream_clear_uses_role_family() -> None:
    """PlayerRole.stream/clear uses unversioned role family."""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop.time.return_value = 0.0
    record = PlayerRecord(client_id="p1", loop=loop, buffer_capacity_bytes=100_000)
    conn = MagicMock()
    conn.send_message = MagicMock()

    role = PlayerRole(_record=record, _connection=conn)
    role.clear_stream()

    msg = conn.send_message.call_args.args[0]
    assert isinstance(msg, StreamClearMessage)
    assert msg.payload.roles == ["player"]


def test_player_role_stream_end_uses_role_family() -> None:
    """PlayerRole.stream/end uses unversioned role family."""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop.time.return_value = 0.0
    record = PlayerRecord(client_id="p1", loop=loop, buffer_capacity_bytes=100_000)
    conn = MagicMock()
    conn.send_message = MagicMock()

    role = PlayerRole(_record=record, _connection=conn)
    role.end_stream()

    msg = conn.send_message.call_args.args[0]
    assert isinstance(msg, StreamEndMessage)
    assert msg.payload.roles == ["player"]
