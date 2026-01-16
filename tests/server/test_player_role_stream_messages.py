"""Tests for PlayerRole stream lifecycle message payloads."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from aiosendspin.models import unpack_binary_header
from aiosendspin.models.core import StreamClearMessage, StreamEndMessage
from aiosendspin.models.types import BinaryMessageType
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


def test_player_role_send_cached_chunk_packs_header_and_tracks_duration() -> None:
    """Catch-up uses role-controlled header packing and accurate duration tracking."""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop.time.return_value = 0.0
    record = PlayerRecord(client_id="p1", loop=loop, buffer_capacity_bytes=100_000)

    class _Tracker:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def register(self, timestamp_us: int, byte_count: int) -> None:
            self.calls.append((timestamp_us, byte_count))

        def reset(self) -> None:
            return

    tracker = _Tracker()
    record._buffer_tracker = tracker  # noqa: SLF001

    sent: list[bytes] = []
    conn = MagicMock()
    conn.queue_high_water = MagicMock(return_value=False)
    conn.try_send_binary = MagicMock(side_effect=lambda data: (sent.append(data), True)[1])

    role = PlayerRole(_record=record, _connection=conn)
    payload = b"\x01\x02\x03"
    timestamp_us = 123_000
    duration_us = 40_000
    byte_count = len(payload)

    assert role.send_cached_chunk(payload, timestamp_us, duration_us, byte_count)
    assert sent, "Expected a binary send"

    header = unpack_binary_header(sent[0])
    assert header.message_type == BinaryMessageType.AUDIO_CHUNK.value
    assert header.timestamp_us == timestamp_us
    assert sent[0][9:] == payload

    assert tracker.calls == [(timestamp_us + duration_us, byte_count)]
