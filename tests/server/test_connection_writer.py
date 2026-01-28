"""Tests for SendspinConnection writer task behavior."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Never
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.models import pack_binary_header_raw
from aiosendspin.models.core import (
    ServerTimeMessage,
    ServerTimePayload,
)
from aiosendspin.models.types import BinaryMessageType
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.connection import SendspinConnection, _BinaryFrame
from aiosendspin.server.roles.base import BinaryHandling


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: LoopClock
    id: str = "srv"
    name: str = "server"

    def get_or_create_client(self, client_id: str) -> Never:
        raise AssertionError(f"unexpected get_or_create_client({client_id}) in this test")


def test_binary_frame_supports_buffer_registration_metadata() -> None:
    """_BinaryFrame should optionally carry buffer registration info."""
    frame_simple = _BinaryFrame(epoch=1, data=b"test", queued_at_us=0)
    assert frame_simple.buffer_end_time_us is None
    assert frame_simple.buffer_byte_count is None

    frame_with_meta = _BinaryFrame(
        epoch=1,
        data=b"test",
        queued_at_us=0,
        buffer_end_time_us=1_000_000,
        buffer_byte_count=1234,
    )
    assert frame_with_meta.buffer_end_time_us == 1_000_000
    assert frame_with_meta.buffer_byte_count == 1234


@pytest.mark.asyncio
async def test_try_send_binary_accepts_buffer_metadata() -> None:
    """try_send_binary should accept optional buffer registration parameters."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    wsock = MagicMock()
    wsock.closed = False

    conn = SendspinConnection(server, wsock_client=wsock)

    result = conn.try_send_binary(
        b"audio_data",
        buffer_end_time_us=1_000_000,
        buffer_byte_count=100,
    )
    assert result is True

    frame = conn._to_write.get_nowait()  # noqa: SLF001
    assert frame.buffer_end_time_us == 1_000_000
    assert frame.buffer_byte_count == 100


@pytest.mark.asyncio
async def test_writer_registers_buffer_after_send() -> None:
    """Writer should call role's buffer_tracker.register() after successful send_bytes."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    wsock = MagicMock()
    wsock.closed = False
    wsock.send_str = AsyncMock()
    wsock.send_bytes = AsyncMock()

    conn = SendspinConnection(server, wsock_client=wsock)
    await conn._setup_connection()  # noqa: SLF001

    # Mock a role that handles AUDIO_CHUNK with buffer tracking
    mock_role = MagicMock()
    mock_buffer_tracker = MagicMock()
    mock_buffer_tracker.time_until_duration_capacity.return_value = 0
    mock_role._buffer_tracker = mock_buffer_tracker  # noqa: SLF001
    mock_role._stream_start_time_us = None  # noqa: SLF001
    mock_role._last_late_log_s = 0.0  # noqa: SLF001
    mock_role._late_skips_since_log = 0  # noqa: SLF001
    mock_role.get_binary_handling.return_value = BinaryHandling(
        drop_late=False,
        buffer_track=True,
    )

    mock_client = MagicMock()
    mock_client.active_roles = [mock_role]
    conn._client = mock_client  # noqa: SLF001

    payload = b"audio_data"
    packed = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, 0) + payload
    conn.try_send_binary(
        packed,
        buffer_end_time_us=1_000_000,
        buffer_byte_count=100,
        duration_us=50_000,
    )

    for _ in range(50):
        if wsock.send_bytes.called:
            break
        await asyncio.sleep(0)

    assert wsock.send_bytes.call_count == 1
    mock_buffer_tracker.register.assert_called_once_with(1_000_000, 100, 50_000)

    await conn.disconnect(retry_connection=False)


@pytest.mark.asyncio
async def test_writer_does_not_register_without_metadata() -> None:
    """Writer should not call register() when metadata is None."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))

    wsock = MagicMock()
    wsock.closed = False
    wsock.send_str = AsyncMock()
    wsock.send_bytes = AsyncMock()

    conn = SendspinConnection(server, wsock_client=wsock)
    await conn._setup_connection()  # noqa: SLF001

    # Mock a role that handles AUDIO_CHUNK with buffer tracking
    mock_role = MagicMock()
    mock_buffer_tracker = MagicMock()
    mock_buffer_tracker.time_until_duration_capacity.return_value = 0
    mock_role._buffer_tracker = mock_buffer_tracker  # noqa: SLF001
    mock_role._stream_start_time_us = None  # noqa: SLF001
    mock_role._last_late_log_s = 0.0  # noqa: SLF001
    mock_role._late_skips_since_log = 0  # noqa: SLF001
    mock_role.get_binary_handling.return_value = BinaryHandling(
        drop_late=False,
        buffer_track=True,
    )

    mock_client = MagicMock()
    mock_client.active_roles = [mock_role]
    conn._client = mock_client  # noqa: SLF001

    payload = b"audio_data"
    packed = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, 0) + payload
    conn.try_send_binary(packed)  # No buffer metadata

    for _ in range(50):
        if wsock.send_bytes.called:
            break
        await asyncio.sleep(0)

    assert wsock.send_bytes.call_count == 1
    mock_buffer_tracker.register.assert_not_called()

    await conn.disconnect(retry_connection=False)


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
