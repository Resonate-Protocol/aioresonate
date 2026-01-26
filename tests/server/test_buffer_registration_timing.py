"""Regression tests for buffer registration timing (queue-time vs send-time)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.pipeline import EncodedChunk


@dataclass(slots=True)
class _DummyServer:
    loop: Any
    clock: Any
    id: str = "srv"
    name: str = "server"

    def get_or_create_client(self, client_id: str) -> Any:  # noqa: ARG002
        raise AssertionError("unexpected get_or_create_client() in this test")


@dataclass(slots=True)
class _DummyGroup:
    clients: list[Any]
    group_id: str = "g1"

    def on_client_connected(self, client: Any) -> None:  # noqa: ARG002
        return


@pytest.mark.asyncio
async def test_buffer_tracker_does_not_backpressure_until_send() -> None:
    """
    Backpressure must reflect bytes that have actually left the server.

    This test blocks websocket send to create a gap between "queued" and "sent".
    Before the send completes, BufferTracker must remain at 0 bytes (no backpressure).
    """
    loop = asyncio.get_running_loop()
    clock = LoopClock(loop)
    server = _DummyServer(loop=loop, clock=clock)

    send_event = asyncio.Event()
    wsock = MagicMock()
    wsock.closed = False
    wsock.send_str = AsyncMock()

    async def slow_send_bytes(_: bytes) -> None:
        await send_event.wait()

    wsock.send_bytes = AsyncMock(side_effect=slow_send_bytes)

    conn = SendspinConnection(server, wsock_client=wsock)
    await conn._setup_connection()  # noqa: SLF001

    group = _DummyGroup(clients=[])
    client = SendspinClient(server, client_id="p1")
    client._group = group  # noqa: SLF001
    group.clients.append(client)

    hello = type("Hello", (), {})()
    hello.client_id = "p1"
    hello.name = "p1"
    hello.player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.PCM,
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            )
        ],
        buffer_capacity=100,
        supported_commands=[PlayerCommand.VOLUME],
    )
    hello.artwork_support = None
    hello.visualizer_support = None

    client.attach_connection(conn, client_info=hello, active_roles=[Roles.PLAYER.value])
    client.mark_connected()
    conn._client = client  # noqa: SLF001

    assert client.player_role is not None
    assert client.buffer_tracker is not None

    now_us = clock.now_us()
    chunk = EncodedChunk(
        timestamp_us=now_us + 100_000,
        data=b"x" * 100,
        byte_count=100,
        sample_count=1,
        duration_us=100_000,
    )

    try:
        sent = client.player_role.send_audio(
            chunk=chunk,
            timestamp_us=chunk.timestamp_us,
            audio_format=AudioFormat(
                sample_rate=48000,
                bit_depth=16,
                channels=2,
            ),
            codec=AudioCodec.PCM,
        )
        assert sent is True

        # Regression assertion: queued-but-not-sent data must not cause backpressure.
        assert client.buffer_tracker.buffered_bytes == 0
        assert client.buffer_tracker.time_until_capacity(1) == 0

        send_event.set()
        for _ in range(50):
            if wsock.send_bytes.called:
                break
            await asyncio.sleep(0)

        assert wsock.send_bytes.call_count == 1
        assert client.buffer_tracker.buffered_bytes == 100
    finally:
        send_event.set()
        await conn.disconnect(retry_connection=False)
