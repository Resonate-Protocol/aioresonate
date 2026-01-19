"""Focused tests for PushStream behavior with persistent SendspinClient objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from aiosendspin.models import unpack_binary_header
from aiosendspin.models.core import StreamClearMessage, StreamEndMessage, StreamStartMessage
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.channels import ChannelRouter
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.push_stream import PushStream


@dataclass(slots=True)
class _DummyServer:
    loop: Any
    clock: Any
    id: str = "srv"
    name: str = "server"


@dataclass(slots=True)
class _DummyGroup:
    clients: list[SendspinClient]

    def on_client_connected(self, client: SendspinClient) -> None:  # noqa: ARG002
        return


class _FakeConnection:
    def __init__(self) -> None:
        self.sent_json: list[object] = []
        self.sent_binary: list[bytes] = []
        self.high_water = False

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:
        self.sent_json.append(message)

    def try_send_binary(self, data: bytes) -> bool:
        self.sent_binary.append(data)
        return True

    def queue_high_water(self, threshold: float = 0.8) -> bool:  # noqa: ARG002
        return self.high_water


def _make_connected_player(
    mock_loop: Any,
    group: _DummyGroup,
    client_id: str,
) -> tuple[SendspinClient, _FakeConnection]:
    """Create a connected player client with a fake connection."""
    server = _DummyServer(loop=mock_loop, clock=LoopClock(mock_loop))
    client = SendspinClient(server, client_id=client_id)
    client._group = group  # noqa: SLF001
    group.clients.append(client)

    conn = _FakeConnection()
    hello = type("Hello", (), {})()
    hello.client_id = client_id
    hello.name = client_id
    hello.player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.PCM,
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            ),
            SupportedAudioFormat(
                codec=AudioCodec.FLAC,
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            ),
        ],
        buffer_capacity=200_000,
        supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
    )
    hello.artwork_support = None
    hello.visualizer_support = None

    client.attach_connection(conn, client_info=hello, active_roles=[Roles.PLAYER.value])
    client.mark_connected()
    return client, conn


@pytest.mark.asyncio
async def test_commit_audio_sends_stream_start_and_binary(mock_loop: Any) -> None:
    """commit_audio sends stream/start and at least one binary audio chunk."""
    group = _DummyGroup(clients=[])
    client, conn = _make_connected_player(mock_loop, group, "p1")

    stream = PushStream(
        loop=mock_loop, clock=LoopClock(mock_loop), group=group, channel_router=ChannelRouter()
    )
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM),
    )
    await stream.commit_audio()

    assert any(isinstance(m, StreamStartMessage) for m in conn.sent_json)
    assert conn.sent_binary, "expected at least one binary chunk"
    header = unpack_binary_header(conn.sent_binary[0])
    assert header.message_type == 4  # BinaryMessageType.AUDIO_CHUNK
    assert client.buffer_tracker is not None
    assert client.buffer_tracker.buffered_bytes > 0


@pytest.mark.asyncio
async def test_queue_high_water_drops_audio_even_for_blocking_player(mock_loop: Any) -> None:
    """When the connection queue is congested, audio is dropped and the player is resynced."""
    group = _DummyGroup(clients=[])
    client, conn = _make_connected_player(mock_loop, group, "p1")
    conn.high_water = True

    stream = PushStream(
        loop=mock_loop, clock=LoopClock(mock_loop), group=group, channel_router=ChannelRouter()
    )
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM),
    )
    await stream.commit_audio()

    assert not conn.sent_binary
    assert client.player_role is not None
    assert client.player_role.get_send_state().needs_resync


@pytest.mark.asyncio
async def test_stop_sends_stream_end_and_resets_buffer_tracker(mock_loop: Any) -> None:
    """Stop sends stream/end and resets BufferTracker state."""
    group = _DummyGroup(clients=[])
    client, conn = _make_connected_player(mock_loop, group, "p1")

    stream = PushStream(
        loop=mock_loop, clock=LoopClock(mock_loop), group=group, channel_router=ChannelRouter()
    )
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM),
    )
    await stream.commit_audio()
    assert client.buffer_tracker is not None
    assert client.buffer_tracker.buffered_bytes > 0

    stream.stop()
    assert any(isinstance(m, StreamEndMessage) for m in conn.sent_json)
    assert client.buffer_tracker.buffered_bytes == 0


@pytest.mark.asyncio
async def test_clear_sends_stream_clear(mock_loop: Any) -> None:
    """Clear sends stream/clear to connected players."""
    group = _DummyGroup(clients=[])
    _, conn = _make_connected_player(mock_loop, group, "p1")

    stream = PushStream(
        loop=mock_loop, clock=LoopClock(mock_loop), group=group, channel_router=ChannelRouter()
    )
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM),
    )
    await stream.commit_audio()

    stream.clear()
    assert any(isinstance(m, StreamClearMessage) for m in conn.sent_json)


@pytest.mark.asyncio
async def test_on_player_join_sends_catchup_chunks(mock_loop: Any) -> None:
    """Late join triggers stream/start and cached audio catch-up."""
    group = _DummyGroup(clients=[])
    _, conn1 = _make_connected_player(mock_loop, group, "p1")
    stream = PushStream(
        loop=mock_loop, clock=LoopClock(mock_loop), group=group, channel_router=ChannelRouter()
    )

    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM),
    )
    await stream.commit_audio()
    assert conn1.sent_binary

    _, conn2 = _make_connected_player(mock_loop, group, "p2")
    stream.on_player_join("p2")

    assert any(isinstance(m, StreamStartMessage) for m in conn2.sent_json)
    assert conn2.sent_binary, "expected catch-up binary chunks"


@pytest.mark.asyncio
async def test_non_blocking_player_resync_waits_for_queue_to_drain(mock_loop: Any) -> None:
    """Resync waits for queue to drain before sending stream/clear."""
    group = _DummyGroup(clients=[])
    client, conn = _make_connected_player(mock_loop, group, "p1")
    client.blocking = False

    stream = PushStream(
        loop=mock_loop, clock=LoopClock(mock_loop), group=group, channel_router=ChannelRouter()
    )

    conn.high_water = True
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM),
    )
    await stream.commit_audio()
    assert not any(isinstance(m, StreamClearMessage) for m in conn.sent_json)

    conn.high_water = False
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.PCM),
    )
    await stream.commit_audio()
    assert any(isinstance(m, StreamClearMessage) for m in conn.sent_json)
