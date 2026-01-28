"""Focused tests for PushStream behavior with persistent SendspinClient objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

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
from aiosendspin.server.roles import AudioRequirements
from aiosendspin.server.transformers import PcmPassthrough, TransformerPool

# Shared transformer for tests that need consistent cache keys
_SHARED_PCM_PASSTHROUGH = PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2)


@dataclass(slots=True)
class _DummyServer:
    loop: Any
    clock: Any
    id: str = "srv"
    name: str = "server"


class _DummyGroup:
    def __init__(self, clients: list[SendspinClient]) -> None:
        self.clients = clients
        self.transformer_pool = TransformerPool()

    def on_client_connected(self, client: SendspinClient) -> None:  # noqa: ARG002
        return


class _FakeConnection:
    def __init__(self) -> None:
        self.sent_json: list[object] = []
        self.sent_binary: list[bytes] = []
        self.buffer_tracker = None

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:
        self.sent_json.append(message)

    def try_send_binary(
        self,
        data: bytes,
        *,
        role_family: str,  # noqa: ARG002
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
    ) -> bool:
        self.sent_binary.append(data)
        if (
            self.buffer_tracker is not None
            and buffer_end_time_us is not None
            and buffer_byte_count is not None
        ):
            self.buffer_tracker.register(buffer_end_time_us, buffer_byte_count, duration_us or 0)
        return True


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
    role = client.role("player@v1")
    if role is not None:
        conn.buffer_tracker = role.get_buffer_tracker()

    # Set up audio requirements on the player role for hook-based streaming
    # Use shared transformer to ensure consistent cache keys across tests
    if role is not None:
        role._audio_requirements = AudioRequirements(  # noqa: SLF001
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=_SHARED_PCM_PASSTHROUGH,
        )

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
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    assert any(isinstance(m, StreamStartMessage) for m in conn.sent_json)
    assert conn.sent_binary, "expected at least one binary chunk"
    header = unpack_binary_header(conn.sent_binary[0])
    assert header.message_type == 4  # BinaryMessageType.AUDIO_CHUNK
    role = client.role("player@v1")
    assert role is not None
    buffer_tracker = role.get_buffer_tracker()
    assert buffer_tracker is not None
    assert buffer_tracker.buffered_bytes > 0


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
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()
    role = client.role("player@v1")
    assert role is not None
    buffer_tracker = role.get_buffer_tracker()
    assert buffer_tracker is not None
    assert buffer_tracker.buffered_bytes > 0

    stream.stop()
    assert any(isinstance(m, StreamEndMessage) for m in conn.sent_json)
    assert buffer_tracker.buffered_bytes == 0


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
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    stream.clear()
    assert any(isinstance(m, StreamClearMessage) for m in conn.sent_json)


@pytest.mark.asyncio
async def test_on_role_join_sends_catchup_chunks(mock_loop: Any) -> None:
    """Late join via on_role_join triggers stream/start and cached audio catch-up."""
    group = _DummyGroup(clients=[])
    _, conn1 = _make_connected_player(mock_loop, group, "p1")
    stream = PushStream(
        loop=mock_loop, clock=LoopClock(mock_loop), group=group, channel_router=ChannelRouter()
    )

    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()
    assert conn1.sent_binary

    client2, conn2 = _make_connected_player(mock_loop, group, "p2")
    role2 = client2.role("player@v1")
    assert role2 is not None
    role2.get_join_delay_s = MagicMock(return_value=0.0)
    stream.on_role_join(role2)

    assert any(isinstance(m, StreamStartMessage) for m in conn2.sent_json)
    assert conn2.sent_binary, "expected catch-up binary chunks"
