"""Focused tests for PushStream behavior with persistent SendspinClient objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from aiosendspin.models import unpack_binary_header
from aiosendspin.models.core import StreamClearMessage, StreamEndMessage, StreamStartMessage
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.channels import MAIN_CHANNEL, ChannelRouter
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.push_stream import PushStream
from aiosendspin.server.roles import AudioChunk, AudioRequirements
from aiosendspin.server.transformers import PcmPassthrough, TransformerPool


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

    def group_role(self, family: str) -> None:  # noqa: ARG002
        return None


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
        timestamp_us: int,  # noqa: ARG002
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
        queued_at_us: int | None = None,  # noqa: ARG002
    ) -> bool:
        self.sent_binary.append(data)
        if (
            self.buffer_tracker is not None
            and buffer_end_time_us is not None
            and buffer_byte_count is not None
        ):
            self.buffer_tracker.register(buffer_end_time_us, buffer_byte_count, duration_us or 0)
        return True


class _DummyRole:
    def __init__(self, requirements: AudioRequirements) -> None:
        self._requirements = requirements
        self.received: list[AudioChunk] = []
        self.started = 0

    def get_audio_requirements(self) -> AudioRequirements | None:
        return self._requirements

    def get_join_delay_s(self) -> float:
        return 0.0

    def on_stream_start(self) -> None:
        self.started += 1

    def on_audio_chunk(self, chunk: AudioChunk) -> bool:
        self.received.append(chunk)
        return True

    def on_stream_end(self) -> None:
        return

    def on_stream_clear(self) -> None:
        return


class _DummyClient:
    def __init__(self, roles: list[_DummyRole]) -> None:
        self.is_connected = True
        self.active_roles = roles
        self.connection = _FakeConnection()


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
    if role is not None:
        transformer = group.transformer_pool.get_or_create(
            PcmPassthrough,
            channel_id=MAIN_CHANNEL,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
        )
        role._audio_requirements = AudioRequirements(  # noqa: SLF001
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=transformer,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
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


@pytest.mark.asyncio
async def test_transform_dedup_uses_transform_key_not_instance(mock_loop: Any) -> None:
    """Transformer dedupe should be based on TransformKey, not instance id."""

    class CountingTransformer:
        calls = 0

        def __init__(self) -> None:
            self._frame_duration_us = 25_000

        @property
        def frame_duration_us(self) -> int:
            return self._frame_duration_us

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[bytes]:
            CountingTransformer.calls += 1
            return [pcm]

        def flush(self) -> list[bytes]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    CountingTransformer.calls = 0
    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=CountingTransformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=CountingTransformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.extend([_DummyClient([role1]), _DummyClient([role2])])

    stream = PushStream(
        loop=mock_loop, clock=LoopClock(mock_loop), group=group, channel_router=ChannelRouter()
    )
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    assert CountingTransformer.calls == 1


@pytest.mark.asyncio
async def test_transform_key_separates_frame_duration(mock_loop: Any) -> None:
    """Different frame_duration_us should not share transformer work."""

    class CountingTransformer:
        calls = 0

        def __init__(self, frame_duration_us: int) -> None:
            self._frame_duration_us = frame_duration_us

        @property
        def frame_duration_us(self) -> int:
            return self._frame_duration_us

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[bytes]:
            CountingTransformer.calls += 1
            return [pcm]

        def flush(self) -> list[bytes]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    CountingTransformer.calls = 0
    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=CountingTransformer(25_000),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=CountingTransformer(50_000),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=50_000,
        )
    )
    group.clients.extend([_DummyClient([role1]), _DummyClient([role2])])

    stream = PushStream(
        loop=mock_loop, clock=LoopClock(mock_loop), group=group, channel_router=ChannelRouter()
    )
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    assert CountingTransformer.calls == 2


@pytest.mark.asyncio
async def test_late_join_uses_cached_chunks_across_role_recreation(mock_loop: Any) -> None:
    """Late join uses cache even if transformer instance changes."""

    class PassTransformer:
        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[bytes]:
            return [pcm]

        def flush(self) -> list[bytes]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=PassTransformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    client1 = _DummyClient([role1])
    group.clients.append(client1)

    stream = PushStream(
        loop=mock_loop, clock=LoopClock(mock_loop), group=group, channel_router=ChannelRouter()
    )
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()
    # Batched broadcast sends directly to connection, not via role.on_audio_chunk()
    assert client1.connection.sent_binary

    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=PassTransformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    stream.on_role_join(role2)

    assert role2.started == 1
    assert role2.received


@pytest.mark.asyncio
async def test_stop_flush_fans_out_to_all_roles(mock_loop: Any) -> None:
    """stop() flush frames to all roles sharing a TransformKey."""

    class FlushingTransformer:
        @property
        def frame_duration_us(self) -> int:
            return 25_000

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[bytes]:
            return [pcm]

        def flush(self) -> list[bytes]:
            return [b"final"]

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=FlushingTransformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=FlushingTransformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.extend([_DummyClient([role1]), _DummyClient([role2])])

    stream = PushStream(
        loop=mock_loop, clock=LoopClock(mock_loop), group=group, channel_router=ChannelRouter()
    )
    stream.stop()

    assert len(role1.received) == 1
    assert len(role2.received) == 1


@pytest.mark.asyncio
async def test_transform_key_separates_channels(mock_loop: Any) -> None:
    """TransformKey includes channel_id to avoid cross-channel sharing."""

    class CountingTransformer:
        calls = 0

        def __init__(self) -> None:
            self._frame_duration_us = 25_000

        @property
        def frame_duration_us(self) -> int:
            return self._frame_duration_us

        def process(self, pcm: bytes, _ts: int, _dur: int) -> list[bytes]:
            CountingTransformer.calls += 1
            return [pcm]

        def flush(self) -> list[bytes]:
            return []

        def get_header(self) -> bytes | None:
            return None

        def reset(self) -> None:
            return

    CountingTransformer.calls = 0
    group = _DummyGroup(clients=[])
    other_channel = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=CountingTransformer(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=CountingTransformer(),
            channel_id=other_channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.extend([_DummyClient([role1]), _DummyClient([role2])])

    stream = PushStream(
        loop=mock_loop, clock=LoopClock(mock_loop), group=group, channel_router=ChannelRouter()
    )
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
        channel_id=MAIN_CHANNEL,
    )
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
        channel_id=other_channel,
    )
    await stream.commit_audio()

    assert CountingTransformer.calls == 2
