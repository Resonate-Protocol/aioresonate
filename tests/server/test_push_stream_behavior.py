"""Focused tests for PushStream behavior with persistent SendspinClient objects."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from aiosendspin.models import unpack_binary_header
from aiosendspin.models.core import (
    StreamClearMessage,
    StreamEndMessage,
    StreamRequestFormatPayload,
    StreamStartMessage,
)
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    StreamRequestFormatPlayer,
    SupportedAudioFormat,
)
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.audio_transformers import PcmPassthrough, TransformerPool
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.push_stream import PushStream
from aiosendspin.server.roles import AudioChunk, AudioRequirements


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
        self._push_stream: PushStream | None = None
        self.has_active_stream = False

    def on_client_connected(self, client: SendspinClient) -> None:  # noqa: ARG002
        return

    def group_role(self, family: str) -> None:  # noqa: ARG002
        return None

    def get_channel_for_player(self, player_id: str) -> UUID:  # noqa: ARG002
        return MAIN_CHANNEL

    def on_role_format_changed(self, role: Any) -> None:
        if self._push_stream is not None and not self._push_stream.is_stopped:
            self._push_stream.on_role_format_changed(role)


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
        role: str,  # noqa: ARG002
        timestamp_us: int,  # noqa: ARG002
        message_type: int,  # noqa: ARG002
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

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
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

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
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

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
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
    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)

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
async def test_pcm_cache_catchup_for_uncached_codec() -> None:
    """PCM cache should enable catch-up when TransformKey cache is empty."""

    class TransformerA:
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

    class TransformerB(TransformerA):
        pass

    group = _DummyGroup(clients=[])
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=TransformerA(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role1]))

    loop = asyncio.get_running_loop()
    stream = PushStream(
        loop=loop,
        clock=LoopClock(loop),
        group=group,
    )
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=TransformerB(),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role2]))
    stream.on_role_join(role2)
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()
    for _ in range(50):
        if role2.received:
            break
        await asyncio.sleep(0.01)

    assert role2.started == 1
    assert role2.received


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

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
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

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
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

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()
    assert role1.received

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

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
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

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)
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


def _make_connected_player_multi_format(
    mock_loop: Any,
    group: _DummyGroup,
    client_id: str,
) -> tuple[SendspinClient, _FakeConnection]:
    """Create a connected player client that supports PCM 48kHz and PCM 44.1kHz."""
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
                codec=AudioCodec.PCM,
                channels=2,
                sample_rate=44100,
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
async def test_format_change_during_active_stream(mock_loop: Any) -> None:
    """Mid-stream format change sends stream/start (deferred) with no stream/clear.

    Full PushStream flow:
    1. Create player with PCM 48kHz, start PushStream
    2. Commit audio N times
    3. Trigger format change via on_stream_request_format with stream_active=True
    4. Commit more audio
    5. Assert: StreamStartMessage (with new format) in sent_json, NO StreamClearMessage
    6. Binary audio continues after format change
    7. Gap between last pre-change chunk and first post-change chunk ≤ 100ms
    """
    group = _DummyGroup(clients=[])
    client, conn = _make_connected_player_multi_format(mock_loop, group, "p1")
    clock = LoopClock(mock_loop)

    stream = PushStream(loop=mock_loop, clock=clock, group=group)
    group._push_stream = stream  # noqa: SLF001

    # Commit several chunks at 48kHz PCM
    for _ in range(3):
        stream.prepare_audio(
            bytes(4800),
            AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
        )
        await stream.commit_audio()

    pre_change_binary_count = len(conn.sent_binary)
    assert pre_change_binary_count > 0

    # Record the last pre-change chunk's end timestamp
    last_pre_header = unpack_binary_header(conn.sent_binary[-1])
    # Duration of a 4800-byte PCM chunk at 48kHz stereo 16bit = 25ms = 25000us
    pre_change_end_us = last_pre_header.timestamp_us + 25_000

    # Clear sent_json to isolate format change messages
    conn.sent_json.clear()

    # Trigger mid-stream format change: PCM 48kHz -> PCM 44.1kHz
    request = StreamRequestFormatPayload(
        player=StreamRequestFormatPlayer(
            codec=AudioCodec.PCM,
            sample_rate=44100,
            channels=2,
            bit_depth=16,
        )
    )
    role = client.role("player@v1")
    assert role is not None
    role.on_stream_request_format(request, stream_active=True)

    # No immediate stream/start or stream/clear
    assert not any(isinstance(msg, StreamStartMessage) for msg in conn.sent_json)
    assert not any(isinstance(msg, StreamClearMessage) for msg in conn.sent_json)

    # Commit audio at the new format (44.1kHz)
    # 1102 samples * 2 bytes * 2 channels = 4408 bytes (~24.99ms)
    stream.prepare_audio(
        bytes(4408),
        AudioFormat(sample_rate=44100, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    # Stream/start should now be sent (deferred until first chunk)
    stream_starts = [msg for msg in conn.sent_json if isinstance(msg, StreamStartMessage)]
    assert len(stream_starts) == 1
    start_msg = stream_starts[0]
    assert start_msg.payload.player is not None
    assert start_msg.payload.player.sample_rate == 44100
    assert start_msg.payload.player.codec == AudioCodec.PCM

    # No stream/clear should have been sent
    assert not any(isinstance(msg, StreamClearMessage) for msg in conn.sent_json)

    # Binary audio continued after the format change
    assert len(conn.sent_binary) > pre_change_binary_count

    # Check the gap: first post-change chunk start vs last pre-change chunk end
    post_change_binary = conn.sent_binary[pre_change_binary_count:]
    first_post_header = unpack_binary_header(post_change_binary[0])
    gap_us = first_post_header.timestamp_us - pre_change_end_us
    assert gap_us <= 100_000, f"Gap between pre/post format change chunks is {gap_us}us (> 100ms)"
