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
from aiosendspin.server.audio_transformers import TransformerPool
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock, ManualClock
from aiosendspin.server.push_stream import CachedChunk, PushStream
from aiosendspin.server.roles import AudioChunk, AudioRequirements
from aiosendspin.server.roles.player.audio_transformers import PcmPassthrough


@dataclass(slots=True)
class _DummyServer:
    loop: Any
    clock: Any
    id: str = "srv"
    name: str = "server"

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False


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

    def send_role_message(self, role: str, message: object) -> None:  # noqa: ARG002
        self.sent_json.append(message)

    def send_binary(
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

    def on_audio_chunk(self, chunk: AudioChunk) -> None:
        self.received.append(chunk)

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
            channel_id=MAIN_CHANNEL.int,
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
async def test_transient_disconnect_keeps_role_in_audio_pipeline(mock_loop: Any) -> None:
    """Transient disconnect keeps role processing active, but transport send remains no-op."""
    group = _DummyGroup(clients=[])
    client, conn = _make_connected_player(mock_loop, group, "p1")
    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)

    role = client.role("player@v1")
    assert role is not None
    original_on_audio_chunk = role.on_audio_chunk
    on_audio_chunk_spy = MagicMock(side_effect=original_on_audio_chunk)
    role.on_audio_chunk = on_audio_chunk_spy  # type: ignore[method-assign]

    client.detach_connection(None)
    assert client.has_warm_disconnected_roles

    conn.sent_binary.clear()
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    assert on_audio_chunk_spy.call_count > 0
    assert not conn.sent_binary


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
        pending_timestamp_us: int | None = None

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
        pending_timestamp_us: int | None = None

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
        pending_timestamp_us: int | None = None

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
async def test_long_gap_reset_is_handled_in_push_stream() -> None:
    """PushStream resets transformer state after long production gaps."""

    class ResetTrackingTransformer:
        pending_timestamp_us: int | None = None

        def __init__(self) -> None:
            self.reset_calls = 0

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
            self.reset_calls += 1

    transformer = ResetTrackingTransformer()
    group = _DummyGroup(clients=[])
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=transformer,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    stream.prepare_audio(bytes(4800), fmt)
    await stream.commit_audio()
    assert transformer.reset_calls == 0

    clock.advance_us(2_000_000)
    stream.prepare_audio(bytes(4800), fmt)
    await stream.commit_audio()
    assert transformer.reset_calls == 1


@pytest.mark.asyncio
async def test_medium_gap_does_not_reset_transformer() -> None:
    """PushStream does not reset transformer state for medium gaps."""

    class ResetTrackingTransformer:
        pending_timestamp_us: int | None = None

        def __init__(self) -> None:
            self.reset_calls = 0

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
            self.reset_calls += 1

    transformer = ResetTrackingTransformer()
    group = _DummyGroup(clients=[])
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=transformer,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    stream.prepare_audio(bytes(4800), fmt)
    await stream.commit_audio()
    assert transformer.reset_calls == 0

    clock.advance_us(500_000)
    stream.prepare_audio(bytes(4800), fmt)
    await stream.commit_audio()
    assert transformer.reset_calls == 0


@pytest.mark.asyncio
async def test_late_join_uses_cached_chunks_across_role_recreation(mock_loop: Any) -> None:
    """Late join uses cache even if transformer instance changes."""

    class PassTransformer:
        pending_timestamp_us: int | None = None

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
async def test_send_cached_chunks_keeps_chunk_overlapping_now(mock_loop: Any) -> None:
    """Cached replay should keep/send chunks that overlap now, not only future chunks."""
    group = _DummyGroup(clients=[])
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))
    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)

    now_us = stream._clock.now_us()  # noqa: SLF001
    overlapping = CachedChunk(
        timestamp_us=now_us - 500_000,
        duration_us=1_000_000,
        payload=b"a",
        byte_count=1,
    )
    future = CachedChunk(
        timestamp_us=now_us + 100_000,
        duration_us=25_000,
        payload=b"b",
        byte_count=1,
    )

    stream._send_cached_chunks_to_role(  # noqa: SLF001
        role, [overlapping, future], now_us
    )
    assert len(role.received) == 2
    assert role.received[0].timestamp_us == overlapping.timestamp_us


@pytest.mark.asyncio
async def test_stop_flush_fans_out_to_all_roles(mock_loop: Any) -> None:
    """stop() flush frames to all roles sharing a TransformKey."""

    class FlushingTransformer:
        pending_timestamp_us: int | None = None

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
        pending_timestamp_us: int | None = None

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
    3. Trigger format change via on_stream_request_format during active playback
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
    group.has_active_stream = True

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
    role.on_stream_request_format(request)

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


# --- Historical Audio Tests ---


@pytest.mark.asyncio
async def test_historical_audio_raises_on_active_channel(mock_loop: Any) -> None:
    """prepare_historical_audio() raises ValueError on channel with active timing."""
    group = _DummyGroup(clients=[])
    _make_connected_player(mock_loop, group, "p1")

    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)

    # Commit audio to establish timing on MAIN_CHANNEL
    stream.prepare_audio(
        bytes(4800),
        AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    # Now historical audio on MAIN_CHANNEL should raise
    with pytest.raises(ValueError, match="already has active timing"):
        stream.prepare_historical_audio(
            bytes(4800),
            AudioFormat(sample_rate=48000, bit_depth=16, channels=2),
            channel_id=MAIN_CHANNEL,
        )


@pytest.mark.asyncio
async def test_historical_audio_allows_synthetic_timing_channel() -> None:
    """Historical audio is allowed when channel timing exists but no audio was committed on it."""
    group = _DummyGroup(clients=[])
    other_channel = UUID("99999999-9999-9999-9999-999999999999")
    role_main = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    role_other = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=other_channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.extend([_DummyClient([role_main]), _DummyClient([role_other])])

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    # Commit MAIN only: other_channel gets synthetic timing, but no committed audio.
    stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)
    await stream.commit_audio()

    synthetic_tail_us = stream._channel_timing[other_channel]  # noqa: SLF001
    assert synthetic_tail_us > 0

    # Should not raise: channel has synthetic timing only.
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=other_channel)
    await stream.commit_audio()

    assert role_other.received
    first_hist = role_other.received[0]
    assert first_hist.timestamp_us + first_hist.duration_us == synthetic_tail_us
    assert stream._channel_timing[other_channel] == synthetic_tail_us  # noqa: SLF001


@pytest.mark.asyncio
async def test_historical_audio_raises_after_historical_commit_on_channel() -> None:
    """Reject historical injection after the channel has committed historical audio."""
    group = _DummyGroup(clients=[])
    channel = UUID("88888888-8888-8888-8888-888888888888")
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)
    await stream.commit_audio()

    with pytest.raises(ValueError, match="already has active timing"):
        stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)


@pytest.mark.asyncio
async def test_historical_audio_respects_explicit_start_time() -> None:
    """Historical audio can be anchored to an explicit start timestamp."""
    group = _DummyGroup(clients=[])
    channel = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)

    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    explicit_start_us = 5_000_000
    stream.prepare_historical_audio(
        bytes(4800),
        fmt,
        channel_id=channel,
        start_time_us=explicit_start_us,
    )
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)

    await stream.commit_audio()

    assert role.received
    assert role.received[0].timestamp_us == explicit_start_us
    assert role.received[1].timestamp_us == explicit_start_us + role.received[0].duration_us


@pytest.mark.asyncio
async def test_historical_audio_skips_stale_delivery_but_advances_timing() -> None:
    """Historical chunks that are already stale are not delivered to active roles."""
    group = _DummyGroup(clients=[])
    channel = UUID("abababab-abab-abab-abab-aaaaaaaaaaaa")
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    clock = ManualClock(now_us_value=1_000_000)
    stream = PushStream(loop=loop, clock=clock, group=group)

    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel, start_time_us=100_000)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)

    await stream.commit_audio()

    # Delivery is skipped because chunks are fully before now + DEFAULT_INITIAL_DELAY_US.
    assert not role.received
    assert role.started == 0

    # Timing/cache still advance so future live chunks stay aligned.
    assert stream._channel_timing[channel] == 150_000  # noqa: SLF001


@pytest.mark.asyncio
async def test_historical_audio_only_no_live() -> None:
    """Historical-only commit (no prepare_audio) bootstraps channel with correct timing."""
    group = _DummyGroup(clients=[])
    channel = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    stream = PushStream(loop=loop, clock=LoopClock(loop), group=group)

    # Queue two historical chunks
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)

    await stream.commit_audio()

    # Channel should have received audio
    assert role.started == 1
    assert len(role.received) >= 2

    # Timestamps should be consecutive
    first_ts = role.received[0].timestamp_us
    second_ts = role.received[1].timestamp_us
    expected_duration = 25_000  # 4800 bytes at 48kHz/16bit/stereo = 25ms
    assert second_ts == first_ts + expected_duration


@pytest.mark.asyncio
async def test_historical_plus_live_seamless_transition(mock_loop: Any) -> None:
    """Historical audio followed by live audio has seamless timestamps."""
    group = _DummyGroup(clients=[])
    channel = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    clock = LoopClock(mock_loop)
    stream = PushStream(loop=mock_loop, clock=clock, group=group)

    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    # Queue historical chunk
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)
    # Queue live chunk
    stream.prepare_audio(bytes(4800), fmt, channel_id=channel)

    await stream.commit_audio()

    assert role.started == 1
    assert len(role.received) >= 2

    # With frozen clock, live chunk timestamp should exactly follow historical
    historical_end = role.received[0].timestamp_us + role.received[0].duration_us
    live_start = role.received[1].timestamp_us
    assert live_start == historical_end


@pytest.mark.asyncio
async def test_historical_on_one_channel_live_on_another() -> None:
    """Historical on one channel, live-only on another in same commit."""
    group = _DummyGroup(clients=[])
    hist_channel = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")

    role_hist = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=hist_channel,
            frame_duration_us=25_000,
        )
    )
    role_live = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.extend([_DummyClient([role_hist]), _DummyClient([role_live])])

    loop = asyncio.get_running_loop()
    stream = PushStream(loop=loop, clock=LoopClock(loop), group=group)

    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    # Historical on hist_channel, live on MAIN_CHANNEL
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=hist_channel)
    stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)

    await stream.commit_audio()

    assert role_hist.started == 1
    assert role_hist.received
    assert role_live.started == 1
    assert role_live.received


@pytest.mark.asyncio
async def test_missing_channel_commits_keep_channel_timing_aligned() -> None:
    """Channels that miss commits should still advance on the shared timeline."""
    group = _DummyGroup(clients=[])
    other_channel = UUID("abababab-abab-abab-abab-abababababab")
    role_main = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    role_other = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=other_channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.extend([_DummyClient([role_main]), _DummyClient([role_other])])

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    # Two commits with only MAIN_CHANNEL prepared.
    stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)
    play_start_1 = await stream.commit_audio()
    clock.advance_us(25_000)
    stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)
    play_start_2 = await stream.commit_audio()
    clock.advance_us(25_000)

    # DSP channel resumes: first timestamp should be aligned to current shared timeline.
    stream.prepare_audio(bytes(4800), fmt, channel_id=other_channel)
    play_start_3 = await stream.commit_audio()

    assert play_start_2 == play_start_1 + 25_000
    assert play_start_3 == play_start_2 + 25_000
    assert role_other.received
    assert role_other.received[0].timestamp_us == play_start_3


@pytest.mark.asyncio
async def test_late_introduced_channel_aligns_with_existing_timeline() -> None:
    """A channel added late should start on the same timeline as existing channels."""
    group = _DummyGroup(clients=[])
    other_channel = UUID("cdcdcdcd-cdcd-cdcd-cdcd-cdcdcdcdcdcd")
    role_main = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=MAIN_CHANNEL,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role_main]))

    loop = asyncio.get_running_loop()
    clock = ManualClock()
    stream = PushStream(loop=loop, clock=clock, group=group)
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)

    # Let MAIN_CHANNEL get far ahead of wall clock (no clock advancement).
    for _ in range(40):
        stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)
        await stream.commit_audio()

    role_other = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=other_channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role_other]))
    stream.on_role_join(role_other)

    stream.prepare_audio(bytes(4800), fmt, channel_id=MAIN_CHANNEL)
    stream.prepare_audio(bytes(4800), fmt, channel_id=other_channel)
    await stream.commit_audio()

    assert role_main.received
    assert role_other.received
    assert role_main.received[-1].timestamp_us == role_other.received[-1].timestamp_us


@pytest.mark.asyncio
async def test_historical_pcm_cache_populated() -> None:
    """Historical audio populates PCM cache when enabled for the channel."""
    group = _DummyGroup(clients=[])
    channel = UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    stream = PushStream(loop=loop, clock=LoopClock(loop), group=group)
    stream.enable_pcm_cache_for_channel(channel)

    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)

    await stream.commit_audio()

    cached = stream.get_cached_pcm_chunks(channel)
    assert len(cached) == 2
    assert cached[0].pcm_data == bytes(4800)
    assert cached[1].timestamp_us == cached[0].timestamp_us + cached[0].duration_us


@pytest.mark.asyncio
async def test_historical_no_pcm_cache_without_enable() -> None:
    """Historical audio does not populate PCM cache when not enabled."""
    group = _DummyGroup(clients=[])
    channel = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    role = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role]))

    loop = asyncio.get_running_loop()
    stream = PushStream(loop=loop, clock=LoopClock(loop), group=group)

    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)

    await stream.commit_audio()

    cached = stream.get_cached_pcm_chunks(channel)
    assert len(cached) == 0


@pytest.mark.asyncio
async def test_clear_clears_historical_buffers(mock_loop: Any) -> None:
    """clear() discards pending historical audio."""
    group = _DummyGroup(clients=[])
    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)

    channel = UUID("11111111-1111-1111-1111-111111111111")
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)

    stream.clear()
    assert not stream._historical_buffers  # noqa: SLF001


@pytest.mark.asyncio
async def test_stop_clears_historical_buffers(mock_loop: Any) -> None:
    """stop() discards pending historical audio."""
    group = _DummyGroup(clients=[])
    stream = PushStream(loop=mock_loop, clock=LoopClock(mock_loop), group=group)

    channel = UUID("22222222-2222-2222-2222-222222222222")
    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)

    stream.stop()
    assert not stream._historical_buffers  # noqa: SLF001


@pytest.mark.asyncio
async def test_late_joiner_after_historical_injection() -> None:
    """Late joiner gets cached chunks after historical audio was injected."""
    group = _DummyGroup(clients=[])
    channel = UUID("33333333-3333-3333-3333-333333333333")
    role1 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role1]))

    loop = asyncio.get_running_loop()
    stream = PushStream(loop=loop, clock=LoopClock(loop), group=group)

    fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    stream.prepare_historical_audio(bytes(4800), fmt, channel_id=channel)
    stream.prepare_audio(bytes(4800), fmt, channel_id=channel)
    await stream.commit_audio()

    assert role1.received

    # Late joiner on the same channel
    role2 = _DummyRole(
        AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=None,
            channel_id=channel,
            frame_duration_us=25_000,
        )
    )
    group.clients.append(_DummyClient([role2]))
    stream.on_role_join(role2)

    assert role2.started == 1
    assert role2.received
