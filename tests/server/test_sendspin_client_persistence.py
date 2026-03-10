"""Tests for persistent SendspinClient device state across reconnects."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.models.core import ClientHelloPayload, StreamStartMessage
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, GoodbyeReason, PlayerCommand, Roles
from aiosendspin.server import client as client_module
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.roles.player.v1 import PlayerPersistentState


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: LoopClock
    id: str = "srv"
    name: str = "server"

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False


class _DummyConnection:
    def __init__(self) -> None:
        self.sent_json: list[object] = []
        self.sent_binary: list[bytes] = []

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
        buffer_end_time_us: int | None = None,  # noqa: ARG002
        buffer_byte_count: int | None = None,  # noqa: ARG002
        duration_us: int | None = None,  # noqa: ARG002
    ) -> bool:
        self.sent_binary.append(data)
        return True


def _player_hello(
    client_id: str,
    *,
    supported_formats: list[SupportedAudioFormat] | None = None,
) -> ClientHelloPayload:
    if supported_formats is None:
        supported_formats = [
            SupportedAudioFormat(
                codec=AudioCodec.PCM,
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            )
        ]

    return ClientHelloPayload(
        client_id=client_id,
        name=client_id,
        version=1,
        supported_roles=[Roles.PLAYER.value],
        player_support=ClientHelloPlayerSupport(
            supported_formats=supported_formats,
            buffer_capacity=100_000,
            supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
        ),
    )


@pytest.mark.asyncio
async def test_goodbye_disconnect_delays_buffer_tracker_reset() -> None:
    """Goodbye disconnect follows the same delayed reset policy."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = SendspinClient(server, client_id="player-1")
    SendspinGroup(server, client)

    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    state = client.get_role_state("player", PlayerPersistentState)
    assert state is not None
    assert state.buffer_tracker is not None
    state.buffer_tracker.register(end_time_us=1_000_000, byte_count=1234)
    assert state.buffer_tracker.buffered_bytes == 1234

    client.detach_connection(GoodbyeReason.USER_REQUEST)
    assert state.buffer_tracker.buffered_bytes == 1234
    await asyncio.sleep(2.2)
    assert state.buffer_tracker.buffered_bytes == 0


@pytest.mark.asyncio
async def test_ungraceful_disconnect_delays_buffer_tracker_reset() -> None:
    """Ungraceful disconnect delays BufferTracker reset to tolerate brief blips."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = SendspinClient(server, client_id="player-1")
    SendspinGroup(server, client)

    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    state = client.get_role_state("player", PlayerPersistentState)
    assert state is not None
    assert state.buffer_tracker is not None
    state.buffer_tracker.register(end_time_us=1_000_000, byte_count=1234)
    client.detach_connection(None)

    assert state.buffer_tracker.buffered_bytes == 1234
    await asyncio.sleep(2.2)
    assert state.buffer_tracker.buffered_bytes == 0


@pytest.mark.asyncio
async def test_reconnect_resets_buffer_tracker() -> None:
    """Reconnect resets buffer tracker immediately (client buffer is empty after reconnect)."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = SendspinClient(server, client_id="player-1")
    SendspinGroup(server, client)

    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    state = client.get_role_state("player", PlayerPersistentState)
    assert state is not None
    assert state.buffer_tracker is not None
    state.buffer_tracker.register(end_time_us=1_000_000, byte_count=1234)
    assert state.buffer_tracker.buffered_bytes == 1234

    client.detach_connection(None)

    # Reconnect before the delayed reset callback fires
    await asyncio.sleep(1.0)
    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    # Buffer tracker should be reset immediately on reconnect
    # (client's actual buffer is empty after reconnect)
    assert state.buffer_tracker.buffered_bytes == 0


@pytest.mark.asyncio
async def test_reconnect_refreshes_audio_requirements_from_new_hello() -> None:
    """Warm reconnect should rebuild cached audio requirements from the new hello."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = SendspinClient(server, client_id="player-1")
    SendspinGroup(server, client)

    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello(
            "player-1",
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.FLAC,
                    channels=2,
                    sample_rate=48_000,
                    bit_depth=32,
                )
            ],
        ),
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    role = client.role("player@v1")
    assert role is not None
    first_req = role.get_audio_requirements()
    assert first_req is not None
    assert first_req.sample_rate == 48_000
    assert first_req.bit_depth == 32
    assert first_req.channels == 2

    client.detach_connection(None)

    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello(
            "player-1",
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.FLAC,
                    channels=2,
                    sample_rate=48_000,
                    bit_depth=24,
                )
            ],
        ),
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    reconnected_role = client.role("player@v1")
    assert reconnected_role is role

    refreshed_req = reconnected_role.get_audio_requirements()
    assert refreshed_req is not None
    assert refreshed_req.sample_rate == 48_000
    assert refreshed_req.bit_depth == 24
    assert refreshed_req.channels == 2


@pytest.mark.asyncio
async def test_reconnect_with_new_format_drops_stale_cached_audio() -> None:
    """Warm reconnect with changed requirements should not replay old-format backlog."""
    mock_loop = MagicMock()
    mock_loop.time.return_value = 1000.0
    server = _DummyServer(loop=mock_loop, clock=LoopClock(mock_loop))
    client = SendspinClient(server, client_id="player-1")
    group = SendspinGroup(server, client)

    first_conn = _DummyConnection()
    client.attach_connection(
        first_conn,
        client_info=_player_hello(
            "player-1",
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM,
                    channels=2,
                    sample_rate=48_000,
                    bit_depth=16,
                )
            ],
        ),
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    stream = group.start_stream()
    stream.prepare_audio(
        bytes(4_800),
        AudioFormat(sample_rate=48_000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    client.detach_connection(None)

    # Keep producing old-format backlog while the role is warm-disconnected.
    stream.prepare_audio(
        bytes(4_800),
        AudioFormat(sample_rate=48_000, bit_depth=16, channels=2),
    )
    await stream.commit_audio()

    reconnect_conn = _DummyConnection()
    client.attach_connection(
        reconnect_conn,
        client_info=_player_hello(
            "player-1",
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM,
                    channels=2,
                    sample_rate=48_000,
                    bit_depth=24,
                )
            ],
        ),
        active_roles=[Roles.PLAYER.value],
    )
    client.mark_connected()

    stream_starts = [msg for msg in reconnect_conn.sent_json if isinstance(msg, StreamStartMessage)]
    assert len(stream_starts) == 1
    assert stream_starts[0].payload.player is not None
    assert stream_starts[0].payload.player.bit_depth == 24
    assert reconnect_conn.sent_binary
    assert all(len(chunk) == 7_209 for chunk in reconnect_conn.sent_binary)


@pytest.mark.asyncio
async def test_transient_disconnect_reuses_role_instance_and_preserves_lifecycle_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient reconnect should reuse role object but still run disconnect/connect hooks."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = SendspinClient(server, client_id="player-1")
    SendspinGroup(server, client)

    class _TrackedRole:
        def __init__(self) -> None:
            self.role_id = "player@v1"
            self.role_family = "player"
            self.on_connect = MagicMock()
            self.on_disconnect = MagicMock()

        def get_binary_handling(self, _message_type: int) -> None:
            return None

    tracked_role = _TrackedRole()

    def _create_role(role_id: str, _client: SendspinClient) -> _TrackedRole | None:
        return tracked_role if role_id == "player@v1" else None

    monkeypatch.setattr(client_module, "create_role", _create_role)

    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    first_role = client.role("player@v1")
    assert first_role is tracked_role
    assert tracked_role.on_connect.call_count == 1
    assert tracked_role.on_disconnect.call_count == 0

    client.detach_connection(None)
    assert client.has_warm_disconnected_roles
    assert client.role("player@v1") is tracked_role
    assert tracked_role.on_disconnect.call_count == 1

    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    assert client.role("player@v1") is first_role
    assert tracked_role.on_connect.call_count == 2
    assert tracked_role.on_disconnect.call_count == 1


@pytest.mark.asyncio
async def test_hard_disconnect_clears_roles() -> None:
    """Non-transient disconnect reasons should clear role instances."""
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = SendspinClient(server, client_id="player-1")
    SendspinGroup(server, client)

    client.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    assert client.role("player@v1") is not None

    client.detach_connection(GoodbyeReason.USER_REQUEST)
    assert not client.has_warm_disconnected_roles
    assert client.role("player@v1") is None


@pytest.mark.asyncio
async def test_stale_connection_disconnect_does_not_wipe_newer_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old async disconnect must not detach a newer replacement connection (PR #168 regression).

    Race condition:
      1. Client has ``old_conn`` (e.g. discovery connection).
      2. ``new_conn`` arrives; ``attach_connection()`` schedules ``old_conn.disconnect()``
         as an async task and immediately sets ``client._connection = new_conn``.
      3. ``new_conn`` completes its handshake (``mark_connected()``).
      4. ``old_conn.disconnect()`` resumes after an async gap and must NOT call
         ``detach_connection()`` because ``client.connection`` is now ``new_conn``, not ``self``.
    """
    loop = asyncio.get_running_loop()
    server = _DummyServer(loop=loop, clock=LoopClock(loop))
    client = SendspinClient(server, client_id="player-1")
    SendspinGroup(server, client)

    # Step 1: attach first connection.
    old_wsock = MagicMock()
    old_wsock.closed = False
    old_wsock.close = AsyncMock()
    old_conn = SendspinConnection(server, wsock_client=old_wsock)
    client.attach_connection(
        old_conn,
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    old_conn._client = client  # mirror what SendspinConnection sets after attach  # noqa: SLF001
    client.mark_connected()
    assert client.connection is old_conn
    assert client.is_connected

    stale_disconnect_done = asyncio.Event()
    old_disconnect = old_conn.disconnect

    async def _disconnect_and_signal(*, retry_connection: bool = True) -> None:
        try:
            await old_disconnect(retry_connection=retry_connection)
        finally:
            stale_disconnect_done.set()

    monkeypatch.setattr(old_conn, "disconnect", _disconnect_and_signal)

    # Step 2: new connection replaces old one.
    # attach_connection() schedules old_conn.disconnect() as a task (eager_start may
    # begin it immediately, but it suspends at the first await inside disconnect()).
    new_wsock = MagicMock()
    new_wsock.closed = False
    new_wsock.close = AsyncMock()
    new_conn = SendspinConnection(server, wsock_client=new_wsock)
    client.attach_connection(
        new_conn,
        client_info=_player_hello("player-1"),
        active_roles=[Roles.PLAYER.value],
    )
    new_conn._client = client  # mirror what SendspinConnection sets after attach  # noqa: SLF001
    client.mark_connected()
    assert client.connection is new_conn

    # Step 3: wait for old_conn.disconnect() to run to completion.
    await asyncio.wait_for(stale_disconnect_done.wait(), timeout=1.0)

    # The new connection must still be the active one.
    assert client.connection is new_conn, (
        "Old connection's async disconnect wiped the newer live connection (PR #168 regression)"
    )
    assert client.is_connected
