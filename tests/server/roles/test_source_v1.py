"""Tests for the server-side source@v1 role."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import aiosendspin.server  # noqa: F401  (import triggers role registration)
from aiosendspin.models.core import (
    ClientStatePayload,
    ClientStreamEndPayload,
    ClientStreamStartPayload,
    ServerCommandMessage,
)
from aiosendspin.models.source import (
    ClientStreamStartSource,
    SourceStatePayload,
)
from aiosendspin.models.types import (
    AudioCodec,
    ClientStateType,
    SourceCommand,
    SourceSignal,
)
from aiosendspin.server.events import GroupEvent
from aiosendspin.server.roles.negotiation import negotiate_active_roles
from aiosendspin.server.roles.registry import GROUP_ROLE_FACTORIES, ROLE_FACTORIES
from aiosendspin.server.roles.source.decoder import SourceDecoder
from aiosendspin.server.roles.source.events import (
    SourceSignalChangedEvent,
    SourceStreamEndedEvent,
    SourceStreamStartedEvent,
)
from aiosendspin.server.roles.source.group import SourceGroupRole
from aiosendspin.server.roles.source.stream import SourceAudioStream
from aiosendspin.server.roles.source.v1 import SourceRoleState, SourceV1Role

if TYPE_CHECKING:
    from aiosendspin.models.types import ServerMessage

_PCM_FORMAT = ClientStreamStartSource(
    codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16
)


class _FakeGroup:
    """Minimal stand-in for SendspinGroup that records emitted events."""

    def __init__(self) -> None:
        self.events: list[GroupEvent] = []
        self.group_roles: dict[str, object] = {}

    def _signal_event(self, event: GroupEvent) -> None:
        self.events.append(event)

    def group_role(self, family: str) -> object | None:
        return self.group_roles.get(family)


class _FakeClient:
    """Minimal stand-in for the server-side SendspinClient."""

    def __init__(self, client_id: str, group: object) -> None:
        self.client_id = client_id
        self.connection = MagicMock()
        self.connection.disconnect = AsyncMock()
        self.group = group
        self.client_state = ClientStateType.SYNCHRONIZED
        self._role_state: dict[str, object] = {}
        self.sent: list[tuple[str, ServerMessage]] = []

    def send_role_message(self, family: str, message: object) -> None:
        self.sent.append((family, message))  # type: ignore[arg-type]

    def get_or_create_role_state(self, family: str, cls: type[object]) -> object:
        if family not in self._role_state:
            self._role_state[family] = cls()
        return self._role_state[family]


def _make_group_role() -> tuple[SourceGroupRole, _FakeGroup]:
    group = _FakeGroup()
    group_role = SourceGroupRole(group)  # type: ignore[arg-type]
    group.group_roles["source"] = group_role
    return group_role, group


def _make_role(group_role: SourceGroupRole | None = None) -> SourceV1Role:
    group = group_role._group if group_role is not None else _FakeGroup()  # noqa: SLF001
    role = SourceV1Role(client=_FakeClient("src-1", group))  # type: ignore[arg-type]
    if group_role is not None:
        role._group_role = group_role  # noqa: SLF001
    return role


def _started_role(group_role: SourceGroupRole | None = None) -> SourceV1Role:
    """Build a role that was told to start and has announced a PCM stream."""
    role = _make_role(group_role)
    role.send_start_command()
    role.on_client_stream_start(ClientStreamStartPayload(source=_PCM_FORMAT))
    return role


def _sent_commands(role: SourceV1Role) -> list[SourceCommand]:
    client: _FakeClient = role._client  # type: ignore[assignment]  # noqa: SLF001
    commands: list[SourceCommand] = []
    for _, message in client.sent:
        assert isinstance(message, ServerCommandMessage)
        assert message.payload.source is not None
        commands.append(message.payload.source.command)
    return commands


def test_source_is_registered_and_negotiated() -> None:
    """The source role auto-registers and is picked during negotiation."""
    assert "source@v1" in ROLE_FACTORIES
    assert "source" in GROUP_ROLE_FACTORIES
    active = negotiate_active_roles(["player@v1", "source@v1"])
    assert "source@v1" in active


def test_role_identity() -> None:
    """role_id / role_family / client_id expose the expected values."""
    role = _make_role()
    assert role.role_id == "source@v1"
    assert role.role_family == "source"
    assert role.client_id == "src-1"


# --- Start/stop policy ---


def test_send_start_command_sends_once() -> None:
    """send_start_command emits a server/command START and dedupes."""
    role = _make_role()
    role.send_start_command()
    role.send_start_command()  # deduped
    client: _FakeClient = role._client  # type: ignore[assignment]  # noqa: SLF001
    assert len(client.sent) == 1
    family, message = client.sent[0]
    assert family == "source"
    assert isinstance(message, ServerCommandMessage)
    assert message.payload.source is not None
    assert message.payload.source.command == SourceCommand.START


def test_send_stop_command_after_start() -> None:
    """send_stop_command emits STOP and can follow a START."""
    role = _make_role()
    role.send_start_command()
    role.send_stop_command()
    assert _sent_commands(role) == [SourceCommand.START, SourceCommand.STOP]


def test_group_join_does_not_start_streaming() -> None:
    """Starting a source is host policy: joining a group must not trigger capture."""
    group_role, _group = _make_group_role()
    role = _make_role(group_role)
    group_role.subscribe(role)
    assert _sent_commands(role) == []


def test_line_sense_is_surfaced_but_does_not_command() -> None:
    """A reported signal is published as an event; acting on it is host policy."""
    group_role, group = _make_group_role()
    role = _make_role(group_role)

    role.on_client_state(ClientStatePayload(source=SourceStatePayload(signal=SourceSignal.PRESENT)))
    role.on_client_state(ClientStatePayload(source=SourceStatePayload(signal=SourceSignal.ABSENT)))

    assert _sent_commands(role) == []
    assert role.signal == SourceSignal.ABSENT
    signals = [e for e in group.events if isinstance(e, SourceSignalChangedEvent)]
    assert [e.signal for e in signals] == [SourceSignal.PRESENT, SourceSignal.ABSENT]
    assert all(e.client_id == "src-1" for e in signals)


def test_repeated_signal_is_not_re_emitted() -> None:
    """Only signal changes are published."""
    group_role, group = _make_group_role()
    role = _make_role(group_role)
    for _ in range(3):
        role.on_client_state(
            ClientStatePayload(source=SourceStatePayload(signal=SourceSignal.PRESENT))
        )
    assert len([e for e in group.events if isinstance(e, SourceSignalChangedEvent)]) == 1


def test_start_command_does_not_survive_reconnect() -> None:
    """Streaming state is per-connection, so a reconnect clears the sent start."""
    group_role, _group = _make_group_role()
    role = _started_role(group_role)
    assert role.is_streaming_requested

    role.on_disconnect()
    role.on_connect()

    assert not role.is_streaming_requested
    assert role.stream_source is None


# --- Stream lifecycle ---


def test_stream_start_opens_a_stream_and_publishes_it() -> None:
    """An announced stream is published to the host as an event."""
    group_role, group = _make_group_role()
    role = _started_role(group_role)

    assert role.stream_source == _PCM_FORMAT
    started = [e for e in group.events if isinstance(e, SourceStreamStartedEvent)]
    assert len(started) == 1
    assert started[0].client_id == "src-1"
    assert started[0].audio_format.sample_rate == 48000
    assert started[0].audio_format.channels == 2
    assert group_role.active_streams == {"src-1": started[0].stream}


async def test_unsolicited_stream_start_is_a_protocol_error() -> None:
    """A source that streams without being asked is rejected and disconnected."""
    group_role, group = _make_group_role()
    role = _make_role(group_role)

    role.on_client_stream_start(ClientStreamStartPayload(source=_PCM_FORMAT))

    assert role.stream_source is None
    assert group_role.active_streams == {}
    assert not [e for e in group.events if isinstance(e, SourceStreamStartedEvent)]
    client: _FakeClient = role._client  # type: ignore[assignment]  # noqa: SLF001
    client.connection.disconnect.assert_called_once_with(retry_connection=False)


def test_stream_end_closes_the_stream() -> None:
    """client_stream/end clears the format and ends the published stream."""
    group_role, group = _make_group_role()
    role = _started_role(group_role)
    stream = group_role.active_streams["src-1"]

    role.on_client_stream_end(ClientStreamEndPayload())

    assert role.stream_source is None
    assert group_role.active_streams == {}
    assert [e.client_id for e in group.events if isinstance(e, SourceStreamEndedEvent)] == ["src-1"]
    assert stream._ended  # noqa: SLF001


def test_restart_replaces_the_previous_stream() -> None:
    """A client_stream/start while open closes the old stream and publishes a new one."""
    group_role, group = _make_group_role()
    role = _started_role(group_role)
    first = group_role.active_streams["src-1"]

    role.on_client_stream_start(
        ClientStreamStartPayload(
            source=ClientStreamStartSource(
                codec=AudioCodec.PCM, channels=1, sample_rate=44100, bit_depth=16
            )
        )
    )

    second = group_role.active_streams["src-1"]
    assert second is not first
    assert first._ended  # noqa: SLF001
    assert second.audio_format.sample_rate == 44100
    assert len([e for e in group.events if isinstance(e, SourceStreamStartedEvent)]) == 2
    assert len([e for e in group.events if isinstance(e, SourceStreamEndedEvent)]) == 1


def test_member_leave_closes_the_stream() -> None:
    """A source leaving the group ends its stream."""
    group_role, _group = _make_group_role()
    role = _started_role(group_role)
    group_role.subscribe(role)

    group_role.unsubscribe(role)

    assert group_role.active_streams == {}


def test_group_change_republishes_an_open_stream() -> None:
    """Moving a capturing source to another group hands the stream to that group's host."""
    old_role, old_group = _make_group_role()
    new_role, new_group = _make_group_role()
    role = _started_role(old_role)
    old_role.subscribe(role)
    client: _FakeClient = role._client  # type: ignore[assignment]  # noqa: SLF001

    client.group = new_group
    role.on_group_changed(new_group)

    assert old_role.active_streams == {}
    assert [e.client_id for e in old_group.events if isinstance(e, SourceStreamEndedEvent)] == [
        "src-1"
    ]
    started = [e for e in new_group.events if isinstance(e, SourceStreamStartedEvent)]
    assert len(started) == 1
    assert new_role.active_streams == {"src-1": started[0].stream}


def test_group_close_ends_every_stream() -> None:
    """Tearing down the group closes open streams instead of leaking them."""
    group_role, group = _make_group_role()
    _started_role(group_role)
    stream = group_role.active_streams["src-1"]

    group_role.close()

    assert group_role.active_streams == {}
    assert stream._ended  # noqa: SLF001
    assert [e.client_id for e in group.events if isinstance(e, SourceStreamEndedEvent)] == ["src-1"]


def test_undecodable_format_does_not_open_a_stream() -> None:
    """An announced format the server cannot decode is refused, not half-opened."""
    group_role, group = _make_group_role()
    role = _make_role(group_role)
    role.send_start_command()

    role.on_client_stream_start(
        ClientStreamStartPayload(
            source=ClientStreamStartSource(
                codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=7
            )
        )
    )

    assert group_role.active_streams == {}
    assert not [e for e in group.events if isinstance(e, SourceStreamStartedEvent)]


# --- Inbound binary ---


def test_binary_routes_only_valid_frames() -> None:
    """Audio frames reach the group only for the right type with an open stream."""
    group_role, _group = _make_group_role()
    group_role.push_audio = MagicMock()  # type: ignore[method-assign]
    role = _make_role(group_role)

    # Wrong message type: ignored.
    role.on_client_binary(99, 1000, b"\x00\x00\x00\x00")
    # Right type but no open stream: ignored.
    role.on_client_binary(12, 1000, b"\x00\x00\x00\x00")
    assert group_role.push_audio.call_count == 0

    role.send_start_command()
    role.on_client_stream_start(ClientStreamStartPayload(source=_PCM_FORMAT))
    role.on_client_binary(12, 2000, b"\x01\x02\x03\x04")
    group_role.push_audio.assert_called_once_with(role, 2000, b"\x01\x02\x03\x04")


def test_binary_is_rejected_while_unavailable() -> None:
    """Chunks from an unavailable source are dropped: its timestamps are untrustworthy."""
    group_role, _group = _make_group_role()
    group_role.push_audio = MagicMock()  # type: ignore[method-assign]
    role = _started_role(group_role)
    client: _FakeClient = role._client  # type: ignore[assignment]  # noqa: SLF001

    for state in (ClientStateType.EXTERNAL_SOURCE, ClientStateType.ERROR):
        client.client_state = state
        role.on_client_binary(12, 2000, b"\x01\x02\x03\x04")
    assert group_role.push_audio.call_count == 0

    client.client_state = ClientStateType.SYNCHRONIZED
    role.on_client_binary(12, 3000, b"\x01\x02\x03\x04")
    assert group_role.push_audio.call_count == 1


def test_push_audio_without_open_stream_is_a_noop() -> None:
    """Frames arriving after the stream closed are discarded, not queued."""
    group_role, _group = _make_group_role()
    role = _make_role(group_role)
    group_role.push_audio(role, 1000, b"\x01\x02")  # no exception


# --- Decoding and the host-facing stream ---


def test_source_decoder_pcm_passthrough() -> None:
    """PCM frames pass through unchanged with the declared format."""
    decoder = SourceDecoder(_PCM_FORMAT)
    assert decoder.decode(b"\x01\x02\x03\x04") == [b"\x01\x02\x03\x04"]
    assert decoder.decode(b"") == []
    assert decoder.audio_format.sample_rate == 48000
    assert decoder.audio_format.channels == 2
    assert decoder.wire_bytes == 2
    assert decoder.frame_bytes == 4


async def test_stream_yields_decoded_chunks_with_capture_time() -> None:
    """The host receives PCM tagged with the reported capture timestamp."""
    group_role, group = _make_group_role()
    role = _started_role(group_role)
    started = next(e for e in group.events if isinstance(e, SourceStreamStartedEvent))

    pcm = b"\x00\x01" * 4  # 8 bytes = 2 stereo frames (2ch * 2 bytes)
    group_role.push_audio(role, 5000, pcm)
    group_role.end_stream(role)

    chunks = [chunk async for chunk in started.stream]
    assert len(chunks) == 1
    assert chunks[0].pcm == pcm
    assert chunks[0].timestamp_us == 5000
    assert chunks[0].duration_us == round(2 * 1_000_000 / 48000)


async def test_stream_reanchors_on_every_capture_timestamp() -> None:
    """Each frame carries its own reported time, so gaps stay visible to the host."""
    group_role, group = _make_group_role()
    role = _started_role(group_role)
    started = next(e for e in group.events if isinstance(e, SourceStreamStartedEvent))

    pcm = b"\x00\x01" * 4
    group_role.push_audio(role, 5000, pcm)
    group_role.push_audio(role, 900_000, pcm)  # a gap, not a continuation
    group_role.end_stream(role)

    timestamps = [chunk.timestamp_us async for chunk in started.stream]
    assert timestamps == [5000, 900_000]


async def test_stream_iteration_ends_when_the_source_stops() -> None:
    """A consumer waiting on a live stream is released when the stream ends."""
    group_role, group = _make_group_role()
    role = _started_role(group_role)
    started = next(e for e in group.events if isinstance(e, SourceStreamStartedEvent))

    received: list[int] = []

    async def consume() -> None:
        received.extend([chunk.timestamp_us async for chunk in started.stream])

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    group_role.push_audio(role, 1234, b"\x00\x01" * 4)
    await asyncio.sleep(0)
    group_role.end_stream(role)
    await asyncio.wait_for(task, timeout=1)

    assert received == [1234]


async def test_stream_drops_oldest_when_the_consumer_falls_behind() -> None:
    """A bounded queue keeps live audio flowing instead of growing without limit."""
    decoder = SourceDecoder(_PCM_FORMAT)
    stream = SourceAudioStream(client_id="src-1", decoder=decoder, maxsize=2)

    for ts in (1, 2, 3, 4):
        stream.push(ts, b"\x00\x01" * 4)
    stream.end()

    timestamps = [chunk.timestamp_us async for chunk in stream]
    assert timestamps == [3, 4]
    assert stream.dropped_frames == 2


async def test_stream_survives_an_undecodable_frame() -> None:
    """A frame that fails to decode is skipped without killing the stream."""
    decoder = SourceDecoder(_PCM_FORMAT)
    stream = SourceAudioStream(client_id="src-1", decoder=decoder)
    decoder.decode = MagicMock(side_effect=[RuntimeError("boom"), [b"\x00\x01" * 4]])  # type: ignore[method-assign]

    stream.push(1, b"bad")
    stream.push(2, b"good")
    stream.end()

    timestamps = [chunk.timestamp_us async for chunk in stream]
    assert timestamps == [2]


def test_push_after_end_is_ignored() -> None:
    """Late frames on a closed stream are dropped rather than resurrecting it."""
    stream = SourceAudioStream(client_id="src-1", decoder=SourceDecoder(_PCM_FORMAT))
    stream.end()
    stream.push(1, b"\x00\x01" * 4)
    assert stream._queue.qsize() == 1  # noqa: SLF001  (just the end sentinel)


def test_source_role_state_defaults() -> None:
    """SourceRoleState is constructible with defaults (for get_or_create_role_state)."""
    state = SourceRoleState()
    assert state.stream_source is None
    assert state.signal is None
    assert state.commanded is None
