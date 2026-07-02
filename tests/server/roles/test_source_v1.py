"""Tests for the server-side source@v1 role."""

from __future__ import annotations

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
from aiosendspin.models.types import AudioCodec, SourceCommand, SourceSignal
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.roles.negotiation import negotiate_active_roles
from aiosendspin.server.roles.registry import GROUP_ROLE_FACTORIES, ROLE_FACTORIES
from aiosendspin.server.roles.source.group import SourceDecoder, SourceGroupRole, SourceIngress
from aiosendspin.server.roles.source.v1 import SourceRoleState, SourceV1Role

if TYPE_CHECKING:
    from aiosendspin.models.types import ServerMessage

_PCM_FORMAT = ClientStreamStartSource(
    codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16
)


class _FakeClient:
    """Minimal stand-in for the server-side SendspinClient."""

    def __init__(self, client_id: str, group: object) -> None:
        self.client_id = client_id
        self.connection = object()  # non-None -> has_connection() True
        self.group = group
        self._role_state: dict[str, object] = {}
        self.sent: list[tuple[str, ServerMessage]] = []

    def send_role_message(self, family: str, message: object) -> None:
        self.sent.append((family, message))  # type: ignore[arg-type]

    def get_or_create_role_state(self, family: str, cls: type[object]) -> object:
        if family not in self._role_state:
            self._role_state[family] = cls()
        return self._role_state[family]


def _make_role(group_role: SourceGroupRole | None = None) -> SourceV1Role:
    group = MagicMock()
    role = SourceV1Role(client=_FakeClient("src-1", group))  # type: ignore[arg-type]
    if group_role is not None:
        role._group_role = group_role  # noqa: SLF001
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


def test_on_client_stream_start_records_format() -> None:
    """client_stream/start stores the announced format."""
    role = _make_role()
    role.on_client_stream_start(ClientStreamStartPayload(source=_PCM_FORMAT))
    assert role.stream_source == _PCM_FORMAT


def test_on_client_stream_end_clears_format() -> None:
    """client_stream/end clears the active format."""
    role = _make_role()
    role.on_client_stream_start(ClientStreamStartPayload(source=_PCM_FORMAT))
    role.on_client_stream_end(ClientStreamEndPayload())
    assert role.stream_source is None


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


def test_on_client_state_follows_line_sense() -> None:
    """A reported ABSENT signal stops the source; PRESENT starts it."""
    role = _make_role()
    role.on_client_state(ClientStatePayload(source=SourceStatePayload(signal=SourceSignal.PRESENT)))
    role.on_client_state(ClientStatePayload(source=SourceStatePayload(signal=SourceSignal.ABSENT)))
    assert _sent_commands(role) == [SourceCommand.START, SourceCommand.STOP]
    assert role.signal == SourceSignal.ABSENT


def test_on_client_binary_routes_only_valid_frames() -> None:
    """Audio frames route to the group only for the right type with an active stream."""
    group_role = SourceGroupRole(MagicMock())
    group_role.enqueue = MagicMock()  # type: ignore[method-assign]
    role = _make_role(group_role)

    # Wrong message type: ignored.
    role.on_client_binary(99, 1000, b"\x00\x00\x00\x00")
    # Right type but no active stream: ignored.
    role.on_client_binary(12, 1000, b"\x00\x00\x00\x00")
    assert group_role.enqueue.call_count == 0

    role.on_client_stream_start(ClientStreamStartPayload(source=_PCM_FORMAT))
    role.on_client_binary(12, 2000, b"\x01\x02\x03\x04")
    group_role.enqueue.assert_called_once_with(role, 2000, b"\x01\x02\x03\x04")


def test_group_on_member_join_starts_source() -> None:
    """The default group policy asks a joining source to start streaming."""
    group_role = SourceGroupRole(MagicMock())
    role = _make_role(group_role)
    group_role.on_member_join(role)
    assert _sent_commands(role) == [SourceCommand.START]


def test_source_decoder_pcm_passthrough() -> None:
    """PCM frames pass through unchanged with the declared format."""
    decoder = SourceDecoder(_PCM_FORMAT)
    assert decoder.decode(b"\x01\x02\x03\x04") == [b"\x01\x02\x03\x04"]
    assert decoder.decode(b"") == []
    assert decoder.audio_format.sample_rate == 48000
    assert decoder.audio_format.channels == 2
    assert decoder.wire_bytes == 2


async def test_ingest_pcm_pushes_to_group_stream() -> None:
    """Ingesting a PCM frame starts the group stream and commits with the header time."""
    push_stream = MagicMock()
    push_stream.is_stopped = False
    push_stream.commit_audio = AsyncMock(return_value=0)
    group = MagicMock()
    group.start_stream.return_value = push_stream

    group_role = SourceGroupRole(group)
    role = _make_role(group_role)
    role.on_client_stream_start(ClientStreamStartPayload(source=_PCM_FORMAT))

    pcm = b"\x00\x01" * 4  # 8 bytes = 2 stereo frames (2ch * 2 bytes)
    await group_role._ingest_one(SourceIngress(role=role, timestamp_us=5000, data=pcm))  # noqa: SLF001

    group.start_stream.assert_called_once()
    args, kwargs = push_stream.prepare_audio.call_args
    assert args[0] == pcm
    assert kwargs["channel_id"] == MAIN_CHANNEL
    push_stream.commit_audio.assert_awaited_once_with(play_start_us=5000)
    # 2 frames at 48 kHz -> ~42 us advance for the next chunk.
    assert group_role._next_play_us == 5000 + round(2 * 1_000_000 / 48000)  # noqa: SLF001


def test_stop_source_stops_active_stream() -> None:
    """Ending the active source stops the group stream."""
    push_stream = MagicMock()
    push_stream.is_stopped = False
    group = MagicMock()

    group_role = SourceGroupRole(group)
    group_role._active_source_id = "src-1"  # noqa: SLF001
    group_role._push_stream = push_stream  # noqa: SLF001
    role = _make_role(group_role)

    group_role.stop_source(role)

    group.stop_stream.assert_called_once()
    assert group_role._active_source_id is None  # noqa: SLF001


def test_source_role_state_defaults() -> None:
    """SourceRoleState is constructible with defaults (for get_or_create_role_state)."""
    state = SourceRoleState()
    assert state.stream_source is None
    assert state.signal is None
    assert state.commanded is None
