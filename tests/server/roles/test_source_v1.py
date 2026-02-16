"""Tests for SourceV1Role and SourceGroupRole behavior."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from aiosendspin.models.core import ClientCommandPayload, InputStreamStartPayload
from aiosendspin.models.source import InputStreamStartSource, SourceClientCommandPayload
from aiosendspin.models.types import (
    AudioCodec,
    BinaryMessageType,
    SourceClientCommand,
    SourceStateType,
)
from aiosendspin.server.roles import SourceGroupRole, SourceV1Role


def _make_client_stub() -> MagicMock:
    client = MagicMock()
    state_store: dict[str, object] = {}

    def get_or_create_role_state(family: str, cls: type[object]) -> object:
        state_store.setdefault(family, cls())
        return state_store[family]

    client.get_or_create_role_state.side_effect = get_or_create_role_state
    client.client_id = "source-client"
    server = MagicMock()
    server.clock.now_us.return_value = 123
    client._server = server  # noqa: SLF001
    return client


def _make_stream_start_source() -> InputStreamStartSource:
    return InputStreamStartSource(codec=AudioCodec.PCM, channels=2, sample_rate=48000, bit_depth=16)


def test_source_role_binary_enqueues_when_streaming() -> None:
    """Binary source chunks are enqueued only while source state is streaming."""
    client = _make_client_stub()
    role = SourceV1Role(client=client)
    role._group_role = MagicMock()  # noqa: SLF001
    state = role._get_state()  # noqa: SLF001
    state.state = SourceStateType.STREAMING
    state.input_stream_format = _make_stream_start_source()

    role.on_client_binary(
        message_type=BinaryMessageType.SOURCE_AUDIO_CHUNK.value,
        timestamp_us=999,
        payload=b"frame",
    )

    role._group_role.enqueue.assert_called_once_with(role, 999, b"frame")  # type: ignore[union-attr]  # noqa: SLF001


def test_source_role_binary_ignored_when_not_streaming() -> None:
    """Binary source chunks are ignored when state is not streaming."""
    client = _make_client_stub()
    role = SourceV1Role(client=client)
    role._group_role = MagicMock()  # noqa: SLF001
    state = role._get_state()  # noqa: SLF001
    state.state = SourceStateType.IDLE
    state.input_stream_format = _make_stream_start_source()

    role.on_client_binary(
        message_type=BinaryMessageType.SOURCE_AUDIO_CHUNK.value,
        timestamp_us=999,
        payload=b"frame",
    )

    role._group_role.enqueue.assert_not_called()  # type: ignore[union-attr]  # noqa: SLF001


def test_source_role_input_stream_start_resets_decoder() -> None:
    """input_stream/start updates format and resets decoder state."""
    client = _make_client_stub()
    role = SourceV1Role(client=client)
    role._group_role = MagicMock()  # noqa: SLF001

    role.on_input_stream_start(InputStreamStartPayload(source=_make_stream_start_source()))

    assert role.input_stream_format is not None
    role._group_role.clear_decoder.assert_called_once_with(role)  # type: ignore[union-attr]  # noqa: SLF001


def test_source_role_command_updates_last_event_and_pushes_state() -> None:
    """client/command source events are persisted and trigger controller refresh."""
    client = _make_client_stub()
    role = SourceV1Role(client=client)
    role._group_role = MagicMock()  # noqa: SLF001

    role.on_command(
        ClientCommandPayload(source=SourceClientCommandPayload(command=SourceClientCommand.STARTED))
    )

    state = role._get_state()  # noqa: SLF001
    assert state.last_event == SourceClientCommand.STARTED
    assert state.last_event_ts_us == 123
    role._group_role.push_state.assert_called_once()  # type: ignore[union-attr]  # noqa: SLF001


class _DummyGroup:
    """Minimal group stub for SourceGroupRole tests."""

    def __init__(self) -> None:
        self._push_stream: object | None = object()
        self.stop_calls = 0

    def group_role(self, _family: str) -> None:
        return None

    async def stop(self) -> bool:
        self.stop_calls += 1
        return True


def _make_role_stub(client_id: str) -> MagicMock:
    role = MagicMock()
    role._client = MagicMock()  # noqa: SLF001
    role._client.client_id = client_id  # noqa: SLF001
    return role


def test_source_group_stop_source_ignores_inactive_role() -> None:
    """stop_source should not stop group stream for non-active sources."""
    group = _DummyGroup()
    source_group = SourceGroupRole(group)
    source_group._active_source_id = "active-id"  # noqa: SLF001

    asyncio.run(source_group.stop_source(_make_role_stub("other-id")))

    assert group.stop_calls == 0
    assert source_group._active_source_id == "active-id"  # noqa: SLF001


def test_source_group_stop_source_stops_active_role() -> None:
    """stop_source should stop group stream for the active source."""
    group = _DummyGroup()
    source_group = SourceGroupRole(group)
    source_group._active_source_id = "active-id"  # noqa: SLF001

    asyncio.run(source_group.stop_source(_make_role_stub("active-id")))

    assert group.stop_calls == 1
    assert source_group._active_source_id is None  # noqa: SLF001
