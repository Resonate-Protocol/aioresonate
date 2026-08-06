"""The client ends the input stream when the server drops source from active_roles."""

from __future__ import annotations

import asyncio

import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.core import ServerActivatePayload
from aiosendspin.models.types import Activity, GoodbyeReason, Roles
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk


class _FakeWs:
    closed = False

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.sent_bytes: list[bytes] = []

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)


class _FakeClient:
    def note_playback_activity(self, _conn: object) -> None:
        pass


def _connection(
    ws: _FakeWs,
    *,
    active_roles: list[str],
    stream_active: bool,
    category: PskCategory = PskCategory.LONG_TERM,
    unpaired_access: bool = False,
) -> SendspinConnection:
    conn = SendspinConnection.__new__(SendspinConnection)
    conn._client = _FakeClient()  # type: ignore[assignment]  # noqa: SLF001
    conn._ws = ws  # type: ignore[assignment]  # noqa: SLF001
    conn._connected = True  # noqa: SLF001
    conn._send_lock = asyncio.Lock()  # noqa: SLF001
    conn._exchange_in_progress = False  # noqa: SLF001
    conn._noise_psk = ResolvedPsk("id", b"\x00" * 32, category)  # noqa: SLF001
    conn._active_roles = active_roles  # noqa: SLF001
    conn._source_stream_active = stream_active  # noqa: SLF001
    conn._selected_pair_method = None  # noqa: SLF001

    async def _unpaired_access_enabled() -> bool:
        return unpaired_access

    conn._unpaired_access_enabled = _unpaired_access_enabled  # type: ignore[method-assign]  # noqa: SLF001
    return conn


async def test_source_dropped_from_active_roles_ends_client_stream() -> None:
    """Removing source@v1 from active_roles auto-sends client_stream/end."""
    ws = _FakeWs()
    conn = _connection(ws, active_roles=[Roles.SOURCE.value], stream_active=True)
    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[])
    )
    assert any("client_stream/end" in m for m in ws.sent)
    assert conn._source_stream_active is False  # noqa: SLF001


async def test_no_client_stream_end_when_source_retained() -> None:
    """Keeping source@v1 active does not end the input stream."""
    ws = _FakeWs()
    conn = _connection(ws, active_roles=[Roles.SOURCE.value], stream_active=True)
    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[Roles.SOURCE.value])
    )
    assert not any("client_stream/end" in m for m in ws.sent)
    assert conn._source_stream_active is True  # noqa: SLF001


async def test_no_client_stream_end_when_no_stream_active() -> None:
    """Dropping source with no active input stream sends nothing."""
    ws = _FakeWs()
    conn = _connection(ws, active_roles=[Roles.SOURCE.value], stream_active=False)
    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[])
    )
    assert ws.sent == []


async def test_unpaired_activation_rejects_source_role() -> None:
    """Unpaired connections cannot activate the source role."""
    ws = _FakeWs()
    conn = _connection(
        ws,
        active_roles=[],
        stream_active=False,
        category=PskCategory.SENTINEL,
        unpaired_access=True,
    )

    reason = await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[Roles.SOURCE.value])
    )

    assert reason is GoodbyeReason.UNAUTHORIZED


async def test_source_chunks_rejected_after_role_deactivation() -> None:
    """Deactivated source roles cannot send more audio chunks."""
    ws = _FakeWs()
    conn = _connection(ws, active_roles=[Roles.SOURCE.value], stream_active=True)
    await conn._apply_activation(  # noqa: SLF001
        ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[])
    )

    with pytest.raises(RuntimeError, match="not active"):
        await conn.send_source_chunk(b"audio", timestamp_us=1)

    assert ws.sent_bytes == []
