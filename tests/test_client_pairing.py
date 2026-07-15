"""Tests for the client's pair-method cross-check (spec server/hello enforcement)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.core import ServerActivatePayload
from aiosendspin.models.types import (
    MediaCommand,
    PairAbortReason,
    PairMethod,
    Roles,
)
from aiosendspin.noise.keys import b64url_encode
from aiosendspin.noise.models import (
    PairAbortMessage,
    ServerPairAuthMessage,
    ServerPairAuthPayload,
)
from aiosendspin.noise.pairing import PairingError
from aiosendspin.noise.trust_store import PskCategory, ResolvedPsk

from .conftest import make_sdk_client


class _FakeWS:
    """Captures sent text frames; satisfies the bits of EncryptedWebSocket used here."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> bool:
        self.closed = True
        return True

    def exception(self) -> BaseException | None:
        return None


def _client_with(category: PskCategory) -> tuple[SendspinConnection, _FakeWS]:
    client = make_sdk_client(client_name="C", roles=[Roles.CONTROLLER])
    connection = SendspinConnection(client)
    ws = _FakeWS()
    connection._ws = ws  # type: ignore[assignment]  # noqa: SLF001
    connection._server_id = "server-1"  # noqa: SLF001
    connection._noise_psk = ResolvedPsk("psk-id", b"\x00" * 32, category)  # noqa: SLF001
    return connection, ws


async def test_pairing_psk_method_accepted_on_pairing_psk() -> None:
    """A Pairing-PSK match with selected_pair_method=pairing_psk passes the cross-check."""
    connection, ws = _client_with(PskCategory.PAIRING)
    await connection._validate_pair_method(PairMethod.PAIRING_PSK)  # noqa: SLF001
    assert ws.sent == []


@pytest.mark.parametrize(
    ("category", "method"),
    [
        (PskCategory.PAIRING, PairMethod.DYNAMIC_PIN),  # not offered by this client
        (PskCategory.LONG_TERM, PairMethod.PAIRING_PSK),  # not allowed for long-term PSK
        (PskCategory.PAIRING, None),  # missing when connection_reason is pairing
    ],
)
async def test_invalid_pair_method_aborts(category: PskCategory, method: PairMethod | None) -> None:
    """A disallowed/unoffered/missing method sends pair/abort and raises."""
    connection, ws = _client_with(category)
    with pytest.raises(PairingError):
        await connection._validate_pair_method(method)  # noqa: SLF001
    abort = PairAbortMessage.from_json(ws.sent[0])
    assert abort.payload.reason is PairAbortReason.METHOD_NOT_SUPPORTED


async def test_stray_pairing_frame_is_discarded_quietly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pairing frame arriving outside an exchange is discarded, not treated as an error."""
    connection, ws = _client_with(PskCategory.LONG_TERM)
    frame = ServerPairAuthMessage(
        payload=ServerPairAuthPayload(pake_msg_1=b64url_encode(b"\x00" * 32)),
    ).to_json()
    with caplog.at_level(logging.DEBUG):
        await connection._handle_json_message(frame)  # noqa: SLF001
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert ws.sent == []


async def test_app_and_time_sends_suppressed_during_exchange() -> None:
    """While an in-band exchange owns the wire, app and time-sync sends are withheld.

    Otherwise they would interleave with the unlocked handshake/pairing sends and desync the
    Noise nonce. Player state is still recorded so the post-exchange resync replays it.
    """
    connection, ws = _client_with(PskCategory.LONG_TERM)
    connection._connected = True  # noqa: SLF001

    connection._exchange_in_progress = True  # noqa: SLF001
    await connection.send_player_state(available=True, volume=7, muted=True)
    await connection.send_group_command(MediaCommand.PLAY)
    await connection._send_time_message()  # noqa: SLF001
    assert ws.sent == []
    assert connection._reported_volume == 7  # noqa: SLF001
    assert connection._reported_muted is True  # noqa: SLF001

    connection._exchange_in_progress = False  # noqa: SLF001
    await connection.send_player_state(available=True, volume=7, muted=True)
    assert len(ws.sent) == 1


async def _cancel_time_task(connection: SendspinConnection) -> None:
    task = connection._time_task  # noqa: SLF001
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def test_leave_activate_resumes_time_sync() -> None:
    """A server/activate that returns the connection to normal service restarts time sync."""
    connection, _ws = _client_with(PskCategory.LONG_TERM)
    connection._connected = True  # noqa: SLF001
    assert connection._time_task is None  # noqa: SLF001

    try:
        await connection._handle_server_activate(  # noqa: SLF001
            ServerActivatePayload(activities=[], active_roles=[])
        )
        assert connection._time_task is not None  # noqa: SLF001
        assert not connection._time_task.done()  # noqa: SLF001
    finally:
        await _cancel_time_task(connection)
