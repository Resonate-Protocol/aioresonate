"""Tests for :mod:`aiosendspin.noise.pairing`."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from cpace import CPace, CPaceRole

from aiosendspin.models.core import ServerActivateMessage, ServerActivatePayload
from aiosendspin.models.types import Activity, PairAbortReason, PairMethod
from aiosendspin.noise import pin as pin_mod
from aiosendspin.noise.keys import b64url_decode, b64url_encode, generate_psk, psk_id_for
from aiosendspin.noise.models import (
    ClientPairAuthMessage,
    ClientPairAuthPayload,
    ClientPairConfirmMessage,
    ClientPairConfirmPayload,
    ClientPairFinalizeMessage,
    ClientPairFinalizePayload,
    ClientPairInitMessage,
    ClientPairInitPayload,
    ClientPairPendingMessage,
    ClientPairPendingPayload,
    ServerPairAuthMessage,
    ServerPairAuthPayload,
    ServerPairInitMessage,
)
from aiosendspin.noise.pairing import (
    PairingAbortError,
    PairingAttempt,
    PairingError,
    PairingTimeoutError,
    run_dynamic_pin_client,
    run_dynamic_pin_server,
    run_pairing_psk_client,
    run_pairing_psk_server,
    run_static_pin_client,
    run_static_pin_server,
)
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    InMemoryClientPairingStore,
    InMemoryServerPairingStore,
    ServerPairingRecord,
)
from aiosendspin.noise.wire import EncryptedWebSocket
from tests.noise.conftest import make_paired_encrypted_ws
from tests.pairing_stores import ExhaustedClientStore


def _added_records(records: Sequence[ClientPairingRecord]) -> list[ClientPairingRecord]:
    """Stored-pubkey records added by pairing (excludes the pre-provisioned shared record)."""
    return [r for r in records if r.server_id is not None]


async def _pin() -> str:
    return "000000"


def test_pairing_attempt_verify_is_pin_only() -> None:
    """Verify is a PIN-only re-authentication flag; PAIRING_PSK rejects it."""
    with pytest.raises(ValueError, match="does not support verification"):
        PairingAttempt(method=PairMethod.PAIRING_PSK, pairing_psk=generate_psk(), verify=True)
    assert PairingAttempt(method=PairMethod.STATIC_PIN, pin_provider=_pin, verify=True).verify
    assert PairingAttempt(method=PairMethod.DYNAMIC_PIN, pin_provider=_pin, verify=True).verify


def test_pairing_attempt_pairing_psk_requires_material() -> None:
    """PAIRING_PSK must carry a 32-byte pairing_psk and no PIN-flow hooks."""
    with pytest.raises(ValueError, match="requires pairing_psk"):
        PairingAttempt(method=PairMethod.PAIRING_PSK)
    with pytest.raises(ValueError, match="must be 32 bytes"):
        PairingAttempt(method=PairMethod.PAIRING_PSK, pairing_psk=b"\x01" * 16)
    with pytest.raises(ValueError, match="does not use pin_provider"):
        PairingAttempt(method=PairMethod.PAIRING_PSK, pairing_psk=generate_psk(), pin_provider=_pin)
    with pytest.raises(ValueError, match="does not use on_pair_pending"):
        PairingAttempt(
            method=PairMethod.PAIRING_PSK,
            pairing_psk=generate_psk(),
            on_pair_pending=lambda: None,
        )


@pytest.mark.parametrize("method", [PairMethod.DYNAMIC_PIN, PairMethod.STATIC_PIN])
def test_pairing_attempt_pin_methods_require_pin_provider(method: PairMethod) -> None:
    """PIN methods must carry a pin_provider and must not carry a pairing_psk."""
    with pytest.raises(ValueError, match="requires pin_provider"):
        PairingAttempt(method=method)
    with pytest.raises(ValueError, match="does not use pairing_psk"):
        PairingAttempt(method=method, pin_provider=_pin, pairing_psk=generate_psk())


def test_pairing_attempt_languages_are_dynamic_pin_only() -> None:
    """The spoken-emission hint belongs to dynamic PIN; the other methods reject it."""
    assert PairingAttempt(
        method=PairMethod.DYNAMIC_PIN, pin_provider=_pin, languages=("ca", "en")
    ).languages == ("ca", "en")
    with pytest.raises(ValueError, match="does not use languages"):
        PairingAttempt(method=PairMethod.STATIC_PIN, pin_provider=_pin, languages=("en",))
    with pytest.raises(ValueError, match="does not use languages"):
        PairingAttempt(method=PairMethod.PAIRING_PSK, pairing_psk=generate_psk(), languages=("en",))


_paired_encrypted_ws = make_paired_encrypted_ws


async def test_pairing_psk_finalize_round_trip() -> None:
    """Both sides persist matching records carrying the client-generated long-term PSK."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()

    _client_ret, server_record = await asyncio.gather(
        run_pairing_psk_client(
            client_ews,
            server_id="server-X",
            store=client_store,
        ),
        run_pairing_psk_server(server_ews, client_id="client-A", store=server_store),
    )

    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    # The same long-term PSK is recorded on both sides.
    assert client_record.psk == server_record.psk
    assert client_record.psk_id == server_record.psk_id
    # Directional counterparties.
    assert client_record.server_id == "server-X"
    assert server_record.client_id == "client-A"
    # The server persisted its record too.
    assert await server_store.record_by_client_id("client-A") == server_record
    # A first pairing records the establishing method.
    assert server_record.pair_methods == [PairMethod.PAIRING_PSK]


async def test_pairing_psk_client_times_out_without_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Pairing PSK client aborts with attempt_timeout if the server never finalizes."""
    monkeypatch.setattr("aiosendspin.noise.pairing._CLIENT_ATTEMPT_TIMEOUT_S", 0.05)
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()

    async def silent_server() -> None:
        await server_ews.receive()  # consume client/pair-finalize, then never reply
        await asyncio.sleep(0.5)

    with pytest.raises(PairingAbortError) as excinfo:
        await asyncio.gather(
            run_pairing_psk_client(client_ews, server_id="server-X", store=client_store),
            silent_server(),
        )
    assert excinfo.value.reason is PairAbortReason.ATTEMPT_TIMEOUT
    assert await client_store.record_by_server_id("server-X") is None


async def test_pairing_psk_server_times_out_without_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Pairing PSK server times out locally, sending nothing, if the client never finalizes."""
    monkeypatch.setattr("aiosendspin.noise.pairing.SERVER_FIRST_MESSAGE_TIMEOUT_S", 0.05)
    _client_ews, server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    with pytest.raises(PairingTimeoutError):
        await run_pairing_psk_server(server_ews, client_id="client-X", store=server_store)
    assert server_raw.sent == []
    assert await server_store.record_by_client_id("client-X") is None


async def test_static_pin_server_first_message_wait_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The static-PIN server times out locally awaiting the client's first pairing message."""
    monkeypatch.setattr("aiosendspin.noise.pairing.SERVER_FIRST_MESSAGE_TIMEOUT_S", 0.05)
    _client_ews, server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    with pytest.raises(PairingTimeoutError):
        await run_static_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pin_provider=_pin,
            client_id="client-X",
            store=server_store,
        )
    assert server_raw.sent == []


async def test_pair_pending_extends_the_first_message_wait() -> None:
    """A matching pair-pending switches the server to the gesture timeout; pairing completes."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    pending_signals = 0

    def on_pending() -> None:
        nonlocal pending_signals
        pending_signals += 1

    async def gated_client() -> None:
        await client_ews.send_str(
            ClientPairPendingMessage(payload=ClientPairPendingPayload(pairing_index=0)).to_json()
        )
        await run_static_pin_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            static_pin=_STATIC_PIN,
            server_id="server-X",
            store=client_store,
        )

    async def provide() -> str:
        return _STATIC_PIN

    _client_ret, server_record = await asyncio.gather(
        gated_client(),
        run_static_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pin_provider=provide,
            client_id="client-A",
            store=server_store,
            on_pair_pending=on_pending,
        ),
    )
    assert server_record is not None
    assert await server_store.record_by_client_id("client-A") == server_record
    assert pending_signals == 1


async def test_gesture_wait_times_out_after_pair_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After pair-pending the gesture bound applies; its expiry raises locally."""
    monkeypatch.setattr("aiosendspin.noise.pairing.SERVER_GESTURE_TIMEOUT_S", 0.05)
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(
        ClientPairPendingMessage(payload=ClientPairPendingPayload(pairing_index=0)).to_json()
    )
    with pytest.raises(PairingTimeoutError):
        await run_static_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pin_provider=_pin,
            client_id="client-X",
            store=server_store,
        )


async def test_stale_pair_pending_is_discarded() -> None:
    """A pair-pending left over from a superseded activate is ignored; the fresh init pairs."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    pending_signals = 0

    def on_pending() -> None:
        nonlocal pending_signals
        pending_signals += 1

    await client_ews.send_str(
        ClientPairPendingMessage(payload=ClientPairPendingPayload(pairing_index=0)).to_json()
    )

    async def provide() -> str:
        return _STATIC_PIN

    _client_ret, server_record = await asyncio.gather(
        run_static_pin_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=1,
            static_pin=_STATIC_PIN,
            server_id="server-X",
            store=client_store,
        ),
        run_static_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=1,
            pin_provider=provide,
            client_id="client-A",
            store=server_store,
            on_pair_pending=on_pending,
        ),
    )
    assert server_record is not None
    assert pending_signals == 0  # the stale pending is not surfaced


async def test_repeated_pair_pending_is_protocol_error() -> None:
    """A second matching pair-pending within one attempt is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    for _ in range(2):
        await client_ews.send_str(
            ClientPairPendingMessage(payload=ClientPairPendingPayload(pairing_index=0)).to_json()
        )
    with pytest.raises(PairingError, match="expected ClientPairInitMessage"):
        await run_static_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pin_provider=_pin,
            client_id="client-A",
            store=server_store,
        )


async def test_pair_init_index_mismatch_after_pending_is_protocol_error() -> None:
    """After the matching pair-pending, an init for a different attempt is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(
        ClientPairPendingMessage(payload=ClientPairPendingPayload(pairing_index=0)).to_json()
    )
    await client_ews.send_str(
        ClientPairInitMessage(payload=ClientPairInitPayload(pairing_index=1)).to_json()
    )
    with pytest.raises(PairingError, match="does not match the attempt"):
        await run_static_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pin_provider=_pin,
            client_id="client-A",
            store=server_store,
        )


async def test_pair_pending_ahead_of_server_count_is_protocol_error() -> None:
    """A pair-pending with a pairing_index the server has not reached yet is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(
        ClientPairPendingMessage(payload=ClientPairPendingPayload(pairing_index=1)).to_json()
    )
    with pytest.raises(PairingError, match="ahead of the server's count"):
        await run_static_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pin_provider=_pin,
            client_id="client-A",
            store=server_store,
        )


async def test_static_pin_server_rejects_non_8_digit_operator_pin() -> None:
    """A non-8-digit operator PIN aborts the server before it emits its PAKE share."""
    client_ews, server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    async def bad_pin() -> str:
        return "12345"

    await client_ews.send_str(
        ClientPairInitMessage(payload=ClientPairInitPayload(pairing_index=0)).to_json(),
    )
    with pytest.raises(PairingError, match="8 decimal digits"):
        await run_static_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pin_provider=bad_pin,
            client_id="client-X",
            store=server_store,
        )
    assert server_raw.sent == []
    assert await server_store.record_by_client_id("client-X") is None


async def test_static_pin_server_rejects_dynamic_only_commit_b() -> None:
    """A static-PIN pair-init carrying commit_B is a protocol error."""
    client_ews, server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(
        ClientPairInitMessage(
            payload=ClientPairInitPayload(
                pairing_index=0,
                commit_B=b64url_encode(pin_mod.commit(pin_mod.generate_nonce())),
            )
        ).to_json(),
    )
    with pytest.raises(PairingError, match="commit_B for static PIN"):
        await run_static_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pin_provider=_pin,
            client_id="client-X",
            store=server_store,
        )
    assert server_raw.sent == []
    assert await server_store.record_by_client_id("client-X") is None


async def test_dynamic_pin_server_times_out_mid_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dynamic-PIN server times out locally, with no abort on the wire, if the client stalls."""
    monkeypatch.setattr("aiosendspin.noise.pairing.SERVER_ATTEMPT_TIMEOUT_S", 0.05)
    client_ews, server_ews, client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(
        ClientPairInitMessage(
            payload=ClientPairInitPayload(
                pairing_index=0, commit_B=b64url_encode(pin_mod.commit(pin_mod.generate_nonce()))
            ),
        ).to_json(),
    )
    with pytest.raises(PairingTimeoutError):
        await run_dynamic_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pin_provider=_pin,
            pin_length=6,
            client_id="client-X",
            store=server_store,
        )
    await client_ews.receive()  # server/pair-init
    await client_ews.receive()  # server/pair-auth
    assert client_raw.incoming.qsize() == 0


async def test_finalize_rotate_preserves_birth_and_appends_method() -> None:
    """Re-pairing rotates the PSK but carries over created_at and appends the method."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    born = datetime(2020, 1, 1, tzinfo=UTC)
    seeded = ServerPairingRecord(
        psk_id="old",
        psk=generate_psk(),
        client_id="client-A",
        created_at=born,
        pair_methods=[PairMethod.DYNAMIC_PIN],
    )
    await server_store.store_record(seeded)

    _client_ret, rotated = await asyncio.gather(
        run_pairing_psk_client(client_ews, server_id="server-X", store=client_store),
        run_pairing_psk_server(server_ews, client_id="client-A", store=server_store),
    )

    assert rotated.psk != seeded.psk  # rotated onto a fresh PSK
    assert rotated.created_at == born  # birth time carried over
    assert rotated.pair_methods == [PairMethod.DYNAMIC_PIN, PairMethod.PAIRING_PSK]


async def test_client_finalize_raises_if_server_closes_before_ack() -> None:
    """If the server closes before sending server/pair-finalize, the client raises."""
    client_ews, _server_ews, _client_raw, server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()

    await server_raw.close_outbound()  # server never acks
    with pytest.raises(PairingError, match="closed while awaiting ServerPairFinalizeMessage"):
        await run_pairing_psk_client(
            client_ews,
            server_id="server-X",
            store=client_store,
        )
    # Nothing persisted on failure (only the pre-provisioned shared record remains).
    assert _added_records(await client_store.list_records()) == []


async def test_server_finalize_raises_if_client_closes_first() -> None:
    """If the client closes before sending client/pair-finalize, the server raises."""
    _client_ews, server_ews, client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_raw.close_outbound()  # client never sends client/pair-finalize
    with pytest.raises(PairingError, match="closed while awaiting ClientPairFinalizeMessage"):
        await run_pairing_psk_server(server_ews, client_id="client-A", store=server_store)
    assert await server_store.record_by_client_id("client-A") is None


_HANDSHAKE_HASH = bytes(range(32))


async def test_dynamic_pin_round_trip() -> None:
    """A matching PIN authenticates the PAKE and both sides persist the record."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def emit(pin: str) -> None:
        shown.set_result(pin)

    async def provide() -> str:
        return await shown  # operator types the PIN the client displayed

    _client_ret, server_record = await asyncio.gather(
        run_dynamic_pin_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pin_length=8,
            pin_emitter=emit,
            server_id="server-X",
            store=client_store,
        ),
        run_dynamic_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pin_length=8,
            pin_provider=provide,
            client_id="client-A",
            store=server_store,
        ),
    )

    assert server_record is not None  # a finalized pairing returns a record
    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    assert client_record.psk == server_record.psk
    assert client_record.psk_id == server_record.psk_id
    assert client_record.server_id == "server-X"
    assert server_record.client_id == "client-A"
    assert await server_store.record_by_client_id("client-A") == server_record


async def test_dynamic_pin_server_discards_stale_pair_init() -> None:
    """A pair-init left over from a superseded activate is discarded; the fresh one pairs."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def emit(pin: str) -> None:
        shown.set_result(pin)

    async def provide() -> str:
        return await shown

    stale = ClientPairInitMessage(
        payload=ClientPairInitPayload(
            pairing_index=0, commit_B=b64url_encode(pin_mod.commit(pin_mod.generate_nonce()))
        ),
    )
    await client_ews.send_str(stale.to_json())

    _client_ret, server_record = await asyncio.gather(
        run_dynamic_pin_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=1,
            pin_length=8,
            pin_emitter=emit,
            server_id="server-X",
            store=client_store,
        ),
        run_dynamic_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=1,
            pin_length=8,
            pin_provider=provide,
            client_id="client-A",
            store=server_store,
        ),
    )
    assert server_record is not None
    assert await server_store.record_by_client_id("client-A") == server_record


async def test_pair_init_ahead_of_server_count_is_protocol_error() -> None:
    """A pair-init with a pairing_index the server has not reached yet is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    await client_ews.send_str(
        ClientPairInitMessage(payload=ClientPairInitPayload(pairing_index=1)).to_json()
    )
    with pytest.raises(PairingError, match="ahead of the server's count"):
        await run_static_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pin_provider=_pin,
            client_id="client-A",
            store=server_store,
        )


async def test_dynamic_pin_wrong_pin_aborts_and_persists_nothing() -> None:
    """A PIN mismatch fails confirmation; both sides abort and store nothing."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def emit(pin: str) -> None:
        shown.set_result(pin)

    async def provide_wrong() -> str:
        pin = await shown
        wrong_first = "2" if pin[0] == "1" else "1"  # guaranteed different from the shown PIN
        return wrong_first + pin[1:]

    with pytest.raises(PairingAbortError) as excinfo:
        await asyncio.gather(
            run_dynamic_pin_client(
                client_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pin_length=8,
                pin_emitter=emit,
                server_id="server-X",
                store=client_store,
            ),
            run_dynamic_pin_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pin_length=8,
                pin_provider=provide_wrong,
                client_id="client-A",
                store=server_store,
            ),
        )

    assert excinfo.value.reason is PairAbortReason.PIN_MISMATCH
    assert await client_store.pin_failure_count() == 1
    assert _added_records(await client_store.list_records()) == []
    assert await server_store.record_by_client_id("client-A") is None


async def test_client_relays_leave_pairing_without_storing() -> None:
    """A server that leaves pairing makes the client relay the server/activate and store nothing."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    await client_store.record_pin_failure()  # a prior failure to be reset
    shown: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def emit(pin: str) -> None:
        shown.set_result(pin)

    async def provide() -> str:
        return await shown

    async def server_leaves_pairing() -> None:
        # Receive client/pair-finalize without finalizing, then leave pairing with a
        # server/activate (what the connection layer sends in place of an ack).
        await run_dynamic_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pin_length=8,
            pin_provider=provide,
            client_id="client-A",
            store=server_store,
            verify=True,
        )
        await server_ews.send_str(
            ServerActivateMessage(
                payload=ServerActivatePayload(activities=[Activity.PLAYBACK], active_roles=[]),
            ).to_json(),
        )

    leftover, _ = await asyncio.gather(
        run_dynamic_pin_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pin_length=8,
            pin_emitter=emit,
            server_id="server-X",
            store=client_store,
        ),
        server_leaves_pairing(),
    )

    # The client relayed the raw server/activate frame and stored nothing on either side.
    assert leftover is not None
    assert "server/activate" in leftover
    assert _added_records(await client_store.list_records()) == []
    assert await server_store.record_by_client_id("client-A") is None
    # Inner authentication succeeded, so the failure counter resets like any other attempt.
    assert await client_store.pin_failure_count() == 0


_STATIC_PIN = "12345678"


async def test_static_pin_round_trip() -> None:
    """A matching static PIN authenticates the PAKE and both sides persist the record."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()
    await client_store.record_pin_failure()  # a dynamic-PIN failure static pairing ignores

    async def provide() -> str:
        return _STATIC_PIN

    _client_ret, server_record = await asyncio.gather(
        run_static_pin_client(
            client_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            static_pin=_STATIC_PIN,
            server_id="server-X",
            store=client_store,
        ),
        run_static_pin_server(
            server_ews,
            handshake_hash=_HANDSHAKE_HASH,
            pairing_index=0,
            pin_provider=provide,
            client_id="client-A",
            store=server_store,
        ),
    )

    client_record = await client_store.record_by_server_id("server-X")
    assert client_record is not None
    assert client_record.psk == server_record.psk
    assert client_record.psk_id == server_record.psk_id
    assert server_record.client_id == "client-A"
    assert await server_store.record_by_client_id("client-A") == server_record
    # The static flow leaves the dynamic-PIN failure counter alone.
    assert await client_store.pin_failure_count() == 1


async def test_static_pin_wrong_pin_aborts_and_persists_nothing() -> None:
    """A static-PIN mismatch aborts and stores nothing; the counter stays untouched."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()
    server_store = InMemoryServerPairingStore()

    async def provide_wrong() -> str:
        return "87654321"

    with pytest.raises(PairingAbortError) as excinfo:
        await asyncio.gather(
            run_static_pin_client(
                client_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                static_pin=_STATIC_PIN,
                server_id="server-X",
                store=client_store,
            ),
            run_static_pin_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pin_provider=provide_wrong,
                client_id="client-A",
                store=server_store,
            ),
        )

    assert excinfo.value.reason is PairAbortReason.PIN_MISMATCH
    assert await client_store.pin_failure_count() == 0
    assert _added_records(await client_store.list_records()) == []
    assert await server_store.record_by_client_id("client-A") is None


@pytest.mark.parametrize(
    "pake_msg_1",
    [
        pytest.param("!!!notbase64!!!", id="not-base64"),
        pytest.param(b64url_encode(bytes(31)), id="wrong-length"),
        pytest.param(b64url_encode(bytes(32)), id="low-order"),
    ],
)
async def test_static_pin_invalid_server_share_is_protocol_error(pake_msg_1: str) -> None:
    """An invalid CPace share from the server is a protocol error, not a PIN guess."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = InMemoryClientPairingStore()

    async def malicious_server() -> None:
        await server_ews.receive()  # client/pair-init
        await server_ews.send_str(
            ServerPairAuthMessage(
                payload=ServerPairAuthPayload(pake_msg_1=pake_msg_1),
            ).to_json(),
        )

    with pytest.raises(PairingError) as excinfo:
        await asyncio.gather(
            run_static_pin_client(
                client_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                static_pin=_STATIC_PIN,
                server_id="server-X",
                store=client_store,
            ),
            malicious_server(),
        )

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await client_store.pin_failure_count() == 0
    assert _added_records(await client_store.list_records()) == []


async def test_static_pin_malformed_client_share_raises() -> None:
    """A non-base64 CPace share from the client aborts the server without persisting a record."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    async def provide() -> str:
        return _STATIC_PIN

    async def malicious_client() -> None:
        await client_ews.send_str(
            ClientPairInitMessage(payload=ClientPairInitPayload(pairing_index=0)).to_json()
        )
        await client_ews.receive()  # server/pair-auth
        await client_ews.send_str(
            ClientPairAuthMessage(
                payload=ClientPairAuthPayload(pake_msg_2="!!!notbase64!!!"),
            ).to_json(),
        )

    with pytest.raises(PairingError) as excinfo:
        await asyncio.gather(
            run_static_pin_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pin_provider=provide,
                client_id="client-A",
                store=server_store,
            ),
            malicious_client(),
        )

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await server_store.record_by_client_id("client-A") is None


async def _honest_pake_to_finalize(
    client_ews: EncryptedWebSocket, *, nonce_b: str | None = None
) -> None:
    """Drive an honest static-PIN PAKE round, stopping just before ``client/pair-finalize``."""
    sid = b"sendspin-pair-pake-v1" + _HANDSHAKE_HASH + (0).to_bytes(4, "big")
    await client_ews.send_str(
        ClientPairInitMessage(payload=ClientPairInitPayload(pairing_index=0)).to_json()
    )
    cpace = CPace.start(
        role=CPaceRole.RESPONDER, prs=_STATIC_PIN.encode("ascii"), sid=sid, ad=b"client"
    )
    auth = ServerPairAuthMessage.from_json((await client_ews.receive()).data)
    await client_ews.send_str(
        ClientPairAuthMessage(
            payload=ClientPairAuthPayload(pake_msg_2=b64url_encode(cpace.public_share)),
        ).to_json(),
    )
    cpace.derive(b64url_decode(auth.payload.pake_msg_1), b"server")
    await client_ews.receive()  # server/pair-confirm
    await client_ews.send_str(
        ClientPairConfirmMessage(
            payload=ClientPairConfirmPayload(client_kc=b64url_encode(cpace.tag()), nonce_B=nonce_b),
        ).to_json(),
    )


async def test_static_pin_server_rejects_dynamic_only_nonce_b() -> None:
    """A static-PIN pair-confirm carrying nonce_B is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    async def provide() -> str:
        return _STATIC_PIN

    with pytest.raises(PairingError, match="nonce_B for static PIN"):
        await asyncio.gather(
            run_static_pin_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pin_provider=provide,
                client_id="client-A",
                store=server_store,
            ),
            _honest_pake_to_finalize(client_ews, nonce_b=b64url_encode(pin_mod.generate_nonce())),
        )

    assert await server_store.record_by_client_id("client-A") is None


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            ClientPairFinalizePayload(long_term_psk=b64url_encode(bytes(32))),
            id="unwrapped_psk",
        ),
        pytest.param(
            ClientPairFinalizePayload(wrapped_psk=b64url_encode(bytes(48))),
            id="undecryptable_wrap",
        ),
    ],
)
async def test_pin_finalize_without_valid_wrap_is_protocol_error(
    payload: ClientPairFinalizePayload,
) -> None:
    """A PIN-flow finalize whose PSK isn't wrapped under the CPace output is a protocol error."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()

    async def provide() -> str:
        return _STATIC_PIN

    async def client_with_bad_finalize() -> None:
        await _honest_pake_to_finalize(client_ews)
        await client_ews.send_str(ClientPairFinalizeMessage(payload=payload).to_json())

    with pytest.raises(PairingError) as excinfo:
        await asyncio.gather(
            run_static_pin_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pin_provider=provide,
                client_id="client-A",
                store=server_store,
            ),
            client_with_bad_finalize(),
        )

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await server_store.record_by_client_id("client-A") is None


async def _dynamic_pake_client(
    client_ews: EncryptedWebSocket,
    pin_future: asyncio.Future[str],
    *,
    mangle_pin: bool = False,
    mangle_nonce: bool = False,
) -> None:
    """Drive a dynamic-PIN PAKE round through ``client/pair-confirm``, optionally cheating.

    ``mangle_pin`` emits (and uses) a PIN not bound to the handshake; ``mangle_nonce``
    reveals a nonce that does not match the commitment.
    """
    sid = b"sendspin-pair-pake-v1" + _HANDSHAKE_HASH + (0).to_bytes(4, "big")
    nonce_b = pin_mod.generate_nonce()
    await client_ews.send_str(
        ClientPairInitMessage(
            payload=ClientPairInitPayload(
                pairing_index=0, commit_B=b64url_encode(pin_mod.commit(nonce_b))
            ),
        ).to_json(),
    )
    init = ServerPairInitMessage.from_json((await client_ews.receive()).data)
    nonce_a = b64url_decode(init.payload.nonce_A)
    pin = pin_mod.derive_pin(_HANDSHAKE_HASH, nonce_a, nonce_b, 8)
    if mangle_pin:
        pin = ("2" if pin[0] == "1" else "1") + pin[1:]
    pin_future.set_result(pin)
    cpace = CPace.start(role=CPaceRole.RESPONDER, prs=pin.encode("ascii"), sid=sid, ad=b"client")
    auth = ServerPairAuthMessage.from_json((await client_ews.receive()).data)
    await client_ews.send_str(
        ClientPairAuthMessage(
            payload=ClientPairAuthPayload(pake_msg_2=b64url_encode(cpace.public_share)),
        ).to_json(),
    )
    cpace.derive(b64url_decode(auth.payload.pake_msg_1), b"server")
    await client_ews.receive()  # server/pair-confirm
    revealed = pin_mod.generate_nonce() if mangle_nonce else nonce_b
    await client_ews.send_str(
        ClientPairConfirmMessage(
            payload=ClientPairConfirmPayload(
                client_kc=b64url_encode(cpace.tag()),
                nonce_B=b64url_encode(revealed),
            ),
        ).to_json(),
    )


async def test_dynamic_pin_mismatched_commit_is_protocol_error() -> None:
    """A revealed nonce_B that doesn't match commit_B is a protocol error, not pin_mismatch."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()
    pin_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    with pytest.raises(PairingError) as excinfo:
        await asyncio.gather(
            run_dynamic_pin_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pin_length=8,
                pin_provider=lambda: pin_future,
                client_id="client-A",
                store=server_store,
            ),
            _dynamic_pake_client(client_ews, pin_future, mangle_nonce=True),
        )

    assert not isinstance(excinfo.value, PairingAbortError)
    assert await server_store.record_by_client_id("client-A") is None


async def test_dynamic_pin_unbound_pin_aborts_pin_mismatch() -> None:
    """A PIN not derived from the handshake fails the binding check with pin_mismatch."""
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    server_store = InMemoryServerPairingStore()
    pin_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    with pytest.raises(PairingAbortError) as excinfo:
        await asyncio.gather(
            run_dynamic_pin_server(
                server_ews,
                handshake_hash=_HANDSHAKE_HASH,
                pairing_index=0,
                pin_length=8,
                pin_provider=lambda: pin_future,
                client_id="client-A",
                store=server_store,
            ),
            _dynamic_pake_client(client_ews, pin_future, mangle_pin=True),
        )

    assert excinfo.value.reason is PairAbortReason.PIN_MISMATCH
    assert await server_store.record_by_client_id("client-A") is None


async def test_pairing_psk_falls_back_to_shared_when_storage_exhausted() -> None:
    """On storage exhaustion the client hands the server its configured shared PSK.

    No new record is created on the client; the server stores the shared PSK as
    its own long-term record keyed by client_id.
    """
    client_ews, server_ews, _client_raw, _server_raw = _paired_encrypted_ws()
    client_store = ExhaustedClientStore()
    server_store = InMemoryServerPairingStore()

    shared_psk = generate_psk()
    shared = ClientPairingRecord(psk_id=psk_id_for(shared_psk), psk=shared_psk, server_id=None)
    await client_store.store_record(shared)
    await client_store.set_record_mode_psk_id(shared.psk_id)

    _client_ret, server_record = await asyncio.gather(
        run_pairing_psk_client(
            client_ews,
            server_id="server-X",
            store=client_store,
        ),
        run_pairing_psk_server(server_ews, client_id="client-A", store=server_store),
    )

    # The client admitted the server under the shared record: no new stored-pubkey record.
    assert _added_records(await client_store.list_records()) == []
    assert await client_store.record_by_psk_id(shared.psk_id) is not None
    # The server received and stored the shared PSK.
    assert server_record.psk == shared_psk
    assert await server_store.record_by_client_id("client-A") == server_record
