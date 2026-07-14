"""Unit tests for the client-side management command handlers."""

from __future__ import annotations

from dataclasses import replace

from aiosendspin.client.management import (
    ManagementEffect,
    handle_add_record,
    handle_get_pairing_config,
    handle_list_records,
    handle_remove_record,
    handle_set_pairing_config,
    handle_unpair,
    with_storage,
)
from aiosendspin.models.management import (
    ManagementAddRecordPayload,
    ManagementRemoveRecordPayload,
    ManagementResultPayload,
    ManagementSetPairingConfigPayload,
    RecordModeConfig,
    SetDynamicPinConfig,
    SetPairingPskConfig,
    SetStaticPinConfig,
    SetUnpairedAccessConfig,
    StorageAccounting,
)
from aiosendspin.models.types import ManagementResult, PairMethod
from aiosendspin.noise.keys import b64url_encode, generate_psk, psk_id_for
from aiosendspin.noise.trust_store import (
    PIN_LOCKOUT_THRESHOLD,
    ClientPairingRecord,
    InMemoryClientPairingStore,
)
from tests.pairing_stores import BoundedClientStore, ExhaustedClientStore

_ALL_METHODS = frozenset({PairMethod.PAIRING_PSK, PairMethod.STATIC_PIN, PairMethod.DYNAMIC_PIN})
_WITHOUT_STATIC_PIN = frozenset({PairMethod.PAIRING_PSK, PairMethod.DYNAMIC_PIN})

SERVER_ID = "srv-self"


async def _seed_record(
    store: InMemoryClientPairingStore,
    *,
    server_id: str | None,
) -> ClientPairingRecord:
    psk = generate_psk()
    record = ClientPairingRecord(psk_id=psk_id_for(psk), psk=psk, server_id=server_id)
    await store.store_record(record)
    return record


# --- list-records -------------------------------------------------------------


async def test_list_records_summarizes_records() -> None:
    """list-records projects each record, omitting server_id for shared-PSK records."""
    store = InMemoryClientPairingStore()
    stored = await _seed_record(store, server_id=SERVER_ID)
    shared = await _seed_record(store, server_id=None)
    await store.mark_record_used(stored.psk_id)
    payload, effect = await handle_list_records(store)
    assert effect is ManagementEffect.NONE
    assert payload.result is ManagementResult.OK
    assert payload.data is not None
    by_id = {s.psk_id: s for s in payload.data.records or []}
    assert by_id[stored.psk_id].server_id == SERVER_ID
    assert by_id[stored.psk_id].used is True
    assert by_id[shared.psk_id].server_id is None
    assert by_id[shared.psk_id].used is False


# --- add-record ---------------------------------------------------------------


async def test_add_record_ok() -> None:
    """A valid add-record persists a new stored-pubkey record."""
    store = InMemoryClientPairingStore()
    psk = generate_psk()
    payload, _ = await handle_add_record(
        store,
        ManagementAddRecordPayload(psk=b64url_encode(psk), server_id="srv2"),
    )
    assert payload.result is ManagementResult.OK
    record = await store.record_by_psk_id(psk_id_for(psk))
    assert record is not None
    assert record.server_id == "srv2"


async def test_add_record_already_exists() -> None:
    """Adding a record whose psk_id is already stored is rejected."""
    store = InMemoryClientPairingStore()
    existing = await _seed_record(store, server_id="srv2")
    payload, _ = await handle_add_record(
        store,
        ManagementAddRecordPayload(psk=b64url_encode(existing.psk), server_id="srv2"),
    )
    assert payload.result is ManagementResult.ALREADY_EXISTS


async def test_add_record_invalid_psk() -> None:
    """A PSK that does not decode to 32 bytes is rejected as invalid."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_add_record(
        store,
        ManagementAddRecordPayload(psk="not-valid-base64-len"),
    )
    assert payload.result is ManagementResult.INVALID


async def test_add_record_storage_exhausted() -> None:
    """Add-record reports storage_exhausted when the store cannot persist."""
    store = ExhaustedClientStore()
    payload, _ = await handle_add_record(
        store,
        ManagementAddRecordPayload(psk=b64url_encode(generate_psk())),
    )
    assert payload.result is ManagementResult.STORAGE_EXHAUSTED


# --- remove-record ------------------------------------------------------------


async def test_remove_record_ok() -> None:
    """Removing an existing record succeeds and deletes it from the store."""
    store = InMemoryClientPairingStore()
    record = await _seed_record(store, server_id="srv2")
    payload, effect = await handle_remove_record(
        store,
        ManagementRemoveRecordPayload(psk_id=record.psk_id),
        requester_server_id=SERVER_ID,
    )
    assert payload.result is ManagementResult.OK
    assert effect is ManagementEffect.NONE
    assert await store.record_by_psk_id(record.psk_id) is None


async def test_remove_record_not_found() -> None:
    """Removing an unknown psk_id reports not_found."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_remove_record(
        store,
        ManagementRemoveRecordPayload(psk_id="missing"),
        requester_server_id=SERVER_ID,
    )
    assert payload.result is ManagementResult.NOT_FOUND


async def test_remove_record_referenced_by_record_mode_is_invalid() -> None:
    """A record referenced by a record_mode.psk_id cannot be removed."""
    store = InMemoryClientPairingStore()
    shared = await _seed_record(store, server_id=None)
    await store.set_record_mode_psk_id(shared.psk_id)
    payload, _ = await handle_remove_record(
        store,
        ManagementRemoveRecordPayload(psk_id=shared.psk_id),
        requester_server_id=SERVER_ID,
    )
    assert payload.result is ManagementResult.INVALID
    assert await store.record_by_psk_id(shared.psk_id) is not None


async def test_remove_record_self_closes_session() -> None:
    """Removing the requester's own record signals a goodbye-unauthorized effect."""
    store = InMemoryClientPairingStore()
    record = await _seed_record(store, server_id=SERVER_ID)
    payload, effect = await handle_remove_record(
        store,
        ManagementRemoveRecordPayload(psk_id=record.psk_id),
        requester_server_id=SERVER_ID,
    )
    assert payload.result is ManagementResult.OK
    assert effect is ManagementEffect.GOODBYE_UNAUTHORIZED


# --- server/unpair ------------------------------------------------------------


async def test_unpair_removes_stored_pubkey_record() -> None:
    """server/unpair drops the matched stored-pubkey record."""
    store = InMemoryClientPairingStore()
    record = await _seed_record(store, server_id=SERVER_ID)
    await handle_unpair(store, matched_psk_id=record.psk_id)
    assert await store.record_by_psk_id(record.psk_id) is None


async def test_unpair_keeps_shared_psk_record() -> None:
    """server/unpair must not remove a shared-PSK record (it may back other servers)."""
    store = InMemoryClientPairingStore()
    shared = await _seed_record(store, server_id=None)
    await handle_unpair(store, matched_psk_id=shared.psk_id)
    assert await store.record_by_psk_id(shared.psk_id) is not None


async def test_unpair_unknown_psk_id_is_noop() -> None:
    """server/unpair for an unknown record leaves the store unchanged."""
    store = InMemoryClientPairingStore()
    record = await _seed_record(store, server_id=SERVER_ID)
    await handle_unpair(store, matched_psk_id="missing")
    assert await store.record_by_psk_id(record.psk_id) is not None


# --- get-pairing-config -------------------------------------------------------


async def test_get_pairing_config_projects_state() -> None:
    """get-config reflects enabled flags + lockout, omits unimplemented methods and secrets."""
    store = InMemoryClientPairingStore()
    await store.store_pairing_config(
        replace(
            await store.get_pairing_config(),
            pairing_psk_enabled=False,
            unpaired_access_enabled=True,
        )
    )
    for _ in range(PIN_LOCKOUT_THRESHOLD):
        await store.record_pin_failure(PairMethod.DYNAMIC_PIN)
    payload, effect = await handle_get_pairing_config(
        store, implemented_pair_methods=_WITHOUT_STATIC_PIN
    )
    assert effect is ManagementEffect.NONE
    data = payload.data
    assert data is not None
    assert data.pairing_psk is not None
    assert data.pairing_psk.enabled is False
    assert data.pairing_psk.locked_out is None  # not a PIN method
    assert data.dynamic_pin is not None
    assert data.dynamic_pin.locked_out is True
    assert data.dynamic_pin.min_pin_length == 6  # default floor
    assert data.static_pin is None  # not implemented
    assert data.unpaired_access is not None
    assert data.unpaired_access.enabled is True
    # The mandatory shared-PSK fallback is always reported.
    assert data.record_mode == RecordModeConfig(
        psk_id=(await store.get_pairing_config()).record_mode_psk_id
    )


async def test_get_pairing_config_shows_static_pin_when_implemented() -> None:
    """A client that implements static PIN includes it and reports its lockout."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_get_pairing_config(store, implemented_pair_methods=_ALL_METHODS)
    assert payload.data is not None
    assert payload.data.static_pin is not None


# --- set-pairing-config -------------------------------------------------------


async def test_set_pairing_config_toggles_enabled() -> None:
    """Toggling a method's enabled flag persists to the config."""
    store = InMemoryClientPairingStore()
    payload, effect = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(pairing_psk=SetPairingPskConfig(enabled=False)),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.OK
    assert effect is ManagementEffect.NONE
    assert (await store.get_pairing_config()).pairing_psk_enabled is False


async def test_set_pairing_config_rotates_psk() -> None:
    """A valid psk replaces the configured Pairing PSK."""
    store = InMemoryClientPairingStore()
    psk = generate_psk()
    payload, _ = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(pairing_psk=SetPairingPskConfig(psk=b64url_encode(psk))),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.OK
    stored = await store.pairing_psk()
    assert stored is not None
    assert stored.psk == psk


async def test_set_pairing_config_invalid_psk() -> None:
    """A malformed psk is rejected and nothing is stored."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(pairing_psk=SetPairingPskConfig(psk="too-short")),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.INVALID
    assert await store.pairing_psk() is None


async def test_set_pairing_config_sets_min_pin_length() -> None:
    """A valid dynamic-PIN min_pin_length is persisted to the config."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(dynamic_pin=SetDynamicPinConfig(min_pin_length=8)),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.OK
    assert (await store.get_pairing_config()).dynamic_pin_min_length == 8


async def test_set_pairing_config_invalid_min_pin_length() -> None:
    """An out-of-range min_pin_length is rejected and the stored value is unchanged."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(dynamic_pin=SetDynamicPinConfig(min_pin_length=3)),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.INVALID
    assert (await store.get_pairing_config()).dynamic_pin_min_length == 6  # unchanged default


async def test_set_pairing_config_stores_static_pin() -> None:
    """A valid 8-digit pin is stored as the configured static PIN."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(static_pin=SetStaticPinConfig(pin="12345678")),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.OK
    assert await store.static_pin() == "12345678"


async def test_set_pairing_config_invalid_static_pin() -> None:
    """A non-8-digit or non-ASCII pin is rejected and nothing is stored."""
    store = InMemoryClientPairingStore()
    for pin in ("12ab", "١٢٣٤٥٦٧٨"):
        payload, _ = await handle_set_pairing_config(
            store,
            ManagementSetPairingConfigPayload(static_pin=SetStaticPinConfig(pin=pin)),
            implemented_pair_methods=_ALL_METHODS,
        )
        assert payload.result is ManagementResult.INVALID
    assert await store.static_pin() is None


async def test_get_pairing_config_omits_static_pin_secret() -> None:
    """get-config exposes static-PIN policy but never the configured PIN itself."""
    store = InMemoryClientPairingStore()
    await store.set_static_pin("12345678")
    payload, _ = await handle_get_pairing_config(store, implemented_pair_methods=_ALL_METHODS)
    assert payload.data is not None
    assert payload.data.static_pin is not None
    assert "12345678" not in payload.to_json()


async def test_set_pairing_config_unimplemented_method_is_invalid() -> None:
    """A patch on a method the client does not implement is rejected."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(static_pin=SetStaticPinConfig(enabled=True)),
        implemented_pair_methods=_WITHOUT_STATIC_PIN,
    )
    assert payload.result is ManagementResult.INVALID


async def test_set_pairing_config_clears_lockout() -> None:
    """locked_out=false clears the PIN failure counter."""
    store = InMemoryClientPairingStore()
    for _ in range(PIN_LOCKOUT_THRESHOLD):
        await store.record_pin_failure(PairMethod.DYNAMIC_PIN)
    assert await store.is_pin_locked_out(PairMethod.DYNAMIC_PIN)
    payload, _ = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(dynamic_pin=SetDynamicPinConfig(locked_out=False)),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.OK
    assert not await store.is_pin_locked_out(PairMethod.DYNAMIC_PIN)


async def test_set_pairing_config_rejects_lockout_true() -> None:
    """locked_out=true is rejected (only false clears)."""
    store = InMemoryClientPairingStore()
    payload, _ = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(dynamic_pin=SetDynamicPinConfig(locked_out=True)),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.INVALID


async def test_set_pairing_config_persists_unpaired_access() -> None:
    """An unpaired-access patch is written to the persisted config."""
    store = InMemoryClientPairingStore()
    payload, effect = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(unpaired_access=SetUnpairedAccessConfig(enabled=True)),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert payload.result is ManagementResult.OK
    assert effect is ManagementEffect.NONE
    assert (await store.get_pairing_config()).unpaired_access_enabled is True


async def test_set_pairing_config_record_mode_requires_shared_record() -> None:
    """A record_mode referencing a missing/non-shared record is invalid; a shared one applies."""
    store = InMemoryClientPairingStore()
    missing = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(record_mode=RecordModeConfig(psk_id="absent")),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert missing[0].result is ManagementResult.INVALID

    stored = await _seed_record(store, server_id="srv-other")
    non_shared = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(record_mode=RecordModeConfig(psk_id=stored.psk_id)),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert non_shared[0].result is ManagementResult.INVALID

    shared = await _seed_record(store, server_id=None)
    ok = await handle_set_pairing_config(
        store,
        ManagementSetPairingConfigPayload(record_mode=RecordModeConfig(psk_id=shared.psk_id)),
        implemented_pair_methods=_ALL_METHODS,
    )
    assert ok[0].result is ManagementResult.OK
    assert (await store.get_pairing_config()).record_mode_psk_id == shared.psk_id


# --- storage accounting -------------------------------------------------------


async def test_with_storage_unbounded_store_leaves_payload_unchanged() -> None:
    """A store reporting no accounting (default, unbounded) attaches no storage object."""
    store = InMemoryClientPairingStore()
    payload = ManagementResultPayload(result=ManagementResult.OK)
    result = await with_storage(payload, store, include_static=True)
    assert result.storage is None


async def test_with_storage_free_only_for_mutations() -> None:
    """A non-read result carries only free; capacity and the per-kind costs are absent."""
    store = BoundedClientStore()
    await _seed_record(store, server_id="srv1")
    result = await with_storage(
        ManagementResultPayload(result=ManagementResult.OK), store, include_static=False
    )
    assert result.storage is not None
    # capacity 4, minus the pre-provisioned shared record and the seeded one.
    assert result.storage.free == 2
    assert result.storage.capacity is None
    assert result.storage.cost_individual is None
    assert result.storage.cost_shared is None


async def test_with_storage_full_for_reads() -> None:
    """A read result (include_static) carries free plus capacity and the per-kind costs."""
    store = BoundedClientStore()
    result = await with_storage(
        ManagementResultPayload(result=ManagementResult.OK), store, include_static=True
    )
    # capacity 4, minus the pre-provisioned shared record.
    assert result.storage == StorageAccounting(free=3, capacity=4, cost_individual=1, cost_shared=1)
