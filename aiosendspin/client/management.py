"""Client-side handling of management commands.

Pure logic that maps a parsed server→client request onto the client's
``ClientPairingStore``, returning the
``ManagementResultPayload`` to reply with and an
``ManagementEffect`` telling the connection what to do after the reply. Kept free of
transport concerns so it is unit-testable against an ``InMemoryClientPairingStore``.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum, auto
from typing import TYPE_CHECKING

from aiosendspin.models.core import UnpairedAccess
from aiosendspin.models.management import (
    ManagementResultData,
    ManagementResultPayload,
    PairingMethodConfig,
    RecordModeConfig,
    RecordSummary,
    StorageAccounting,
)
from aiosendspin.models.types import ManagementResult, PairMethod
from aiosendspin.noise.keys import PSK_SIZE, b64url_decode, psk_id_for
from aiosendspin.noise.pin import MAX_PIN_DIGITS, MIN_PIN_DIGITS
from aiosendspin.noise.trust_store import (
    ClientPairingRecord,
    PairingPsk,
    PskCategory,
)

if TYPE_CHECKING:
    from collections.abc import Container

    from aiosendspin.models.management import (
        ManagementAddRecordPayload,
        ManagementRemoveRecordPayload,
        ManagementSetPairingConfigPayload,
        SetDynamicPinConfig,
        SetPairingPskConfig,
        SetStaticPinConfig,
        SetUnpairedAccessConfig,
    )
    from aiosendspin.noise.trust_store import ClientPairingStore

_STATIC_PIN_DIGITS = 8


class ManagementEffect(Enum):
    """Action the connection takes after sending the management/result."""

    NONE = auto()
    GOODBYE_UNAUTHORIZED = auto()
    """The requester demoted/removed its own record; close with goodbye 'unauthorized'."""


def _result(
    result: ManagementResult, data: ManagementResultData | None = None
) -> ManagementResultPayload:
    return ManagementResultPayload(result=result, data=data)


async def handle_unpair(store: ClientPairingStore, *, matched_psk_id: str) -> None:
    """Drop the matched record on server/unpair; a shared-PSK record is never removed here."""
    record = await store.record_by_psk_id(matched_psk_id)
    if record is None or record.server_id is None:
        return
    await store.remove_record(matched_psk_id)


async def handle_list_records(
    store: ClientPairingStore,
) -> tuple[ManagementResultPayload, ManagementEffect]:
    """Return summaries of all stored long-term records."""
    summaries = [
        RecordSummary(psk_id=r.psk_id, server_id=r.server_id, used=r.used)
        for r in await store.list_records()
    ]
    data = ManagementResultData(records=summaries)
    return _result(ManagementResult.OK, data), ManagementEffect.NONE


async def handle_add_record(
    store: ClientPairingStore, payload: ManagementAddRecordPayload
) -> tuple[ManagementResultPayload, ManagementEffect]:
    """Add a stored-pubkey or shared-PSK record from a provided PSK."""
    try:
        psk = b64url_decode(payload.psk)
    except ValueError:
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    if len(psk) != PSK_SIZE:
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    psk_id = psk_id_for(psk)
    if await store.resolve_by_psk_id(psk_id) is not None:
        return _result(ManagementResult.ALREADY_EXISTS), ManagementEffect.NONE
    if not await store.can_store_record():
        return _result(ManagementResult.STORAGE_EXHAUSTED), ManagementEffect.NONE
    await store.store_record(
        ClientPairingRecord(
            psk_id=psk_id,
            psk=psk,
            server_id=payload.server_id,
        )
    )
    return _result(ManagementResult.OK), ManagementEffect.NONE


async def handle_remove_record(
    store: ClientPairingStore,
    payload: ManagementRemoveRecordPayload,
    *,
    requester_psk_id: str | None,
) -> tuple[ManagementResultPayload, ManagementEffect]:
    """Remove a record; removing one's own record closes the session."""
    record = await store.record_by_psk_id(payload.psk_id)
    if record is None:
        return _result(ManagementResult.NOT_FOUND), ManagementEffect.NONE
    if not await store.can_remove_record(payload.psk_id):
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    effect = (
        ManagementEffect.GOODBYE_UNAUTHORIZED
        if record.psk_id == requester_psk_id
        else ManagementEffect.NONE
    )
    await store.remove_record(payload.psk_id)
    return _result(ManagementResult.OK), effect


async def handle_get_pairing_config(
    store: ClientPairingStore,
    *,
    implemented_pair_methods: Container[PairMethod],
) -> tuple[ManagementResultPayload, ManagementEffect]:
    """Assemble the pairing-config view; secrets are never included."""
    config = await store.get_pairing_config()
    data = ManagementResultData(
        pairing_psk=await _method_config(
            store, PairMethod.PAIRING_PSK, enabled=config.pairing_psk_enabled, is_pin=False
        ),
        static_pin=(
            await _method_config(
                store, PairMethod.STATIC_PIN, enabled=config.static_pin_enabled, is_pin=True
            )
            if PairMethod.STATIC_PIN in implemented_pair_methods
            else None
        ),
        dynamic_pin=(
            await _method_config(
                store,
                PairMethod.DYNAMIC_PIN,
                enabled=config.dynamic_pin_enabled,
                is_pin=True,
                min_pin_length=config.dynamic_pin_min_length,
            )
            if PairMethod.DYNAMIC_PIN in implemented_pair_methods
            else None
        ),
        record_mode=RecordModeConfig(psk_id=config.record_mode_psk_id),
        unpaired_access=UnpairedAccess(enabled=config.unpaired_access_enabled),
    )
    return _result(ManagementResult.OK, data), ManagementEffect.NONE


async def handle_set_pairing_config(
    store: ClientPairingStore,
    payload: ManagementSetPairingConfigPayload,
    *,
    implemented_pair_methods: Container[PairMethod],
) -> tuple[ManagementResultPayload, ManagementEffect]:
    """Apply a validated config patch (enabled flags, secrets, record mode, unpaired access)."""
    methods = (
        (PairMethod.PAIRING_PSK, payload.pairing_psk),
        (PairMethod.STATIC_PIN, payload.static_pin),
        (PairMethod.DYNAMIC_PIN, payload.dynamic_pin),
    )
    # 1. A patch on a method the client does not implement is invalid.
    if any(cfg is not None and method not in implemented_pair_methods for method, cfg in methods):
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    # 2. Validate secrets, lockout, and record mode before mutating anything.
    psk_bytes: bytes | None = None
    if payload.pairing_psk is not None and payload.pairing_psk.psk is not None:
        psk_bytes = _decode_psk(payload.pairing_psk.psk)
        if psk_bytes is None:
            return _result(ManagementResult.INVALID), ManagementEffect.NONE
    if (
        payload.static_pin is not None
        and payload.static_pin.pin is not None
        and not _valid_static_pin(payload.static_pin.pin)
    ):
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    if _rejects_lockout(payload.static_pin) or _rejects_lockout(payload.dynamic_pin):
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    if (
        payload.dynamic_pin is not None
        and payload.dynamic_pin.min_pin_length is not None
        and not MIN_PIN_DIGITS <= payload.dynamic_pin.min_pin_length <= MAX_PIN_DIGITS
    ):
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    if payload.record_mode is not None and not await _is_shared_record(
        store, payload.record_mode.psk_id
    ):
        return _result(ManagementResult.INVALID), ManagementEffect.NONE
    # 3. Apply (all inputs validated; nothing below fails).
    if payload.record_mode is not None:
        await store.set_record_mode_psk_id(payload.record_mode.psk_id)
    config = await store.get_pairing_config()
    await store.store_pairing_config(
        replace(
            config,
            pairing_psk_enabled=_merge_enabled(
                payload.pairing_psk, current=config.pairing_psk_enabled
            ),
            static_pin_enabled=_merge_enabled(
                payload.static_pin, current=config.static_pin_enabled
            ),
            dynamic_pin_enabled=_merge_enabled(
                payload.dynamic_pin, current=config.dynamic_pin_enabled
            ),
            unpaired_access_enabled=_merge_enabled(
                payload.unpaired_access, current=config.unpaired_access_enabled
            ),
            dynamic_pin_min_length=(
                payload.dynamic_pin.min_pin_length
                if payload.dynamic_pin is not None
                and payload.dynamic_pin.min_pin_length is not None
                else config.dynamic_pin_min_length
            ),
        )
    )
    if psk_bytes is not None:
        await store.set_pairing_psk(PairingPsk(psk_id=psk_id_for(psk_bytes), psk=psk_bytes))
    if payload.static_pin is not None and payload.static_pin.pin is not None:
        await store.set_static_pin(payload.static_pin.pin)
    if payload.static_pin is not None and payload.static_pin.locked_out is False:
        await store.reset_pin_failures(PairMethod.STATIC_PIN)
    if payload.dynamic_pin is not None and payload.dynamic_pin.locked_out is False:
        await store.reset_pin_failures(PairMethod.DYNAMIC_PIN)
    return _result(ManagementResult.OK), ManagementEffect.NONE


async def with_storage(
    payload: ManagementResultPayload, store: ClientPairingStore, *, include_static: bool
) -> ManagementResultPayload:
    """Attach storage accounting to a result, or leave it absent.

    ``free`` is always set; ``capacity`` and the per-kind costs are added only when
    ``include_static`` (list-records and get-pairing-config). A store that reports no
    accounting (unbounded/unknown storage) leaves the payload unchanged.
    """
    report = await store.storage_accounting()
    if report is None:
        return payload
    storage = (
        StorageAccounting(
            free=report.free,
            capacity=report.capacity,
            cost_individual=report.cost_individual,
            cost_shared=report.cost_shared,
        )
        if include_static
        else StorageAccounting(free=report.free)
    )
    return replace(payload, storage=storage)


async def _method_config(
    store: ClientPairingStore,
    method: PairMethod,
    *,
    enabled: bool,
    is_pin: bool,
    min_pin_length: int | None = None,
) -> PairingMethodConfig:
    """Project a method's stored state into its wire config (no secrets)."""
    return PairingMethodConfig(
        enabled=enabled,
        locked_out=(await store.is_pin_locked_out(method)) if is_pin else None,
        min_pin_length=min_pin_length,
    )


async def _is_shared_record(store: ClientPairingStore, psk_id: str) -> bool:
    """Return whether ``psk_id`` names a shared-PSK record (long-term, no counterparty)."""
    resolved = await store.resolve_by_psk_id(psk_id)
    return (
        resolved is not None
        and resolved.category is PskCategory.LONG_TERM
        and resolved.counterparty_id is None
    )


def _decode_psk(value: str) -> bytes | None:
    """Decode a base64url PSK, returning ``None`` if it is not a 32-byte key."""
    try:
        psk = b64url_decode(value)
    except ValueError:
        return None
    return psk if len(psk) == PSK_SIZE else None


def _valid_static_pin(pin: str) -> bool:
    """Return whether ``pin`` is exactly 8 ASCII decimal digits."""
    return len(pin) == _STATIC_PIN_DIGITS and pin.isascii() and pin.isdigit()


def _rejects_lockout(cfg: SetStaticPinConfig | SetDynamicPinConfig | None) -> bool:
    """Return whether ``cfg`` sets ``locked_out`` to anything but ``false`` (only false clears)."""
    return cfg is not None and cfg.locked_out is True


def _merge_enabled(
    cfg: SetPairingPskConfig
    | SetStaticPinConfig
    | SetDynamicPinConfig
    | SetUnpairedAccessConfig
    | None,
    *,
    current: bool,
) -> bool:
    """Return ``cfg.enabled`` when present, else the current value."""
    if cfg is None or cfg.enabled is None:
        return current
    return cfg.enabled
