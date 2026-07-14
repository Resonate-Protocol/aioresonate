"""Tests for the pairing-related fields on client/hello and server/hello."""

from __future__ import annotations

from aiosendspin.models.core import (
    ClientHelloPayload,
    PairMethodDescriptor,
    ServerActivatePayload,
    ServerHelloPayload,
    UnpairedAccess,
)
from aiosendspin.models.types import Activity, PairMethod, TrustLevel


def test_client_hello_pairing_fields_round_trip() -> None:
    """client/hello carries trust, pairing methods, and unpaired-access flag."""
    payload = ClientHelloPayload(
        client_id="c1",
        name="Client",
        version=1,
        supported_roles=["controller@v1"],
        trust_level=TrustLevel.USER,
        supported_pair_methods=[PairMethodDescriptor(method=PairMethod.PAIRING_PSK)],
        unpaired_access=UnpairedAccess(enabled=True),
    )
    restored = ClientHelloPayload.from_json(payload.to_json())
    assert restored == payload
    assert restored.trust_level is TrustLevel.USER
    assert restored.supported_pair_methods == [PairMethodDescriptor(method=PairMethod.PAIRING_PSK)]
    assert restored.unpaired_access.enabled is True


def test_client_hello_defaults_when_pairing_fields_absent() -> None:
    """A hello without the new fields deserializes to spec-safe defaults (legacy clients)."""
    legacy = '{"client_id":"c1","name":"Client","version":1,"supported_roles":["controller@v1"]}'
    payload = ClientHelloPayload.from_json(legacy)
    assert payload.trust_level is TrustLevel.NONE
    assert payload.supported_pair_methods is None
    assert payload.unpaired_access == UnpairedAccess(enabled=False)


def test_server_hello_round_trips() -> None:
    """server/hello carries the name."""
    payload = ServerHelloPayload(name="Server")
    restored = ServerHelloPayload.from_json(payload.to_json())
    assert restored == payload
    assert restored.name == "Server"


def test_server_activate_selected_pair_method_round_trips() -> None:
    """server/activate carries the selected pairing method when present, else omits it."""
    payload = ServerActivatePayload(
        activities=[Activity.PAIRING],
        selected_pair_method=PairMethod.PAIRING_PSK,
    )
    restored = ServerActivatePayload.from_json(payload.to_json())
    assert restored.selected_pair_method is PairMethod.PAIRING_PSK
    assert restored.activities == [Activity.PAIRING]
    assert (
        ServerActivatePayload.from_json('{"activities":["playback"]}').selected_pair_method is None
    )
