"""Tests for the pairing-related fields on client/hello and server/hello."""

from __future__ import annotations

import orjson

from aiosendspin.models.core import (
    ActivatePairing,
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


def test_server_activate_pairing_object_round_trips() -> None:
    """server/activate carries the pairing object when present, else omits it."""
    payload = ServerActivatePayload(
        activities=[Activity.PAIRING],
        pairing=ActivatePairing(method=PairMethod.DYNAMIC_PIN, pin_length=6),
    )
    restored = ServerActivatePayload.from_json(payload.to_json())
    assert restored.pairing == ActivatePairing(method=PairMethod.DYNAMIC_PIN, pin_length=6)
    assert restored.activities == [Activity.PAIRING]
    assert ServerActivatePayload.from_json('{"activities":["playback"]}').pairing is None


def test_activate_pairing_omits_pin_length_for_non_dynamic_methods() -> None:
    """The pairing object omits pin_length when the method carries none."""
    payload = ServerActivatePayload(
        activities=[Activity.PAIRING],
        pairing=ActivatePairing(method=PairMethod.STATIC_PIN),
    )
    raw = orjson.loads(payload.to_json())
    assert raw["pairing"] == {"method": "static_pin"}


def test_activate_pairing_languages_round_trip() -> None:
    """The dynamic-PIN pairing object carries the spoken-emission language hint, or omits it."""
    payload = ServerActivatePayload(
        activities=[Activity.PAIRING],
        pairing=ActivatePairing(
            method=PairMethod.DYNAMIC_PIN, pin_length=6, languages=["ca", "es", "en"]
        ),
    )
    restored = ServerActivatePayload.from_json(payload.to_json())
    assert restored.pairing is not None
    assert restored.pairing.languages == ["ca", "es", "en"]
    bare = ActivatePairing(method=PairMethod.DYNAMIC_PIN, pin_length=6)
    assert "languages" not in orjson.loads(bare.to_json())


def test_pair_method_descriptor_locations_round_trip() -> None:
    """A static-secret descriptor carries the locations hint, others omit it."""
    descriptor = PairMethodDescriptor(method=PairMethod.STATIC_PIN, locations=["device", "leaflet"])
    restored = PairMethodDescriptor.from_json(descriptor.to_json())
    assert restored.locations == ["device", "leaflet"]
    bare = PairMethodDescriptor(method=PairMethod.PAIRING_PSK)
    assert orjson.loads(bare.to_json()) == {"method": "pairing_psk"}
