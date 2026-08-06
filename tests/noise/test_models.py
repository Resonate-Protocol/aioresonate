"""Tests for :mod:`aiosendspin.noise.models`."""

from __future__ import annotations

import json

from aiosendspin.noise.models import (
    ClientInitMessage,
    ClientInitPayload,
    NoiseHandshakeMessage,
    NoiseHandshakePayload,
    NoiseMsg1Payload,
    NoiseMsg2Payload,
    ServerInitMessage,
    ServerInitPayload,
)


def test_client_init_message_round_trip() -> None:
    """ClientInitMessage serializes to spec-shaped JSON and roundtrips through from_json."""
    msg = ClientInitMessage(
        payload=ClientInitPayload(
            client_id="GFsV9tLaSQm9HcFWpKsgYQOr7wFTvNUtkmFwuVz3zoo",
            version=1,
            suite="25519_ChaChaPoly_SHA256",
        ),
    )
    raw = msg.to_json()
    assert '"type":"client/init"' in raw
    assert '"suite":"25519_ChaChaPoly_SHA256"' in raw
    parsed = ClientInitMessage.from_json(raw)
    assert parsed == msg


def test_server_init_has_no_suite_field() -> None:
    """ServerInitPayload only carries server_id and version (spec: no suite)."""
    msg = ServerInitMessage(
        payload=ServerInitPayload(
            server_id="GFsV9tLaSQm9HcFWpKsgYQOr7wFTvNUtkmFwuVz3zoo",
            version=1,
        ),
    )
    raw = msg.to_json()
    assert "suite" not in raw
    assert '"type":"server/init"' in raw


def test_noise_integer_fields_serialize_as_integers() -> None:
    """Noise messages emit integer-typed wire fields."""
    msg = ClientInitMessage(
        payload=ClientInitPayload(
            client_id="GFsV9tLaSQm9HcFWpKsgYQOr7wFTvNUtkmFwuVz3zoo",
            version=1.0,
            suite="25519_ChaChaPoly_SHA256",
        ),
    )
    version = json.loads(msg.to_json())["payload"]["version"]
    assert version == 1
    assert type(version) is int


def test_noise_handshake_message_round_trip() -> None:
    """NoiseHandshakeMessage carries a base64url ``data`` field."""
    msg = NoiseHandshakeMessage(payload=NoiseHandshakePayload(data="aGVsbG8"))
    parsed = NoiseHandshakeMessage.from_json(msg.to_json())
    assert parsed.payload.data == "aGVsbG8"
    assert parsed.type == "noise/handshake"


def test_noise_msg1_payload_carries_psk_id() -> None:
    """The encrypted msg-1 inner payload exposes ``psk_id`` exactly."""
    payload = NoiseMsg1Payload(psk_id="GFsV9tLaSQm9HcFWpKsgYQOr7wFTvNUtkmFwuVz3zoo")
    raw = payload.to_json()
    assert raw == '{"psk_id":"GFsV9tLaSQm9HcFWpKsgYQOr7wFTvNUtkmFwuVz3zoo"}'


def test_noise_msg2_payload_serializes_to_empty_object() -> None:
    """The encrypted msg-2 inner payload is the literal empty object ``{}`` per spec."""
    assert NoiseMsg2Payload().to_json() == "{}"
    # And empty input roundtrips cleanly.
    assert NoiseMsg2Payload.from_json("{}") == NoiseMsg2Payload()
