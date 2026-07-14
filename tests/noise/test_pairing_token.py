"""Tests for :mod:`aiosendspin.noise.pairing_token`."""

import pytest

from aiosendspin.noise.keys import b64url_encode
from aiosendspin.noise.pairing_token import PSKPairingToken, decode_token, encode_token

# Reference vector for client_id=b64url(bytes(range(32))), pairing_psk=bytes(range(32, 64))
# — guards the wire format against accidental changes.
REFERENCE_TOKEN = (
    "SP:1AAAQEAYEAUDAOCAJBIFQYDIOB4IBCEQTCQKRMFYYDENBWHA5DYPSAIJCEMSCKJRHFAUSUKZ"  # noqa: S105 - fixed test vector, not a credential
    "MFUXC6MBRGIZTINJWG44DSOR3HQ6T4PY"
)
REFERENCE_PAIRING_TOKEN = PSKPairingToken(
    client_id=b64url_encode(bytes(range(32))),
    pairing_psk=bytes(range(32, 64)),
)


def test_roundtrip() -> None:
    """A token survives an encode/decode round-trip."""
    token = PSKPairingToken(client_id=b64url_encode(b"\x01" * 32), pairing_psk=b"\x02" * 32)
    assert decode_token(encode_token(token)) == token


def test_reference_vector() -> None:
    """The codec decodes and reproduces the reference token byte-for-byte."""
    assert decode_token(REFERENCE_TOKEN) == REFERENCE_PAIRING_TOKEN
    assert encode_token(REFERENCE_PAIRING_TOKEN) == REFERENCE_TOKEN


def test_encoded_charset_is_qr_alphanumeric_safe() -> None:
    """Encoded tokens stay uppercase with 2 transliterated to 9 and no padding."""
    assert REFERENCE_TOKEN.upper() == REFERENCE_TOKEN
    assert "2" not in REFERENCE_TOKEN
    assert "=" not in REFERENCE_TOKEN


@pytest.mark.parametrize(
    "value",
    [
        REFERENCE_TOKEN.lower(),
        f"  {REFERENCE_TOKEN}\n",
        REFERENCE_TOKEN.removeprefix("SP:"),
        REFERENCE_TOKEN.removeprefix("SP:").lower(),
    ],
)
def test_decode_leniency(value: str) -> None:
    """Case, surrounding whitespace, and a missing SP: prefix are tolerated."""
    assert decode_token(value) == REFERENCE_PAIRING_TOKEN


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("", "malformed"),
        ("SP:", "malformed"),
        ("SP:2" + REFERENCE_TOKEN[4:], "unsupported"),
        ("SP:1NOT!VALID", "malformed"),
        ("SP:1" + REFERENCE_TOKEN[4:40], "malformed"),  # truncated payload
    ],
)
def test_decode_rejects_malformed(value: str, match: str) -> None:
    """Malformed or unsupported tokens raise ValueError."""
    with pytest.raises(ValueError, match=match):
        decode_token(value)


def test_encode_validates_field_sizes() -> None:
    """encode_token rejects wrong-size client_id and pairing_psk."""
    with pytest.raises(ValueError, match="client_id"):
        encode_token(
            PSKPairingToken(client_id=b64url_encode(b"\x01" * 16), pairing_psk=b"\x02" * 32)
        )
    with pytest.raises(ValueError, match="pairing_psk"):
        encode_token(
            PSKPairingToken(client_id=b64url_encode(b"\x01" * 32), pairing_psk=b"\x02" * 16)
        )


def test_encode_rejects_non_base64url_client_id() -> None:
    """A client_id that isn't valid base64url is rejected before size checks."""
    # A single character is an invalid base64 length (1 more than a multiple of 4).
    with pytest.raises(ValueError, match="not valid base64url"):
        encode_token(PSKPairingToken(client_id="A", pairing_psk=b"\x02" * 32))


def test_decode_fuzz_single_char_mutations_never_crash() -> None:
    """Every single-character mutation of a valid token decodes cleanly or raises ValueError.

    QR/paste input is attacker-influenced, so ``decode_token`` must only ever raise
    ``ValueError`` (its documented contract) — never an unhandled binascii/base64/index
    error. Deterministic: walks every position against a fixed replacement alphabet.
    """
    replacements = "ABYZ2679!@ \t/=+-.0"  # b32 chars, transliterated digits, and junk
    for pos in range(len(REFERENCE_TOKEN)):
        for repl in replacements:
            if repl == REFERENCE_TOKEN[pos]:
                continue
            mutated = REFERENCE_TOKEN[:pos] + repl + REFERENCE_TOKEN[pos + 1 :]
            try:
                result = decode_token(mutated)
            except ValueError:
                continue
            # A mutation that still decodes must yield a well-formed token.
            assert len(result.pairing_psk) == 32, mutated
