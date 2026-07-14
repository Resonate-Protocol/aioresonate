"""Tests for :mod:`aiosendspin.noise.keys`."""

import pytest

from aiosendspin.noise.keys import (
    PEER_ID_SIZE,
    PSK_SIZE,
    Identity,
    b64url_decode,
    b64url_encode,
    generate_psk,
    psk_id_for,
)


def test_generate_psk_size_and_uniqueness() -> None:
    """generate_psk returns 32 bytes from a CSPRNG (no collisions)."""
    a = generate_psk()
    b = generate_psk()
    assert len(a) == PSK_SIZE
    assert a != b


def test_psk_id_rejects_wrong_size() -> None:
    """psk_id_for raises on non-32-byte input."""
    with pytest.raises(ValueError, match="PSK must be 32 bytes"):
        psk_id_for(b"\x00" * 16)


def test_psk_id_is_43_chars_no_padding() -> None:
    """psk_id is base64url SHA-256 — 43 chars, no '=' padding."""
    pid = psk_id_for(b"\x00" * PSK_SIZE)
    assert len(pid) == PEER_ID_SIZE
    assert "=" not in pid


def test_b64url_roundtrip() -> None:
    """b64url_encode / b64url_decode invert each other on arbitrary bytes."""
    original = b"\xde\xad\xbe\xef" * 8
    assert b64url_decode(b64url_encode(original)) == original


def test_b64url_decode_tolerates_missing_padding() -> None:
    """b64url_decode accepts unpadded input (the format we always emit)."""
    assert b64url_decode("Zm9vYg") == b"foob"
    assert b64url_decode("Zm9vYmE") == b"fooba"


def test_b64url_uses_url_safe_alphabet() -> None:
    """b64url_encode uses '-' and '_' rather than '+' and '/'."""
    data = bytes([0xFB, 0xFF, 0xBF])
    enc = b64url_encode(data)
    assert "+" not in enc
    assert "/" not in enc


def test_identity_generate_shapes() -> None:
    """Identity.generate returns 32-byte keys and a 43-char peer_id."""
    identity = Identity.generate()
    assert len(identity.private_bytes) == 32
    assert len(identity.public_bytes) == 32
    assert len(identity.peer_id) == PEER_ID_SIZE


def test_identity_from_private_bytes_roundtrip() -> None:
    """Identity.from_private_bytes reproduces the original public key."""
    original = Identity.generate()
    rehydrated = Identity.from_private_bytes(original.private_bytes)
    assert rehydrated.public_bytes == original.public_bytes
    assert rehydrated.peer_id == original.peer_id


def test_identity_from_private_bytes_rejects_wrong_size() -> None:
    """from_private_bytes raises on non-32-byte input."""
    with pytest.raises(ValueError, match="32 bytes"):
        Identity.from_private_bytes(b"\x00" * 16)


def test_identity_private_b64u_roundtrips_through_b64url_decode() -> None:
    """private_b64u serializes the raw private key in base64url."""
    identity = Identity.generate()
    decoded = b64url_decode(identity.private_b64u)
    assert decoded == identity.private_bytes
