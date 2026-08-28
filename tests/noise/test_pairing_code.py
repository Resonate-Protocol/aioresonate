"""Tests for dynamic-PAIRING_CODE derivation and commitment (:mod:`aiosendspin.noise.pairing_code`).

The known-answer values are computed from the spec formula over fixed inputs
(``h = 0x00..1f``, ``nonce_A = 0x01*32``, ``nonce_B = 0x02*32``) and pinned here
to catch regressions in the derivation.
"""

from __future__ import annotations

import pytest

from aiosendspin.noise.pairing_code import (
    COMMIT_SIZE,
    NONCE_SIZE,
    QR_CODE_SIZE,
    commit,
    decode_qr_token,
    derive_digits,
    derive_qr_code,
    encode_qr_token,
    generate_nonce,
    verify_commit,
)

H = bytes(range(32))
NONCE_A = bytes([1]) * 32
NONCE_B = bytes([2]) * 32


def test_derive_digits_known_answer() -> None:
    """Dynamic digits are always the six-digit spec derivation."""
    assert derive_digits(H, NONCE_A, NONCE_B) == "305673"


def test_derive_digits_is_fixed_width_and_input_sensitive() -> None:
    """The six-digit code changes when any derivation input changes."""
    base = derive_digits(H, NONCE_A, NONCE_B)
    assert len(base) == 6
    assert base.isascii()
    assert base.isdigit()
    assert derive_digits(bytes([9]) * 32, NONCE_A, NONCE_B) != base
    assert derive_digits(H, bytes([9]) * 32, NONCE_B) != base
    assert derive_digits(H, NONCE_A, bytes([9]) * 32) != base


def test_derive_qr_code_round_trips_as_version_one_token() -> None:
    """The first 24 digest bytes encode and decode as an SP:1 pairing token."""
    code = derive_qr_code(H, NONCE_A, NONCE_B)
    assert len(code) == QR_CODE_SIZE
    token = encode_qr_token(code)
    assert token.startswith("SP:1")
    assert decode_qr_token(token) == code


def test_qr_token_rejects_wrong_version_and_size() -> None:
    """Pairing tokens reject unsupported versions and payload sizes."""
    with pytest.raises(ValueError, match="version"):
        decode_qr_token("SP:0AAAA")
    with pytest.raises(ValueError, match="qr_code"):
        encode_qr_token(b"short")


def test_commit_known_answer_and_size() -> None:
    """Commit returns the pinned SHA-256 digest of the labeled nonce."""
    digest = commit(NONCE_B)
    assert len(digest) == COMMIT_SIZE
    assert digest.hex() == "03882b6c6c622d0d347626dad0e7957853e3cd5830163a2688823bba9443dbca"


def test_verify_commit_round_trip() -> None:
    """verify_commit accepts the matching nonce and rejects others."""
    nonce = generate_nonce()
    assert len(nonce) == NONCE_SIZE
    assert verify_commit(nonce, commit(nonce))
    assert not verify_commit(generate_nonce(), commit(nonce))


def test_size_validation() -> None:
    """Nonces of the wrong length are rejected by the derivation helpers."""
    with pytest.raises(ValueError, match="nonce"):
        commit(b"short")
    with pytest.raises(ValueError, match="nonce_a"):
        derive_digits(H, b"short", NONCE_B)
    with pytest.raises(ValueError, match="nonce_b"):
        derive_qr_code(H, NONCE_A, b"short")
