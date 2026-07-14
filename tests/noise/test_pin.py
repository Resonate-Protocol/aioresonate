"""Tests for dynamic-PIN derivation and commitment (:mod:`aiosendspin.noise.pin`).

The known-answer values are computed from the spec formula over fixed inputs
(``h = 0x00..1f``, ``nonce_A = 0x01*32``, ``nonce_B = 0x02*32``) and pinned here
to catch regressions in the derivation.
"""

from __future__ import annotations

import pytest

from aiosendspin.noise.pin import (
    COMMIT_SIZE,
    MAX_PIN_DIGITS,
    MIN_PIN_DIGITS,
    NONCE_SIZE,
    commit,
    derive_pin,
    generate_nonce,
    verify_commit,
)

H = bytes(range(32))
NONCE_A = bytes([1]) * 32
NONCE_B = bytes([2]) * 32


@pytest.mark.parametrize(
    ("pin_length", "expected"),
    [(4, "8562"), (6, "638562"), (8, "97638562"), (12, "437497638562")],
)
def test_derive_pin_known_answer(pin_length: int, expected: str) -> None:
    """Derivation matches the pinned known-answer value for each length."""
    assert derive_pin(H, NONCE_A, NONCE_B, pin_length) == expected


@pytest.mark.parametrize("pin_length", range(MIN_PIN_DIGITS, MAX_PIN_DIGITS + 1))
def test_derive_pin_has_requested_length(pin_length: int) -> None:
    """The PIN is always exactly ``pin_length`` ASCII decimal digits."""
    for seed in range(20):
        pin = derive_pin(H, NONCE_A, bytes([seed]) * 32, pin_length)
        assert len(pin) == pin_length
        assert pin.isdigit()


def test_derive_pin_depends_on_each_input() -> None:
    """Changing the handshake hash, either nonce, or the length changes the PIN."""
    base = derive_pin(H, NONCE_A, NONCE_B, 8)
    assert derive_pin(bytes([9]) * 32, NONCE_A, NONCE_B, 8) != base
    assert derive_pin(H, bytes([9]) * 32, NONCE_B, 8) != base
    assert derive_pin(H, NONCE_A, bytes([9]) * 32, 8) != base
    assert derive_pin(H, NONCE_A, NONCE_B, 6) != base


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
    """Nonces of the wrong length and out-of-range pin lengths are rejected."""
    with pytest.raises(ValueError, match="nonce"):
        commit(b"short")
    with pytest.raises(ValueError, match="nonce_a"):
        derive_pin(H, b"short", NONCE_B, 8)
    with pytest.raises(ValueError, match="nonce_b"):
        derive_pin(H, NONCE_A, b"short", 8)
    with pytest.raises(ValueError, match="pin_length"):
        derive_pin(H, NONCE_A, NONCE_B, MIN_PIN_DIGITS - 1)
    with pytest.raises(ValueError, match="pin_length"):
        derive_pin(H, NONCE_A, NONCE_B, MAX_PIN_DIGITS + 1)
