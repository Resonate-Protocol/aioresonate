"""Dynamic-PIN derivation and commitment."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Final

PIN_DERIVE_LABEL: Final[bytes] = b"sendspin-pin-derive-v1"
COMMIT_LABEL: Final[bytes] = b"sendspin-pair-commit-v1"
NONCE_SIZE: Final[int] = 32
COMMIT_SIZE: Final[int] = 32
MIN_PIN_DIGITS: Final[int] = 4
MAX_PIN_DIGITS: Final[int] = 12
DEFAULT_MIN_PIN_DIGITS: Final[int] = 6
SHORT_PIN_DIGITS: Final[int] = 6
STATIC_PIN_DIGITS: Final[int] = 8


def is_valid_static_pin(pin: str) -> bool:
    """Return whether ``pin`` is exactly 8 ASCII decimal digits (the static-PIN format)."""
    return len(pin) == STATIC_PIN_DIGITS and pin.isascii() and pin.isdigit()


def generate_nonce() -> bytes:
    """Return a fresh 32-byte CSPRNG nonce (``nonce_A`` or ``nonce_B``)."""
    return secrets.token_bytes(NONCE_SIZE)


def commit(nonce: bytes) -> bytes:
    """Return ``SHA-256(label || nonce)`` — the commitment ``commit_B`` to ``nonce_B``."""
    _check_size(nonce, NONCE_SIZE, "nonce")
    return hashlib.sha256(COMMIT_LABEL + nonce).digest()


def verify_commit(nonce: bytes, commitment: bytes) -> bool:
    """Return whether ``commitment`` is ``commit(nonce)`` (constant-time)."""
    return hmac.compare_digest(commit(nonce), commitment)


def derive_pin(handshake_hash: bytes, nonce_a: bytes, nonce_b: bytes, pin_length: int) -> str:
    """Derive the ``pin_length``-digit dynamic PIN from the handshake hash and both nonces."""
    _check_pin_length(pin_length)
    _check_size(nonce_a, NONCE_SIZE, "nonce_a")
    _check_size(nonce_b, NONCE_SIZE, "nonce_b")
    digest = hashlib.sha256(PIN_DERIVE_LABEL + handshake_hash + nonce_a + nonce_b).digest()
    pin_int = int.from_bytes(digest, "big") % (10**pin_length)
    return f"{pin_int:0{pin_length}d}"


def _check_size(value: bytes, size: int, name: str) -> None:
    if len(value) != size:
        msg = f"{name} must be {size} bytes, got {len(value)}"
        raise ValueError(msg)


def _check_pin_length(pin_length: int) -> None:
    if not MIN_PIN_DIGITS <= pin_length <= MAX_PIN_DIGITS:
        msg = f"pin_length must be in [{MIN_PIN_DIGITS}, {MAX_PIN_DIGITS}], got {pin_length}"
        raise ValueError(msg)
