"""Pairing-token encoding for the Pairing PSK flow."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Final

from .keys import (
    PSK_SIZE,
    X25519_KEY_SIZE,
    b64url_decode,
    b64url_encode,
)

_TOKEN_PREFIX: Final[str] = "SP:"  # noqa: S105 - format marker, not a credential
_PSK_TOKEN_VERSION: Final[str] = "1"  # noqa: S105 - format version, not a credential
_PSK_TOKEN_SIZE: Final[int] = X25519_KEY_SIZE + PSK_SIZE


@dataclass(frozen=True, slots=True)
class PSKPairingToken:
    """Pairing Token."""

    client_id: str
    pairing_psk: bytes


def encode_token(token: PSKPairingToken) -> str:
    """Encode a ``PSKPairingToken`` as a single QR- and paste-friendly string."""
    try:
        client_id = b64url_decode(token.client_id)
    except ValueError as exc:
        raise ValueError("client_id is not valid base64url") from exc
    if len(client_id) != X25519_KEY_SIZE:
        msg = f"client_id must decode to {X25519_KEY_SIZE} bytes"
        raise ValueError(msg)
    if len(token.pairing_psk) != PSK_SIZE:
        msg = f"pairing_psk must be {PSK_SIZE} bytes"
        raise ValueError(msg)
    contents = client_id + token.pairing_psk
    body = base64.b32encode(contents).decode("ascii").rstrip("=").replace("2", "9")
    return f"{_TOKEN_PREFIX}{_PSK_TOKEN_VERSION}{body}"


def decode_token(value: str) -> PSKPairingToken:
    """Decode a pairing token (leniently), raising ``ValueError`` if malformed or unsupported."""
    text = value.strip().upper().removeprefix(_TOKEN_PREFIX)
    if not text:
        raise ValueError("malformed pairing token")
    version, body = text[0], text[1:]
    if version != _PSK_TOKEN_VERSION:
        raise ValueError("unsupported pairing token version")
    pad = (-len(body)) % 8
    try:
        payload = base64.b32decode(body.replace("9", "2") + "=" * pad)
    except ValueError as exc:
        raise ValueError("malformed pairing token") from exc
    if len(payload) != _PSK_TOKEN_SIZE:
        raise ValueError("malformed pairing token")
    return PSKPairingToken(
        client_id=b64url_encode(payload[:X25519_KEY_SIZE]),
        pairing_psk=payload[X25519_KEY_SIZE:],
    )
