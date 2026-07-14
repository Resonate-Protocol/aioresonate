"""Pure Noise KKpsk2 protocol object — handshake and transport, no I/O."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, cast

from cryptography.exceptions import InvalidTag
from noise.connection import Keypair, NoiseConnection
from noise.exceptions import NoiseInvalidMessage

from .keys import PSK_SIZE, X25519_KEY_SIZE


class NoiseCipherSuite(StrEnum):
    """The Noise KKpsk2 cipher suites this implementation supports."""

    CHACHAPOLY = "25519_ChaChaPoly_SHA256"
    AESGCM = "25519_AESGCM_SHA256"

    @property
    def pattern_name(self) -> bytes:
        """The full Noise pattern string fed into the underlying library."""
        return f"Noise_KKpsk2_{self.value}".encode("ascii")


class NoiseSession:
    """A Noise KKpsk2 session — handshake-and-transport state machine."""

    def __init__(self, conn: NoiseConnection, *, suite: NoiseCipherSuite) -> None:
        """Wrap an already-configured ``NoiseConnection``; prefer the factories."""
        self._conn = conn
        self._suite = suite

    @property
    def suite(self) -> NoiseCipherSuite:
        """The cipher suite negotiated for this session."""
        return self._suite

    @classmethod
    def as_initiator(
        cls,
        *,
        suite: NoiseCipherSuite,
        local_static_priv: bytes,
        remote_static_pub: bytes,
        prologue: bytes,
        psk: bytes,
    ) -> NoiseSession:
        """Build the **server-side** session and start the handshake."""
        _check_key_size("local_static_priv", local_static_priv)
        _check_key_size("remote_static_pub", remote_static_pub)
        _check_psk_size(psk)
        conn = _build_conn(
            suite=suite,
            initiator=True,
            local_static_priv=local_static_priv,
            remote_static_pub=remote_static_pub,
            prologue=prologue,
            psk=psk,
        )
        return cls(conn, suite=suite)

    @classmethod
    def as_responder(
        cls,
        *,
        suite: NoiseCipherSuite,
        local_static_priv: bytes,
        remote_static_pub: bytes,
        prologue: bytes,
    ) -> NoiseSession:
        """Build the **client-side** session and start the handshake."""
        _check_key_size("local_static_priv", local_static_priv)
        _check_key_size("remote_static_pub", remote_static_pub)
        conn = _build_conn(
            suite=suite,
            initiator=False,
            local_static_priv=local_static_priv,
            remote_static_pub=remote_static_pub,
            prologue=prologue,
            psk=_PLACEHOLDER_PSK,
        )
        return cls(conn, suite=suite)

    def write_message(self, payload: bytes = b"") -> bytes:
        """Produce the next outgoing handshake message containing ``payload``."""
        return bytes(self._conn.write_message(payload))

    def read_message(self, ciphertext: bytes) -> bytes:
        """Consume the next incoming handshake message; return the decrypted payload."""
        try:
            return bytes(self._conn.read_message(ciphertext))
        except InvalidTag as exc:
            raise NoiseInvalidMessage("Failed authentication of handshake message") from exc

    def mix_psk(self, psk: bytes) -> None:
        """Swap in the real PSK between reading message 1 and writing message 2."""
        _check_psk_size(psk)
        self._conn.set_psks(psks=[psk])

    @property
    def handshake_complete(self) -> bool:
        """``True`` once both handshake messages have been processed."""
        return cast("bool", self._conn.handshake_finished)

    @property
    def handshake_hash(self) -> bytes:
        """The 32-byte Noise handshake hash ``h``."""
        if not self.handshake_complete:
            msg = "handshake_hash is only available after the handshake completes"
            raise RuntimeError(msg)
        return cast("bytes", self._conn.get_handshake_hash())

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt and authenticate ``plaintext`` for transport mode."""
        return cast("bytes", self._conn.encrypt(plaintext))

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt and authenticate ``ciphertext`` for transport mode."""
        return cast("bytes", self._conn.decrypt(ciphertext))


# --- private helpers -----------------------------------------------------

_PLACEHOLDER_PSK: Final[bytes] = b"\x00" * PSK_SIZE


def _check_key_size(name: str, value: bytes) -> None:
    if len(value) != X25519_KEY_SIZE:
        msg = f"{name} must be {X25519_KEY_SIZE} bytes, got {len(value)}"
        raise ValueError(msg)


def _check_psk_size(psk: bytes) -> None:
    if len(psk) != PSK_SIZE:
        msg = f"PSK must be {PSK_SIZE} bytes, got {len(psk)}"
        raise ValueError(msg)


def _build_conn(
    *,
    suite: NoiseCipherSuite,
    initiator: bool,
    local_static_priv: bytes,
    remote_static_pub: bytes,
    prologue: bytes,
    psk: bytes,
) -> NoiseConnection:
    conn = NoiseConnection.from_name(suite.pattern_name)
    if initiator:
        conn.set_as_initiator()
    else:
        conn.set_as_responder()
    conn.set_keypair_from_private_bytes(Keypair.STATIC, local_static_priv)
    conn.set_keypair_from_public_bytes(Keypair.REMOTE_STATIC, remote_static_pub)
    conn.set_prologue(prologue)
    conn.set_psks(psks=[psk])
    conn.start_handshake()
    return conn
