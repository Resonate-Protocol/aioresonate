"""Tests for :mod:`aiosendspin.noise.session`."""

from __future__ import annotations

import pytest
from noise.exceptions import NoiseHandshakeError, NoiseInvalidMessage

from aiosendspin.noise.constants import SENTINEL_PSK
from aiosendspin.noise.keys import Identity, generate_psk
from aiosendspin.noise.session import NoiseCipherSuite, NoiseSession

CHACHAPOLY = NoiseCipherSuite.CHACHAPOLY
AESGCM = NoiseCipherSuite.AESGCM
ALL_SUITES = [CHACHAPOLY, AESGCM]


def _new_pair() -> tuple[Identity, Identity]:
    return Identity.generate(), Identity.generate()


def _drive_handshake(
    initiator: NoiseSession,
    responder: NoiseSession,
    *,
    msg1_payload: bytes = b"",
    msg2_payload: bytes = b"",
    swap_psk: bytes | None = None,
) -> tuple[bytes, bytes]:
    """Drive a full KKpsk2 handshake; return decrypted payloads as (msg1, msg2)."""
    msg1 = initiator.write_message(msg1_payload)
    received_msg1 = responder.read_message(msg1)
    if swap_psk is not None:
        responder.mix_psk(swap_psk)
    msg2 = responder.write_message(msg2_payload)
    received_msg2 = initiator.read_message(msg2)
    return received_msg1, received_msg2


def test_suite_wire_value_and_pattern_name() -> None:
    """Both suites' wire values and Noise pattern strings match the spec."""
    assert NoiseCipherSuite.CHACHAPOLY.value == "25519_ChaChaPoly_SHA256"
    assert NoiseCipherSuite.CHACHAPOLY.pattern_name == b"Noise_KKpsk2_25519_ChaChaPoly_SHA256"
    assert NoiseCipherSuite.AESGCM.value == "25519_AESGCM_SHA256"
    assert NoiseCipherSuite.AESGCM.pattern_name == b"Noise_KKpsk2_25519_AESGCM_SHA256"


def test_suite_lookup_by_wire_string() -> None:
    """NoiseCipherSuite(wire_string) resolves each implemented suite."""
    assert NoiseCipherSuite("25519_ChaChaPoly_SHA256") is NoiseCipherSuite.CHACHAPOLY
    assert NoiseCipherSuite("25519_AESGCM_SHA256") is NoiseCipherSuite.AESGCM


def test_unknown_suite_raises_value_error() -> None:
    """Unknown suite strings raise ValueError."""
    with pytest.raises(ValueError, match="25519_AESGCM_SHA512"):
        NoiseCipherSuite("25519_AESGCM_SHA512")


@pytest.mark.parametrize("suite", ALL_SUITES)
def test_kkpsk2_full_handshake_completes_with_matching_psk(suite: NoiseCipherSuite) -> None:
    """Initiator and responder agree on transport keys with matching PSK."""
    server, client = _new_pair()
    psk = generate_psk()

    initiator = NoiseSession.as_initiator(
        suite=suite,
        local_static_priv=server.private_bytes,
        remote_static_pub=client.public_bytes,
        prologue=b"hello",
        psk=psk,
    )
    responder = NoiseSession.as_responder(
        suite=suite,
        local_static_priv=client.private_bytes,
        remote_static_pub=server.public_bytes,
        prologue=b"hello",
    )
    _drive_handshake(initiator, responder, swap_psk=psk)

    assert initiator.handshake_complete
    assert responder.handshake_complete
    # Both sides derive the same handshake hash.
    assert initiator.handshake_hash == responder.handshake_hash
    assert len(initiator.handshake_hash) == 32


@pytest.mark.parametrize("suite", ALL_SUITES)
def test_handshake_payloads_round_trip_through_encrypted_handshake_messages(
    suite: NoiseCipherSuite,
) -> None:
    """Handshake-message payloads (psk_id in msg1, ack in msg2) decrypt correctly."""
    server, client = _new_pair()
    psk = generate_psk()

    initiator = NoiseSession.as_initiator(
        suite=suite,
        local_static_priv=server.private_bytes,
        remote_static_pub=client.public_bytes,
        prologue=b"",
        psk=psk,
    )
    responder = NoiseSession.as_responder(
        suite=suite,
        local_static_priv=client.private_bytes,
        remote_static_pub=server.public_bytes,
        prologue=b"",
    )

    received_msg1, received_msg2 = _drive_handshake(
        initiator,
        responder,
        msg1_payload=b'{"psk_id":"x"}',
        msg2_payload=b"{}",
        swap_psk=psk,
    )
    assert received_msg1 == b'{"psk_id":"x"}'
    assert received_msg2 == b"{}"


@pytest.mark.parametrize("suite", ALL_SUITES)
def test_transport_encrypt_decrypt_roundtrip(suite: NoiseCipherSuite) -> None:
    """After handshake, encrypt+decrypt roundtrips arbitrary bytes in both directions."""
    server, client = _new_pair()
    psk = generate_psk()
    initiator = NoiseSession.as_initiator(
        suite=suite,
        local_static_priv=server.private_bytes,
        remote_static_pub=client.public_bytes,
        prologue=b"",
        psk=psk,
    )
    responder = NoiseSession.as_responder(
        suite=suite,
        local_static_priv=client.private_bytes,
        remote_static_pub=server.public_bytes,
        prologue=b"",
    )
    _drive_handshake(initiator, responder, swap_psk=psk)

    # Server → client
    plaintext_s2c = b"\x00" + b'{"type":"server/hello"}'
    ct = initiator.encrypt(plaintext_s2c)
    assert responder.decrypt(ct) == plaintext_s2c

    # Client → server
    plaintext_c2s = b"\x04" + (b"\x00" * 8) + b"audio frame"
    ct = responder.encrypt(plaintext_c2s)
    assert initiator.decrypt(ct) == plaintext_c2s


def test_mix_psk_with_sentinel_then_real_psk_does_not_recover() -> None:
    """If responder swaps to a PSK that differs from the initiator's, handshake fails."""
    server, client = _new_pair()
    real_psk = generate_psk()
    wrong_psk = generate_psk()

    initiator = NoiseSession.as_initiator(
        suite=CHACHAPOLY,
        local_static_priv=server.private_bytes,
        remote_static_pub=client.public_bytes,
        prologue=b"",
        psk=real_psk,
    )
    responder = NoiseSession.as_responder(
        suite=CHACHAPOLY,
        local_static_priv=client.private_bytes,
        remote_static_pub=server.public_bytes,
        prologue=b"",
    )

    msg1 = initiator.write_message(b"")
    responder.read_message(msg1)
    responder.mix_psk(wrong_psk)
    msg2 = responder.write_message(b"")

    # Initiator authenticates msg2 with real_psk; responder mixed wrong_psk →
    # AEAD tag mismatch ⇒ NoiseInvalidMessage.
    with pytest.raises(NoiseInvalidMessage):
        initiator.read_message(msg2)


def test_prologue_mismatch_fails_handshake() -> None:
    """If the two sides disagree on the prologue, the handshake fails."""
    server, client = _new_pair()
    psk = generate_psk()
    initiator = NoiseSession.as_initiator(
        suite=CHACHAPOLY,
        local_static_priv=server.private_bytes,
        remote_static_pub=client.public_bytes,
        prologue=b"server-saw-this",
        psk=psk,
    )
    responder = NoiseSession.as_responder(
        suite=CHACHAPOLY,
        local_static_priv=client.private_bytes,
        remote_static_pub=server.public_bytes,
        prologue=b"client-saw-this",
    )

    msg1 = initiator.write_message(b"")
    with pytest.raises(NoiseInvalidMessage):
        responder.read_message(msg1)


def test_wrong_remote_static_pub_fails_handshake() -> None:
    """If the responder expects a different server static key, handshake fails."""
    server, client = _new_pair()
    impostor = Identity.generate()
    psk = generate_psk()

    initiator = NoiseSession.as_initiator(
        suite=CHACHAPOLY,
        local_static_priv=server.private_bytes,
        remote_static_pub=client.public_bytes,
        prologue=b"",
        psk=psk,
    )
    responder = NoiseSession.as_responder(
        suite=CHACHAPOLY,
        local_static_priv=client.private_bytes,
        remote_static_pub=impostor.public_bytes,  # not the real server key
        prologue=b"",
    )

    msg1 = initiator.write_message(b"")
    with pytest.raises(NoiseInvalidMessage):
        responder.read_message(msg1)


def test_encrypt_before_handshake_completes_raises() -> None:
    """encrypt() raises if called before the handshake completes."""
    server, client = _new_pair()
    psk = generate_psk()
    initiator = NoiseSession.as_initiator(
        suite=CHACHAPOLY,
        local_static_priv=server.private_bytes,
        remote_static_pub=client.public_bytes,
        prologue=b"",
        psk=psk,
    )
    with pytest.raises(NoiseHandshakeError):
        initiator.encrypt(b"too early")


def test_handshake_hash_before_completion_raises() -> None:
    """handshake_hash raises RuntimeError until the handshake completes."""
    server, client = _new_pair()
    psk = generate_psk()
    initiator = NoiseSession.as_initiator(
        suite=CHACHAPOLY,
        local_static_priv=server.private_bytes,
        remote_static_pub=client.public_bytes,
        prologue=b"",
        psk=psk,
    )
    with pytest.raises(RuntimeError, match="handshake"):
        _ = initiator.handshake_hash


def test_write_message_at_wrong_turn_raises() -> None:
    """Calling write_message when it's our turn to read raises."""
    server, client = _new_pair()
    psk = generate_psk()
    initiator = NoiseSession.as_initiator(
        suite=CHACHAPOLY,
        local_static_priv=server.private_bytes,
        remote_static_pub=client.public_bytes,
        prologue=b"",
        psk=psk,
    )
    initiator.write_message(b"")
    # Now it's initiator's turn to read, not write.
    with pytest.raises(NoiseHandshakeError):
        initiator.write_message(b"again")


def test_initiator_rejects_wrong_private_key_size() -> None:
    """Factory raises ValueError on a non-32-byte private key."""
    other = Identity.generate()
    with pytest.raises(ValueError, match="32 bytes"):
        NoiseSession.as_initiator(
            suite=CHACHAPOLY,
            local_static_priv=b"\x00" * 16,
            remote_static_pub=other.public_bytes,
            prologue=b"",
            psk=SENTINEL_PSK,
        )


def test_initiator_rejects_wrong_psk_size() -> None:
    """Factory raises ValueError on a non-32-byte PSK."""
    server, client = _new_pair()
    with pytest.raises(ValueError, match="PSK must be 32 bytes"):
        NoiseSession.as_initiator(
            suite=CHACHAPOLY,
            local_static_priv=server.private_bytes,
            remote_static_pub=client.public_bytes,
            prologue=b"",
            psk=b"short",
        )


def test_mix_psk_rejects_wrong_size() -> None:
    """mix_psk raises ValueError on a non-32-byte PSK."""
    server, client = _new_pair()
    responder = NoiseSession.as_responder(
        suite=CHACHAPOLY,
        local_static_priv=client.private_bytes,
        remote_static_pub=server.public_bytes,
        prologue=b"",
    )
    with pytest.raises(ValueError, match="PSK must be 32 bytes"):
        responder.mix_psk(b"short")
