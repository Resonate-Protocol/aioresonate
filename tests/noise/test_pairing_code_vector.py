"""Cross-validation for the SendspinKit dynamic pairing-code known-answer vector."""

from aiosendspin.noise.pairing_code import (
    decode_qr_token,
    derive_digits,
    derive_qr_code,
    encode_qr_token,
)


# Provenance: SendspinKit/Tests/SendspinKitTests/Resources/cpace-mcf-known-answer.json,
# dynamic_transcript; independently verified against cpace-py and the Sendspin spec.
def test_sendspinkit_dynamic_pairing_code_vector() -> None:
    """Match the independently generated SendspinKit dynamic transcript values."""
    handshake_hash = bytes.fromhex(
        "00112233445566778899aabbccddeeff102132435465768798a9bacbdcedfe0f"
    )
    nonce_a = bytes.fromhex("101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f")
    nonce_b = bytes.fromhex("303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f")

    assert derive_digits(handshake_hash, nonce_a, nonce_b) == "268386"
    qr_bytes = derive_qr_code(handshake_hash, nonce_a, nonce_b)
    assert qr_bytes.hex() == "3e2e937a82ea414f686a6155b3628640ee30d3fda85dc931"
    pairing_token = encode_qr_token(qr_bytes)
    assert pairing_token == "SP:1HYXJG6UC5JAU69DKMFK3GYUGIDXDBU75VBO4SMI"  # noqa: S105
    assert decode_qr_token(pairing_token) == qr_bytes
