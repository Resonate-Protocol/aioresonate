"""Tests for :mod:`aiosendspin.noise.wire`."""

from __future__ import annotations

import pytest
from aiohttp import WSMessage, WSMsgType

from aiosendspin.noise.constants import (
    MAX_TRANSPORT_PLAINTEXT,
    MSG_TYPE_FRAGMENT_END,
    MSG_TYPE_FRAGMENT_MORE,
)
from aiosendspin.noise.keys import Identity
from aiosendspin.noise.session import NoiseCipherSuite, NoiseSession
from aiosendspin.noise.wire import EncryptedWebSocket
from tests.noise.conftest import FakeWebSocket, make_paired_sessions


def test_wrapper_refuses_pre_handshake_session() -> None:
    """Constructing EncryptedWebSocket on a not-yet-complete session raises."""
    server, client = Identity.generate(), Identity.generate()
    incomplete = NoiseSession.as_responder(
        suite=NoiseCipherSuite.CHACHAPOLY,
        local_static_priv=client.private_bytes,
        remote_static_pub=server.public_bytes,
        prologue=b"",
    )
    with pytest.raises(RuntimeError, match="transport mode"):
        EncryptedWebSocket(FakeWebSocket(), incomplete)


async def test_send_str_emits_type_zero_binary_frame() -> None:
    r"""send_str encrypts ``b'\x00' + utf8(payload)`` and emits one BINARY frame."""
    initiator, responder = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, initiator)

    await wrapper.send_str('{"type":"server/hello"}')

    assert len(ws.sent) == 1
    # Decrypt on the peer side to verify wire format.
    plaintext = responder.decrypt(ws.sent[0])
    assert plaintext[:1] == b"\x00"
    assert plaintext[1:].decode("utf-8") == '{"type":"server/hello"}'


async def test_send_bytes_keeps_caller_type_prefix() -> None:
    """send_bytes does not add a type byte; it encrypts the bytes verbatim."""
    initiator, responder = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, initiator)

    # Caller prefixes type-4 (player audio) + 8-byte timestamp + frame data.
    payload = b"\x04" + (b"\x00" * 8) + b"audio-data"
    await wrapper.send_bytes(payload)

    plaintext = responder.decrypt(ws.sent[0])
    assert plaintext == payload


async def test_send_bytes_rejects_empty_payload() -> None:
    """An empty payload has no type byte — caller misuse, raise."""
    initiator, _ = make_paired_sessions()
    wrapper = EncryptedWebSocket(FakeWebSocket(), initiator)
    with pytest.raises(ValueError, match="leading type byte"):
        await wrapper.send_bytes(b"")


async def test_iter_decodes_type_zero_into_synthesized_text_message() -> None:
    """Incoming BINARY with type byte 0 yields TEXT WSMessage with UTF-8 body."""
    initiator, responder = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, responder)

    # Initiator-side encrypts a JSON body the same way send_str would; we
    # push it as a BINARY frame onto the fake ws so wrapper.__aiter__ sees it.
    ct = initiator.encrypt(b"\x00" + b'{"type":"server/hello"}')
    await ws.push(WSMessage(WSMsgType.BINARY, ct, ""))
    await ws.push(None)

    seen = [m async for m in wrapper]
    assert len(seen) == 1
    assert seen[0].type is WSMsgType.TEXT
    assert seen[0].data == '{"type":"server/hello"}'


async def test_iter_keeps_role_type_byte_in_synthesized_binary_message() -> None:
    """Incoming BINARY with a non-zero type byte yields BINARY with the type byte intact.

    The existing per-role dispatch reads the type at offset 0, so the wrapper
    must NOT strip it from synthesized BINARY messages.
    """
    initiator, responder = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, responder)

    role_frame = b"\x04" + (b"\x00" * 8) + b"audio"  # type 4 = player audio
    ct = initiator.encrypt(role_frame)
    await ws.push(WSMessage(WSMsgType.BINARY, ct, ""))
    await ws.push(None)

    seen = [m async for m in wrapper]
    assert seen[0].type is WSMsgType.BINARY
    assert seen[0].data == role_frame  # type byte retained


async def test_iter_passes_transport_error_through_unchanged() -> None:
    """ERROR frames (aiohttp transport-layer signal) pass through verbatim."""
    initiator, _ = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, initiator)

    error_msg = WSMessage(WSMsgType.ERROR, RuntimeError("boom"), "")
    await ws.push(error_msg)
    await ws.push(None)

    seen = [m async for m in wrapper]
    assert seen == [error_msg]


async def test_iter_rejects_cleartext_text_frame_as_error() -> None:
    """A cleartext TEXT frame post-handshake is a spec violation; surface as ERROR."""
    initiator, _ = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, initiator)

    await ws.push(WSMessage(WSMsgType.TEXT, "unauthenticated", ""))
    await ws.push(None)

    seen = [m async for m in wrapper]
    assert len(seen) == 1
    assert seen[0].type is WSMsgType.ERROR
    assert isinstance(seen[0].data, RuntimeError)
    assert "TEXT" in str(seen[0].data)


async def test_send_bytes_below_limit_is_a_single_frame() -> None:
    """A payload that fits in one transport frame is not fragmented."""
    initiator, responder = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, initiator)

    payload = b"\x04" + b"x" * (MAX_TRANSPORT_PLAINTEXT - 1)
    await wrapper.send_bytes(payload)

    assert len(ws.sent) == 1
    assert responder.decrypt(ws.sent[0]) == payload


async def test_send_bytes_fragments_oversized_payload() -> None:
    """An oversized payload is split into more/end frames that reassemble exactly."""
    initiator, responder = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, initiator)

    body = bytes(range(256)) * 600  # 153600 bytes -> spans three frames after the type byte
    payload = b"\x04" + body
    await wrapper.send_bytes(payload)

    assert len(ws.sent) > 1
    frames = [responder.decrypt(ct) for ct in ws.sent]
    assert all(len(f) <= MAX_TRANSPORT_PLAINTEXT for f in frames)
    assert frames[0][0] == MSG_TYPE_FRAGMENT_MORE
    assert frames[0][1] == 0x04  # orig_type carried on the opening frame only
    assert all(f[0] == MSG_TYPE_FRAGMENT_MORE for f in frames[1:-1])
    assert frames[-1][0] == MSG_TYPE_FRAGMENT_END

    reassembled = frames[0][2:] + b"".join(f[1:] for f in frames[1:])
    assert reassembled == body


async def test_receive_reassembles_fragmented_round_trip() -> None:
    """A fragmented send is delivered as one BINARY message on the peer."""
    initiator, responder = make_paired_sessions()
    sender_ws = FakeWebSocket()
    sender = EncryptedWebSocket(sender_ws, initiator)
    receiver_ws = FakeWebSocket()
    receiver = EncryptedWebSocket(receiver_ws, responder)

    body = b"".join(bytes([i % 256]) for i in range(200_000))
    payload = b"\x04" + body
    await sender.send_bytes(payload)
    assert len(sender_ws.sent) > 1

    for ct in sender_ws.sent:
        await receiver_ws.push(WSMessage(WSMsgType.BINARY, ct, ""))
    await receiver_ws.push(None)

    seen = [m async for m in receiver]
    assert len(seen) == 1
    assert seen[0].type is WSMsgType.BINARY
    assert seen[0].data == payload


async def test_receive_reassembles_fragmented_json_into_text() -> None:
    """A fragmented JSON body (orig_type 0) is delivered as a single TEXT message."""
    initiator, responder = make_paired_sessions()
    sender = EncryptedWebSocket(FakeWebSocket(), initiator)
    receiver_ws = FakeWebSocket()
    receiver = EncryptedWebSocket(receiver_ws, responder)

    body = '{"big":"' + "z" * 100_000 + '"}'
    await sender.send_str(body)
    for ct in sender._ws.sent:  # type: ignore[attr-defined]  # noqa: SLF001
        await receiver_ws.push(WSMessage(WSMsgType.BINARY, ct, ""))
    await receiver_ws.push(None)

    seen = [m async for m in receiver]
    assert len(seen) == 1
    assert seen[0].type is WSMsgType.TEXT
    assert seen[0].data == body


async def test_receive_rejects_fragmented_message_exceeding_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reassembly buffer that would exceed the cap is rejected before it grows."""
    monkeypatch.setattr("aiosendspin.noise.wire.MAX_REASSEMBLED_MESSAGE_BYTES", 10)
    initiator, responder = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, responder)

    start = initiator.encrypt(bytes([MSG_TYPE_FRAGMENT_MORE, 0x04]) + b"12345")
    overflow = initiator.encrypt(bytes([MSG_TYPE_FRAGMENT_MORE]) + b"67890ABC")
    await ws.push(WSMessage(WSMsgType.BINARY, start, ""))
    await ws.push(WSMessage(WSMsgType.BINARY, overflow, ""))
    await ws.push(None)

    seen = [m async for m in wrapper]
    assert seen[0].type is WSMsgType.ERROR
    assert "maximum reassembly size" in str(seen[0].data)


async def test_receive_rejects_fragment_end_with_nothing_in_flight() -> None:
    """A fragment-end frame with no message in flight surfaces as ERROR."""
    initiator, responder = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, responder)

    ct = initiator.encrypt(bytes([MSG_TYPE_FRAGMENT_END]) + b"orphan")
    await ws.push(WSMessage(WSMsgType.BINARY, ct, ""))
    await ws.push(None)

    seen = [m async for m in wrapper]
    assert seen[0].type is WSMsgType.ERROR
    assert "no fragmented message in flight" in str(seen[0].data)


async def test_receive_rejects_non_fragment_frame_mid_reassembly() -> None:
    """A normal frame arriving while reassembly is in flight is a protocol violation."""
    initiator, responder = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, responder)

    more = initiator.encrypt(bytes([MSG_TYPE_FRAGMENT_MORE, 0x04]) + b"start")
    interloper = initiator.encrypt(b"\x00" + b'{"type":"x"}')
    await ws.push(WSMessage(WSMsgType.BINARY, more, ""))
    await ws.push(WSMessage(WSMsgType.BINARY, interloper, ""))
    await ws.push(None)

    seen = [m async for m in wrapper]
    assert seen[0].type is WSMsgType.ERROR
    assert "in flight" in str(seen[0].data)


async def test_iter_surfaces_empty_plaintext_as_error_message() -> None:
    """If a decrypted plaintext is empty (protocol violation), yield an ERROR WSMessage."""
    initiator, responder = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, responder)

    ct = initiator.encrypt(b"")  # Noise allows empty plaintext; our protocol does not.
    await ws.push(WSMessage(WSMsgType.BINARY, ct, ""))
    await ws.push(None)

    seen = [m async for m in wrapper]
    assert len(seen) == 1
    assert seen[0].type is WSMsgType.ERROR
    assert isinstance(seen[0].data, RuntimeError)
    assert "empty plaintext" in str(seen[0].data)


def _tamper_tag(ct: bytes) -> bytes:
    """Flip the final byte of a ciphertext (its Poly1305/GCM auth tag)."""
    return ct[:-1] + bytes([ct[-1] ^ 0x01])


def _tamper_body(ct: bytes) -> bytes:
    """Flip the first byte of a ciphertext (its encrypted payload)."""
    return bytes([ct[0] ^ 0x01]) + ct[1:]


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("tampered_tag", _tamper_tag),
        ("tampered_body", _tamper_body),
        ("truncated", lambda ct: ct[: len(ct) // 2]),
        ("empty", lambda _ct: b""),
    ],
)
async def test_corrupt_transport_frame_surfaces_as_error(
    name: str,
    mutate: object,
) -> None:
    """A frame that fails Noise authentication becomes an ERROR, not a raised exception.

    Post-handshake, a bit-flipped tag, a mutated body, a truncated frame, and an
    empty frame all fail the AEAD check; the wrapper must surface each as an ERROR
    ``WSMessage`` so the read loop tears the connection down (rather than escaping
    as an unhandled ``NoiseInvalidMessage``).
    """
    initiator, responder = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, responder)

    ct = initiator.encrypt(b"\x00" + b'{"type":"server/hello"}')
    await ws.push(WSMessage(WSMsgType.BINARY, mutate(ct), ""))  # type: ignore[operator]
    await ws.push(None)

    seen = [m async for m in wrapper]
    assert len(seen) == 1, name
    assert seen[0].type is WSMsgType.ERROR, name
    assert isinstance(seen[0].data, RuntimeError)
    assert "failed authentication" in str(seen[0].data)


async def test_replayed_transport_frame_surfaces_as_error() -> None:
    """A verbatim replay of an already-accepted frame fails the Noise nonce check.

    Noise's monotonic transport nonce means the receiver has advanced past the
    replayed frame's nonce, so re-delivering an accepted frame authenticates as
    corruption — the wrapper surfaces it as ERROR.
    """
    initiator, responder = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, responder)

    frame = initiator.encrypt(b"\x00first")
    await ws.push(WSMessage(WSMsgType.BINARY, frame, ""))  # accepted at nonce 0
    await ws.push(WSMessage(WSMsgType.BINARY, frame, ""))  # replay -> nonce mismatch
    await ws.push(None)

    seen = [m async for m in wrapper]
    assert seen[0].type is WSMsgType.TEXT
    assert seen[0].data == "first"
    assert seen[1].type is WSMsgType.ERROR
    assert "failed authentication" in str(seen[1].data)


async def test_reordered_transport_frame_surfaces_as_error() -> None:
    """Delivering frames out of order desynchronizes the nonce and fails authentication."""
    initiator, responder = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, responder)

    initiator.encrypt(b"\x00zero")  # burn nonce 0 without delivering it
    frame_one = initiator.encrypt(b"\x00one")  # nonce 1
    await ws.push(WSMessage(WSMsgType.BINARY, frame_one, ""))  # receiver still expects nonce 0
    await ws.push(None)

    seen = [m async for m in wrapper]
    assert seen[0].type is WSMsgType.ERROR
    assert "failed authentication" in str(seen[0].data)


async def test_corrupt_frame_mid_reassembly_clears_partial_plaintext() -> None:
    """An auth failure while a fragmented message is in flight discards the partial buffer.

    Routing decrypt failures through the same ``_error`` path as other protocol
    violations means no half-reassembled plaintext lingers after the ERROR.
    """
    initiator, responder = make_paired_sessions()
    ws = FakeWebSocket()
    wrapper = EncryptedWebSocket(ws, responder)

    start = initiator.encrypt(bytes([MSG_TYPE_FRAGMENT_MORE, 0x04]) + b"partial")  # nonce 0
    corrupt = _tamper_tag(initiator.encrypt(bytes([MSG_TYPE_FRAGMENT_MORE]) + b"more"))  # nonce 1
    await ws.push(WSMessage(WSMsgType.BINARY, start, ""))
    await ws.push(WSMessage(WSMsgType.BINARY, corrupt, ""))
    await ws.push(None)

    seen = [m async for m in wrapper]
    assert seen[0].type is WSMsgType.ERROR
    assert "failed authentication" in str(seen[0].data)
    # No partial plaintext is retained once the frame is rejected.
    assert wrapper._reasm_buf is None  # noqa: SLF001
    assert wrapper._reasm_type is None  # noqa: SLF001


@pytest.mark.parametrize(
    "body_len",
    [
        0,
        1,
        MAX_TRANSPORT_PLAINTEXT - 2,  # last byte that fits before the type byte forces a split
        MAX_TRANSPORT_PLAINTEXT - 1,
        MAX_TRANSPORT_PLAINTEXT,  # first size that must fragment
        MAX_TRANSPORT_PLAINTEXT + 1,
        2 * MAX_TRANSPORT_PLAINTEXT,
        3 * MAX_TRANSPORT_PLAINTEXT + 7,
    ],
)
async def test_fragmentation_round_trip_at_boundaries(body_len: int) -> None:
    """A payload of any length round-trips intact across the single-frame/fragment boundary."""
    initiator, responder = make_paired_sessions()
    sender_ws = FakeWebSocket()
    sender = EncryptedWebSocket(sender_ws, initiator)
    receiver_ws = FakeWebSocket()
    receiver = EncryptedWebSocket(receiver_ws, responder)

    payload = b"\x04" + bytes((i * 31 + 7) % 256 for i in range(body_len))
    await sender.send_bytes(payload)
    for ct in sender_ws.sent:
        assert isinstance(ct, bytes)
        await receiver_ws.push(WSMessage(WSMsgType.BINARY, ct, ""))
    await receiver_ws.push(None)

    seen = [m async for m in receiver]
    assert len(seen) == 1
    assert seen[0].type is WSMsgType.BINARY
    assert seen[0].data == payload


def test_swap_session_refuses_pre_handshake_session() -> None:
    """Swapping in a session that isn't in transport mode raises."""
    initiator, _ = make_paired_sessions()
    wrapper = EncryptedWebSocket(FakeWebSocket(), initiator)
    server, client = Identity.generate(), Identity.generate()
    incomplete = NoiseSession.as_responder(
        suite=NoiseCipherSuite.CHACHAPOLY,
        local_static_priv=client.private_bytes,
        remote_static_pub=server.public_bytes,
        prologue=b"",
    )
    with pytest.raises(RuntimeError, match="transport mode"):
        wrapper.swap_session(incomplete)


def test_swap_session_replaces_transport_session() -> None:
    """A completed replacement session is accepted and becomes the active session."""
    initiator, _ = make_paired_sessions()
    wrapper = EncryptedWebSocket(FakeWebSocket(), initiator)
    new_initiator, _ = make_paired_sessions()
    wrapper.swap_session(new_initiator)
    assert wrapper.session is new_initiator
