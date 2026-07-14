"""Post-handshake encrypted WebSocket wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from aiohttp import WSMessage, WSMsgType
from noise.exceptions import NoiseInvalidMessage

from .constants import (
    MAX_TRANSPORT_PLAINTEXT,
    MSG_TYPE_FRAGMENT_END,
    MSG_TYPE_FRAGMENT_MORE,
    MSG_TYPE_JSON_BODY,
)

if TYPE_CHECKING:
    from .session import NoiseSession

# Bounds a single connection's reassembly buffer against a peer streaming endless fragments.
MAX_REASSEMBLED_MESSAGE_BYTES = 64 * 1024 * 1024


class RawWebSocket(Protocol):
    """Structural typing for the subset of aiohttp WS APIs we use.

    Satisfied structurally by both ``aiohttp.web.WebSocketResponse``
    (server-side) and ``aiohttp.ClientWebSocketResponse`` (client-side); also
    by lightweight in-memory fakes in tests.
    """

    async def send_bytes(self, data: bytes) -> None:
        """Send a binary WebSocket frame."""

    async def receive(self) -> WSMessage:
        """Receive a single WebSocket message."""

    @property
    def closed(self) -> bool:
        """Whether the underlying connection has been closed."""

    @property
    def close_code(self) -> int | None:
        """The WebSocket close code once closed, or ``None``."""

    async def close(self) -> bool:
        """Close the underlying connection."""

    def exception(self) -> BaseException | None:
        """Return the exception that closed the connection, if any."""


class EncryptedWebSocket:
    """Wraps a raw WebSocket post-handshake; routes I/O through a ``NoiseSession``.

    Construct only after ``NoiseSession.handshake_complete`` is true.
    """

    def __init__(self, ws: RawWebSocket, session: NoiseSession) -> None:
        """Initialize the wrapper; ``session`` must already be in transport mode."""
        if not session.handshake_complete:
            msg = "NoiseSession must be in transport mode before wrapping a WebSocket"
            raise RuntimeError(msg)
        self._ws = ws
        self._session = session
        self._reasm_buf: bytearray | None = None
        self._reasm_type: int | None = None

    @property
    def session(self) -> NoiseSession:
        """The underlying ``NoiseSession`` (transport-mode)."""
        return self._session

    def swap_session(self, new_session: NoiseSession) -> None:
        """Replace the transport session with ``new_session`` after a re-handshake."""
        if not new_session.handshake_complete:
            msg = "new NoiseSession must be in transport mode before swapping"
            raise RuntimeError(msg)
        self._session = new_session

    @property
    def closed(self) -> bool:
        """Whether the underlying transport is closed."""
        return self._ws.closed

    @property
    def close_code(self) -> int | None:
        """The underlying transport's close code."""
        return self._ws.close_code

    async def close(self) -> bool:
        """Close the underlying transport."""
        return await self._ws.close()

    def exception(self) -> BaseException | None:
        """Return the underlying transport's exception, if any."""
        return self._ws.exception()

    async def send_str(self, data: str) -> None:
        """Encrypt and send a JSON control body (transport type byte ``0``)."""
        await self._send_plaintext(bytes([MSG_TYPE_JSON_BODY]) + data.encode("utf-8"))

    async def send_bytes(self, data: bytes) -> None:
        """Encrypt and send pre-typed binary data (caller has prefixed the role type byte)."""
        if not data:
            msg = "binary payload must include a leading type byte"
            raise ValueError(msg)
        await self._send_plaintext(data)

    async def _send_plaintext(self, plaintext: bytes) -> None:
        """Encrypt and send ``plaintext``, fragmenting it if it exceeds one frame."""
        if len(plaintext) <= MAX_TRANSPORT_PLAINTEXT:
            await self._ws.send_bytes(self._session.encrypt(plaintext))
            return
        for frame in _fragment(plaintext):
            await self._ws.send_bytes(self._session.encrypt(frame))

    def __aiter__(self) -> EncryptedWebSocket:
        """Iterate decrypted messages (the wrapper is its own async iterator)."""
        return self

    async def __anext__(self) -> WSMessage:
        """Return the next decrypted ``WSMessage``, raising on close."""
        msg = await self.receive()
        if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
            raise StopAsyncIteration
        return msg

    async def receive(self) -> WSMessage:
        """Return the next decrypted message, reassembling fragments across frames."""
        while True:
            raw = await self._ws.receive()
            if raw.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
                return raw
            decoded = self._decode(raw)
            if decoded is not None:
                return decoded

    def _decode(self, msg: WSMessage) -> WSMessage | None:
        """Decrypt one frame; return a ``WSMessage`` or ``None`` to await more fragments."""
        if msg.type is WSMsgType.BINARY:
            try:
                plaintext = self._session.decrypt(msg.data)
            except NoiseInvalidMessage:
                return self._error("Noise transport frame failed authentication")
            if not plaintext:
                return self._error("empty plaintext after Noise decrypt")
            type_byte = plaintext[0]
            if type_byte == MSG_TYPE_FRAGMENT_MORE:
                return self._on_fragment_more(plaintext)
            if type_byte == MSG_TYPE_FRAGMENT_END:
                return self._on_fragment_end(plaintext)
            if self._reasm_buf is not None:
                return self._error("non-fragment frame while a fragmented message is in flight")
            return self._dispatch(plaintext)
        if msg.type is WSMsgType.ERROR:
            return msg
        return self._error(f"unexpected {msg.type.name} frame after Noise handshake")

    def _on_fragment_more(self, plaintext: bytes) -> WSMessage | None:
        """Begin or continue a fragmented message."""
        if self._reasm_buf is None:
            orig_type = plaintext[1:2]
            if not orig_type:
                return self._error("fragment-more start frame missing orig_type")
            self._reasm_type = orig_type[0]
            self._reasm_buf = bytearray(plaintext[2:])
        else:
            data = plaintext[1:]
            if len(self._reasm_buf) + len(data) > MAX_REASSEMBLED_MESSAGE_BYTES:
                return self._error("fragmented message exceeds maximum reassembly size")
            self._reasm_buf += data
        return None

    def _on_fragment_end(self, plaintext: bytes) -> WSMessage:
        """Close a fragmented message and dispatch the reassembled payload."""
        if self._reasm_buf is None or self._reasm_type is None:
            return self._error("fragment-end frame with no fragmented message in flight")
        data = plaintext[1:]
        if len(self._reasm_buf) + len(data) > MAX_REASSEMBLED_MESSAGE_BYTES:
            return self._error("fragmented message exceeds maximum reassembly size")
        self._reasm_buf += data
        reassembled = bytes([self._reasm_type]) + bytes(self._reasm_buf)
        self._reasm_buf = None
        self._reasm_type = None
        return self._dispatch(reassembled)

    def _dispatch(self, plaintext: bytes) -> WSMessage:
        """Synthesize a ``WSMessage`` from a complete, type-prefixed plaintext."""
        if plaintext[0] == MSG_TYPE_JSON_BODY:
            return WSMessage(WSMsgType.TEXT, plaintext[1:].decode("utf-8"), "")
        return WSMessage(WSMsgType.BINARY, plaintext, "")

    def _error(self, message: str) -> WSMessage:
        """Discard any reassembly state and build an ERROR ``WSMessage``."""
        self._reasm_buf = None
        self._reasm_type = None
        return WSMessage(WSMsgType.ERROR, RuntimeError(message), "")


def _fragment(plaintext: bytes) -> list[bytes]:
    """Split an oversized type-prefixed plaintext into fragment frames."""
    orig_type = plaintext[0]
    data = memoryview(plaintext)[1:]
    first_cap = MAX_TRANSPORT_PLAINTEXT - 2
    cont_cap = MAX_TRANSPORT_PLAINTEXT - 1

    frames = [bytes([MSG_TYPE_FRAGMENT_MORE, orig_type]) + bytes(data[:first_cap])]
    rest = data[first_cap:]
    chunks = [rest[i : i + cont_cap] for i in range(0, len(rest), cont_cap)]
    last = len(chunks) - 1
    for index, chunk in enumerate(chunks):
        tag = MSG_TYPE_FRAGMENT_END if index == last else MSG_TYPE_FRAGMENT_MORE
        frames.append(bytes([tag]) + bytes(chunk))
    return frames
