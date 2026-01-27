"""WebSocket connection handling for a Sendspin client."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from aiohttp import ClientWebSocketResponse, WSMsgType, web

from aiosendspin.models import unpack_binary_header
from aiosendspin.models.core import (
    ClientCommandMessage,
    ClientGoodbyeMessage,
    ClientHelloMessage,
    ClientHelloPayload,
    ClientStateMessage,
    ClientTimeMessage,
    ServerHelloMessage,
    ServerHelloPayload,
    ServerTimeMessage,
    ServerTimePayload,
    StreamClearMessage,
    StreamEndMessage,
    StreamRequestFormatMessage,
)
from aiosendspin.models.types import (
    BinaryMessageType,
    ClientMessage,
    ConnectionReason,
    GoodbyeReason,
    Roles,
    ServerMessage,
    has_role,
    negotiate_active_roles,
)

from .client import SendspinClient

if TYPE_CHECKING:
    from .server import SendspinServer


logger = logging.getLogger(__name__)

MAX_PENDING_MSG = 4096


@dataclass(frozen=True, slots=True)
class _BinaryFrame:
    """Binary payload with an epoch for droppable queue semantics."""

    epoch: int
    data: bytes
    buffer_end_time_us: int | None = None
    buffer_byte_count: int | None = None
    duration_us: int | None = None


class SendspinConnection:
    """A single WebSocket connection to a Sendspin client device."""

    def __init__(
        self,
        server: SendspinServer,
        *,
        request: web.Request | None = None,
        wsock_client: ClientWebSocketResponse | None = None,
    ) -> None:
        """Initialize a SendspinConnection.

        Exactly one of `request` (client-initiated) or `wsock_client` (server-initiated)
        must be provided.
        """
        self._server = server
        self._wsock_client = wsock_client
        self._wsock_server: web.WebSocketResponse | None = None
        self._request = request

        if request is not None:
            if wsock_client is not None:
                raise ValueError("Only one of request or wsock_client may be provided")
            self._wsock_server = web.WebSocketResponse(heartbeat=55)
            self._logger = logger.getChild(f"unknown-{request.remote}")
        elif wsock_client is not None:
            self._logger = logger.getChild("unknown-client")
        else:
            raise ValueError("Either request or wsock_client must be provided")

        self._to_write: asyncio.Queue[ServerMessage | _BinaryFrame] = asyncio.Queue(
            maxsize=MAX_PENDING_MSG
        )
        self._writer_task: asyncio.Task[None] | None = None
        self._message_loop_task: asyncio.Task[None] | None = None

        self._client_id: str | None = None
        self._client_info: ClientHelloPayload | None = None
        self._active_roles: list[str] = []
        self._client: SendspinClient | None = None

        self._closing = False
        self._disconnecting = False

        self._server_hello_sent = False
        self._initial_state_received = False
        self._initial_state_timeout_handle: asyncio.TimerHandle | None = None

        self._last_goodbye_reason: GoodbyeReason | None = None
        self._binary_epoch = 0
        # FB: remove player role specific things from here
        self._stream_start_time_us: int | None = None
        self._last_late_audio_log_s: float = 0.0
        self._late_audio_skips_since_log: int = 0

    @property
    def websocket_connection(self) -> web.WebSocketResponse | ClientWebSocketResponse:
        """Return the underlying aiohttp WebSocket connection object."""
        wsock = self._wsock_server or self._wsock_client
        assert wsock is not None
        return wsock

    @property
    def is_server_initiated(self) -> bool:
        """Return True if this connection was initiated by the server."""
        return self._wsock_client is not None

    def requires_initial_state(self) -> bool:
        """Whether this connection must receive initial client/state before being 'connected'."""
        # FB: make this role independent, expand the role ABC so this can check if any role needs it
        # FB: also, look if we got the state of all roles that need the initial state, not just if
        # we got the message, but if it had the subobject for all needed roles.
        # (can be sent in multiple messages or combined)
        return has_role(Roles.PLAYER.value, self._active_roles)

    def drop_pending_binary(self) -> None:
        """Drop any queued (not-yet-sent) binary payloads for this connection."""
        self._binary_epoch += 1

    def try_send_binary(
        self,
        data: bytes,
        *,
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
    ) -> bool:
        """Try to enqueue a binary message without disconnecting on queue overflow."""
        try:
            self._to_write.put_nowait(
                _BinaryFrame(
                    epoch=self._binary_epoch,
                    data=data,
                    buffer_end_time_us=buffer_end_time_us,
                    buffer_byte_count=buffer_byte_count,
                    duration_us=duration_us,
                )
            )
        except asyncio.QueueFull:
            return False
        return True

    # FB: remove client side backpressure, assume that clients have enough throughput
    def queue_high_water(self, threshold: float = 0.8) -> bool:
        """Return True if the outgoing queue is at/above a high water mark."""
        max_size = self._to_write.maxsize
        if max_size <= 0:
            return False
        return self._to_write.qsize() >= max_size * threshold

    def queue_status(self) -> tuple[int, int]:
        """Return (qsize, maxsize) for the outgoing queue."""
        return self._to_write.qsize(), self._to_write.maxsize

    def send_message(self, message: ServerMessage | bytes) -> None:
        """
        Enqueue a JSON or binary message to be sent to the client.

        Binary payloads are considered droppable. Prefer try_send_binary for audio/art/vis.
        """
        if isinstance(message, bytes):
            if (not self.try_send_binary(message)) and (not self._disconnecting):
                self._logger.error("Message queue full, client too slow - disconnecting")
                task = self._server.loop.create_task(self.disconnect(retry_connection=True))
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            return

        # FB: only drop from the role that is addressed by end/clear messages
        # we dont want to drop artwork when playback stops
        if isinstance(message, StreamClearMessage | StreamEndMessage):
            self.drop_pending_binary()

        try:
            self._to_write.put_nowait(message)
        except asyncio.QueueFull:
            if not self._disconnecting:
                self._logger.error("Message queue full, client too slow - disconnecting")
                task = self._server.loop.create_task(self.disconnect(retry_connection=True))
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            return

        if isinstance(message, StreamClearMessage | StreamEndMessage):
            self._stream_start_time_us = None
        elif not isinstance(message, ServerTimeMessage):
            self._logger.debug("Enqueueing message: %s", type(message).__name__)

    async def disconnect(self, *, retry_connection: bool = True) -> None:
        """Disconnect this connection and detach from its persistent client."""
        if not retry_connection:
            self._closing = True
        if self._disconnecting:
            return
        self._disconnecting = True

        if self._initial_state_timeout_handle is not None:
            self._initial_state_timeout_handle.cancel()
            self._initial_state_timeout_handle = None

        if self._writer_task and not self._writer_task.done():
            self._writer_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._writer_task
        if self._message_loop_task and not self._message_loop_task.done():
            self._message_loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._message_loop_task

        wsock = self._wsock_client or self._wsock_server
        if wsock is not None and not wsock.closed:
            with suppress(Exception):
                await wsock.close()

        if self._client is not None:
            self._client.detach_connection(self._last_goodbye_reason)
            self._client = None

        self._logger.info("Connection disconnected")

    def _initial_state_timeout_callback(self) -> None:
        if self._initial_state_received:
            return
        self._initial_state_timeout_handle = None
        self._logger.warning(
            "Client %s failed to send required initial state within timeout (spec violation)",
            self._client_id or "unknown",
        )
        # Be lenient: keep the connection and mark the client as connected anyway.
        # Some clients may not send an initial state update promptly.
        if self._client is not None:
            self._initial_state_received = True
            self._client.mark_connected()

    async def _setup_connection(self) -> None:
        """Prepare a server-side WebSocketResponse, if applicable."""
        if self._wsock_server is not None:
            assert self._request is not None
            async with asyncio.timeout(10):
                await self._wsock_server.prepare(self._request)

        # Start writer task for both client-initiated and server-initiated connections.
        self._logger.info("Connection established")
        self._writer_task = self._server.loop.create_task(self._writer())

    async def _cleanup_connection(self) -> None:
        wsock = self._wsock_client or self._wsock_server
        if wsock is not None and not wsock.closed:
            with suppress(Exception):
                await wsock.close()
        await self.disconnect(retry_connection=not self._closing)

    async def _run_message_loop(self) -> None:
        wsock = self._wsock_server or self._wsock_client
        assert wsock is not None
        try:
            async for msg in wsock:
                timestamp_us = self._server.clock.now_us()

                if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
                    break

                if msg.type == WSMsgType.BINARY:
                    self._logger.warning("Received binary message from client (spec violation)")
                    continue

                if msg.type != WSMsgType.TEXT:
                    continue

                await self._handle_message(
                    ClientMessage.from_json(cast("str", msg.data)), timestamp_us
                )
        except asyncio.CancelledError:
            self._logger.debug("Message loop cancelled")
        except Exception:
            self._logger.exception("Unexpected error inside websocket API")
        finally:
            if self._writer_task and not self._writer_task.done():
                self._writer_task.cancel()

    async def _handle_message(self, message: ClientMessage, timestamp_us: int) -> None:  # noqa: PLR0915
        if self._client_info is None and not isinstance(message, ClientHelloMessage):
            raise ValueError("First message must be client/hello")
        if (
            self._client_info is not None
            and not self._server_hello_sent
            and not isinstance(message, ClientHelloMessage)
        ):
            raise ValueError("Client must wait for server/hello before sending other messages")
        if isinstance(message, ClientHelloMessage):
            client_info = message.payload
            if client_info.version != 1:
                self._logger.error(
                    "Incompatible protocol version %s (only '1' is supported)",
                    client_info.version,
                )
                await self.disconnect(retry_connection=False)
                return

            self._client_info = client_info
            self._client_id = client_info.client_id
            self._active_roles = negotiate_active_roles(client_info.supported_roles)
            self._logger = logger.getChild(self._client_id)
            self._logger.info("Received client/hello: %s", client_info)

            client = self._server.get_or_create_client(self._client_id)
            client.attach_connection(self, client_info=client_info, active_roles=self._active_roles)
            self._client = client

            self.send_message(
                ServerHelloMessage(
                    payload=ServerHelloPayload(
                        server_id=self._server.id,
                        name=self._server.name,
                        version=1,
                        active_roles=self._active_roles,
                        connection_reason=ConnectionReason.DISCOVERY,
                    )
                )
            )
            self._server_hello_sent = True

            if self.requires_initial_state():
                self._initial_state_timeout_handle = self._server.loop.call_later(
                    5.0, self._initial_state_timeout_callback
                )
            else:
                client.mark_connected()
            return

        if isinstance(message, ClientTimeMessage):
            client_time = message.payload
            self.send_message(
                ServerTimeMessage(
                    ServerTimePayload(
                        client_transmitted=client_time.client_transmitted,
                        server_received=timestamp_us,
                        server_transmitted=self._server.clock.now_us(),
                    )
                )
            )
            return

        if isinstance(message, ClientStateMessage):
            payload = message.payload
            if self._client is None:
                return

            if self.requires_initial_state() and not self._initial_state_received:
                self._initial_state_received = True
                if self._initial_state_timeout_handle is not None:
                    self._initial_state_timeout_handle.cancel()
                    self._initial_state_timeout_handle = None
                self._client.mark_connected()

            new_state = payload.state
            # DEPRECATED(before-spec-pr-50): fall back to player.state for older clients.
            if new_state is None and payload.player is not None:
                new_state = payload.player.state

            if new_state is not None and new_state != self._client.client_state:
                await self._client.handle_state_transition(new_state)

            if payload.player is not None and self._client.check_role(Roles.PLAYER):
                self._client.handle_player_state_update(payload.player)
            return

        if isinstance(message, StreamRequestFormatMessage):
            if self._client is None:
                return
            await self._client.group.handle_stream_format_request(self._client, message.payload)
            return

        if isinstance(message, ClientCommandMessage):
            if self._client is None:
                return
            if message.payload.controller is not None and self._client.check_role(Roles.CONTROLLER):
                await self._client.handle_controller_command(message.payload.controller)
            return

        if isinstance(message, ClientGoodbyeMessage):
            self._logger.info(
                "Received client/goodbye with reason: %s",
                message.payload.reason,
            )
            self._last_goodbye_reason = message.payload.reason
            retry = message.payload.reason == GoodbyeReason.RESTART
            await self.disconnect(retry_connection=retry)
            return

    async def _writer(self) -> None:
        wsock = self._wsock_server or self._wsock_client
        assert wsock is not None
        try:
            while not wsock.closed and not self._closing:
                item = await self._to_write.get()

                if isinstance(item, _BinaryFrame):
                    if item.epoch != self._binary_epoch:
                        continue
                    data = item.data
                    header = unpack_binary_header(data)

                    # FB: the connection.py should be role independent, remove player specific logic
                    # or make this more general
                    if header.message_type == BinaryMessageType.AUDIO_CHUNK.value:
                        now = self._server.clock.now_us()
                        if self._stream_start_time_us is None:
                            self._stream_start_time_us = now
                        in_grace_period = (now - self._stream_start_time_us) < 2_000_000
                        late_by_us = now - header.timestamp_us
                        if late_by_us > 0 and not in_grace_period:
                            self._late_audio_skips_since_log += 1
                            now_s = time.monotonic()
                            if now_s - self._last_late_audio_log_s >= 1.0:
                                qsize, qmax = self.queue_status()
                                self._logger.warning(
                                    "Late audio: skipping %s chunk(s); "
                                    "late_by_us=%s ts_us=%s now_us=%s "
                                    "queue=%s/%s",
                                    self._late_audio_skips_since_log,
                                    late_by_us,
                                    header.timestamp_us,
                                    now,
                                    qsize,
                                    qmax,
                                )
                                self._late_audio_skips_since_log = 0
                                self._last_late_audio_log_s = now_s
                            continue

                    await wsock.send_bytes(data)
                    if (
                        item.buffer_end_time_us is not None
                        and item.buffer_byte_count is not None
                        and self._client is not None
                        and (buffer_tracker := self._client.buffer_tracker) is not None
                    ):
                        buffer_tracker.register(item.buffer_end_time_us, item.buffer_byte_count)

                    # FB: same here
                    # Rate limit audio to ~110% of real-time to avoid bursty delivery
                    if (
                        header.message_type == BinaryMessageType.AUDIO_CHUNK.value
                        and item.duration_us is not None
                    ):
                        delay_s = item.duration_us / 1.1 / 1_000_000
                        await asyncio.sleep(delay_s)
                    continue

                await wsock.send_str(item.to_json())
        except asyncio.CancelledError:
            self._logger.debug("Writer cancelled")
        except Exception:
            self._logger.exception("Writer failed")

    async def _handle_client(self) -> None:
        """Run the complete websocket connection lifecycle (internal)."""
        try:
            await self._setup_connection()
            self._message_loop_task = self._server.loop.create_task(self._run_message_loop())
            await self._message_loop_task
        finally:
            await self._cleanup_connection()
