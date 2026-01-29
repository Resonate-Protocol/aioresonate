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
    ClientMessage,
    ConnectionReason,
    GoodbyeReason,
    Roles,
    ServerMessage,
    negotiate_active_roles,
)

from .client import SendspinClient

if TYPE_CHECKING:
    from .roles.base import BinaryHandling, Role
    from .server import SendspinServer


logger = logging.getLogger(__name__)

MAX_PENDING_MSG = 4096


@dataclass(frozen=True, slots=True)
class _BinaryFrame:
    """Binary payload with an epoch for droppable queue semantics."""

    epoch_all: int
    epoch_family: int
    role_family: str
    data: bytes
    timestamp_us: int  # playback timestamp from header (cached to avoid unpacking)
    message_type: int  # binary message type for role lookup (cached)
    buffer_end_time_us: int | None = None
    buffer_byte_count: int | None = None
    duration_us: int | None = None


@dataclass(frozen=True, slots=True)
class _PriorityItem:
    """Wrapper for priority queue ordering.

    Priority 0 = high (time sync), 1 = normal.
    Sequence provides FIFO tie-breaking within the same priority.
    """

    priority: int
    sequence: int
    item: ServerMessage | _BinaryFrame

    def __lt__(self, other: _PriorityItem) -> bool:
        return (self.priority, self.sequence) < (other.priority, other.sequence)


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

        self._to_write: asyncio.PriorityQueue[_PriorityItem] = asyncio.PriorityQueue(
            maxsize=MAX_PENDING_MSG
        )
        self._queue_sequence: int = 0  # FIFO tie-breaker for priority queue
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
        self._binary_epoch_all = 0
        self._binary_epoch_by_family: dict[str, int] = {}

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
        if self._client is None:
            return False
        return any(role.requires_initial_state() for role in self._client.active_roles)

    def drop_pending_binary(self, roles: list[str] | None) -> None:
        """Drop queued binary payloads for the specified role families."""
        if roles is None:
            self._binary_epoch_all += 1
            return
        for family in roles:
            self._binary_epoch_by_family[family] = self._binary_epoch_by_family.get(family, 0) + 1

    def try_send_binary(
        self,
        data: bytes,
        *,
        role_family: str,
        timestamp_us: int,
        message_type: int,
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
    ) -> bool:
        """Try to enqueue a binary message without disconnecting on queue overflow.

        Args:
            data: Binary data to send.
            role_family: Role family for epoch tracking.
            timestamp_us: Playback timestamp from binary header (cached to avoid unpacking).
            message_type: Binary message type for role lookup (cached).
            buffer_end_time_us: End timestamp for buffer tracking.
            buffer_byte_count: Byte count for buffer tracking.
            duration_us: Duration for buffer tracking.
        """
        epoch_family = self._binary_epoch_by_family.get(role_family, 0)
        frame = _BinaryFrame(
            epoch_all=self._binary_epoch_all,
            epoch_family=epoch_family,
            role_family=role_family,
            data=data,
            timestamp_us=timestamp_us,
            message_type=message_type,
            buffer_end_time_us=buffer_end_time_us,
            buffer_byte_count=buffer_byte_count,
            duration_us=duration_us,
        )
        seq = self._queue_sequence
        self._queue_sequence += 1
        try:
            self._to_write.put_nowait(_PriorityItem(priority=1, sequence=seq, item=frame))
        except asyncio.QueueFull:
            return False
        return True

    def queue_status(self) -> tuple[int, int]:
        """Return (qsize, maxsize) for the outgoing queue."""
        return self._to_write.qsize(), self._to_write.maxsize

    def send_message(self, message: ServerMessage | bytes) -> None:
        """
        Enqueue a JSON or binary message to be sent to the client.

        Binary payloads are considered droppable. Prefer try_send_binary for audio/art/vis.
        """
        if isinstance(message, bytes):
            # Legacy path: parse header to get timestamp/type for binary messages
            header = unpack_binary_header(message)
            if (
                not self.try_send_binary(
                    message,
                    role_family="unknown",
                    timestamp_us=header.timestamp_us,
                    message_type=header.message_type,
                )
            ) and (not self._disconnecting):
                self._logger.error("Message queue full, client too slow - disconnecting")
                task = self._server.loop.create_task(self.disconnect(retry_connection=True))
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            return

        if isinstance(message, StreamClearMessage | StreamEndMessage):
            self.drop_pending_binary(message.payload.roles)

        seq = self._queue_sequence
        self._queue_sequence += 1
        try:
            self._to_write.put_nowait(_PriorityItem(priority=1, sequence=seq, item=message))
        except asyncio.QueueFull:
            if not self._disconnecting:
                self._logger.error("Message queue full, client too slow - disconnecting")
                task = self._server.loop.create_task(self.disconnect(retry_connection=True))
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            return

        if not isinstance(message, ServerTimeMessage):
            self._logger.debug("Enqueueing message: %s", type(message).__name__)

    def send_priority_message(self, message: ServerMessage) -> None:
        """Enqueue a high-priority message (processed before regular queue)."""
        seq = self._queue_sequence
        self._queue_sequence += 1
        self._to_write.put_nowait(_PriorityItem(priority=0, sequence=seq, item=message))

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
            self.send_priority_message(
                ServerTimeMessage(
                    payload=ServerTimePayload(
                        client_transmitted=client_time.client_transmitted,
                        server_received=timestamp_us,
                        server_transmitted=0,  # Set at actual send time
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
            if new_state is not None and new_state != self._client.client_state:
                await self._client.handle_state_transition(new_state)
            for role in self._client.active_roles:
                role.on_client_state(payload)
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

    def _check_late_binary(
        self, handling: BinaryHandling | None, role: Role | None, timestamp_us: int
    ) -> bool:
        """Check if binary message is late and should be dropped. Returns True to drop."""
        if handling is None or role is None or not handling.drop_late:
            return False

        now = self._server.clock.now_us()
        if role._stream_start_time_us is None:  # noqa: SLF001
            role._stream_start_time_us = now  # noqa: SLF001
        elapsed = now - role._stream_start_time_us  # noqa: SLF001
        in_grace_period = elapsed < handling.grace_period_us
        late_by_us = now - timestamp_us

        if late_by_us > 0 and not in_grace_period:
            role._late_skips_since_log += 1  # noqa: SLF001
            self._logger.debug(
                "Discarding late chunk: late_by=%.1fms, plays_in=%.1fms",
                late_by_us / 1000,
                -late_by_us / 1000,
            )
            now_s = time.monotonic()
            if now_s - role._last_late_log_s >= 1.0:  # noqa: SLF001
                qsize, qmax = self.queue_status()
                self._logger.warning(
                    "Late binary: skipping %s chunk(s); "
                    "late_by_us=%s ts_us=%s now_us=%s queue=%s/%s",
                    role._late_skips_since_log,  # noqa: SLF001
                    late_by_us,
                    timestamp_us,
                    now,
                    qsize,
                    qmax,
                )
                role._late_skips_since_log = 0  # noqa: SLF001
                role._last_late_log_s = now_s  # noqa: SLF001
            return True
        return False

    async def _send_message(
        self,
        wsock: web.WebSocketResponse | ClientWebSocketResponse,
        message: ServerMessage,
    ) -> None:
        """Send a single message, handling time message timestamps."""
        if isinstance(message, ServerTimeMessage):
            # Update timestamp to actual send time
            message = ServerTimeMessage(
                payload=ServerTimePayload(
                    client_transmitted=message.payload.client_transmitted,
                    server_received=message.payload.server_received,
                    server_transmitted=self._server.clock.now_us(),
                )
            )
        await wsock.send_str(message.to_json())

    async def _send_binary_frame(
        self,
        wsock: web.WebSocketResponse | ClientWebSocketResponse,
        item: _BinaryFrame,
    ) -> None:
        """Send a binary frame with buffer tracking and late detection."""
        data = item.data
        timestamp_us = item.timestamp_us
        message_type = item.message_type

        # Find the role that handles this message type (O(1) lookup via client cache)
        handling = None
        handling_role = None
        buffer_tracker = None
        if self._client is not None:
            cached = self._client.get_binary_handling_cached(message_type)
            if cached is not None:
                handling, handling_role = cached
                # Cache buffer_tracker lookup (called once instead of twice)
                if handling.buffer_track:
                    buffer_tracker = handling_role.get_buffer_tracker()

        # Drop late messages if role requests it
        if self._check_late_binary(handling, handling_role, timestamp_us):
            return

        # Rate limit: wait if we're too far ahead of playback (duration-based only)
        if buffer_tracker is not None and handling is not None and handling.rate_limit:
            wait_us = buffer_tracker.time_until_duration_capacity(item.duration_us or 0)
            if wait_us > 0:
                await asyncio.sleep(wait_us / 1_000_000)

        await wsock.send_bytes(data)

        # Buffer tracking via role's tracker (framework-managed)
        if (
            buffer_tracker is not None
            and item.buffer_end_time_us is not None
            and item.buffer_byte_count is not None
        ):
            buffer_tracker.register(
                item.buffer_end_time_us,
                item.buffer_byte_count,
                item.duration_us or 0,
            )

    async def _writer(self) -> None:
        wsock = self._wsock_server or self._wsock_client
        assert wsock is not None
        try:
            while not wsock.closed and not self._closing:
                priority_item = await self._to_write.get()
                item = priority_item.item

                if isinstance(item, _BinaryFrame):
                    if item.epoch_all != self._binary_epoch_all:
                        continue
                    if item.epoch_family != self._binary_epoch_by_family.get(item.role_family, 0):
                        continue
                    await self._send_binary_frame(wsock, item)
                    continue

                await self._send_message(wsock, item)
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
