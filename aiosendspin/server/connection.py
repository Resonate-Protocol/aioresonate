"""WebSocket connection handling for a Sendspin client."""
# TODO: how is rate limit handled/mentioned in this file?


# TODO: this is a complicated file, please add comments so nobody gets lost in the message sending

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from aiohttp import ClientWebSocketResponse, WSMsgType, web

from aiosendspin.models.core import (
    ClientCommandMessage,
    ClientGoodbyeMessage,
    ClientHelloMessage,
    ClientHelloPayload,
    ClientStateMessage,
    ClientTimeMessage,
    GroupUpdateServerMessage,
    GroupUpdateServerPayload,
    ServerHelloMessage,
    ServerHelloPayload,
    ServerStateMessage,
    ServerStatePayload,
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
    ServerMessage,
    negotiate_active_roles,
)

from .client import SendspinClient

if TYPE_CHECKING:
    from .audio import BufferTracker
    from .roles.base import BinaryHandling, Role
    from .server import SendspinServer


logger = logging.getLogger(__name__)

# TODO: should we make this per role instead? i mean its still max.
MAX_PENDING_MSG = 4096  # Should be more than enough for ~1 minute of buffering
# TODO: remove again if ws timeout is enough, we have buffer overfill handling
SEND_TIMEOUT_S = 5.0  # Max time to wait for a single send before disconnecting


@dataclass(frozen=True, slots=True)
class _BinaryFrame:
    """Binary payload with an epoch for droppable queue semantics."""

    # TODO: document fields
    epoch_all: int
    epoch_family: int
    role_family: str
    data: bytes
    timestamp_us: int  # playback timestamp from header (cached to avoid unpacking)
    message_type: int  # binary message type for role lookup (cached)
    buffer_end_time_us: int | None = None
    buffer_byte_count: int | None = None
    duration_us: int | None = None


# Max timestamp value used to ensure FIFO ordering for JSON messages
# (they sort after all timestamped binary frames at the same priority)
_FIFO_TIMESTAMP = 2**62


class SendspinConnection:
    """A single WebSocket connection to a Sendspin client device."""

    def __init__(
        self,
        server: SendspinServer,
        *,
        request: web.Request | None = None,
        wsock_client: ClientWebSocketResponse | None = None,
        url: str | None = None,
    ) -> None:
        """Initialize a SendspinConnection.

        Exactly one of `request` (client-initiated) or `wsock_client` (server-initiated)
        must be provided. For server-initiated connections, `url` should be provided
        for connection reason lookup and client URL registration.
        """
        self._server = server
        self._wsock_client = wsock_client
        self._wsock_server: web.WebSocketResponse | None = None
        self._request = request
        self._url = url  # For server-initiated connections

        if request is not None:
            if wsock_client is not None:
                raise ValueError("Only one of request or wsock_client may be provided")
            self._wsock_server = web.WebSocketResponse(heartbeat=30, compress=False)
            self._logger = logger.getChild(f"unknown-{request.remote}")
        elif wsock_client is not None:
            self._logger = logger.getChild("unknown-client")
        else:
            raise ValueError("Either request or wsock_client must be provided")

        self._queue_sequence: int = 0  # FIFO tie-breaker across all queues
        self._queue_size: int = 0
        # Outgoing message queues
        self._priority_messages: deque[ServerMessage] = deque()
        self._normal_messages: deque[tuple[int, ServerMessage]] = deque()
        # Binary queues: per role family min-heap of (sort_ts, seq, frame)
        self._binary_queues: dict[str, list[tuple[int, int, _BinaryFrame]]] = defaultdict(list)
        # Global scheduler heaps for binary families
        self._ready_families: list[tuple[int, int, str]] = []
        self._delayed_families: list[tuple[int, int, str]] = []
        self._blocked_until_us: dict[str, int] = {}
        self._block_generation: defaultdict[str, int] = defaultdict(int)
        self._writer_wakeup = asyncio.Event()
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
        self._binary_epoch_by_family: defaultdict[str, int] = defaultdict(int)

        # Timing tracking for binary frame logging (per role family)
        self._last_send_time_us_by_family: dict[str, int] = {}
        self._last_timestamp_us_by_family: dict[str, int] = {}
        self._send_stats_by_family: dict[str, dict[str, float | int]] = {}
        self._send_summary_last_log_s = time.monotonic()

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
            self._binary_epoch_all += 1  # TODO: just have one by family instead?
            self._writer_wakeup.set()
            return
        for family in roles:
            self._binary_epoch_by_family[family] += 1
        self._writer_wakeup.set()

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
        epoch_family = self._binary_epoch_by_family[role_family]
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
        if self._queue_size >= MAX_PENDING_MSG:
            return False  # TODO: lets make max_pending_msg a hard limit, disconnect instead?

        seq = self._queue_sequence
        self._queue_sequence += 1
        # Use timestamp for ordering if present, otherwise FIFO
        # TODO: does this mean if both timestamped and non-timestamped binary messages are sent,
        # that non-timestamped ones are always delayed?
        sort_ts = timestamp_us if timestamp_us > 0 else _FIFO_TIMESTAMP
        family_queue = self._binary_queues[role_family]
        heapq.heappush(family_queue, (sort_ts, seq, frame))
        self._queue_size += 1

        # If family not blocked and this is the head, schedule it
        if role_family not in self._blocked_until_us:
            head_sort_ts, head_seq, _ = family_queue[0]
            if head_sort_ts == sort_ts and head_seq == seq:
                heapq.heappush(self._ready_families, (head_sort_ts, head_seq, role_family))

        self._writer_wakeup.set()
        return True

    def queue_status(self) -> tuple[int, int]:
        """Return (qsize, maxsize) for the outgoing queue."""
        return self._queue_size, MAX_PENDING_MSG

    def send_message(self, message: ServerMessage) -> None:
        """Enqueue a JSON message to be sent to the client."""
        if isinstance(message, StreamClearMessage | StreamEndMessage):
            self.drop_pending_binary(message.payload.roles)

        # Coalesce consecutive state-like messages to avoid client-side clearing on omitted fields.
        if self._normal_messages:
            last_seq, last_message = self._normal_messages[-1]
            merged = self._merge_state_messages(last_message, message)
            if merged is not None:
                self._normal_messages[-1] = (last_seq, merged)
                return

        if self._queue_size >= MAX_PENDING_MSG:
            if not self._disconnecting:
                self._logger.error("Message queue full, client too slow - disconnecting")
                # TODO: use eager task
                task = self._server.loop.create_task(self.disconnect(retry_connection=True))
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            return

        seq = self._queue_sequence
        self._queue_sequence += 1
        self._normal_messages.append((seq, message))
        self._queue_size += 1
        self._writer_wakeup.set()

        if not isinstance(message, ServerTimeMessage):
            self._logger.debug("Enqueueing message: %s", type(message).__name__)

    def _merge_state_messages(
        self,
        existing: ServerMessage,
        incoming: ServerMessage,
    ) -> ServerMessage | None:
        """Merge consecutive state-like messages where safe."""
        # TODO: this hard codes roles, generically merge fields by name instead
        if isinstance(existing, ServerStateMessage) and isinstance(incoming, ServerStateMessage):
            metadata = incoming.payload.metadata or existing.payload.metadata
            controller = incoming.payload.controller or existing.payload.controller
            return ServerStateMessage(ServerStatePayload(metadata=metadata, controller=controller))
        if isinstance(existing, GroupUpdateServerMessage) and isinstance(
            incoming, GroupUpdateServerMessage
        ):
            payload = GroupUpdateServerPayload(
                playback_state=(
                    incoming.payload.playback_state
                    if incoming.payload.playback_state is not None
                    else existing.payload.playback_state
                ),
                group_id=(
                    incoming.payload.group_id
                    if incoming.payload.group_id is not None
                    else existing.payload.group_id
                ),
                group_name=(
                    incoming.payload.group_name
                    if incoming.payload.group_name is not None
                    else existing.payload.group_name
                ),
            )
            return GroupUpdateServerMessage(payload)
        return None

    def send_priority_message(self, message: ServerMessage) -> None:
        """Enqueue a high-priority message (processed before regular queue)."""
        if self._queue_size >= MAX_PENDING_MSG:
            if not self._disconnecting:
                self._logger.error("Message queue full, client too slow - disconnecting")
                # TODO: use eager task
                task = self._server.loop.create_task(self.disconnect(retry_connection=True))
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            return
        self._queue_sequence += 1
        self._priority_messages.append(message)
        self._queue_size += 1
        self._writer_wakeup.set()

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
        # TODO: use eager task
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
                    self._logger.info(
                        "WebSocket closed: type=%s close_code=%s",
                        msg.type.name,
                        wsock.close_code,
                    )
                    break

                if msg.type == WSMsgType.ERROR:
                    self._logger.warning("WebSocket error: %s", wsock.exception() or "unknown")
                    break

                if msg.type == WSMsgType.BINARY:
                    self._logger.warning("Received binary message from client (spec violation)")
                    continue

                if msg.type != WSMsgType.TEXT:
                    self._logger.debug("Ignoring message type: %s", msg.type.name)
                    continue

                await self._handle_message(
                    ClientMessage.from_json(cast("str", msg.data)), timestamp_us
                )
            else:
                # Loop exited normally (iterator exhausted) - connection closed
                close_code = wsock.close_code
                log_func = (
                    self._logger.debug if close_code in (1000, 1001) else self._logger.warning
                )
                log_func(
                    "WebSocket closed, close_code=%s close_message=%s",
                    close_code,
                    getattr(wsock, "close_message", None),
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

            # Register client_id → URL mapping for server-initiated connections
            if self._url is not None:
                self._server.register_client_url(client_info.client_id, self._url)

            # Look up connection reason for server-initiated connections
            connection_reason = (
                self._server.get_connection_reason(self._url)
                if self._url is not None
                else ConnectionReason.DISCOVERY
            )

            self.send_message(
                ServerHelloMessage(
                    payload=ServerHelloPayload(
                        server_id=self._server.id,
                        name=self._server.name,
                        version=1,
                        active_roles=self._active_roles,
                        connection_reason=connection_reason,
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
            stream_active = self._client.group.has_active_stream
            for role in self._client.active_roles:
                # TODO: why is stream_active passed here?
                role.on_stream_request_format(message.payload, stream_active=stream_active)
            return

        if isinstance(message, ClientCommandMessage):
            if self._client is None:
                return
            for role in self._client.active_roles:
                role.on_command(message.payload)
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

    # TODO: explain me this method
    def _check_late_binary(
        self, handling: BinaryHandling | None, role: Role | None, timestamp_us: int
    ) -> bool:
        """Check if binary message is late and should be dropped. Returns True to drop."""
        # timestamp_us=0 means "no playback semantics" - skip late detection
        if handling is None or role is None or not handling.drop_late or timestamp_us == 0:
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
        async with asyncio.timeout(SEND_TIMEOUT_S):
            await wsock.send_str(message.to_json())

    async def _send_binary_frame(
        self,
        wsock: web.WebSocketResponse | ClientWebSocketResponse,
        item: _BinaryFrame,
        buffer_tracker: BufferTracker | None,
    ) -> None:
        """Send a binary frame with buffer tracking. Assumes rate limit already checked."""
        start_s = time.monotonic()
        async with asyncio.timeout(SEND_TIMEOUT_S):
            await wsock.send_bytes(item.data)
        elapsed_ms = (time.monotonic() - start_s) * 1000
        if elapsed_ms >= 50.0:
            self._logger.error(
                "Slow send_bytes: %.1fms size=%s ts_us=%s role=%s",
                elapsed_ms,
                len(item.data),
                item.timestamp_us,
                item.role_family,
            )

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

    # TODO: explain me all methods handling the heap, and _promote_ready_families as well
    # also how drop_pending_binary exactly interacts with the writer loop

    def _schedule_family_head(self, role_family: str) -> None:
        if role_family in self._blocked_until_us:
            return
        if family_queue := self._binary_queues.get(role_family):
            head_sort_ts, head_seq, _ = family_queue[0]
            heapq.heappush(self._ready_families, (head_sort_ts, head_seq, role_family))

    def _discard_family_head(self, role_family: str) -> None:
        family_queue = self._binary_queues.get(role_family)
        if not family_queue:
            return
        heapq.heappop(family_queue)
        self._queue_size = max(self._queue_size - 1, 0)
        if not family_queue:
            self._binary_queues.pop(role_family, None)

    def _peek_ready_binary(self) -> tuple[str, _BinaryFrame, int, int] | None:
        while self._ready_families:
            sort_ts, seq, role_family = heapq.heappop(self._ready_families)
            if role_family in self._blocked_until_us:
                continue
            family_queue = self._binary_queues.get(role_family)
            if not family_queue:
                continue
            head_sort_ts, head_seq, head_frame = family_queue[0]
            if head_sort_ts != sort_ts or head_seq != seq:
                heapq.heappush(self._ready_families, (head_sort_ts, head_seq, role_family))
                continue
            return role_family, head_frame, head_sort_ts, head_seq
        return None

    def _block_family(self, role_family: str, ready_at_us: int) -> None:
        self._blocked_until_us[role_family] = ready_at_us
        generation = self._block_generation[role_family] + 1
        self._block_generation[role_family] = generation
        heapq.heappush(self._delayed_families, (ready_at_us, generation, role_family))

    def _promote_ready_families(self, now_us: int) -> None:
        while self._delayed_families and self._delayed_families[0][0] <= now_us:
            ready_at_us, generation, role_family = heapq.heappop(self._delayed_families)
            if self._block_generation.get(role_family, 0) != generation:
                continue
            blocked_until = self._blocked_until_us.get(role_family)
            if blocked_until is None or blocked_until != ready_at_us:
                continue
            self._blocked_until_us.pop(role_family, None)
            self._schedule_family_head(role_family)

    async def _writer(self) -> None:  # noqa: C901, PLR0912, PLR0915
        wsock = self._wsock_server or self._wsock_client
        assert wsock is not None

        # Cache hot attributes as locals to avoid repeated attribute lookups
        clock_now_us = self._server.clock.now_us
        binary_epoch_by_family = self._binary_epoch_by_family
        last_send_us_by_family = self._last_send_time_us_by_family
        last_ts_us_by_family = self._last_timestamp_us_by_family

        iterations_since_yield = 0
        now_us = clock_now_us()

        try:
            while not wsock.closed and not self._closing:
                # Periodic yield to prevent event loop starvation
                if iterations_since_yield >= 50:
                    # TODO: try removing this/adding logging to se if this is still required
                    await asyncio.sleep(0)
                    iterations_since_yield = 0
                    now_us = clock_now_us()

                # Priority messages always go first
                if self._priority_messages:
                    message = self._priority_messages.popleft()
                    self._queue_size = max(self._queue_size - 1, 0)
                    await self._send_message(wsock, message)
                    now_us = clock_now_us()
                    iterations_since_yield = 0
                    continue

                now_us = clock_now_us()
                self._promote_ready_families(now_us)

                ready_binary = self._peek_ready_binary()
                normal_entry = self._normal_messages[0] if self._normal_messages else None

                if ready_binary is None and normal_entry is None:
                    # No immediate work; wait for new items or next delayed family
                    self._writer_wakeup.clear()
                    if self._priority_messages or self._normal_messages or self._ready_families:
                        continue

                    sleep_s = None
                    if self._delayed_families:
                        next_ready_us = self._delayed_families[0][0]
                        sleep_s = max((next_ready_us - now_us) / 1_000_000, 0.0)

                    try:
                        if sleep_s is None:
                            await self._writer_wakeup.wait()
                        else:
                            await asyncio.wait_for(self._writer_wakeup.wait(), timeout=sleep_s)
                    except TimeoutError:
                        pass
                    continue

                if ready_binary is None and normal_entry is not None:
                    _, message = self._normal_messages.popleft()
                    self._queue_size = max(self._queue_size - 1, 0)
                    await self._send_message(wsock, message)
                    now_us = clock_now_us()
                    iterations_since_yield = 0
                    continue

                assert ready_binary is not None
                role_family, frame, sort_ts, seq = ready_binary

                # TODO: more explanation, dont understand this
                # Compare with normal messages when timestamps are FIFO-equivalent
                if normal_entry is not None and sort_ts == _FIFO_TIMESTAMP:
                    normal_seq, _ = normal_entry
                    if normal_seq < seq:
                        heapq.heappush(self._ready_families, (sort_ts, seq, role_family))
                        _, message = self._normal_messages.popleft()
                        self._queue_size = max(self._queue_size - 1, 0)
                        await self._send_message(wsock, message)
                        now_us = clock_now_us()
                        iterations_since_yield = 0
                        continue

                # Check epoch - frame may have been invalidated
                if frame.epoch_all != self._binary_epoch_all:
                    self._discard_family_head(role_family)
                    self._schedule_family_head(role_family)
                    iterations_since_yield += 1
                    continue
                if frame.epoch_family != binary_epoch_by_family[role_family]:
                    # TODO: if I understand correctly, this is activated on stream clear and
                    # start, discarding all pending frames for the role family
                    # Does this mean that after a new format is requested, audio data that wasn't
                    # sent already is just dropped? and immediately after that message was passed
                    # to send_message?
                    self._discard_family_head(role_family)
                    self._schedule_family_head(role_family)
                    iterations_since_yield += 1
                    continue

                # TODO: delete the "(single lookup for rate limit + buffer tracking)"
                # TODO: maybe directly unpack to handling and handling_role? no
                # TODO: cached variable
                # Get binary handling info (single lookup for rate limit + buffer tracking)
                cached = (
                    self._client.get_binary_handling_cached(frame.message_type)
                    if self._client
                    else None
                )
                handling = cached[0] if cached else None
                handling_role = cached[1] if cached else None

                # Drop late messages if role requests it
                if (
                    handling is not None
                    and handling_role is not None
                    and self._check_late_binary(handling, handling_role, frame.timestamp_us)
                ):
                    self._discard_family_head(role_family)
                    self._schedule_family_head(role_family)
                    iterations_since_yield += 1
                    continue

                # Check rate limit
                wait_us = 0
                buffer_tracker = None
                # TODO: in this method there are a lot of these ifs,
                # hard to understand, add comments explaining each sections
                # responsibility?
                if handling is not None and handling_role is not None:
                    if handling.buffer_track:
                        buffer_tracker = handling_role.get_buffer_tracker()
                    # Stream-start delay (for clients that need a gap before first binary)
                    if buffer_tracker is not None:
                        wait_us = max(wait_us, buffer_tracker.time_until_unblocked())
                    if handling.rate_limit and buffer_tracker is not None:
                        duration_us = frame.duration_us or 0
                        buffer_tracker.prune_consumed(now_us)
                        buffer_depth_us = buffer_tracker.buffered_duration_us
                        max_dur = buffer_tracker.max_duration_us
                        if max_dur > 0 and duration_us > 0:
                            # Allow burst during initial fill window
                            burst_until = getattr(
                                handling_role, "_stream_start_burst_until_us", None
                            )
                            if burst_until is None or now_us >= burst_until:
                                effective_max = int(max_dur * handling.rate_limit_factor)
                                projected = buffer_depth_us + duration_us
                                if projected > effective_max:
                                    wait_us = max(wait_us, projected - effective_max)

                if wait_us > 0:
                    # TODO: explain me this
                    # Delay this frame - one per role family
                    self._block_family(role_family, now_us + wait_us)
                    iterations_since_yield += 1
                    continue

                # TODO: put all debugging info behind the debug flag, and in a
                # separate method
                # Log timing info (only if debug enabled)
                timestamp_us = frame.timestamp_us
                last_send_us = last_send_us_by_family.get(role_family)
                last_ts_us = last_ts_us_by_family.get(role_family)
                send_gap_ms = (now_us - last_send_us) / 1000 if last_send_us is not None else 0
                ts_gap_ms = (timestamp_us - last_ts_us) / 1000 if last_ts_us is not None else 0
                last_send_us_by_family[role_family] = now_us
                last_ts_us_by_family[role_family] = timestamp_us

                # Send immediately
                self._discard_family_head(role_family)
                await self._send_binary_frame(wsock, frame, buffer_tracker)
                stats = self._send_stats_by_family.setdefault(
                    role_family,
                    {
                        "count": 0,
                        "send_gap_sum_ms": 0.0,
                        "send_gap_min_ms": 1e9,
                        "send_gap_max_ms": 0.0,
                        "ts_gap_sum_ms": 0.0,
                        "ts_gap_min_ms": 1e9,
                        "ts_gap_max_ms": 0.0,
                        "buf_count": 0,
                        "buf_sum_ms": 0.0,
                        "buf_min_ms": 1e9,
                        "buf_max_ms": 0.0,
                    },
                )
                if last_send_us is not None and last_ts_us is not None:
                    stats["count"] += 1
                    stats["send_gap_sum_ms"] += send_gap_ms
                    stats["send_gap_min_ms"] = min(stats["send_gap_min_ms"], send_gap_ms)
                    stats["send_gap_max_ms"] = max(stats["send_gap_max_ms"], send_gap_ms)
                    stats["ts_gap_sum_ms"] += ts_gap_ms
                    stats["ts_gap_min_ms"] = min(stats["ts_gap_min_ms"], ts_gap_ms)
                    stats["ts_gap_max_ms"] = max(stats["ts_gap_max_ms"], ts_gap_ms)
                    if buffer_tracker is not None:
                        buf_ms = buffer_tracker.buffered_duration_us / 1000
                        stats["buf_count"] += 1
                        stats["buf_sum_ms"] += buf_ms
                        stats["buf_min_ms"] = min(stats["buf_min_ms"], buf_ms)
                        stats["buf_max_ms"] = max(stats["buf_max_ms"], buf_ms)

                now_s = time.monotonic()
                if now_s - self._send_summary_last_log_s >= 5.0:
                    self._send_summary_last_log_s = now_s
                    for fam, fam_stats in self._send_stats_by_family.items():
                        count = int(fam_stats["count"])
                        if count <= 0:
                            continue
                        avg_send = fam_stats["send_gap_sum_ms"] / count
                        avg_ts = fam_stats["ts_gap_sum_ms"] / count
                        if fam_stats["buf_count"] > 0:
                            avg_buf = fam_stats["buf_sum_ms"] / fam_stats["buf_count"]
                            self._logger.info(
                                "Send summary role=%s samples=%s "
                                "send_gap_ms(avg=%.1f min=%.1f max=%.1f) "
                                "ts_gap_ms(avg=%.1f min=%.1f max=%.1f) "
                                "buf_ms(avg=%.1f min=%.1f max=%.1f)",
                                fam,
                                count,
                                avg_send,
                                fam_stats["send_gap_min_ms"],
                                fam_stats["send_gap_max_ms"],
                                avg_ts,
                                fam_stats["ts_gap_min_ms"],
                                fam_stats["ts_gap_max_ms"],
                                avg_buf,
                                fam_stats["buf_min_ms"],
                                fam_stats["buf_max_ms"],
                            )
                        else:
                            self._logger.info(
                                "Send summary role=%s samples=%s "
                                "send_gap_ms(avg=%.1f min=%.1f max=%.1f) "
                                "ts_gap_ms(avg=%.1f min=%.1f max=%.1f)",
                                fam,
                                count,
                                avg_send,
                                fam_stats["send_gap_min_ms"],
                                fam_stats["send_gap_max_ms"],
                                avg_ts,
                                fam_stats["ts_gap_min_ms"],
                                fam_stats["ts_gap_max_ms"],
                            )
                        fam_stats["count"] = 0
                        fam_stats["send_gap_sum_ms"] = 0.0
                        fam_stats["send_gap_min_ms"] = 1e9
                        fam_stats["send_gap_max_ms"] = 0.0
                        fam_stats["ts_gap_sum_ms"] = 0.0
                        fam_stats["ts_gap_min_ms"] = 1e9
                        fam_stats["ts_gap_max_ms"] = 0.0
                        fam_stats["buf_count"] = 0
                        fam_stats["buf_sum_ms"] = 0.0
                        fam_stats["buf_min_ms"] = 1e9
                        fam_stats["buf_max_ms"] = 0.0
                self._schedule_family_head(role_family)
                now_us = clock_now_us()
                iterations_since_yield = 0
        except asyncio.CancelledError:
            self._logger.debug("Writer cancelled")
        except TimeoutError:
            self._logger.warning("Send timed out - client too slow, disconnecting")
            if not wsock.closed:
                with suppress(Exception):
                    await wsock.close()
        except Exception:
            self._logger.exception("Writer failed")
            # Close the websocket to signal the message loop to exit
            if not wsock.closed:
                with suppress(Exception):
                    await wsock.close()

    async def _handle_client(self) -> None:
        """Run the complete websocket connection lifecycle (internal)."""
        try:
            await self._setup_connection()
            # TODO: use eager task
            self._message_loop_task = self._server.loop.create_task(self._run_message_loop())
            await self._message_loop_task
        finally:
            await self._cleanup_connection()
