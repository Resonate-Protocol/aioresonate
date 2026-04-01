"""WebSocket connection handling for a Sendspin client.

Message Sending Architecture
----------------------------
This module implements a priority-based message queue with timestamp ordering for sync.

**Queue Structure:**
- Priority messages: ServerHello, time sync - sent immediately (FIFO deque)
- Normal messages: Non-role JSON control messages - sent in FIFO order (deque)
- Role queues: Per-role min-heaps holding both binary and JSON messages, sorted by
  (timestamp, sequence). Binary messages use their playback timestamp; JSON messages
  inherit the timestamp of the previous message in that role's queue.

**Message Ordering:**
Messages are grouped by role (e.g., player, artwork). Within each role, binary and
JSON messages share the same min-heap, ensuring strict ordering. Binary messages sort
by playback timestamp for correct sequencing even when chunks are encoded out-of-order.
JSON messages inherit the previous message's timestamp so they stay in position relative
to surrounding binary data.

**Epoch-Based Invalidation:**
Each role has an epoch counter. When a stream is cleared or ends, the epoch increments,
causing binary entries with the old epoch to be silently discarded. JSON entries in the
same queue are NOT affected - they skip epoch validation and are always delivered.

**Backpressure:**
Roles can be "blocked" until a future time (e.g., waiting for client buffer space).
Blocked roles are tracked in a separate heap and promoted back when ready.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import orjson
from aiohttp import ClientWebSocketResponse, WSMsgType, web

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
    StreamStartMessage,
)
from aiosendspin.models.types import (
    ClientMessage,
    ConnectionReason,
    GoodbyeReason,
    Roles,
    ServerMessage,
    role_family,
)
from aiosendspin.util import create_task

from .client import SendspinClient
from .roles.negotiation import negotiate_active_roles
from .roles.registry import ROLE_FACTORIES, ROLE_SUPPORT_SPECS

if TYPE_CHECKING:
    from .audio import BufferTracker
    from .roles.base import BinaryHandling, Role
    from .server import SendspinServer


logger = logging.getLogger(__name__)

MAX_PENDING_MSG = 4096  # Default queue cap (per role queues, and global control queues)


@dataclass(frozen=True, slots=True)
class _BinaryData:
    """Binary payload metadata for buffer tracking."""

    data: bytes
    message_type: int
    buffer_end_time_us: int | None = None
    buffer_byte_count: int | None = None
    duration_us: int | None = None


@dataclass(frozen=True, slots=True)
class _RoleQueueEntry:
    """Unified queue entry for binary or JSON messages within a role.

    Both binary and JSON messages for a role go through the same min-heap,
    sorted by (timestamp, sequence). JSON messages inherit the timestamp of the
    previous message in the role queue, ensuring they maintain their position
    relative to surrounding timed binary. If no previous message exists, timestamp is 0.
    """

    epoch: int
    timestamp_us: int
    # Exactly one of these is set
    binary: _BinaryData | None = None
    json_message: ServerMessage | None = None


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
        self._normal_messages: deque[ServerMessage] = deque()
        # Role queues: per role min-heap of (sort_ts, seq, entry)
        # Both binary and JSON messages for a role go through the same heap.
        self._role_queues: dict[str, list[tuple[int, int, _RoleQueueEntry]]] = defaultdict(list)
        self._max_pending_msg_by_role: defaultdict[str, int] = defaultdict(lambda: MAX_PENDING_MSG)
        # Last timestamp per role for JSON inheritance (JSON gets previous message's timestamp)
        self._last_enqueued_ts_by_role: dict[str, int] = {}
        # Global scheduler heaps for families
        self._ready_roles: list[tuple[int, int, str]] = []
        self._delayed_roles: list[tuple[int, int, str]] = []
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
        self._epoch_by_role: defaultdict[str, int] = defaultdict(int)

        # Timing tracking for binary frame logging (per role)
        self._last_send_time_us_by_role: dict[str, int] = {}
        self._last_timestamp_us_by_role: dict[str, int] = {}
        self._send_stats_by_role: dict[str, dict[str, float | int]] = {}
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

    @property
    def should_retry_server_initiated_connection(self) -> bool:
        """Whether the server should reconnect this URL after disconnect."""
        return not self._closing and self._last_goodbye_reason != GoodbyeReason.ANOTHER_SERVER

    @property
    def goodbye_reason(self) -> GoodbyeReason | None:
        """Disconnect reason reported by client/goodbye, if available."""
        return self._last_goodbye_reason

    def requires_initial_state(self) -> bool:
        """Whether this connection must receive initial client/state before being 'connected'."""
        if self._client is None:
            return False
        return any(role.requires_initial_state() for role in self._client.active_roles)

    def drop_pending_binary(self, roles: list[str] | None) -> None:
        """Drop queued binary payloads for the specified roles.

        Uses epoch-based lazy invalidation: increments the epoch counter for each role,
        causing the writer loop to discard binary entries with the old epoch.
        JSON entries in the same queue are NOT affected (they skip epoch validation).
        """
        roles_to_drop = list(self._epoch_by_role.keys()) if roles is None else roles
        for role in roles_to_drop:
            self._epoch_by_role[role] += 1
        self._writer_wakeup.set()

    def send_binary(
        self,
        data: bytes,
        *,
        role: str,
        timestamp_us: int,
        message_type: int,
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
    ) -> None:
        """Enqueue a binary message.

        Args:
            data: Binary data to send.
            role: Role for epoch tracking and queue routing.
            timestamp_us: Playback timestamp from binary header (cached to avoid unpacking).
            message_type: Binary message type for role lookup (cached).
            buffer_end_time_us: End timestamp for buffer tracking.
            buffer_byte_count: Byte count for buffer tracking.
            duration_us: Duration for buffer tracking.
        """
        if self._is_role_queue_full(role):
            self._disconnect_due_to_queue_overflow(
                f"Role queue full for {role} ({len(self._role_queues.get(role, []))}/"
                f"{self._max_pending_msg_by_role[role]}), client too slow"
            )
            return

        # Keep per-role queue ordering monotonic so role-scoped lifecycle JSON
        # (stream/start, stream/end, stream/clear) cannot be overtaken by binary
        # packets that carry an older playback timestamp (e.g. historical backfill).
        sort_ts = max(0, timestamp_us, self._last_enqueued_ts_by_role.get(role, 0))
        entry = _RoleQueueEntry(
            epoch=self._epoch_by_role[role],
            timestamp_us=timestamp_us,
            binary=_BinaryData(
                data=data,
                message_type=message_type,
                buffer_end_time_us=buffer_end_time_us,
                buffer_byte_count=buffer_byte_count,
                duration_us=duration_us,
            ),
        )
        self._last_enqueued_ts_by_role[role] = sort_ts
        self._enqueue_role_entry(role, sort_ts, entry)

    def queue_status(self) -> tuple[int, int]:
        """Return (qsize, maxsize) for the outgoing queue."""
        maxsize = MAX_PENDING_MSG + (len(self._role_queues) * MAX_PENDING_MSG)
        return self._queue_size, maxsize

    def _disconnect_due_to_queue_overflow(self, message: str) -> None:
        if self._disconnecting:
            return
        self._logger.error("%s - disconnecting", message)
        create_task(self.disconnect(retry_connection=True))

    def _is_role_queue_full(self, role: str) -> bool:
        return len(self._role_queues.get(role, [])) >= self._max_pending_msg_by_role[role]

    def _enqueue_role_entry(self, role: str, sort_ts: int, entry: _RoleQueueEntry) -> None:
        """Push an entry into a role's heap and schedule it if it becomes the new head."""
        seq = self._queue_sequence
        self._queue_sequence += 1
        role_queue = self._role_queues[role]
        heapq.heappush(role_queue, (sort_ts, seq, entry))
        self._queue_size += 1

        if role not in self._blocked_until_us:
            head_sort_ts, head_seq, _ = role_queue[0]
            if head_sort_ts == sort_ts and head_seq == seq:
                heapq.heappush(self._ready_roles, (head_sort_ts, head_seq, role))

        self._writer_wakeup.set()

    def send_role_message(self, role: str, message: ServerMessage) -> None:
        """Enqueue a JSON message into a role's queue with inherited timestamp.

        The message inherits the timestamp of the last message enqueued for this role,
        so it maintains its position relative to surrounding timed binary. If no previous
        message exists, it uses timestamp 0 (sent before any timed binary).

        Exception: StreamEnd and StreamStart use current time instead of inheriting,
        ensuring they are ordered correctly across stream boundaries.
        """
        if isinstance(message, StreamClearMessage | StreamEndMessage):
            self.drop_pending_binary(message.payload.roles)

        if self._is_role_queue_full(role):
            self._disconnect_due_to_queue_overflow(
                f"Role queue full for {role} ({len(self._role_queues.get(role, []))}/"
                f"{self._max_pending_msg_by_role[role]}), client too slow"
            )
            return

        # Stream lifecycle messages use current time to ensure correct ordering
        # across stream boundaries (prevents old stream timestamps from affecting new stream)
        if isinstance(message, StreamEndMessage | StreamStartMessage):
            sort_ts = self._server.clock.now_us()
            # Update tracker so subsequent messages inherit this timestamp
            self._last_enqueued_ts_by_role[role] = sort_ts
        else:
            sort_ts = self._last_enqueued_ts_by_role.get(role, 0)

        entry = _RoleQueueEntry(
            epoch=self._epoch_by_role[role],
            timestamp_us=sort_ts,
            json_message=message,
        )
        self._enqueue_role_entry(role, sort_ts, entry)

        if not isinstance(message, ServerTimeMessage):
            self._logger.debug("Enqueueing role message: %s", type(message).__name__)

    def send_message(self, message: ServerMessage) -> None:
        """Enqueue a non-role JSON message (sent in FIFO order, not tied to any role)."""
        if isinstance(message, StreamClearMessage | StreamEndMessage):
            self.drop_pending_binary(message.payload.roles)

        if self._queue_size >= MAX_PENDING_MSG:
            self._disconnect_due_to_queue_overflow("Control message queue full, client too slow")
            return

        self._normal_messages.append(message)
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
        return existing.merge(incoming)

    def send_priority_message(self, message: ServerMessage) -> None:
        """Enqueue a high-priority message (processed before regular queue)."""
        if self._queue_size >= MAX_PENDING_MSG:
            self._disconnect_due_to_queue_overflow("Priority message queue full, client too slow")
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
            # Only detach if this connection is still the active one.
            if self._client.connection is self:
                self._client.detach_connection(self._last_goodbye_reason)
            self._client = None

        self._logger.debug("Connection disconnected")

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
            self._server.on_client_first_connect(self._client.client_id)

    @staticmethod
    def _first_registered_role_id_in_family(
        supported_roles: list[str], *, family: str
    ) -> str | None:
        """Return first client-preferred, server-registered role id in a role family."""
        for role_id in supported_roles:
            if role_family(role_id) == family and role_id in ROLE_FACTORIES:
                return role_id
        return None

    @classmethod
    def _primary_role_id_for_family(cls, family: str) -> str | None:
        """Return built-in primary role id for a role family, if defined."""
        for role in Roles:
            if role_family(role.value) == family:
                return role.value
        return None

    @classmethod
    def _extract_custom_role_supports(cls, message: dict[str, Any]) -> dict[str, tuple[str, Any]]:
        """Extract custom support objects from raw client/hello JSON without mutating payload."""
        payload = message.get("payload")
        if not isinstance(payload, dict):
            return {}
        supported_roles = payload.get("supported_roles")
        if not (
            isinstance(supported_roles, list) and all(isinstance(v, str) for v in supported_roles)
        ):
            return {}

        custom_supports: dict[str, tuple[str, Any]] = {}
        for family in ROLE_SUPPORT_SPECS:
            selected_role = cls._first_registered_role_id_in_family(supported_roles, family=family)
            if selected_role is None:
                # No registered role for this family — check if client advertises
                # any custom role in the family so we still parse its support key.
                selected_role = next((r for r in supported_roles if role_family(r) == family), None)
            if selected_role is None:
                continue
            primary_role_id = cls._primary_role_id_for_family(family)
            if selected_role == primary_role_id:
                continue

            custom_support_key = f"{selected_role}_support"
            custom_support = payload.get(custom_support_key)
            primary_support_key = (
                f"{primary_role_id}_support" if primary_role_id is not None else None
            )
            legacy_support_key = f"{family}_support"
            if custom_support is None and (
                (primary_support_key is not None and payload.get(primary_support_key) is not None)
                or payload.get(legacy_support_key) is not None
            ):
                # If built-in/legacy support keys are present for custom role IDs, keep
                # legacy behavior but make it explicit that custom roles must use their
                # versioned support key.
                logger.warning(
                    "Ignoring %s/%s for custom role %s; expected %s",
                    primary_support_key,
                    legacy_support_key,
                    selected_role,
                    custom_support_key,
                )
            custom_supports[family] = (selected_role, custom_support)
        return custom_supports

    @classmethod
    def _apply_custom_role_support(
        cls, hello: ClientHelloPayload, custom_supports: dict[str, tuple[str, Any]]
    ) -> None:
        """Apply parsed custom role support objects onto ClientHelloPayload fields."""
        for family, spec in ROLE_SUPPORT_SPECS.items():
            custom = custom_supports.get(family)
            if custom is None:
                continue
            custom_role, raw_support = custom
            if raw_support is None:
                raise ValueError(
                    f"{custom_role}_support must be provided when "
                    f"'{custom_role}' is in supported_roles"
                )
            if not isinstance(raw_support, dict):
                raise TypeError(
                    f"{custom_role}_support must be an object for role family '{family}'"
                )
            setattr(hello, f"{family}_support", spec.parse_support(raw_support))

    @classmethod
    def _deserialize_client_message(cls, raw_message: str) -> ClientMessage:
        """Deserialize inbound client message with custom support-key normalization."""
        parsed = ClientMessage.from_json(raw_message)
        if isinstance(parsed, ClientHelloMessage):
            decoded = orjson.loads(raw_message)
            if not isinstance(decoded, dict):
                return parsed
            custom_supports = cls._extract_custom_role_supports(decoded)
            if isinstance(parsed, ClientHelloMessage):
                cls._apply_custom_role_support(parsed.payload, custom_supports)
            return parsed
        return parsed

    async def _setup_connection(self) -> None:
        """Prepare a server-side WebSocketResponse, if applicable."""
        if self._wsock_server is not None:
            assert self._request is not None
            async with asyncio.timeout(10):
                await self._wsock_server.prepare(self._request)

        # Start writer task for both client-initiated and server-initiated connections.
        self._logger.debug("Connection established")
        self._writer_task = create_task(self._writer())

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
                    close_code = wsock.close_code
                    log_func = (
                        self._logger.debug if close_code in (1000, 1001) else self._logger.warning
                    )
                    log_func(
                        "WebSocket closed: type=%s close_code=%s",
                        msg.type.name,
                        close_code,
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
                    self._deserialize_client_message(cast("str", msg.data)), timestamp_us
                )
            else:
                # Loop exited normally (iterator exhausted) - connection closed
                close_code = wsock.close_code
                log_func = (
                    self._logger.debug if close_code in (1000, 1001) else self._logger.warning
                )
                log_func(
                    "WebSocket closed, close_code=%s",
                    close_code,
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
            self._logger.debug("Received client/hello: %s", client_info)

            # Look up connection reason for server-initiated connections
            connection_reason = (
                self._server.get_connection_reason(self._url)
                if self._url is not None
                else ConnectionReason.DISCOVERY
            )

            self.send_priority_message(
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

            client = self._server.get_or_create_client(self._client_id)
            client.attach_connection(self, client_info=client_info, active_roles=self._active_roles)
            self._client = client

            # Register client_id → URL mapping for server-initiated connections
            if self._url is not None:
                self._server.register_client_url(client_info.client_id, self._url)

            if self.requires_initial_state():
                self._initial_state_timeout_handle = self._server.loop.call_later(
                    5.0, self._initial_state_timeout_callback
                )
            else:
                client.mark_connected()
                self._server.on_client_first_connect(client.client_id)
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
                self._server.on_client_first_connect(self._client.client_id)

            new_state = payload.state
            if new_state is not None and new_state != self._client.client_state:
                await self._client.handle_state_transition(new_state)
            for role in self._client.active_roles:
                role.on_client_state(payload)
            return

        if isinstance(message, StreamRequestFormatMessage):
            if self._client is None:
                return
            for role in self._client.active_roles:
                role.on_stream_request_format(message.payload)
            return

        if isinstance(message, ClientCommandMessage):
            if self._client is None:
                return
            for role in self._client.active_roles:
                role.on_command(message.payload)
            return

        if isinstance(message, ClientGoodbyeMessage):
            self._logger.debug(
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
        """Check if a binary message's playback time has passed and should be dropped.

        Compares the message's playback timestamp against the current clock. During the
        grace period (configurable per-role), late messages are allowed through to give
        clients time to build their initial buffer.
        """
        # timestamp_us=0 means "no playback semantics" - skip late detection
        if handling is None or role is None or not handling.drop_late or timestamp_us == 0:
            return False

        now = self._server.clock.now_us()
        if role._stream_start_time_us is None:  # noqa: SLF001
            role._stream_start_time_us = now  # noqa: SLF001
        elapsed = now - role._stream_start_time_us  # noqa: SLF001
        in_grace_period = elapsed < handling.grace_period_us
        late_by_us = now - (timestamp_us - role.get_static_delay_us())

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

    async def _send_binary_data(
        self,
        wsock: web.WebSocketResponse | ClientWebSocketResponse,
        role: str,
        entry: _RoleQueueEntry,
        buffer_tracker: BufferTracker | None,
    ) -> None:
        """Send a binary frame with buffer tracking."""
        assert entry.binary is not None
        binary = entry.binary
        start_s = time.monotonic()
        await wsock.send_bytes(binary.data)
        elapsed_ms = (time.monotonic() - start_s) * 1000
        if elapsed_ms >= 50.0:
            # Slow writes indicate transport/backpressure issues but are not fatal.
            self._logger.debug(
                "Slow send_bytes: %.1fms size=%s ts_us=%s role=%s",
                elapsed_ms,
                len(binary.data),
                entry.timestamp_us,
                role,
            )

        # Buffer tracking via role's tracker (framework-managed)
        if (
            buffer_tracker is not None
            and binary.buffer_end_time_us is not None
            and binary.buffer_byte_count is not None
        ):
            buffer_tracker.register(
                binary.buffer_end_time_us,
                binary.buffer_byte_count,
                binary.duration_us or 0,
            )

    #### Role Queue Heap Management ####
    #
    # Two-level heap: per-role min-heaps hold entries sorted by (timestamp, seq).
    # A global _ready_roles heap tracks which role has the earliest head entry.
    # _delayed_roles tracks roles blocked by backpressure until a future time;
    # _promote_ready_roles moves them back to _ready_roles when their time comes.
    # Generation counters prevent stale delayed entries from unblocking a re-blocked role.

    def _schedule_role_head(self, role: str) -> None:
        if role in self._blocked_until_us:
            return
        if role_queue := self._role_queues.get(role):
            head_sort_ts, head_seq, _ = role_queue[0]
            heapq.heappush(self._ready_roles, (head_sort_ts, head_seq, role))

    def _discard_role_head(self, role: str) -> None:
        role_queue = self._role_queues.get(role)
        if not role_queue:
            return
        heapq.heappop(role_queue)
        self._queue_size = max(self._queue_size - 1, 0)
        if not role_queue:
            self._role_queues.pop(role, None)

    def _peek_ready_entry(self) -> tuple[str, _RoleQueueEntry, int, int] | None:
        # TODO: any reason why a peek method does a full pop and push operation?
        # TODO: or is it most of the time not pushing back? i mean does this peek
        # TODO: mutate anything or not?
        while self._ready_roles:
            sort_ts, seq, role = heapq.heappop(self._ready_roles)
            if role in self._blocked_until_us:
                continue
            role_queue = self._role_queues.get(role)
            if not role_queue:
                continue
            head_sort_ts, head_seq, head_entry = role_queue[0]
            if head_sort_ts != sort_ts or head_seq != seq:
                heapq.heappush(self._ready_roles, (head_sort_ts, head_seq, role))
                continue
            return role, head_entry, head_sort_ts, head_seq
        return None

    def _block_role(self, role: str, ready_at_us: int) -> None:
        self._blocked_until_us[role] = ready_at_us
        generation = self._block_generation[role] + 1
        self._block_generation[role] = generation
        heapq.heappush(self._delayed_roles, (ready_at_us, generation, role))

    def _promote_ready_roles(self, now_us: int) -> None:
        while self._delayed_roles and self._delayed_roles[0][0] <= now_us:
            ready_at_us, generation, role = heapq.heappop(self._delayed_roles)
            if self._block_generation.get(role, 0) != generation:
                continue
            blocked_until = self._blocked_until_us.get(role)
            if blocked_until is None or blocked_until != ready_at_us:
                continue
            self._blocked_until_us.pop(role, None)
            self._schedule_role_head(role)

    async def _process_priority_messages(
        self,
        wsock: web.WebSocketResponse | ClientWebSocketResponse,
    ) -> bool:
        """Send one queued priority message if available."""
        if not self._priority_messages:
            return False
        message = self._priority_messages.popleft()
        self._queue_size = max(self._queue_size - 1, 0)
        await self._send_message(wsock, message)
        return True

    async def _process_normal_messages(
        self,
        wsock: web.WebSocketResponse | ClientWebSocketResponse,
        ready_entry: tuple[str, _RoleQueueEntry, int, int] | None,
    ) -> bool:
        """Send one queued non-role message when no role entry is ready."""
        if ready_entry is not None or not self._normal_messages:
            return False
        message = self._normal_messages.popleft()
        self._queue_size = max(self._queue_size - 1, 0)
        await self._send_message(wsock, message)
        return True

    def _fresh_send_stats(self) -> dict[str, float | int]:
        return {
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
        }

    def _update_send_stats(
        self,
        role: str,
        *,
        send_gap_ms: float,
        ts_gap_ms: float,
        buffer_tracker: BufferTracker | None,
        now_us: int,
    ) -> None:
        stats = self._send_stats_by_role.setdefault(role, self._fresh_send_stats())
        stats["count"] += 1
        stats["send_gap_sum_ms"] += send_gap_ms
        stats["send_gap_min_ms"] = min(stats["send_gap_min_ms"], send_gap_ms)
        stats["send_gap_max_ms"] = max(stats["send_gap_max_ms"], send_gap_ms)
        stats["ts_gap_sum_ms"] += ts_gap_ms
        stats["ts_gap_min_ms"] = min(stats["ts_gap_min_ms"], ts_gap_ms)
        stats["ts_gap_max_ms"] = max(stats["ts_gap_max_ms"], ts_gap_ms)
        if buffer_tracker is not None:
            buf_ms = buffer_tracker.buffered_horizon_us(now_us) / 1000
            stats["buf_count"] += 1
            stats["buf_sum_ms"] += buf_ms
            stats["buf_min_ms"] = min(stats["buf_min_ms"], buf_ms)
            stats["buf_max_ms"] = max(stats["buf_max_ms"], buf_ms)

    def _log_send_summaries_if_due(self) -> None:
        if not self._logger.isEnabledFor(logging.DEBUG):
            return
        now_s = time.monotonic()
        if now_s - self._send_summary_last_log_s < 5.0:
            return
        self._send_summary_last_log_s = now_s
        for role_name, role_stats in self._send_stats_by_role.items():
            count = int(role_stats["count"])
            if count <= 0:
                continue
            avg_send = role_stats["send_gap_sum_ms"] / count
            avg_ts = role_stats["ts_gap_sum_ms"] / count
            if role_stats["buf_count"] > 0:
                avg_buf = role_stats["buf_sum_ms"] / role_stats["buf_count"]
                self._logger.debug(
                    "Send summary role=%s samples=%s "
                    "send_gap_ms(avg=%.1f min=%.1f max=%.1f) "
                    "ts_gap_ms(avg=%.1f min=%.1f max=%.1f) "
                    "buf_ms(avg=%.1f min=%.1f max=%.1f)",
                    role_name,
                    count,
                    avg_send,
                    role_stats["send_gap_min_ms"],
                    role_stats["send_gap_max_ms"],
                    avg_ts,
                    role_stats["ts_gap_min_ms"],
                    role_stats["ts_gap_max_ms"],
                    avg_buf,
                    role_stats["buf_min_ms"],
                    role_stats["buf_max_ms"],
                )
            else:
                self._logger.debug(
                    "Send summary role=%s samples=%s "
                    "send_gap_ms(avg=%.1f min=%.1f max=%.1f) "
                    "ts_gap_ms(avg=%.1f min=%.1f max=%.1f)",
                    role_name,
                    count,
                    avg_send,
                    role_stats["send_gap_min_ms"],
                    role_stats["send_gap_max_ms"],
                    avg_ts,
                    role_stats["ts_gap_min_ms"],
                    role_stats["ts_gap_max_ms"],
                )
            self._send_stats_by_role[role_name] = self._fresh_send_stats()

    async def _process_binary_role_messages(
        self,
        wsock: web.WebSocketResponse | ClientWebSocketResponse,
        role: str,
        entry: _RoleQueueEntry,
        now_us: int,
    ) -> tuple[bool, int]:
        assert entry.binary is not None

        # Look up handling info for late detection + buffer tracking
        cached = None
        if self._client is not None:
            cached = self._client.get_binary_handling_cached(entry.binary.message_type)
        handling = cached[0] if cached else None
        handling_role = cached[1] if cached else None

        # Drop late messages if role requests it
        if (
            handling is not None
            and handling_role is not None
            and self._check_late_binary(handling, handling_role, entry.timestamp_us)
        ):
            self._discard_role_head(role)
            self._schedule_role_head(role)
            return False, now_us

        # Check backpressure from buffer tracker
        wait_us = 0
        buffer_tracker = None
        if handling is not None and handling_role is not None:
            if handling.buffer_track:
                buffer_tracker = handling_role.get_buffer_tracker()
            if buffer_tracker is not None:
                buffer_tracker.prune_consumed(now_us)
                wait_us = max(wait_us, buffer_tracker.time_until_unblocked())
                bytes_needed = entry.binary.buffer_byte_count or 0
                duration_needed_us = entry.binary.duration_us or 0
                wait_us = max(
                    wait_us,
                    buffer_tracker.time_until_ready(
                        bytes_needed,
                        duration_needed_us,
                        end_time_us=entry.binary.buffer_end_time_us,
                    ),
                )

        if wait_us > 0:
            # Block this role until buffer has space
            self._block_role(role, now_us + wait_us)
            return False, now_us

        debug_enabled = self._logger.isEnabledFor(logging.DEBUG)
        last_send_us: int | None = None
        last_ts_us: int | None = None
        send_gap_ms = 0.0
        ts_gap_ms = 0.0
        if debug_enabled:
            timestamp_us = entry.timestamp_us
            last_send_us = self._last_send_time_us_by_role.get(role)
            last_ts_us = self._last_timestamp_us_by_role.get(role)
            send_gap_ms = (now_us - last_send_us) / 1000 if last_send_us is not None else 0
            ts_gap_ms = (timestamp_us - last_ts_us) / 1000 if last_ts_us is not None else 0
            self._last_send_time_us_by_role[role] = now_us
            self._last_timestamp_us_by_role[role] = timestamp_us

        self._discard_role_head(role)
        await self._send_binary_data(wsock, role, entry, buffer_tracker)

        if debug_enabled and last_send_us is not None and last_ts_us is not None:
            self._update_send_stats(
                role,
                send_gap_ms=send_gap_ms,
                ts_gap_ms=ts_gap_ms,
                buffer_tracker=buffer_tracker,
                now_us=now_us,
            )
        if debug_enabled:
            self._log_send_summaries_if_due()
        self._schedule_role_head(role)
        return True, self._server.clock.now_us()

    async def _process_role_messages(
        self,
        wsock: web.WebSocketResponse | ClientWebSocketResponse,
        ready_entry: tuple[str, _RoleQueueEntry, int, int],
        now_us: int,
    ) -> tuple[bool, int]:
        """Process one ready role entry."""
        role, entry, _sort_ts, _seq = ready_entry

        # Binary entries with a stale epoch are discarded (stream was cleared/ended).
        # JSON entries skip this check - they are always delivered.
        if entry.binary is not None and entry.epoch != self._epoch_by_role[role]:
            self._discard_role_head(role)
            self._schedule_role_head(role)
            return False, now_us

        if entry.json_message is not None:
            self._discard_role_head(role)
            # Merge consecutive state-like messages at send time.
            message = entry.json_message
            while True:
                role_queue = self._role_queues.get(role)
                if not role_queue:
                    break
                _, _, next_entry = role_queue[0]
                if next_entry.json_message is None:
                    break
                merged = self._merge_state_messages(message, next_entry.json_message)
                if merged is None:
                    break
                message = merged
                self._discard_role_head(role)
            await self._send_message(wsock, message)
            self._schedule_role_head(role)
            return True, self._server.clock.now_us()

        return await self._process_binary_role_messages(wsock, role, entry, now_us)

    async def _wait_for_writer_work(self, now_us: int) -> None:
        """Sleep until new work arrives or next delayed role becomes ready."""
        self._writer_wakeup.clear()
        if self._priority_messages or self._normal_messages or self._ready_roles:
            return

        sleep_s = None
        if self._delayed_roles:
            next_ready_us = self._delayed_roles[0][0]
            sleep_s = max((next_ready_us - now_us) / 1_000_000, 0.0)

        try:
            if sleep_s is None:
                await self._writer_wakeup.wait()
            else:
                await asyncio.wait_for(self._writer_wakeup.wait(), timeout=sleep_s)
        except TimeoutError:
            pass

    async def _writer(self) -> None:
        wsock = self._wsock_server or self._wsock_client
        assert wsock is not None

        clock_now_us = self._server.clock.now_us

        iterations_since_yield = 0
        now_us = clock_now_us()

        try:
            while not wsock.closed and not self._closing:
                # Periodic yield to prevent event loop starvation
                if iterations_since_yield >= 50:
                    await asyncio.sleep(0)
                    iterations_since_yield = 0
                    now_us = clock_now_us()

                if await self._process_priority_messages(wsock):
                    now_us = clock_now_us()
                    iterations_since_yield = 0
                    continue

                now_us = clock_now_us()
                self._promote_ready_roles(now_us)

                ready_entry = self._peek_ready_entry()
                has_normal = bool(self._normal_messages)

                if ready_entry is None and not has_normal:
                    await self._wait_for_writer_work(now_us)
                    continue

                if await self._process_normal_messages(wsock, ready_entry):
                    now_us = clock_now_us()
                    iterations_since_yield = 0
                    continue

                assert ready_entry is not None
                sent, now_us = await self._process_role_messages(wsock, ready_entry, now_us)
                if sent:
                    iterations_since_yield = 0
                    continue
                iterations_since_yield += 1
        except asyncio.CancelledError:
            self._logger.debug("Writer cancelled")
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
            self._message_loop_task = create_task(self._run_message_loop())
            await self._message_loop_task
        finally:
            await self._cleanup_connection()
