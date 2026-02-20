"""Persistent Sendspin client (device) state.

SendspinClient represents a client device across reconnects. It may have an active
WebSocket connection (SendspinConnection) or be disconnected while still retaining
its identity, group membership, and per-role persistent state (e.g. BufferTracker).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

from aiosendspin.models.core import (
    ClientHelloPayload,
    StreamStartMessage,
)
from aiosendspin.models.types import (
    BinaryMessageType,
    ClientStateType,
    GoodbyeReason,
    Roles,
    has_role,
    negotiate_active_roles,
)
from aiosendspin.util import create_task

from .events import ClientEvent, ClientGroupChangedEvent
from .roles import Role
from .roles.base import BinaryHandling
from .roles.registry import create_role

if TYPE_CHECKING:
    from aiosendspin.models.types import ServerMessage

    from .connection import SendspinConnection
    from .group import SendspinGroup
    from .server import SendspinServer


logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Cleanup delay for reconnect-friendly disconnect reasons (seconds)
CLIENT_CLEANUP_DELAY = 30.0

# Reasons that trigger immediate client cleanup from the registry.
# Note: ANOTHER_SERVER is intentionally excluded and never auto-cleaned up.
IMMEDIATE_CLEANUP_REASONS: frozenset[GoodbyeReason] = frozenset(
    {
        GoodbyeReason.SHUTDOWN,
        GoodbyeReason.USER_REQUEST,
    }
)


class DisconnectBehaviour(Enum):
    """Enum for disconnect behaviour options."""

    UNGROUP = "ungroup"
    """
    The client will ungroup itself from its current group when it gets disconnected.

    Playback will continue on the remaining group members.
    """

    STOP = "stop"
    """
    The client will stop playback of the whole group when disconnecting.
    """


class SendspinClient:
    """Persistent client/device object."""

    def __init__(self, server: SendspinServer, client_id: str) -> None:
        """Create a new persistent client/device object."""
        self._server = server
        self._client_id = client_id
        self._name = client_id
        self._info: ClientHelloPayload | None = None
        self._negotiated_roles: list[str] = []
        self._roles: dict[str, Role] = {}
        self._group: SendspinGroup | None = None

        self._connection: SendspinConnection | None = None
        self._connected: bool = False
        self._roles_warm_disconnected: bool = False

        self.disconnect_behaviour = DisconnectBehaviour.UNGROUP

        self._event_cbs: list[Callable[[SendspinClient, ClientEvent], None]] = []
        self._logger = logger.getChild(client_id)

        # Client-level state (reported by client/state). Persists across reconnects until updated.
        self._client_state: ClientStateType = ClientStateType.SYNCHRONIZED

        # Role-owned persistent state (per role family).
        self._role_state: dict[str, object] = {}

        # Cache for binary handling lookup: message_type -> (BinaryHandling, Role)
        # Built when roles are attached for cached lookup in _send_binary_frame().
        self._binary_handling_cache: dict[int, tuple[BinaryHandling, Role]] = {}

        # Pending cleanup handle (scheduled via loop.call_soon/call_later on disconnect)
        self._cleanup_handle: asyncio.Handle | None = None
        # True when we intentionally retain a disconnected client after ANOTHER_SERVER goodbye.
        self._cleanup_on_mdns_removal: bool = False

    @property
    def client_id(self) -> str:
        """Return the stable unique identifier for this device."""
        return self._client_id

    @property
    def name(self) -> str:
        """Return the human-readable device name."""
        return self._name

    @property
    def info(self) -> ClientHelloPayload:
        """Return the most recent `client/hello` payload."""
        assert self._info is not None, "client/hello has not been processed yet"
        return self._info

    @property
    def negotiated_roles(self) -> list[str]:
        """Return the negotiated active roles for this connection (versioned role IDs)."""
        return self._negotiated_roles

    def role(self, role_id: str) -> Role | None:
        """Get active role by versioned ID (e.g., 'player@v1')."""
        return self._roles.get(role_id)

    def roles_by_family(self, family: str) -> list[Role]:
        """Return all active roles for a role family."""
        return [role for role in self._roles.values() if role.role_family == family]

    def get_role_state(self, family: str, cls: type[_T]) -> _T | None:
        """Return persistent role state for a family, or None if unset."""
        state = self._role_state.get(family)
        if state is None:
            return None
        if not isinstance(state, cls):
            raise TypeError(
                f"Role state for {family} is {type(state).__name__}, expected {cls.__name__}"
            )
        return state

    def set_role_state(self, family: str, state: object) -> None:
        """Store persistent role state for a family."""
        self._role_state[family] = state

    def get_or_create_role_state(self, family: str, cls: type[_T]) -> _T:
        """Return persistent role state, creating a default if missing."""
        existing = self.get_role_state(family, cls)
        if existing is not None:
            return existing
        created = cls()
        self._role_state[family] = created
        return created

    @property
    def group(self) -> SendspinGroup:
        """Return the current group this client belongs to."""
        assert self._group is not None, "client group has not been initialized"
        return self._group

    @property
    def is_connected(self) -> bool:
        """Return True if this device currently has an active WebSocket connection."""
        return self._connected and self._connection is not None

    @property
    def connection(self) -> SendspinConnection | None:
        """Return the active connection for this device, if connected."""
        return self._connection

    @property
    def has_warm_disconnected_roles(self) -> bool:
        """Whether role instances are retained while transport is down."""
        return self._roles_warm_disconnected and self._connection is None

    @property
    def cleanup_on_mdns_removal(self) -> bool:
        """Whether this retained client should be removed when mDNS record disappears."""
        return self._cleanup_on_mdns_removal

    @property
    def client_state(self) -> ClientStateType:
        """Return the current client operational state reported by `client/state`."""
        return self._client_state

    async def handle_state_transition(self, new_state: ClientStateType) -> None:
        """Handle client state transitions by notifying all roles."""
        old_state = self._client_state
        self._client_state = new_state

        for role in self._roles.values():
            coro = role.on_state_transition(old_state, new_state)
            if coro is not None:
                await coro

    def check_role(self, role: Roles) -> bool:
        """Check if the client has a role active (by role family)."""
        return has_role(role.value, self._negotiated_roles)

    def attach_connection(
        self,
        connection: SendspinConnection,
        *,
        client_info: ClientHelloPayload,
        active_roles: list[str],
    ) -> None:
        """Attach a new WebSocket connection to this client."""
        # Cancel pending cleanup if client reconnected before cleanup fired
        if self._cleanup_handle is not None:
            self._logger.debug("Cancelling pending cleanup due to reconnect")
            self._cleanup_handle.cancel()
            self._cleanup_handle = None

        if self._connection is not None and self._connection is not connection:
            # Replace an existing connection for the same device.
            self._logger.debug("Replacing existing connection for %s", self._client_id)
            create_task(self._connection.disconnect(retry_connection=False))

        self._connection = connection
        self._connected = False  # set True once initial state is received (spec)
        self._cleanup_on_mdns_removal = False
        on_transport_attached = getattr(self._server, "on_client_transport_attached", None)
        if callable(on_transport_attached):
            on_transport_attached(self._client_id)

        self._info = client_info
        self._name = client_info.name
        previous_roles = list(self._negotiated_roles)
        self._negotiated_roles = active_roles
        self._logger = logger.getChild(self._client_id)

        # Reuse warm-disconnected roles when the negotiated role IDs are unchanged.
        can_reuse_warm_roles = (
            self._roles_warm_disconnected
            and bool(self._roles)
            and set(previous_roles) == set(self._negotiated_roles)
            and set(self._roles.keys()) == set(self._negotiated_roles)
        )
        if can_reuse_warm_roles:
            for role in self._roles.values():
                role.on_connect()
        else:
            if self._roles_warm_disconnected:
                # Roles already received on_disconnect() during detach_connection();
                # just clear the mappings to avoid invoking disconnect hooks twice.
                self._roles.clear()
                self._binary_handling_cache.clear()
            else:
                self._hard_detach_roles()

            # Create and register active roles via registry.
            for role_id in self._negotiated_roles:
                new_role = create_role(role_id, self)
                if new_role is None:
                    continue
                new_role.on_connect()
                self._roles[new_role.role_id] = new_role

        self._roles_warm_disconnected = False

        # Build binary handling cache for fast lookup
        self._binary_handling_cache.clear()
        for msg_type in BinaryMessageType:
            for role in self._roles.values():
                handling = role.get_binary_handling(msg_type.value)
                if handling is not None:
                    self._binary_handling_cache[msg_type.value] = (handling, role)
                    break  # First role that handles it wins

        # Ensure group exists (server creates it on first sight).
        if self._group is None:
            raise RuntimeError("SendspinClient.group must be initialized by the server")

    def preload_hello(self, client_info: ClientHelloPayload) -> None:
        """Seed persistent client identity/capabilities without an active connection."""
        self._info = client_info
        self._name = client_info.name
        self._negotiated_roles = negotiate_active_roles(client_info.supported_roles)

    def mark_connected(self) -> None:
        """Mark this client as fully connected (after initial client/state if required)."""
        if self._connection is None:
            return
        self._connected = True
        self.group.on_client_connected(self)

    def detach_connection(self, goodbye_reason: GoodbyeReason | None) -> None:
        """Detach the current connection and apply BufferTracker reset policy."""
        self._connected = False

        warm_disconnect = goodbye_reason in {None, GoodbyeReason.RESTART}
        if warm_disconnect:
            # Keep role instances alive for reconnect-aware processing, but run
            # role disconnect hooks so reconnect always preserves lifecycle order.
            for role in self._roles.values():
                role.on_disconnect()
            self._binary_handling_cache.clear()
            self._roles_warm_disconnected = True
        else:
            self._hard_detach_roles()
            self._roles_warm_disconnected = False

        self._connection = None

        if goodbye_reason == GoodbyeReason.ANOTHER_SERVER:
            create_task(self._handle_takeover_disconnect())

        # Schedule client cleanup from registry
        self._schedule_cleanup(goodbye_reason)

    async def _handle_takeover_disconnect(self) -> None:
        """Handle ANOTHER_SERVER disconnect by ungrouping first, then stopping."""
        old_group_id = self.group.group_id
        try:
            await self.ungroup()
            await self.group.stop()
        except Exception:
            self._logger.exception(
                "Takeover disconnect sequence failed for %s (old_group=%s)",
                self._client_id,
                old_group_id,
            )

    def _schedule_cleanup(self, goodbye_reason: GoodbyeReason | None) -> None:
        """Schedule cleanup from server registry based on disconnect reason."""
        if self._server.is_external_player(self._client_id):
            self._logger.debug("Skipping cleanup scheduling for external player")
            self._cleanup_on_mdns_removal = False
            return

        if goodbye_reason == GoodbyeReason.ANOTHER_SERVER:
            self._logger.debug("Skipping cleanup scheduling (reason: %s)", goodbye_reason)
            self._cleanup_on_mdns_removal = True
            return

        self._cleanup_on_mdns_removal = False
        if goodbye_reason in IMMEDIATE_CLEANUP_REASONS:
            # Immediate cleanup for explicit disconnect reasons
            self._logger.debug("Scheduling immediate cleanup (reason: %s)", goodbye_reason)
            self._cleanup_handle = self._server.loop.call_soon(self._do_cleanup)
        else:
            # Delayed cleanup for reconnect-friendly scenarios
            self._logger.debug(
                "Scheduling delayed cleanup in %.0fs (reason: %s)",
                CLIENT_CLEANUP_DELAY,
                goodbye_reason,
            )
            self._cleanup_handle = self._server.loop.call_later(
                CLIENT_CLEANUP_DELAY, self._do_cleanup
            )

    def _do_cleanup(self) -> None:
        """Remove this client from the server registry."""
        self._cleanup_handle = None
        if self._connected:
            # Client reconnected, don't clean up
            return
        self._hard_detach_roles()
        self._roles_warm_disconnected = False
        self._logger.debug("Cleaning up client from registry")
        create_task(self._server.remove_client(self._client_id))

    def _hard_detach_roles(self) -> None:
        """Run role disconnect hooks and clear role-related caches."""
        for role in self._roles.values():
            role.on_disconnect()
        self._roles.clear()
        self._binary_handling_cache.clear()

    # ---- Messaging (delegates to connection) ----

    def send_message(self, message: ServerMessage) -> None:
        """Send a message if connected; otherwise no-op."""
        if self._connection is None:
            return
        if isinstance(message, StreamStartMessage):
            self._logger.debug("Sending stream/start: %s", message.payload)
        self._connection.send_message(message)

    def send_role_message(self, role: str, message: ServerMessage) -> None:
        """Send a role-scoped message if connected; otherwise no-op."""
        if self._connection is None:
            return
        if isinstance(message, StreamStartMessage):
            self._logger.debug("Sending stream/start: %s", message.payload)
        self._connection.send_role_message(role, message)

    def send_binary(
        self,
        data: bytes,
        *,
        role_family: str,
        timestamp_us: int,
        message_type: int,
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
    ) -> None:
        """Enqueue a binary payload for this client, or no-op when disconnected."""
        if self._connection is None:
            return
        self._connection.send_binary(
            data,
            role=role_family,
            timestamp_us=timestamp_us,
            message_type=message_type,
            buffer_end_time_us=buffer_end_time_us,
            buffer_byte_count=buffer_byte_count,
            duration_us=duration_us,
        )

    @property
    def active_roles(self) -> list[Role]:
        """All active roles for iteration."""
        return list(self._roles.values())

    def get_binary_handling_cached(self, message_type: int) -> tuple[BinaryHandling, Role] | None:
        """Return cached binary handling for a message type."""
        return self._binary_handling_cache.get(message_type)

    # ---- Events + grouping ----

    def add_event_listener(
        self, callback: Callable[[SendspinClient, ClientEvent], None]
    ) -> Callable[[], None]:
        """Register a callback for client-scoped events.

        The second callback argument is ``ClientEvent`` to cover both client-core
        events and role-emitted client events.

        Returns an unsubscribe callable.
        """
        self._event_cbs.append(callback)

        def _remove() -> None:
            with suppress(ValueError):
                self._event_cbs.remove(callback)

        return _remove

    def _signal_event(self, event: ClientEvent) -> None:
        for cb in self._event_cbs:
            try:
                cb(self, event)
            except Exception:
                logger.exception("Error in event listener")

    def _set_group(self, group: SendspinGroup) -> None:
        """Set the group for this client. For internal use by SendspinGroup only."""
        if self._group is not None:
            self._group._unregister_client_events(self)  # noqa: SLF001
        self._group = group
        self._group._register_client_events(self)  # noqa: SLF001
        self._signal_event(ClientGroupChangedEvent(group))
        for role in self._roles.values():
            role.on_group_changed(group)

    async def ungroup(self) -> None:
        """Remove the client from the group (no-op if already solo)."""
        if len(self.group.clients) > 1:
            await self.group.remove_client(self)
