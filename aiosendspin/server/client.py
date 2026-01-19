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
from typing import TYPE_CHECKING

from aiosendspin.models.core import ClientHelloPayload
from aiosendspin.models.types import ClientStateType, Roles, has_role
from aiosendspin.server.audio import AudioFormat, BufferTracker
from aiosendspin.server.events import ClientEvent, ClientGroupChangedEvent

from .controller import ControllerClient
from .metadata import MetadataClient
from .player import PlayerClient
from .roles import PlayerRole
from .visualizer import VisualizerClient

if TYPE_CHECKING:
    from aiosendspin.models.types import GoodbyeReason, ServerMessage

    from .connection import SendspinConnection
    from .group import SendspinGroup
    from .server import SendspinServer


logger = logging.getLogger(__name__)


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
        self._roles: list[str] = []
        self._group: SendspinGroup | None = None

        self._connection: SendspinConnection | None = None
        self._connected: bool = False

        self.disconnect_behaviour = DisconnectBehaviour.UNGROUP

        self._event_cbs: list[Callable[[SendspinClient, ClientEvent], None]] = []
        self._logger = logger.getChild(client_id)

        # Client-level state (reported by client/state). Persists across reconnects until updated.
        self._client_state: ClientStateType = ClientStateType.SYNCHRONIZED

        # State used by controller "switch" semantics for external_source recovery.
        self._previous_group_id: str | None = None
        self._external_source_solo_group_id: str | None = None

        # Role helpers (persistent)
        self._player: PlayerClient | None = None
        self._controller: ControllerClient | None = None
        self._metadata_client: MetadataClient | None = None
        self._visualizer: VisualizerClient | None = None

        # Player role persistent state
        self._player_role: PlayerRole | None = None
        self._buffer_tracker: BufferTracker | None = None
        self._preferred_format: AudioFormat | None = None
        self._blocking: bool = True

        # Disconnect bookkeeping for delayed BufferTracker reset policy.
        self._disconnect_time_us: int | None = None
        self._buffer_reset_handle: asyncio.TimerHandle | None = None

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
    def roles(self) -> list[str]:
        """Return the negotiated active roles for this connection (versioned role IDs)."""
        return self._roles

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
    def client_state(self) -> ClientStateType:
        """Return the current client operational state reported by `client/state`."""
        return self._client_state

    async def handle_state_transition(self, new_state: ClientStateType) -> None:
        """
        Handle client state transitions.

        When transitioning to external_source:
        - If in multi-client group: remember previous group, move to solo group
        - If already in solo group: stop playback
        """
        old_state = self._client_state
        self._client_state = new_state

        self._logger.info(
            "Client state transition: %s -> %s",
            old_state.value,
            new_state.value,
        )

        if new_state != ClientStateType.EXTERNAL_SOURCE:
            return

        is_multi_client_group = len(self.group.clients) > 1

        if is_multi_client_group:
            self._previous_group_id = self.group.group_id
            self._logger.debug(
                "Storing previous group %s for external_source client",
                self._previous_group_id,
            )
            await self.group.remove_client(self)
            self._external_source_solo_group_id = self.group.group_id
            return

        self._logger.debug("Client already in solo group, stopping playback for external_source")
        await self.group.stop()

    def check_role(self, role: Roles) -> bool:
        """Check if the client has a role active (by role family)."""
        return has_role(role.value, self._roles)

    def attach_connection(
        self,
        connection: SendspinConnection,
        *,
        client_info: ClientHelloPayload,
        active_roles: list[str],
    ) -> None:
        """Attach a new WebSocket connection to this client."""
        if self._connection is not None and self._connection is not connection:
            # Replace an existing connection for the same device.
            self._logger.debug("Replacing existing connection for %s", self._client_id)
            task = self._server.loop.create_task(
                self._connection.disconnect(retry_connection=False)
            )
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

        # Cancel any pending delayed BufferTracker reset from a previous disconnect.
        if self._buffer_reset_handle is not None:
            self._buffer_reset_handle.cancel()
            self._buffer_reset_handle = None

        self._connection = connection
        self._connected = False  # set True once initial state is received (spec)
        self._disconnect_time_us = None

        self._info = client_info
        self._name = client_info.name
        self._roles = active_roles
        self._logger = logger.getChild(self._client_id)

        # Initialize role helpers based on negotiated active roles.
        self._player = PlayerClient(self) if has_role(Roles.PLAYER.value, self._roles) else None
        self._controller = (
            ControllerClient(self) if has_role(Roles.CONTROLLER.value, self._roles) else None
        )
        self._metadata_client = (
            MetadataClient(self) if has_role(Roles.METADATA.value, self._roles) else None
        )
        self._visualizer = (
            VisualizerClient(self) if has_role(Roles.VISUALIZER.value, self._roles) else None
        )

        # Player persistent state.
        if self._player is not None and self.info.player_support is not None:
            capacity = self.info.player_support.buffer_capacity
            if self._buffer_tracker is None:
                self._buffer_tracker = BufferTracker(
                    loop=self._server.loop,
                    client_id=self._client_id,
                    capacity_bytes=capacity,
                )
            else:
                self._buffer_tracker.capacity_bytes = capacity

            supported = self.info.player_support.supported_formats
            default_format = AudioFormat(
                codec=supported[0].codec,
                sample_rate=supported[0].sample_rate,
                bit_depth=supported[0].bit_depth,
                channels=supported[0].channels,
            )
            if self._preferred_format is None or not any(
                fmt.codec == self._preferred_format.codec
                and fmt.sample_rate == self._preferred_format.sample_rate
                and fmt.bit_depth == self._preferred_format.bit_depth
                and fmt.channels == self._preferred_format.channels
                for fmt in supported
            ):
                self._preferred_format = default_format

            self._player_role = PlayerRole(_client=self)
            self._player_role.on_connect()
        else:
            self._player_role = None

        # Ensure group exists (server creates it on first sight).
        if self._group is None:
            raise RuntimeError("SendspinClient.group must be initialized by the server")

    def mark_connected(self) -> None:
        """Mark this client as fully connected (after initial client/state if required)."""
        if self._connection is None:
            return
        self._connected = True
        self.group.on_client_connected(self)

    def detach_connection(self, goodbye_reason: GoodbyeReason | None) -> None:
        """Detach the current connection and apply BufferTracker reset policy."""
        self._connected = False

        if self._player_role is not None:
            self._player_role.on_disconnect()
            self._player_role = None

        self._connection = None
        self._disconnect_time_us = int(self._server.loop.time() * 1_000_000)

        if self._buffer_tracker is None:
            return

        # Policy:
        # - client/goodbye => reset immediately
        # - ungraceful disconnect => delayed reset to tolerate brief blips
        if goodbye_reason is not None:
            self._buffer_tracker.reset()
            return

        disconnect_time_us = self._disconnect_time_us

        def _maybe_reset() -> None:
            self._buffer_reset_handle = None
            if self._connection is not None:
                return
            if disconnect_time_us != self._disconnect_time_us:
                return
            assert self._buffer_tracker is not None
            self._buffer_tracker.reset()

        # Duration threshold (seconds)
        reset_after_s = 2.0
        if self._buffer_reset_handle is not None:
            self._buffer_reset_handle.cancel()
        self._buffer_reset_handle = self._server.loop.call_later(reset_after_s, _maybe_reset)

    # ---- Messaging (delegates to connection) ----

    def send_message(self, message: ServerMessage | bytes) -> None:
        """Send a message if connected; otherwise no-op."""
        if self._connection is None:
            return
        self._connection.send_message(message)

    def try_send_binary(self, data: bytes) -> bool:
        """Try to enqueue a droppable binary payload for this client."""
        if self._connection is None:
            return False
        return self._connection.try_send_binary(data)

    def queue_high_water(self, threshold: float = 0.8) -> bool:
        """Return True if the outgoing queue is above a high-water mark."""
        if self._connection is None:
            return False
        return self._connection.queue_high_water(threshold=threshold)

    # ---- Player streaming state ----

    @property
    def player_role(self) -> PlayerRole | None:
        """Return the active PlayerRole instance for this connection, if any."""
        return self._player_role

    @property
    def buffer_tracker(self) -> BufferTracker | None:
        """Return the persistent buffer tracker for the player role, if any."""
        return self._buffer_tracker

    @property
    def preferred_format(self) -> AudioFormat | None:
        """Return the preferred audio format for the player role, if set."""
        return self._preferred_format

    @preferred_format.setter
    def preferred_format(self, value: AudioFormat | None) -> None:
        self._preferred_format = value

    @property
    def blocking(self) -> bool:
        """Return whether this player participates in backpressure timing."""
        return self._blocking

    @blocking.setter
    def blocking(self, value: bool) -> None:
        self._blocking = value

    # ---- Role helpers ----

    @property
    def player(self) -> PlayerClient | None:
        """Return the PlayerClient helper if this client has the player role."""
        return self._player

    @property
    def controller(self) -> ControllerClient | None:
        """Return the ControllerClient helper if this client has the controller role."""
        return self._controller

    @property
    def metadata(self) -> MetadataClient | None:
        """Return the MetadataClient helper if this client has the metadata role."""
        return self._metadata_client

    @property
    def visualizer(self) -> VisualizerClient | None:
        """Return the VisualizerClient helper if this client has the visualizer role."""
        return self._visualizer

    # ---- Events + grouping ----

    def add_event_listener(
        self, callback: Callable[[SendspinClient, ClientEvent], None]
    ) -> Callable[[], None]:
        """Register a callback for client-scoped events and return an unsubscribe callable."""
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

    async def ungroup(self) -> None:
        """Remove the client from the group (no-op if already solo)."""
        if len(self.group.clients) > 1:
            await self.group.remove_client(self)
