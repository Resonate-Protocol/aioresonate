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

from aiosendspin.models import AudioCodec
from aiosendspin.models.controller import ControllerCommandPayload
from aiosendspin.models.core import (
    ClientHelloPayload,
    StreamStartMessage,
)
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    PlayerStatePayload,
)
from aiosendspin.models.types import (
    ClientStateType,
    MediaCommand,
    PlaybackStateType,
    PlayerCommand,
    Roles,
    has_role,
)
from aiosendspin.server.audio import AudioFormat, BufferTracker
from aiosendspin.server.events import ClientEvent, ClientGroupChangedEvent, VolumeChangedEvent

from .roles import PlayerRole, Role

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
        self._negotiated_roles: list[str] = []
        self._roles: dict[str, Role] = {}
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

        # Player volume/mute state (persistent across reconnects)
        self._player_volume: int = 100
        self._player_muted: bool = False

        # Player role persistent state
        self._buffer_tracker: BufferTracker | None = None
        self._preferred_format: AudioFormat | None = None
        self._preferred_codec: AudioCodec | None = None

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
    def negotiated_roles(self) -> list[str]:
        """Return the negotiated active roles for this connection (versioned role IDs)."""
        return self._negotiated_roles

    def role(self, role_id: str) -> Role | None:
        """Get active role by versioned ID (e.g., 'player@v1')."""
        return self._roles.get(role_id)

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
        return has_role(role.value, self._negotiated_roles)

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
        self._negotiated_roles = active_roles
        self._logger = logger.getChild(self._client_id)

        # Clear previous roles
        self._roles.clear()

        # Player persistent state (survives reconnects, role gets a reference).
        has_player_role = has_role(Roles.PLAYER.value, self._negotiated_roles)
        if has_player_role and self.info.player_support is not None:
            capacity = self.info.player_support.buffer_capacity
            if self._buffer_tracker is None:
                self._buffer_tracker = BufferTracker(
                    clock=self._server.clock,
                    client_id=self._client_id,
                    capacity_bytes=capacity,
                )
            else:
                self._buffer_tracker.capacity_bytes = capacity

            supported = self.info.player_support.supported_formats
            preferred = next(
                (fmt for fmt in supported if fmt.codec == AudioCodec.OPUS),
                supported[0],
            )
            default_format = AudioFormat(
                sample_rate=preferred.sample_rate,
                bit_depth=preferred.bit_depth,
                channels=preferred.channels,
            )
            default_codec = preferred.codec
            # Check if current preferred format+codec is still supported
            current_codec = self._preferred_codec or default_codec
            if self._preferred_format is None or not any(
                fmt.codec == current_codec
                and fmt.sample_rate == self._preferred_format.sample_rate
                and fmt.bit_depth == self._preferred_format.bit_depth
                and fmt.channels == self._preferred_format.channels
                for fmt in supported
            ):
                self._preferred_format = default_format
                self._preferred_codec = default_codec

            # Create and register player role
            player_role = PlayerRole(client=self)
            player_role._buffer_tracker = self._buffer_tracker  # noqa: SLF001
            player_role.on_connect()
            player_role.on_transport_attach()
            self._roles["player@v1"] = player_role

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

        # Notify all roles about detachment
        for role in self._roles.values():
            role.on_transport_detach()
            role.on_disconnect()
        self._roles.clear()

        self._connection = None
        self._disconnect_time_us = self._server.clock.now_us()

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
        if isinstance(message, StreamStartMessage):
            self._logger.info("Sending stream/start: %s", message.payload)
        self._connection.send_message(message)

    def try_send_binary(
        self,
        data: bytes,
        *,
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
    ) -> bool:
        """Try to enqueue a droppable binary payload for this client."""
        if self._connection is None:
            return False
        return self._connection.try_send_binary(
            data,
            buffer_end_time_us=buffer_end_time_us,
            buffer_byte_count=buffer_byte_count,
            duration_us=duration_us,
        )

    # ---- Player streaming state (convenience accessors) ----

    @property
    def player_role(self) -> PlayerRole | None:
        """Return the active PlayerRole instance for this connection, if any."""
        role = self._roles.get("player@v1")
        if role is not None and isinstance(role, PlayerRole):
            return role
        return None

    @property
    def active_roles(self) -> list[Role]:
        """All active roles for iteration."""
        return list(self._roles.values())

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
    def preferred_codec(self) -> AudioCodec | None:
        """Return the preferred audio codec for the player role, if set."""
        return self._preferred_codec

    @preferred_codec.setter
    def preferred_codec(self, value: AudioCodec | None) -> None:
        self._preferred_codec = value

    # ---- Player volume/mute state and commands ----

    @property
    def player_support(self) -> ClientHelloPlayerSupport | None:
        """Return player capabilities advertised in the hello payload."""
        return self.info.player_support

    @property
    def player_volume(self) -> int:
        """Current volume of this player (0-100)."""
        return self._player_volume

    @property
    def player_muted(self) -> bool:
        """Current mute state of this player."""
        return self._player_muted

    def set_player_volume(self, volume: int) -> None:
        """Set the volume of this player.

        DEPRECATED: Use client.role('player@v1').set_volume() instead.
        """
        player = self.role("player@v1")
        if player is not None and isinstance(player, PlayerRole):
            player.set_volume(volume)

    def set_player_mute(self, muted: bool) -> None:  # noqa: FBT001
        """Set the mute state of this player.

        DEPRECATED: Use client.role('player@v1').set_mute() instead.
        """
        player = self.role("player@v1")
        if player is not None and isinstance(player, PlayerRole):
            player.set_mute(muted)

    def handle_player_state_update(self, state: PlayerStatePayload) -> None:
        """Update internal mute/volume state from client report and emit event."""
        support = self.player_support
        changed = False

        if state.volume is not None:
            if not support or PlayerCommand.VOLUME not in support.supported_commands:
                self._logger.warning(
                    "Client sent volume field without declaring 'volume' in supported_commands"
                )
            elif self._player_volume != state.volume:
                self._player_volume = state.volume
                changed = True

        if state.muted is not None:
            if not support or PlayerCommand.MUTE not in support.supported_commands:
                self._logger.warning(
                    "Client sent muted field without declaring 'mute' in supported_commands"
                )
            elif self._player_muted != state.muted:
                self._player_muted = state.muted
                changed = True

        if changed:
            self._signal_event(
                VolumeChangedEvent(volume=self._player_volume, muted=self._player_muted)
            )

    # ---- Controller command handling ----

    async def handle_controller_command(self, payload: ControllerCommandPayload) -> None:
        """Handle controller commands from this client."""
        # Get supported commands from the group
        supported_commands = self.group._get_supported_commands()  # noqa: SLF001

        # Validate command is supported
        if payload.command not in supported_commands:
            self._logger.warning(
                "Client %s sent unsupported command '%s'. Supported commands: %s",
                self._client_id,
                payload.command.value,
                [cmd.value for cmd in supported_commands],
            )
            # Silently ignore unsupported commands (spec doesn't define error responses)
            return

        if payload.command == MediaCommand.SWITCH:
            await self._handle_switch_command()
        else:
            # Forward other commands to the group
            self.group._handle_group_command(payload)  # noqa: SLF001

    async def _handle_switch_command(self) -> None:
        """Handle the switch command to cycle through groups."""
        # Clients in external_source can't participate in playback; don't allow switching groups
        # until they report a normal operational state again.
        if self._client_state == ClientStateType.EXTERNAL_SOURCE:
            self._logger.warning("Ignoring switch command while client is in external_source state")
            return

        # Check if client should rejoin previous group (external_source recovery priority)
        if await self._try_rejoin_previous_group():
            return

        current_group = self.group

        # Get all unique groups from all connected clients
        all_groups = self._get_all_groups()

        # Build the cycle list based on client's player role
        has_player_role = self.check_role(Roles.PLAYER)
        cycle_groups = self._build_group_cycle(all_groups, current_group, has_player_role)

        if not cycle_groups:
            self._logger.debug("No groups available to switch to")
            return

        # Find current position in cycle and move to next
        try:
            current_index = cycle_groups.index(current_group)
            next_index = (current_index + 1) % len(cycle_groups)
        except ValueError:
            # Current group not in cycle, start from beginning
            next_index = 0

        next_group = cycle_groups[next_index]

        # Move client to the next group
        if next_group is None:
            # The group.remove_client will create a new solo group for the client
            self._logger.info(
                "Switching client %s to solo group",
                self._client_id,
            )
            await current_group.remove_client(self)
        elif next_group != current_group:
            self._logger.info(
                "Switching client %s to group %s",
                self._client_id,
                next_group.group_id,
            )
            await current_group.remove_client(self)
            await next_group.add_client(self)

    def _get_all_groups(self) -> list[SendspinGroup]:
        """Get all unique groups from all connected clients."""
        groups_seen: set[str] = set()
        unique_groups: list[SendspinGroup] = []

        for client in self._server.connected_clients:
            group = client.group
            group_id = group.group_id
            if group_id not in groups_seen:
                groups_seen.add(group_id)
                unique_groups.append(group)

        return unique_groups

    def _build_group_cycle(
        self,
        all_groups: list[SendspinGroup],
        current_group: SendspinGroup,
        has_player_role: bool,  # noqa: FBT001
    ) -> list[SendspinGroup | None]:
        """
        Build the cycle of groups based on the spec.

        Returns a list of groups to cycle through. For player clients, the list
        may contain None indicating to "go to a new solo group".
        """
        # Separate groups into categories
        multi_client_playing: list[SendspinGroup] = []
        single_client: list[SendspinGroup] = []

        for group in all_groups:
            client_count = len(group.clients)
            is_playing = group.state == PlaybackStateType.PLAYING

            if client_count > 1 and is_playing:
                # Verify the group has at least one player
                # (groups with only controllers/metadata can't actually be "playing")
                has_player = any(c.check_role(Roles.PLAYER) for c in group.clients)
                if has_player:
                    multi_client_playing.append(group)
            elif client_count == 1 and is_playing:
                # Get the single client in this group
                single_client_obj = group.clients[0]
                # Skip current group, it will be handled as solo option for player clients
                if group != current_group and single_client_obj.check_role(Roles.PLAYER):
                    # Only include single-client groups where the client has player role
                    single_client.append(group)

        # Sort for stable ordering (by group ID)
        multi_client_playing.sort(key=lambda g: g.group_id)
        single_client.sort(key=lambda g: g.group_id)

        # Build cycle based on client's player role
        if has_player_role:
            # With player role: multi-client playing -> single-client -> own solo
            current_is_solo = len(current_group.clients) == 1
            # Use current group if solo, otherwise switch to new solo group (None)
            solo_option: list[SendspinGroup | None] = [current_group] if current_is_solo else [None]
            return multi_client_playing + single_client + solo_option
        # Without player role: multi-client playing -> single-client (no own solo)
        return [*multi_client_playing, *single_client]

    def _should_rejoin_previous_group(self) -> bool:
        """
        Check if client should rejoin previous group (external_source recovery).

        Per spec: "If the client is still in the solo group from its 'external_source'
        transition, the switch command prioritizes rejoining the previous group."
        """
        return (
            self._previous_group_id is not None
            and self._client_state != ClientStateType.EXTERNAL_SOURCE
            and self._external_source_solo_group_id == self.group.group_id
            and len(self.group.clients) == 1  # Still in the solo group
        )

    async def _try_rejoin_previous_group(self) -> bool:
        """Try to rejoin the previous group after external_source ended."""
        if not self._should_rejoin_previous_group():
            return False

        previous_group_id = self._previous_group_id
        # Clear external_source tracking after attempt (regardless of success)
        self._previous_group_id = None
        self._external_source_solo_group_id = None

        previous_group = self._find_group_by_id(previous_group_id)

        if previous_group is not None and previous_group != self.group:
            self._logger.info(
                "Rejoining previous group %s after external_source",
                previous_group_id,
            )
            await self.group.remove_client(self)
            await previous_group.add_client(self)
            return True
        self._logger.debug(
            "Previous group %s no longer exists or is current group, "
            "falling back to normal switch cycle",
            previous_group_id,
        )
        return False

    def _find_group_by_id(self, group_id: str | None) -> SendspinGroup | None:
        """Find a group by its ID from all connected clients."""
        if group_id is None:
            return None

        for client in self._server.connected_clients:
            if client.group.group_id == group_id:
                return client.group
        return None

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
