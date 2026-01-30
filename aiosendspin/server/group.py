"""Manages and synchronizes playback for a group of one or more clients."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiosendspin.models.core import (
    GroupUpdateServerMessage,
    GroupUpdateServerPayload,
    StreamEndMessage,
    StreamEndPayload,
)
from aiosendspin.models.types import (
    MediaCommand,
    PlaybackStateType,
)
from aiosendspin.server.roles import GroupRole
from aiosendspin.server.roles.registry import create_group_roles

from .audio_transformers import TransformerPool
from .channels import ChannelRouter
from .push_stream import PushStream

if TYPE_CHECKING:
    from .client import SendspinClient
    from .roles.controller.group import ControllerGroupRole
    from .server import SendspinServer

logger = logging.getLogger(__name__)


class GroupEvent:
    """Base event type used by SendspinGroup.add_event_listener()."""


@dataclass
class GroupStateChangedEvent(GroupEvent):
    """Group state has changed."""

    state: PlaybackStateType
    """The new group state."""


@dataclass
class GroupMemberAddedEvent(GroupEvent):
    """A client was added to the group."""

    client_id: str
    """The ID of the client that was added."""


@dataclass
class GroupMemberRemovedEvent(GroupEvent):
    """A client was removed from the group."""

    client_id: str
    """The ID of the client that was removed."""


@dataclass
class GroupDeletedEvent(GroupEvent):
    """This group has no more members and has been deleted."""


class SendspinGroup:
    """
    A group of one or more clients for synchronized playback.

    Handles synchronized audio streaming across multiple clients with automatic
    format conversion and buffer management. Every client is always assigned to
    a group to simplify grouping requests.
    """

    _clients: list[SendspinClient]
    """List of all clients in this group."""
    _server: SendspinServer
    """Reference to the SendspinServer instance."""
    _event_cbs: list[Callable[[SendspinGroup, GroupEvent], None]]
    """List of event callbacks for this group."""
    _current_state: PlaybackStateType = PlaybackStateType.STOPPED
    """Current playback state of the group."""
    _group_id: str
    """Unique identifier for this group."""
    _group_name: str | None
    """Friendly name for this group."""
    _play_start_time_us: int | None
    """Absolute timestamp in microseconds when playback started, None when not streaming."""
    _scheduled_stop_handle: asyncio.TimerHandle | None
    """Timer handle for scheduled stop, None when no stop is scheduled."""
    _playback_lock: asyncio.Lock
    """Lock to serialize play_media() and stop() operations, preventing race conditions."""
    _push_stream: PushStream | None
    """Current PushStream for push-based streaming, None when not active."""
    _transformer_pool: TransformerPool
    """Pool for shared transformer instances (encoders, etc.) across roles."""
    _group_roles: dict[str, GroupRole]
    """Registry of GroupRole instances, keyed by role family."""

    def __init__(self, server: SendspinServer, *args: SendspinClient) -> None:
        """
        DO NOT CALL THIS CONSTRUCTOR. INTERNAL USE ONLY.

        Groups are managed automatically by the server.

        Initialize a new SendspinGroup.

        Args:
            server: The SendspinServer instance this group belongs to.
            *args: Clients to add to this group.
        """
        self._clients = list(args)
        assert len(self._clients) > 0, "A group must have at least one client"
        self._server = server
        self._event_cbs = []
        self._group_id = str(uuid.uuid4())
        self._group_name: str | None = None
        self._play_start_time_us: int | None = None
        self._scheduled_stop_handle: asyncio.TimerHandle | None = None
        self._playback_lock = asyncio.Lock()
        self._push_stream: PushStream | None = None
        self._transformer_pool = TransformerPool()
        self._group_roles = create_group_roles(self)

        # Set group reference for initial clients
        for client in self._clients:
            client._set_group(self)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

        logger.debug(
            "SendspinGroup initialized with %d client(s): %s",
            len(self._clients),
            [type(c).__name__ for c in self._clients],
        )

    def start_stream(
        self,
        *,
        channel_router: ChannelRouter | None = None,
    ) -> PushStream:
        """
        Create a new PushStream for push-based audio streaming.

        Args:
            channel_router: Optional custom channel router. If not provided,
                a new ChannelRouter is created.

        Returns:
            A new PushStream instance configured for this group.
        """
        if channel_router is None:
            channel_router = ChannelRouter()

        self._push_stream = PushStream(
            loop=self._server.loop,
            clock=self._server.clock,
            group=self,
            channel_router=channel_router,
        )
        # Starting a stream implies the group is actively playing.
        if self._current_state != PlaybackStateType.PLAYING:
            self._current_state = PlaybackStateType.PLAYING
            self._signal_event(GroupStateChangedEvent(PlaybackStateType.PLAYING))
            self._send_group_update_to_clients()
        return self._push_stream

    def stop_stream(self) -> None:
        """
        Stop the current push stream.

        Does nothing if no stream is active.
        """
        if self._push_stream is not None:
            self._push_stream.stop()

    def _send_group_update_to_clients(self) -> None:
        """Send group/update messages to all clients."""
        group_message = GroupUpdateServerMessage(
            GroupUpdateServerPayload(
                playback_state=self._current_state,
                group_id=self.group_id,
                group_name=self.group_name,
            )
        )
        for client in self._clients:
            client.send_message(group_message)

    def on_client_connected(self, client: SendspinClient) -> None:
        """Send current group state to a client that just finished handshaking."""
        if client not in self._clients:
            return

        group_message = GroupUpdateServerMessage(
            GroupUpdateServerPayload(
                playback_state=self._current_state,
                group_id=self.group_id,
                group_name=self.group_name,
            )
        )
        client.send_message(group_message)

        if self._push_stream is not None and not self._push_stream.is_stopped:
            for role in client.active_roles:
                if role.get_audio_requirements() is not None:
                    self._push_stream.on_role_join(role)

    def _send_stream_end_msg(self, client: SendspinClient, roles: list[str] | None = None) -> None:
        """Send a stream end message to a client.

        Args:
            client: The client to send the message to.
            roles: Optional list of roles to end streams for. If None, ends all streams.
        """
        logger.debug("ending stream for %s (%s), roles=%s", client.name, client.client_id, roles)
        client.send_message(StreamEndMessage(payload=StreamEndPayload(roles=roles)))

    def _schedule_delayed_stop(self, stop_time_us: int, active: bool, needs_cleanup: bool) -> bool:  # noqa: FBT001
        """Schedule a delayed stop at the specified timestamp.

        Args:
            stop_time_us: Absolute timestamp when stop should occur
            active: Whether stream task is currently active
            needs_cleanup: Whether cleanup is needed

        Returns:
            True if stop was scheduled, False if nothing to do
        """
        now_us = self._server.clock.now_us()
        if stop_time_us <= now_us:
            return False

        # Only schedule if there's something to stop or cleanup
        if not active and not needs_cleanup:
            return False

        delay = (stop_time_us - now_us) / 1_000_000

        async def _delayed_stop() -> None:
            # Store handle locally to detect if it's been replaced
            handle = self._scheduled_stop_handle
            try:
                await self.stop()  # This will clear _scheduled_stop_handle
            except Exception:
                logger.exception("Scheduled stop failed")
            finally:
                # Only clear if this handle is still current (e.g., stop() was interrupted
                # or a new stop was scheduled during the stop() call)
                if self._scheduled_stop_handle is handle:
                    self._scheduled_stop_handle = None

        def _schedule_stop() -> None:
            task = self._server.loop.create_task(_delayed_stop())
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

        self._scheduled_stop_handle = self._server.loop.call_later(delay, _schedule_stop)
        return True

    def _send_stopped_state_to_clients(self) -> None:
        """Send stopped state to all clients."""
        group_message = GroupUpdateServerMessage(
            GroupUpdateServerPayload(
                playback_state=PlaybackStateType.STOPPED,
                group_id=self.group_id,
                group_name=self.group_name,
            )
        )
        for client in self._clients:
            client.send_message(group_message)

    async def stop(self, stop_time_us: int | None = None) -> bool:
        """
        Stop playback for the group and clean up resources.

        Args:
            stop_time_us: Optional absolute timestamp (microseconds) when playback should
                stop. When provided and in the future, the stop request is scheduled and
                this method returns immediately.

        Returns:
            bool: True if an active stream was stopped (or scheduled to stop),
            False if no stream was active and no cleanup was required.
        """
        if len(self._clients) == 0:
            # An empty group cannot have active playback
            return False

        async with self._playback_lock:
            # Cancel any existing scheduled stop first to prevent race conditions
            if self._scheduled_stop_handle is not None:
                logger.debug("Canceling previously scheduled stop in stop()")
                self._scheduled_stop_handle.cancel()
                self._scheduled_stop_handle = None

            active = self._push_stream is not None and not self._push_stream.is_stopped
            needs_cleanup = self._current_state != PlaybackStateType.STOPPED

            # Handle delayed stop if requested
            if stop_time_us is not None and self._schedule_delayed_stop(
                stop_time_us, active, needs_cleanup
            ):
                return active or needs_cleanup

            if not active and not needs_cleanup:
                return False

            logger.debug(
                "Stopping playback for group with clients: %s",
                [c.client_id for c in self._clients],
            )

            # Stop the push stream if active
            if self._push_stream is not None:
                self._push_stream.stop()

            if self._current_state != PlaybackStateType.STOPPED:
                self._signal_event(GroupStateChangedEvent(PlaybackStateType.STOPPED))
                self._current_state = PlaybackStateType.STOPPED

            self._send_stopped_state_to_clients()
            return True

    @property
    def clients(self) -> list[SendspinClient]:
        """All clients that are part of this group."""
        return self._clients

    @property
    def has_active_stream(self) -> bool:
        """Check if there is an active stream running."""
        return self._push_stream is not None and not self._push_stream.is_stopped

    @property
    def transformer_pool(self) -> TransformerPool:
        """Return the transformer pool for encoder deduplication."""
        return self._transformer_pool

    def group_role(self, family: str) -> GroupRole | None:
        """Get the GroupRole for a role family."""
        return self._group_roles.get(family)

    def _controller_group_role(self) -> ControllerGroupRole | None:
        """Get the ControllerGroupRole (type-safe accessor)."""
        from .roles.controller.group import ControllerGroupRole  # noqa: PLC0415

        role = self._group_roles.get("controller")
        if isinstance(role, ControllerGroupRole):
            return role
        return None

    def register_group_role(self, group_role: GroupRole) -> None:
        """Register a GroupRole (called during group initialization)."""
        self._group_roles[group_role.role_family] = group_role

    def add_event_listener(
        self, callback: Callable[[SendspinGroup, GroupEvent], None]
    ) -> Callable[[], None]:
        """
        Register a callback to listen for state changes of this group.

        State changes include:
        - The group started playing
        - The group stopped/finished playing

        Returns a function to remove the listener.
        """
        self._event_cbs.append(callback)

        def _remove() -> None:
            with suppress(ValueError):
                self._event_cbs.remove(callback)

        return _remove

    def _signal_event(self, event: GroupEvent) -> None:
        for cb in self._event_cbs:
            try:
                cb(self, event)
            except Exception:
                logger.exception("Error in event listener")

    def _register_client_events(self, client: SendspinClient) -> None:
        """Register event listeners for client events like volume changes."""
        controller_role = self._controller_group_role()
        if controller_role is not None:
            controller_role.subscribe_to_player_client(client)

    def _unregister_client_events(self, client: SendspinClient) -> None:
        """Unregister event listeners for a client."""
        controller_role = self._controller_group_role()
        if controller_role is not None:
            controller_role.unsubscribe_from_player_client(client)

    @property
    def group_id(self) -> str:
        """Unique identifier for this group."""
        return self._group_id

    @property
    def group_name(self) -> str | None:
        """Friendly name for this group."""
        return self._group_name

    @property
    def state(self) -> PlaybackStateType:
        """Current playback state of the group."""
        return self._current_state

    @property
    def volume(self) -> int:
        """Return current group volume (0-100), delegated to group roles."""
        for role in self._group_roles.values():
            if (volume := role.get_group_volume()) is not None:
                return volume
        return 100

    @property
    def muted(self) -> bool:
        """Return current group mute state, delegated to group roles."""
        for role in self._group_roles.values():
            if (muted := role.get_group_muted()) is not None:
                return muted
        return False

    def set_volume(self, volume_level: int) -> None:
        """Set group volume, delegated to group roles."""
        for role in self._group_roles.values():
            if role.set_group_volume(volume_level) is not None:
                break

    def set_mute(self, muted: bool) -> None:  # noqa: FBT001
        """Set group mute state, delegated to group roles."""
        for role in self._group_roles.values():
            if role.set_group_muted(muted) is not None:
                break

    def set_supported_commands(self, commands: list[MediaCommand]) -> None:
        """
        Set the media commands supported by the application.

        Args:
            commands: List of MediaCommand values that the application can handle.
                Empty list means no commands are supported.
        """
        controller_role = self._controller_group_role()
        if controller_role is not None:
            controller_role.set_supported_commands(commands)

    async def remove_client(self, client: SendspinClient) -> None:
        """
        Remove a client from this group.

        If a stream is active, the client receives a stream end message.
        The client is automatically moved to its own new group since every
        client must belong to a group.
        If the client is not part of this group, this will have no effect.

        Args:
            client: The client to remove from this group.
        """
        if client not in self._clients:
            return

        # Cancel any pending delayed join for this client
        logger.debug("removing %s from group with members: %s", client.client_id, self._clients)
        if len(self._clients) == 1:
            # Delete this group if that was the last client
            await self.stop()
            self._clients = []
        else:
            self._clients.remove(client)
            # End the stream for the removed client via role hooks
            handled = False
            for role in client.active_roles:
                role.on_stream_end()
                if self._push_stream is not None and not self._push_stream.is_stopped:
                    self._push_stream.on_role_leave(role)
                handled = True
            if not handled:
                self._send_stream_end_msg(client)
        if not self._clients:
            # Emit event for group deletion, no clients left
            self._signal_event(GroupDeletedEvent())
        else:
            # Emit event for client removal
            self._signal_event(GroupMemberRemovedEvent(client.client_id))
        # Each client needs to be in a group, add it to a new one
        new_group = SendspinGroup(self._server, client)
        # Send group update to notify client of their new solo group
        new_group.on_client_connected(client)

    async def add_client(self, client: SendspinClient) -> None:
        """
        Add a client to this group.

        The client is first removed from any existing group. If a session is
        currently active, players are immediately joined to the session with
        an appropriate audio format.

        Args:
            client: The client to add to this group.
        """
        logger.debug("adding %s to group with members: %s", client.client_id, self._clients)
        old_group = client.group
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "add_client(%s): stopping previous group=%s active=%s members=%s",
                client.client_id,
                old_group.group_id,
                old_group.has_active_stream,
                [c.client_id for c in old_group.clients],
            )
        stopped = await old_group.stop()
        if stopped and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "add_client(%s): previous group=%s stopped playback",
                client.client_id,
                old_group.group_id,
            )
        if client in self._clients:
            return
        # Remove it from any existing group first
        await client.ungroup()

        # Check for and remove any stale client with the same client_id
        # This handles the case where a client disconnects and reconnects
        # while still being listed in _clients (e.g., solo client disconnect)
        stale_client = next((c for c in self._clients if c.client_id == client.client_id), None)
        if stale_client is not None:
            logger.debug(
                "Removing stale client %s (object %s) before adding new client (object %s)",
                stale_client.client_id,
                id(stale_client),
                id(client),
            )
            self._clients.remove(stale_client)
            self._unregister_client_events(stale_client)

        # Add client to this group's client list
        self._clients.append(client)

        # Emit event for client addition
        self._signal_event(GroupMemberAddedEvent(client.client_id))

        # Then set the group (which will emit ClientGroupChangedEvent)
        client._set_group(self)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

        # Handle player joining/reconnecting with active PushStream
        if self._push_stream is not None and not self._push_stream.is_stopped:
            # Call on_role_join for all roles with audio requirements (hook-based flow)
            for role in client.active_roles:
                if role.get_audio_requirements() is not None:
                    self._push_stream.on_role_join(role)

        # Send current state to the new client
        group_message = GroupUpdateServerMessage(
            GroupUpdateServerPayload(
                playback_state=self._current_state,
                group_id=self.group_id,
                group_name=self.group_name,
            )
        )
        logger.debug("Sending group update to new client %s", client.client_id)
        client.send_message(group_message)

        # Note: Role-specific state (controller, metadata, artwork) is sent
        # via respective GroupRole.on_member_join() methods
