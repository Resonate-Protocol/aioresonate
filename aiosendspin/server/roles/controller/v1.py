"""ControllerV1Role implementation (v1).

This role handles bidirectional communication:
- Inbound: client/command controller messages
- Outbound: server/state controller messages
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aiosendspin.models.controller import ControllerCommandPayload
from aiosendspin.models.types import (
    ClientStateType,
    MediaCommand,
    PlaybackStateType,
    has_role_family,
)
from aiosendspin.server.roles.base import Role
from aiosendspin.util import create_task

if TYPE_CHECKING:
    from aiosendspin.models.core import ClientCommandPayload
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.group import SendspinGroup
    from aiosendspin.server.roles.controller.group import ControllerGroupRole


logger = logging.getLogger(__name__)


@dataclass
class ControllerRoleState:
    """Persistent state for ControllerV1Role across reconnects."""

    previous_group_id: str | None = None
    """Group ID to rejoin after external_source ends."""

    external_source_solo_group_id: str | None = None
    """Solo group ID created when entering external_source."""


class ControllerV1Role(Role):
    """Role implementation for controller clients.

    Receives controller state from ControllerGroupRole and forwards commands
    from the client to the group.
    """

    def __init__(self, client: SendspinClient | None = None) -> None:
        """Initialize ControllerV1Role.

        Args:
            client: The owning SendspinClient.
        """
        if client is None:
            msg = "ControllerV1Role requires a client"
            raise ValueError(msg)
        self._client = client
        self._stream_started = False
        self._buffer_tracker = None
        self._group_role: ControllerGroupRole | None = None
        self._switch_lock = asyncio.Lock()
        self._logger = logger.getChild(str(client.client_id))

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "controller@v1"

    @property
    def role_family(self) -> str:
        """Role family name for protocol messages."""
        return "controller"

    def on_connect(self) -> None:
        """Subscribe to ControllerGroupRole for state updates."""
        self._subscribe_to_group_role()

    def on_disconnect(self) -> None:
        """Unsubscribe from ControllerGroupRole."""
        self._unsubscribe_from_group_role()

    def on_state_transition(
        self,
        old_state: ClientStateType,  # noqa: ARG002
        new_state: ClientStateType,
    ) -> Coroutine[Any, Any, None] | None:
        """Handle external_source transitions by moving client to solo group."""
        if new_state != ClientStateType.EXTERNAL_SOURCE:
            return None
        return self._handle_external_source_transition()

    async def _handle_external_source_transition(self) -> None:
        """Handle transition to external_source state.

        When transitioning to external_source:
        - If in multi-client group: remember previous group, move to solo group
        - If already in solo group: stop playback
        """
        is_multi_client_group = len(self._client.group.clients) > 1

        if is_multi_client_group:
            # Store previous group in controller role state (persists across reconnects)
            state = self._get_state()
            state.previous_group_id = self._client.group.group_id
            self._logger.debug(
                "Storing previous group %s for external_source client",
                state.previous_group_id,
            )
            await self._client.group.remove_client(self._client)
            state.external_source_solo_group_id = self._client.group.group_id
            return

        self._logger.debug("Client already in solo group, stopping playback for external_source")
        await self._client.group.stop()

    def on_command(self, payload: ClientCommandPayload) -> None:
        """Handle client/command payload."""
        controller_cmd = payload.controller
        if controller_cmd is None:
            return

        if controller_cmd.command == MediaCommand.SWITCH:
            create_task(self._handle_switch_command())
            return

        # Forward other commands to group role
        if self._group_role is not None:
            self._group_role.handle_command(controller_cmd)

    def handle_command(self, cmd: ControllerCommandPayload) -> None:
        """Forward a controller command to the group role.

        Args:
            cmd: The controller command from the client.
        """
        if self._group_role is not None:
            self._group_role.handle_command(cmd)

    # --- Switch command handling ---

    def _get_state(self) -> ControllerRoleState:
        """Get or create persistent state for this role."""
        return self._client.get_or_create_role_state(self.role_family, ControllerRoleState)

    async def _handle_switch_command(self) -> None:
        """Handle the switch command to cycle through groups."""
        try:
            await asyncio.wait_for(self._switch_lock.acquire(), timeout=0)
        except TimeoutError:
            self._logger.debug("Ignoring switch command; switch already in progress")
            return
        try:
            await self._handle_switch_command_locked()
        finally:
            self._switch_lock.release()

    async def _handle_switch_command_locked(self) -> None:
        """Handle the switch command to cycle through groups (locked)."""
        # Clients in external_source can't participate in playback
        if self._client.client_state == ClientStateType.EXTERNAL_SOURCE:
            self._logger.debug("Ignoring switch command while client is in external_source state")
            return

        # Check if client should rejoin previous group (external_source recovery priority)
        if await self._try_rejoin_previous_group():
            return

        current_group = self._client.group

        # Get all unique groups from all connected clients
        all_groups = self._get_all_groups()

        # Build the cycle list based on client's player role
        has_player_role = has_role_family("player", self._client.negotiated_roles)
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
                self._client.client_id,
            )
            await current_group.remove_client(self._client)
        elif next_group != current_group:
            self._logger.info(
                "Switching client %s to group %s",
                self._client.client_id,
                next_group.group_id,
            )
            await current_group.remove_client(self._client)
            await next_group.add_client(self._client)

    def _get_all_groups(self) -> list[SendspinGroup]:
        """Get all unique groups from all connected clients."""
        groups_seen: set[str] = set()
        unique_groups: list[SendspinGroup] = []

        for client in self._client._server.connected_clients:  # noqa: SLF001
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
        """Build the cycle of groups based on the spec.

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
                has_player = any(
                    has_role_family("player", c.negotiated_roles) for c in group.clients
                )
                if has_player:
                    multi_client_playing.append(group)
            elif client_count == 1 and is_playing:
                # Get the single client in this group
                single_client_obj = group.clients[0]
                # Skip current group, it will be handled as solo option for player clients
                if group != current_group and has_role_family(
                    "player", single_client_obj.negotiated_roles
                ):
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
        """Check if client should rejoin previous group (external_source recovery).

        Per spec: "If the client is still in the solo group from its 'external_source'
        transition, the switch command prioritizes rejoining the previous group."
        """
        state = self._get_state()
        return (
            state.previous_group_id is not None
            and self._client.client_state != ClientStateType.EXTERNAL_SOURCE
            and state.external_source_solo_group_id == self._client.group.group_id
            and len(self._client.group.clients) == 1  # Still in the solo group
        )

    async def _try_rejoin_previous_group(self) -> bool:
        """Try to rejoin the previous group after external_source ended."""
        if not self._should_rejoin_previous_group():
            return False

        state = self._get_state()
        previous_group_id = state.previous_group_id
        # Clear external_source tracking after attempt (regardless of success)
        state.previous_group_id = None
        state.external_source_solo_group_id = None

        previous_group = self._find_group_by_id(previous_group_id)

        if previous_group is not None and previous_group != self._client.group:
            self._logger.info(
                "Rejoining previous group %s after external_source",
                previous_group_id,
            )
            await self._client.group.remove_client(self._client)
            await previous_group.add_client(self._client)
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

        for client in self._client._server.connected_clients:  # noqa: SLF001
            if client.group.group_id == group_id:
                return client.group
        return None
