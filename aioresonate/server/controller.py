"""Helpers for clients supporting the controller role."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aioresonate.models.controller import ControllerCommandPayload
from aioresonate.models.types import MediaCommand, PlaybackStateType

if TYPE_CHECKING:
    from .client import ResonateClient
    from .group import ResonateGroup
    from .server import ResonateServer


class ControllerClient:
    """Encapsulates controller role behaviour for a client."""

    def __init__(self, client: ResonateClient) -> None:
        """Attach to a client that exposes controller capabilities."""
        self.client = client
        self._logger = client._logger.getChild("controller")  # noqa: SLF001

    async def handle_command(self, payload: ControllerCommandPayload) -> None:
        """Handle controller commands."""
        if payload.command == MediaCommand.SWITCH:
            await self._handle_switch()
        else:
            # Forward other commands to the group
            self.client.group._handle_group_command(payload)  # noqa: SLF001

    async def _handle_switch(self) -> None:
        """Handle the switch command to cycle through groups."""
        # TODO: this is untested, who knows if it works as described in the spec
        server = self.client._server  # noqa: SLF001
        current_group = self.client.group

        # Get all unique groups from all connected clients
        all_groups = self._get_all_groups(server)

        # Build the cycle list based on client's player role
        has_player_role = self.client.player is not None
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
        if next_group != current_group:
            self._logger.info(
                "Switching client %s to group %s",
                self.client.client_id,
                next_group._group_id,  # noqa: SLF001
            )
            await current_group.remove_client(self.client)
            await next_group.add_client(self.client)

    def _get_all_groups(self, server: ResonateServer) -> list[ResonateGroup]:
        """Get all unique groups from all connected clients."""
        groups_seen: set[str] = set()
        unique_groups: list[ResonateGroup] = []

        for client in server._clients:  # noqa: SLF001
            group = client.group
            group_id = group._group_id  # noqa: SLF001
            if group_id not in groups_seen:
                groups_seen.add(group_id)
                unique_groups.append(group)

        return unique_groups

    def _build_group_cycle(
        self,
        all_groups: list[ResonateGroup],
        current_group: ResonateGroup,
        has_player_role: bool,  # noqa: FBT001
    ) -> list[ResonateGroup]:
        """Build the cycle of groups based on the spec."""
        # Separate groups into categories
        multi_client_playing: list[ResonateGroup] = []
        single_client: list[ResonateGroup] = []
        current_solo: list[ResonateGroup] = []

        for group in all_groups:
            client_count = len(group._clients)  # noqa: SLF001
            is_playing = group._current_state == PlaybackStateType.PLAYING  # noqa: SLF001

            if client_count > 1 and is_playing:
                multi_client_playing.append(group)
            elif client_count == 1:
                if group == current_group:
                    current_solo.append(group)
                else:
                    single_client.append(group)

        # Sort for stable ordering (by group ID)
        multi_client_playing.sort(key=lambda g: g._group_id)  # noqa: SLF001
        single_client.sort(key=lambda g: g._group_id)  # noqa: SLF001

        # Build cycle based on client's player role
        if has_player_role:
            # With player role: multi-client playing -> single-client -> own solo
            return multi_client_playing + single_client + current_solo
        # Without player role: multi-client playing -> single-client (no own solo)
        return multi_client_playing + single_client
