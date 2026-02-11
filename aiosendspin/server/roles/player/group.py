"""PlayerGroupRole - group-level player coordination."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.server.roles.base import GroupRole
from aiosendspin.server.roles.player.events import (
    PlayerGroupMuteChangedEvent,
    PlayerGroupVolumeChangedEvent,
)
from aiosendspin.server.roles.player.types import PlayerRoleProtocol

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient


class PlayerGroupRole(GroupRole):
    """Coordinate player roles across a group."""

    role_family = "player"

    def _player_roles(self) -> list[PlayerRoleProtocol]:
        """Return player role members.

        All members of PlayerGroupRole are PlayerV1Role instances since only
        roles with role_family="player" subscribe to this GroupRole.
        """
        return list(self._members)

    def get_group_volume(self) -> int | None:
        """Return current group volume (average of player volumes)."""
        players = self._player_roles()
        if not players:
            return 100
        total = 0
        count = 0
        for p in players:
            vol = p.get_player_volume()
            if vol is not None:
                total += vol
                count += 1
        return round(total / count) if count else 100

    def get_group_muted(self) -> bool | None:
        """Return current group mute state (true only when ALL players muted)."""
        players = self._player_roles()
        if not players:
            return False
        for p in players:
            m = p.get_player_muted()
            if m is None or not m:
                return False
        return True

    def set_group_volume(self, level: int) -> bool | None:
        """Set group volume using redistribution algorithm."""
        level = max(0, min(100, level))
        players = self._player_roles()
        if not players:
            return True
        previous_volume = self.get_group_volume()

        # Build mapping of player -> current volume (only players with volume support)
        player_volumes: dict[PlayerRoleProtocol, float] = {}
        for p in players:
            vol = p.get_player_volume()
            if vol is not None:
                player_volumes[p] = float(vol)

        if not player_volumes:
            return True

        # Calculate initial delta
        current_avg = sum(player_volumes.values()) / len(player_volumes)
        delta = level - current_avg

        # Redistribution iterations
        active_players = list(player_volumes.keys())
        for _ in range(5):
            lost_delta_sum = 0.0
            next_active: list[PlayerRoleProtocol] = []

            for player in active_players:
                current = player_volumes[player]
                proposed = current + delta

                if proposed > 100:
                    clamped = 100.0
                    lost_delta_sum += proposed - clamped
                elif proposed < 0:
                    clamped = 0.0
                    lost_delta_sum += proposed - clamped
                else:
                    clamped = proposed
                    next_active.append(player)

                player_volumes[player] = clamped

            if not next_active or abs(lost_delta_sum) < 0.01:
                break

            delta = lost_delta_sum / len(next_active)
            active_players = next_active

        # Apply to players
        for player, final_vol in player_volumes.items():
            player.set_player_volume(round(final_vol))
        new_volume = self.get_group_volume()
        if previous_volume is not None and new_volume is not None and previous_volume != new_volume:
            self.emit_group_event(
                PlayerGroupVolumeChangedEvent(
                    previous_volume=previous_volume,
                    volume=new_volume,
                )
            )
        return True

    def set_group_muted(self, muted: bool) -> bool | None:  # noqa: FBT001
        """Set mute state on all players."""
        previous_muted = bool(self.get_group_muted())
        for player in self._player_roles():
            player.set_player_mute(muted)
        new_muted = bool(self.get_group_muted())
        if previous_muted != new_muted:
            self.emit_group_event(
                PlayerGroupMuteChangedEvent(previous_muted=previous_muted, muted=new_muted)
            )
        return True

    @property
    def volume(self) -> int:
        """Return current group volume (average of player volumes)."""
        return self.get_group_volume() or 100

    @property
    def muted(self) -> bool:
        """Return current group mute state (true only when ALL players muted)."""
        return bool(self.get_group_muted())

    def set_volume(self, level: int) -> None:
        """Set group volume using redistribution algorithm."""
        self.set_group_volume(level)

    def set_mute(self, muted: bool) -> None:  # noqa: FBT001
        """Set mute state on all players."""
        self.set_group_muted(muted)

    def get_player_clients(self) -> list[SendspinClient]:
        """Return all clients in this group that have an active player role.

        Returns:
            Clients with player roles.
        """
        return [role._client for role in self._player_roles()]  # noqa: SLF001
