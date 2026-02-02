"""PlayerGroupRole - group-level player coordination."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.server.roles.base import GroupRole
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
        return True

    def set_group_muted(self, muted: bool) -> bool | None:  # noqa: FBT001
        """Set mute state on all players."""
        for player in self._player_roles():
            player.set_player_mute(muted)
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

    def suggest_optimal_sample_rate(self, source_sample_rate: int) -> int:
        """Suggest an optimal sample rate for the next track.

        Analyzes all player roles in this group and returns the best sample rate
        that minimizes resampling across group members. Preference order:
        - If there is a common supported rate across all players, choose the one
          closest to the source sample rate (tie-breaker: higher rate).
        - Otherwise, choose the rate supported by the most players; among those,
          pick the closest to the source (tie-breaker: higher rate).

        Args:
            source_sample_rate: The sample rate of the upcoming source media.

        Returns:
            The recommended sample rate in Hz.
        """
        supported_sets: list[set[int]] = []
        for role in self._player_roles():
            rates = role.get_player_supported_sample_rates()
            if rates:
                supported_sets.append(rates)

        if not supported_sets:
            return source_sample_rate

        def choose(candidates: set[int]) -> int:
            best_distance = min(abs(r - source_sample_rate) for r in candidates)
            best_rates = [r for r in candidates if abs(r - source_sample_rate) == best_distance]
            return max(best_rates)

        # 1) Intersection across all players
        if intersection := set.intersection(*supported_sets):
            return choose(intersection)

        # 2) No common rate; pick rate supported by most players, then closest to source
        counts: dict[int, int] = {}
        for s in supported_sets:
            for r in s:
                counts[r] = counts.get(r, 0) + 1
        max_count = max(counts.values())
        top_rates = {r for r, c in counts.items() if c == max_count}
        return choose(top_rates)
