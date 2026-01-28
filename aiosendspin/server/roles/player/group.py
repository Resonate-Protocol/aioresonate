"""PlayerGroupRole - group-level player coordination."""

from __future__ import annotations

from aiosendspin.server.roles.base import GroupRole, Role


class PlayerGroupRole(GroupRole):
    """Coordinate player roles across a group."""

    role_family = "player"

    def _player_roles(self) -> list[Role]:
        """Return player role members.

        All members of PlayerGroupRole are PlayerRole instances since only
        roles with role_family="player" subscribe to this GroupRole.
        """
        return list(self._members)
