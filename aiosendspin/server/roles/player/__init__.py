"""Player role - client and group level."""

from aiosendspin.server.roles.player.group import PlayerGroupRole
from aiosendspin.server.roles.player.v1 import PlayerRole
from aiosendspin.server.roles.registry import register_group_role, register_role

register_group_role("player", lambda group: PlayerGroupRole(group))
register_role("player@v1", lambda client: PlayerRole(client=client))

__all__ = ["PlayerGroupRole", "PlayerRole"]
