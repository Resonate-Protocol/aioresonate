"""Player role - client and group level."""

from aiosendspin.server.roles.group_registry import register_group_role
from aiosendspin.server.roles.player.group import PlayerGroupRole
from aiosendspin.server.roles.player.v1 import PlayerRole

register_group_role("player", lambda group: PlayerGroupRole(group))

__all__ = ["PlayerGroupRole", "PlayerRole"]
