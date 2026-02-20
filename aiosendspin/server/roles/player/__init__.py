"""Player role - client and group level."""

from aiosendspin.server.roles.player.audio_transformers import FlacEncoder, PcmPassthrough
from aiosendspin.server.roles.player.events import (
    PlayerGroupEvent,
    PlayerGroupMuteChangedEvent,
    PlayerGroupVolumeChangedEvent,
    VolumeChangedEvent,
)
from aiosendspin.server.roles.player.group import PlayerGroupRole
from aiosendspin.server.roles.player.types import PlayerRoleProtocol
from aiosendspin.server.roles.player.v1 import PlayerV1Role
from aiosendspin.server.roles.registry import register_group_role, register_role

register_group_role("player", PlayerGroupRole)
register_role("player@v1", lambda client: PlayerV1Role(client=client))

__all__ = [
    "FlacEncoder",
    "PcmPassthrough",
    "PlayerGroupEvent",
    "PlayerGroupMuteChangedEvent",
    "PlayerGroupRole",
    "PlayerGroupVolumeChangedEvent",
    "PlayerRoleProtocol",
    "PlayerV1Role",
    "VolumeChangedEvent",
]
