"""Color role - client and group level."""

from aiosendspin.server.roles.color.events import (
    ColorClearedEvent,
    ColorEvent,
    ColorUpdatedEvent,
)
from aiosendspin.server.roles.color.group import ColorGroupRole
from aiosendspin.server.roles.color.state import Color
from aiosendspin.server.roles.color.types import ColorRoleProtocol
from aiosendspin.server.roles.color.v1 import ColorV1Role
from aiosendspin.server.roles.registry import register_group_role, register_role

register_group_role("color", ColorGroupRole)
register_role("color@v1", lambda client: ColorV1Role(client=client))

__all__ = [
    "Color",
    "ColorClearedEvent",
    "ColorEvent",
    "ColorGroupRole",
    "ColorRoleProtocol",
    "ColorUpdatedEvent",
    "ColorV1Role",
]
