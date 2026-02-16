"""Source role - client and group level."""

from aiosendspin.server.roles.registry import register_group_role, register_role
from aiosendspin.server.roles.source.group import SourceGroupRole
from aiosendspin.server.roles.source.v1 import SourceV1Role

register_group_role("source", lambda group: SourceGroupRole(group))
register_role("source@v1", lambda client: SourceV1Role(client=client))

__all__ = [
    "SourceGroupRole",
    "SourceV1Role",
]
