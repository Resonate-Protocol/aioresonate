"""Source role - client and group level."""

from aiosendspin.server.roles.registry import register_group_role, register_role
from aiosendspin.server.roles.source.group import SourceDecoder, SourceGroupRole, SourceIngress
from aiosendspin.server.roles.source.v1 import SourceRoleState, SourceV1Role

register_group_role("source", SourceGroupRole)
register_role("source@v1", lambda client: SourceV1Role(client=client))

__all__ = [
    "SourceDecoder",
    "SourceGroupRole",
    "SourceIngress",
    "SourceRoleState",
    "SourceV1Role",
]
