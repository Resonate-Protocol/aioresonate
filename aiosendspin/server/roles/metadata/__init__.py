"""Metadata role - client and group level."""

from aiosendspin.server.roles.metadata.group import MetadataGroupRole
from aiosendspin.server.roles.metadata.v1 import MetadataRole
from aiosendspin.server.roles.registry import register_group_role, register_role

register_group_role("metadata", lambda group: MetadataGroupRole(group))
register_role("metadata@v1", lambda client: MetadataRole(client=client))

__all__ = ["MetadataGroupRole", "MetadataRole"]
