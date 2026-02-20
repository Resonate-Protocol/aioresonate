"""Metadata role - client and group level."""

from aiosendspin.server.roles.metadata.events import (
    MetadataClearedEvent,
    MetadataEvent,
    MetadataUpdatedEvent,
)
from aiosendspin.server.roles.metadata.group import MetadataGroupRole
from aiosendspin.server.roles.metadata.state import Metadata
from aiosendspin.server.roles.metadata.types import MetadataRoleProtocol
from aiosendspin.server.roles.metadata.v1 import MetadataV1Role
from aiosendspin.server.roles.registry import register_group_role, register_role

register_group_role("metadata", MetadataGroupRole)
register_role("metadata@v1", lambda client: MetadataV1Role(client=client))

__all__ = [
    "Metadata",
    "MetadataClearedEvent",
    "MetadataEvent",
    "MetadataGroupRole",
    "MetadataRoleProtocol",
    "MetadataUpdatedEvent",
    "MetadataV1Role",
]
