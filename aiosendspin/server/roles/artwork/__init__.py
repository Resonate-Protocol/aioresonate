"""Artwork role - client and group level."""

from aiosendspin.server.roles.artwork.events import (
    ArtworkClearedEvent,
    ArtworkEvent,
    ArtworkUpdatedEvent,
)
from aiosendspin.server.roles.artwork.group import ArtworkGroupRole
from aiosendspin.server.roles.artwork.types import ArtworkRoleProtocol
from aiosendspin.server.roles.artwork.v1 import ArtworkV1Role
from aiosendspin.server.roles.registry import register_group_role, register_role

register_group_role("artwork", lambda group: ArtworkGroupRole(group))
register_role("artwork@v1", lambda client: ArtworkV1Role(client=client))

__all__ = [
    "ArtworkClearedEvent",
    "ArtworkEvent",
    "ArtworkGroupRole",
    "ArtworkRoleProtocol",
    "ArtworkUpdatedEvent",
    "ArtworkV1Role",
]
