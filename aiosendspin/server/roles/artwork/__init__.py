"""Artwork role - client and group level."""

from aiosendspin.models.artwork import ClientHelloArtworkSupport
from aiosendspin.server.roles.artwork.events import (
    ArtworkClearedEvent,
    ArtworkEvent,
    ArtworkUpdatedEvent,
)
from aiosendspin.server.roles.artwork.group import ArtworkGroupRole
from aiosendspin.server.roles.artwork.types import ArtworkRoleProtocol
from aiosendspin.server.roles.artwork.v1 import ArtworkV1Role
from aiosendspin.server.roles.registry import (
    RoleSupportSpec,
    register_group_role,
    register_role,
    register_role_support_spec,
)

register_group_role("artwork", ArtworkGroupRole)
register_role("artwork@v1", lambda client: ArtworkV1Role(client=client))
register_role_support_spec(
    "artwork",
    RoleSupportSpec(
        parse_support=ClientHelloArtworkSupport.from_dict,
    ),
)

__all__ = [
    "ArtworkClearedEvent",
    "ArtworkEvent",
    "ArtworkGroupRole",
    "ArtworkRoleProtocol",
    "ArtworkUpdatedEvent",
    "ArtworkV1Role",
]
