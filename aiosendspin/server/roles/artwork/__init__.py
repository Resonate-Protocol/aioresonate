"""Artwork role - client and group level."""

from aiosendspin.server.roles.artwork.group import ArtworkGroupRole
from aiosendspin.server.roles.artwork.v1 import ArtworkRole
from aiosendspin.server.roles.registry import register_group_role, register_role

register_group_role("artwork", lambda group: ArtworkGroupRole(group))
register_role("artwork@v1", lambda client: ArtworkRole(client=client))

__all__ = ["ArtworkGroupRole", "ArtworkRole"]
