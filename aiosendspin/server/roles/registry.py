"""Role registry for server-side role factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from aiosendspin.server.roles.player import PlayerRole

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.roles.base import Role

RoleFactory = Callable[["SendspinClient"], "Role"]

ROLE_FACTORIES: dict[str, RoleFactory] = {
    "player@v1": lambda client: PlayerRole(client=client),
}


def register_role(role_id: str, factory: RoleFactory) -> None:
    """Register or replace a role factory for a versioned role ID."""
    ROLE_FACTORIES[role_id] = factory


def create_role(role_id: str, client: SendspinClient) -> Role | None:
    """Create a role instance for the given role ID, if registered."""
    factory = ROLE_FACTORIES.get(role_id)
    if factory is None:
        return None
    return factory(client)
