"""GroupRole registry and factory helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from aiosendspin.server.roles.base import GroupRole

if TYPE_CHECKING:
    from aiosendspin.server.group import SendspinGroup

GROUP_ROLE_FACTORIES: dict[str, Callable[[SendspinGroup], GroupRole]] = {}


def register_group_role(role_family: str, factory: Callable[[SendspinGroup], GroupRole]) -> None:
    """Register a group role factory for a role family."""
    GROUP_ROLE_FACTORIES[role_family] = factory


def create_group_roles(group: SendspinGroup) -> dict[str, GroupRole]:
    """Create group roles for a new group from registered factories."""
    return {family: factory(group) for family, factory in GROUP_ROLE_FACTORIES.items()}
