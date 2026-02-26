"""Server-side role negotiation helpers."""

from __future__ import annotations

from aiosendspin.models.types import role_family

from .registry import ROLE_FACTORIES

# Server-defined role family activation order. Families listed here are
# connected first (in the order shown); any unlisted families follow in
# client-provided order.
_FAMILY_ORDER = {
    family: i
    for i, family in enumerate(
        [
            # Player must come before controller so that PlayerGroupRole
            # already contains the player when ControllerGroupRole
            # reads group volume during on_member_join().
            "player",
            "controller",
        ]
    )
}


def negotiate_active_roles(client_supported_roles: list[str]) -> list[str]:
    """Negotiate active roles from the client-supported role list.

    For each role family, pick the first role in client order that is
    registered in ROLE_FACTORIES. The result is sorted by the server-defined
    family activation order.
    """
    active: dict[str, str] = {}

    for client_role_id in client_supported_roles:
        family = role_family(client_role_id)
        if family in active:
            continue

        if client_role_id in ROLE_FACTORIES:
            active[family] = client_role_id

    # Sort by server-defined order: listed families first, then the rest.
    return sorted(
        active.values(), key=lambda rid: _FAMILY_ORDER.get(role_family(rid), len(_FAMILY_ORDER))
    )
