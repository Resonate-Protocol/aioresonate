"""Server-side role negotiation helpers."""

from __future__ import annotations

from aiosendspin.models.types import role_family

from .registry import ROLE_FACTORIES


def negotiate_active_roles(client_supported_roles: list[str]) -> list[str]:
    """Negotiate active roles from the client-supported role list.

    For each role family, pick the first role in client order that is
    registered in ROLE_FACTORIES.
    """
    active: dict[str, str] = {}

    for client_role_id in client_supported_roles:
        family = role_family(client_role_id)
        if family in active:
            continue

        if client_role_id in ROLE_FACTORIES:
            active[family] = client_role_id

    return list(active.values())
