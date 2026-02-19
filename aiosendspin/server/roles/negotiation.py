"""Server-side role negotiation helpers."""

from __future__ import annotations

from aiosendspin.models.types import role_family

from .registry import ROLE_FACTORIES

# Canonical built-in role versions supported by this server.
SUPPORTED_ROLE_VERSIONS: dict[str, str] = {
    "player": "player@v1",
    "controller": "controller@v1",
    "metadata": "metadata@v1",
    "artwork": "artwork@v1",
    "visualizer": "visualizer@v1",
}


def negotiate_active_roles(client_supported_roles: list[str]) -> list[str]:
    """Negotiate active roles from the client-supported role list.

    For each role family, pick the first role in client order that is
    negotiable by this server.

    A role is negotiable when either:
    - It matches this server's canonical built-in role version for that family.
    - It is registered in ROLE_FACTORIES (custom/server-extended roles).
    """
    active: dict[str, str] = {}

    for client_role_id in client_supported_roles:
        family = role_family(client_role_id)
        if family in active:
            continue

        server_role_id = SUPPORTED_ROLE_VERSIONS.get(family)
        if server_role_id is not None and client_role_id == server_role_id:
            active[family] = server_role_id
            continue

        if client_role_id in ROLE_FACTORIES:
            active[family] = client_role_id

    return list(active.values())

