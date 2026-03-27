"""Visualizer role - client and group level."""

from aiosendspin.models.visualizer import ClientHelloVisualizerSupport
from aiosendspin.server.roles.registry import (
    RoleSupportSpec,
    register_group_role,
    register_role,
    register_role_support_spec,
)
from aiosendspin.server.roles.visualizer.group import VisualizerGroupRole
from aiosendspin.server.roles.visualizer.v1 import VisualizerV1Role

register_group_role("visualizer", VisualizerGroupRole)
register_role("visualizer@_draft_r1", lambda client: VisualizerV1Role(client=client))
register_role_support_spec(
    "visualizer",
    RoleSupportSpec(
        parse_support=ClientHelloVisualizerSupport.from_dict,
    ),
)

__all__ = ["VisualizerGroupRole", "VisualizerV1Role"]
