"""Visualizer role - client and group level."""

from aiosendspin.server.roles.registry import register_group_role, register_role
from aiosendspin.server.roles.visualizer.group import VisualizerGroupRole
from aiosendspin.server.roles.visualizer.v1 import VisualizerV1Role

register_group_role("visualizer", lambda group: VisualizerGroupRole(group))
register_role("visualizer@v1", lambda client: VisualizerV1Role(client=client))

__all__ = ["VisualizerGroupRole", "VisualizerV1Role"]
