"""Controller role - client and group level."""

from aiosendspin.server.roles.controller.events import (
    ControllerEvent,
    ControllerMuteEvent,
    ControllerNextEvent,
    ControllerPauseEvent,
    ControllerPlayEvent,
    ControllerPreviousEvent,
    ControllerRepeatEvent,
    ControllerShuffleEvent,
    ControllerStopEvent,
    ControllerSwitchEvent,
    ControllerVolumeEvent,
)
from aiosendspin.server.roles.controller.group import ControllerGroupRole
from aiosendspin.server.roles.controller.v1 import ControllerRole
from aiosendspin.server.roles.registry import register_group_role, register_role

register_group_role("controller", lambda group: ControllerGroupRole(group))
register_role("controller@v1", lambda client: ControllerRole(client=client))

__all__ = [
    "ControllerEvent",
    "ControllerGroupRole",
    "ControllerMuteEvent",
    "ControllerNextEvent",
    "ControllerPauseEvent",
    "ControllerPlayEvent",
    "ControllerPreviousEvent",
    "ControllerRepeatEvent",
    "ControllerRole",
    "ControllerShuffleEvent",
    "ControllerStopEvent",
    "ControllerSwitchEvent",
    "ControllerVolumeEvent",
]
