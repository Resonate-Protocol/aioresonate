"""Role implementations for connection-specific behavior.

This package contains the role implementations:
- Base role classes (Role ABC, dataclasses)
- Specific role implementations (PlayerRole, ControllerRole, etc.)

Roles encapsulate per-connection behavior for different client capabilities.
"""

# Import submodules to trigger auto-registration of roles
from aiosendspin.server.roles import (
    artwork,  # noqa: F401
    controller,  # noqa: F401
    metadata,  # noqa: F401
    player,  # noqa: F401
    visualizer,  # noqa: F401
)

# Re-export role classes for convenience
from aiosendspin.server.roles.artwork import ArtworkGroupRole, ArtworkRole
from aiosendspin.server.roles.base import (
    AudioChunk,
    AudioRequirements,
    GroupRole,
    Role,
    StreamRequirements,
)
from aiosendspin.server.roles.controller import (
    ControllerEvent,
    ControllerGroupRole,
    ControllerMuteEvent,
    ControllerNextEvent,
    ControllerPauseEvent,
    ControllerPlayEvent,
    ControllerPreviousEvent,
    ControllerRepeatEvent,
    ControllerRole,
    ControllerShuffleEvent,
    ControllerStopEvent,
    ControllerSwitchEvent,
    ControllerVolumeEvent,
)
from aiosendspin.server.roles.metadata import MetadataGroupRole, MetadataRole
from aiosendspin.server.roles.player import PlayerGroupRole, PlayerRole
from aiosendspin.server.roles.visualizer import VisualizerGroupRole, VisualizerRole

__all__ = [
    "ArtworkGroupRole",
    "ArtworkRole",
    "AudioChunk",
    "AudioRequirements",
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
    "GroupRole",
    "MetadataGroupRole",
    "MetadataRole",
    "PlayerGroupRole",
    "PlayerRole",
    "Role",
    "StreamRequirements",
    "VisualizerGroupRole",
    "VisualizerRole",
]
