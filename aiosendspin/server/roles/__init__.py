"""Role implementations for connection-specific behavior.

This package contains the role implementations:
- Base role classes (Role ABC, dataclasses)
- Specific role implementations (PlayerRole, etc.)

Roles encapsulate per-connection behavior for different client capabilities.
"""

from aiosendspin.server.roles.base import (
    AudioChunk,
    AudioRequirements,
    GroupRole,
    Role,
    StreamRequirements,
)
from aiosendspin.server.roles.player import PlayerGroupRole, PlayerRole

__all__ = [
    "AudioChunk",
    "AudioRequirements",
    "GroupRole",
    "PlayerGroupRole",
    "PlayerRole",
    "Role",
    "StreamRequirements",
]
