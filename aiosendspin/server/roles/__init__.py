"""Role implementations for connection-specific behavior.

This package contains the role implementations:
- Base role classes (Role ABC, dataclasses)
- Specific role implementations (PlayerRole, etc.)

Roles encapsulate per-connection behavior for different client capabilities.
"""

from aiosendspin.server.roles.base import (
    AudioChunk,
    AudioRequirements,
    Role,
    StreamRequirements,
)
from aiosendspin.server.roles.player_v1 import PlayerRole

__all__ = [
    "AudioChunk",
    "AudioRequirements",
    "PlayerRole",
    "Role",
    "StreamRequirements",
]
