"""Role implementations for connection-specific behavior (v2).

This package contains the role implementations:
- Base role classes (Role ABC, dataclasses)
- Specific role implementations (PlayerRole, etc.)

Roles encapsulate per-connection behavior for different client capabilities.
"""

from aiosendspin.server.roles_v2.base import (
    AudioChunk,
    AudioRequirements,
    Role,
    StreamRequirements,
)
from aiosendspin.server.roles_v2.player_v1 import PlayerRole, PlayerSendState

__all__ = [
    "AudioChunk",
    "AudioRequirements",
    "PlayerRole",
    "PlayerSendState",
    "Role",
    "StreamRequirements",
]
