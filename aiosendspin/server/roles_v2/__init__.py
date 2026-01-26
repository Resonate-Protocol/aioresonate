"""Role implementations for connection-specific behavior (v2).

This package contains the new simplified role implementations:
- Base role classes (Role ABC, dataclasses)
- Specific role implementations (PlayerRole, etc.)

This is the v2 implementation that will replace the legacy roles.py in Phase 5.
Roles encapsulate per-connection behavior for different client capabilities.
"""

from aiosendspin.server.roles_v2.base import (
    AudioChunk,
    AudioRequirements,
    Role,
    StreamRequirements,
)
from aiosendspin.server.roles_v2.player_v1 import PlayerRole

__all__ = [
    "AudioChunk",
    "AudioRequirements",
    "PlayerRole",
    "Role",
    "StreamRequirements",
]
