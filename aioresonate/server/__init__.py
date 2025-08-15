"""Resonate Server implementation to connect to and manage many Resonate Players.

ResonateServer is the core of the music listening experience, responsible for:
- Managing connected players
- Orchestrating synchronized grouped playback
"""

__all__ = [
    "PlayerGroup",
    "PlayerInstance",
    "ResonateServer",
]

from .group import PlayerGroup
from .instance import PlayerInstance
from .server import ResonateServer
