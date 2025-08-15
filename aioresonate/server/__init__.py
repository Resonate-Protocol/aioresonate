"""Resonate Server implementation to connect to and manage many Resonate Players.

ResonateServer is the core of the music listening experience, responsible for:
- Managing connected players
- Orchestrating synchronized grouped playback
"""

__all__ = [
    "AudioFormat",
    "PlayerAddedEvent",
    "PlayerGroup",
    "PlayerInstance",
    "PlayerInstanceEvent",
    "PlayerRemovedEvent",
    "ResonateEvent",
    "ResonateServer",
    "VolumeChangedEvent",
]

from .group import AudioFormat, PlayerGroup
from .instance import PlayerInstance, PlayerInstanceEvent, VolumeChangedEvent
from .server import PlayerAddedEvent, PlayerRemovedEvent, ResonateEvent, ResonateServer
