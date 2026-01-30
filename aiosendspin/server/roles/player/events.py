"""Player role events."""

from __future__ import annotations

from dataclasses import dataclass

from aiosendspin.server.events import ClientEvent


@dataclass
class VolumeChangedEvent(ClientEvent):
    """The volume or mute status of the player was changed."""

    volume: int
    muted: bool
