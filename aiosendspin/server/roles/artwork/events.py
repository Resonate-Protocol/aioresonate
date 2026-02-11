"""Artwork group role events."""

from __future__ import annotations

from dataclasses import dataclass

from aiosendspin.models.types import ArtworkSource
from aiosendspin.server.events import GroupRoleEvent


class ArtworkEvent(GroupRoleEvent):
    """Base event type for artwork group role changes."""


@dataclass
class ArtworkUpdatedEvent(ArtworkEvent):
    """Artwork was set or updated for a source."""

    source: ArtworkSource
    timestamp_us: int
    width: int
    height: int


@dataclass
class ArtworkClearedEvent(ArtworkEvent):
    """Artwork was cleared for a source."""

    source: ArtworkSource
    timestamp_us: int
