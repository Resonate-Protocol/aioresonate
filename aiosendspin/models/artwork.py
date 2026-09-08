"""
Artwork messages for the Sendspin protocol.

This module contains messages specific to clients with the artwork role, which
handle display of artwork images. Artwork clients receive images in their
preferred format and resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import SendspinConfig, SendspinModel
from .types import ArtworkSource, PictureFormat

# Pre-rename dimension keys, superseded by `width`/`height`.
_DIMENSION_ALIASES = {"media_width": "width", "media_height": "height"}


def _rewrite_legacy_dimensions(d: dict[str, Any]) -> dict[str, Any]:
    """Rewrite pre-rename dimension keys onto width/height, recording which were used."""
    normalized = dict(d)
    legacy_keys: list[str] = []
    for legacy_key, current_key in _DIMENSION_ALIASES.items():
        if legacy_key not in normalized:
            continue
        legacy_keys.append(legacy_key)
        value = normalized.pop(legacy_key)
        # Rewrite only when the client didn't also send the current key.
        if current_key not in normalized:
            normalized[current_key] = value
    # Always overwrite so a client cannot spoof the record via the wire.
    normalized["legacy_dimension_keys"] = legacy_keys or None
    return normalized


@dataclass
class ArtworkChannel(SendspinModel):
    """Configuration for a single artwork channel."""

    source: ArtworkSource
    """Artwork source type."""
    format: PictureFormat
    """Image format identifier."""
    width: int
    """Width in pixels of the delivered image."""
    height: int
    """Height in pixels of the delivered image."""
    legacy_dimension_keys: list[str] | None = None
    """Pre-rename dimension keys the parser rewrote, recorded for the server to flag.
    Not part of the wire schema (omitted when None)."""

    @classmethod
    def __pre_deserialize__(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Accept the pre-rename `media_width`/`media_height` spelling."""
        return _rewrite_legacy_dimensions(d)

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True


# Client -> Server: client/hello artwork support object
@dataclass
class ClientHelloArtworkSupport(SendspinModel):
    """Artwork support configuration - only if artwork role is set."""

    channels: list[ArtworkChannel]
    """List of supported artwork channels (length 1-4), array index is the channel number."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if not 1 <= len(self.channels) <= 4:
            raise ValueError(f"channels must have 1-4 elements, got {len(self.channels)}")


@dataclass
class StreamArtworkChannelConfig(SendspinModel):
    """Configuration for an artwork channel in stream/start."""

    source: ArtworkSource
    """Artwork source type."""
    format: PictureFormat
    """Format of the encoded image."""
    width: int
    """Width in pixels of the encoded image."""
    height: int
    """Height in pixels of the encoded image."""


# Server -> Client: stream/start artwork object
@dataclass
class StreamStartArtwork(SendspinModel):
    """
    Artwork object in stream/start message.

    Sent to clients with the artwork role.
    """

    channels: list[StreamArtworkChannelConfig]
    """Configuration for each active artwork channel, array index is the channel number."""


# Client -> Server: stream/request-format artwork object
@dataclass
class StreamRequestFormatArtwork(SendspinModel):
    """Request the server to change artwork format for a specific channel."""

    channel: int
    """Channel number (0-3) corresponding to the channel index declared in artwork client/hello."""
    source: ArtworkSource | None = None
    """Artwork source type."""
    format: PictureFormat | None = None
    """Requested image format identifier."""
    width: int | None = None
    """Requested width in pixels."""
    height: int | None = None
    """Requested height in pixels."""
    legacy_dimension_keys: list[str] | None = None
    """Pre-rename dimension keys the parser rewrote, recorded for the role to flag.
    Not part of the wire schema (omitted when None)."""

    @classmethod
    def __pre_deserialize__(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Accept the pre-rename `media_width`/`media_height` spelling."""
        return _rewrite_legacy_dimensions(d)

    def __post_init__(self) -> None:
        """Validate field values."""
        if not 0 <= self.channel <= 3:
            raise ValueError(f"channel must be 0-3, got {self.channel}")

    class Config(SendspinConfig):
        """Config for parsing json messages."""

        omit_none = True
