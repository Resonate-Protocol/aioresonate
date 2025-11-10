"""
Artwork messages for the Resonate protocol.

This module contains messages specific to clients with the artwork role, which
handle display of artwork images. Artwork clients receive images in their
preferred format and resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import ArtworkSource, PictureFormat


@dataclass
class ArtworkChannel(DataClassORJSONMixin):
    """Configuration for a single artwork channel."""

    source: ArtworkSource
    """Artwork source type."""
    format: PictureFormat
    """Image format identifier."""
    media_width: int
    """Max width in pixels."""
    media_height: int
    """Max height in pixels."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.media_width <= 0:
            raise ValueError(f"media_width must be positive, got {self.media_width}")
        if self.media_height <= 0:
            raise ValueError(f"media_height must be positive, got {self.media_height}")


# Client -> Server: client/hello artwork support object
@dataclass
class ClientHelloArtworkSupport(DataClassORJSONMixin):
    """Artwork support configuration - only if artwork role is set."""

    channels: list[ArtworkChannel]
    """List of supported artwork channels (length 1-4), array index is the channel number."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if not 1 <= len(self.channels) <= 4:
            raise ValueError(f"channels must have 1-4 elements, got {len(self.channels)}")


# Server -> Client: stream/start artwork object
@dataclass
class StreamStartArtwork(DataClassORJSONMixin):
    """
    Artwork object in stream/start message.

    Sent to clients with the artwork role.
    """

    art_format: PictureFormat
    """Format of the encoded image."""


# Server -> Client: stream/update artwork object
@dataclass
class StreamUpdateArtwork(DataClassORJSONMixin):
    """Artwork object in stream/update message with delta updates."""

    art_format: PictureFormat
    """Format of the encoded image."""
