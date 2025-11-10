"""
Artwork messages for the Resonate protocol.

This module contains messages specific to clients with the artwork role, which
handle display of artwork images. Artwork clients receive images in their
preferred format and resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import PictureFormat


# Client -> Server: client/hello artwork support object
@dataclass
class ClientHelloArtworkSupport(DataClassORJSONMixin):
    """Artwork support configuration - only if artwork role is set."""

    support_picture_formats: list[str]
    """Supported media art image formats (empty array if no art desired)."""
    media_width: int | None = None
    """Max width in pixels (if only width set, scales preserving aspect ratio)."""
    media_height: int | None = None
    """Max height in pixels (if only height set, scales preserving aspect ratio)."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.media_width is not None and self.media_width <= 0:
            raise ValueError(f"media_width must be positive, got {self.media_width}")

        if self.media_height is not None and self.media_height <= 0:
            raise ValueError(f"media_height must be positive, got {self.media_height}")

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True


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
