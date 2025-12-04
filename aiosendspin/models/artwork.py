"""
Artwork messages for the Sendspin protocol.

This module contains messages specific to clients with the artwork role, which
handle display of artwork images. Artwork clients receive images in their
preferred format and resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

from mashumaro.config import BaseConfig
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


@dataclass
class StreamArtworkChannelConfig(DataClassORJSONMixin):
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
class StreamStartArtwork(DataClassORJSONMixin):
    """
    Artwork object in stream/start message.

    Sent to clients with the artwork role.
    """

    channels: list[StreamArtworkChannelConfig]
    """Configuration for each active artwork channel, array index is the channel number."""


# Client -> Server: stream/request-format artwork object
@dataclass
class StreamRequestFormatArtwork(DataClassORJSONMixin):
    """Request the server to change artwork format for a specific channel."""

    channel: int
    """Channel number (0-3) corresponding to the channel index declared in artwork client/hello."""
    source: ArtworkSource | None = None
    """Artwork source type."""
    format: PictureFormat | None = None
    """Requested image format identifier."""
    media_width: int | None = None
    """Requested max width in pixels."""
    media_height: int | None = None
    """Requested max height in pixels."""

    def __post_init__(self) -> None:
        """Validate field values."""
        if not 0 <= self.channel <= 3:
            raise ValueError(f"channel must be 0-3, got {self.channel}")

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_none = True
