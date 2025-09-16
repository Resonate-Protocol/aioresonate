"""Models for the resonate audio protocol."""

from __future__ import annotations

from mashumaro.config import BaseConfig

__all__ = [
    "BINARY_HEADER_FORMAT",
    "BINARY_HEADER_SIZE",
    "BinaryHeader",
    "BinaryMessageType",
    "GroupStateType",
    "MediaCommand",
    "PlaybackStateType",
    "PlayerStateType",
    "RepeatMode",
    "controller",
    "core",
    "metadata",
    "pack_binary_header",
    "pack_binary_header_raw",
    "player",
    "types",
    "unpack_binary_header",
]
import struct
from dataclasses import dataclass
from typing import NamedTuple

from mashumaro.mixins.orjson import DataClassORJSONMixin
from mashumaro.types import Discriminator

from . import controller, core, metadata, player, types
from .types import (
    BinaryMessageType,
    GroupStateType,
    MediaCommand,
    PlaybackStateType,
    PlayerStateType,
    RepeatMode,
)

# Binary header (big-endian): message_type(1) + timestamp_us(8) = 9 bytes
BINARY_HEADER_FORMAT = ">BQ"
BINARY_HEADER_SIZE = struct.calcsize(BINARY_HEADER_FORMAT)


# Base message classes
@dataclass
class ClientMessage(DataClassORJSONMixin):
    """Base class for client messages."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclass
class ServerMessage(DataClassORJSONMixin):
    """Base class for server messages."""

    class Config(BaseConfig):
        """Config for parsing json messages."""

        discriminator = Discriminator(field="type", include_subtypes=True)


# Helpers for binary messages
class BinaryHeader(NamedTuple):
    """Header structure for binary messages."""

    message_type: int  # message type identifier (B - unsigned char)
    timestamp_us: int  # timestamp in microseconds (Q - unsigned long long)


def unpack_binary_header(data: bytes) -> BinaryHeader:
    """
    Unpack binary header from bytes.

    Args:
        data: First 9 bytes containing the binary header

    Returns:
        BinaryHeader with typed fields

    Raises:
        struct.error: If data is not exactly 9 bytes or format is invalid
    """
    if len(data) < BINARY_HEADER_SIZE:
        raise ValueError(f"Expected at least {BINARY_HEADER_SIZE} bytes, got {len(data)}")

    unpacked = struct.unpack(BINARY_HEADER_FORMAT, data[:BINARY_HEADER_SIZE])
    return BinaryHeader(message_type=unpacked[0], timestamp_us=unpacked[1])


def pack_binary_header(header: BinaryHeader) -> bytes:
    """
    Pack binary header into bytes.

    Args:
        header: BinaryHeader to pack

    Returns:
        9-byte packed binary header
    """
    return struct.pack(BINARY_HEADER_FORMAT, header.message_type, header.timestamp_us)


def pack_binary_header_raw(message_type: int, timestamp_us: int) -> bytes:
    """
    Pack binary header from raw values into bytes.

    Args:
        message_type: BinaryMessageType value
        timestamp_us: timestamp in microseconds
        size: size in bytes

    Returns:
        9-byte packed binary header
    """
    return struct.pack(BINARY_HEADER_FORMAT, message_type, timestamp_us)
