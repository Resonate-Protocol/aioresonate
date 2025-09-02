"""Models for the resonate audio protocol."""

from __future__ import annotations

__all__ = [
    "BINARY_HEADER_FORMAT",
    "BINARY_HEADER_SIZE",
    "BinaryHeader",
    "BinaryMessageType",
    "MediaCommand",
    "PlayerStateType",
    "RepeatMode",
    "client_messages",
    "pack_binary_header",
    "pack_binary_header_raw",
    "server_messages",
    "types",
    "unpack_binary_header",
]
import struct
from typing import NamedTuple

from . import client_messages, server_messages, types
from .types import BinaryMessageType, MediaCommand, PlayerStateType, RepeatMode

# Binary header (big-endian): message_type(1) + timestamp_us(8) + size(4) = 13 bytes
BINARY_HEADER_FORMAT = ">BQI"
BINARY_HEADER_SIZE = struct.calcsize(BINARY_HEADER_FORMAT)


class BinaryHeader(NamedTuple):
    """Binary header structure for audio chunks."""

    message_type: int  # message type identifier (B - unsigned char)
    timestamp_us: int  # timestamp in microseconds (Q - unsigned long long)
    size: int  # payload size in bytes (I - unsigned int)


def unpack_binary_header(data: bytes) -> BinaryHeader:
    """
    Unpack binary header from bytes.

    Args:
        data: First 13 bytes containing the binary header

    Returns:
        BinaryHeader with typed fields

    Raises:
        struct.error: If data is not exactly 13 bytes or format is invalid
    """
    if len(data) < BINARY_HEADER_SIZE:
        raise ValueError(f"Expected at least {BINARY_HEADER_SIZE} bytes, got {len(data)}")

    unpacked = struct.unpack(BINARY_HEADER_FORMAT, data[:BINARY_HEADER_SIZE])
    return BinaryHeader(message_type=unpacked[0], timestamp_us=unpacked[1], size=unpacked[2])


def pack_binary_header(header: BinaryHeader) -> bytes:
    """
    Pack binary header into bytes.

    Args:
        header: BinaryHeader to pack

    Returns:
        13-byte packed binary header
    """
    return struct.pack(BINARY_HEADER_FORMAT, header.message_type, header.timestamp_us, header.size)


def pack_binary_header_raw(message_type: int, timestamp_us: int, size: int) -> bytes:
    """
    Pack binary header from raw values into bytes.

    Args:
        message_type: BinaryMessageType value
        timestamp_us: timestamp in microseconds
        size: size in bytes

    Returns:
        13-byte packed binary header
    """
    return struct.pack(BINARY_HEADER_FORMAT, message_type, timestamp_us, size)
