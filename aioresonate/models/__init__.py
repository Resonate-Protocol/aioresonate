"""Models for the resonate audio protocol."""

__all__ = [
    "BinaryMessageType",
    "MediaCommand",
    "PlayerStateType",
    "RepeatMode",
    "client_messages",
    "server_messages",
    "types",
]


import struct

from . import client_messages, server_messages, types
from .types import BinaryMessageType, MediaCommand, PlayerStateType, RepeatMode

# TODO: check this
BINARY_HEADER_FORMAT = ">BQI"
BINARY_HEADER_SIZE = struct.calcsize(BINARY_HEADER_FORMAT)
