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
from dataclasses import dataclass
from typing import Any

from mashumaro.mixins.orjson import DataClassORJSONMixin

from . import client_messages, server_messages, types
from .types import BinaryMessageType, MediaCommand, PlayerStateType, RepeatMode


@dataclass
class Message(DataClassORJSONMixin):
    """Message type used by resonate."""

    type: str
    payload: dict[str, Any] | Any


# TODO: check this
BINARY_HEADER_FORMAT = ">BQI"
BINARY_HEADER_SIZE = struct.calcsize(BINARY_HEADER_FORMAT)
