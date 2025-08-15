"""Models for the resonate audio protocol."""

__all__ = [
    "BinaryMessageType",
    "ClientMessages",
    "MediaCommand",
    "PlayerStateType",
    "RepeatMode",
    "ServerMessages",
    "client_messages",
    "server_messages",
    "types",
]


import struct
from dataclasses import dataclass
from typing import Any

from mashumaro.mixins.orjson import DataClassORJSONMixin

from . import client_messages, server_messages, types
from .client_messages import ClientMessages
from .server_messages import ServerMessages
from .types import BinaryMessageType, MediaCommand, PlayerStateType, RepeatMode


@dataclass
class Message(DataClassORJSONMixin):
    """Message type used by resonate."""

    type: str
    payload: dict[str, Any] | Any


# TODO: check this
BINARY_HEADER_FORMAT = ">BQI"
BINARY_HEADER_SIZE = struct.calcsize(BINARY_HEADER_FORMAT)
