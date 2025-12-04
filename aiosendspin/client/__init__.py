"""Public interface for the Sendspin client package."""

from .client import (
    AudioChunkCallback,
    DisconnectCallback,
    GroupUpdateCallback,
    MetadataCallback,
    PCMFormat,
    SendspinClient,
    ServerInfo,
    StreamEndCallback,
    StreamStartCallback,
)
from .time_sync import SendspinTimeFilter

__all__ = [
    "AudioChunkCallback",
    "DisconnectCallback",
    "GroupUpdateCallback",
    "MetadataCallback",
    "PCMFormat",
    "SendspinClient",
    "SendspinTimeFilter",
    "ServerInfo",
    "StreamEndCallback",
    "StreamStartCallback",
]
