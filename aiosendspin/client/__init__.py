"""Public interface for the Sendspin client package."""

from .client import (
    AudioChunkCallback,
    AudioFormat,
    DisconnectCallback,
    GroupUpdateCallback,
    InputStreamRequestFormatCallback,
    MetadataCallback,
    PCMFormat,
    SendspinClient,
    ServerInfo,
    SourceCommandCallback,
    StreamEndCallback,
    StreamStartCallback,
)
from .listener import ClientListener
from .time_sync import SendspinTimeFilter

__all__ = [
    "AudioChunkCallback",
    "AudioFormat",
    "ClientListener",
    "DisconnectCallback",
    "GroupUpdateCallback",
    "InputStreamRequestFormatCallback",
    "MetadataCallback",
    "PCMFormat",
    "SendspinClient",
    "SendspinTimeFilter",
    "ServerInfo",
    "SourceCommandCallback",
    "StreamEndCallback",
    "StreamStartCallback",
]
