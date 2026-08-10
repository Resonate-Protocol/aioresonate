"""Public interface for the Sendspin client package."""

from .client import (
    AudioChunkCallback,
    DisconnectCallback,
    GroupUpdateCallback,
    MetadataCallback,
    SendspinClient,
    StreamEndCallback,
    StreamStartCallback,
    VisualizerCallback,
)
from .listener import ClientListener
from .models import AudioFormat, PairingSupport, PCMFormat, ServerInfo
from .source import SourceCapture
from .time_sync import SendspinTimeFilter

__all__ = [
    "AudioChunkCallback",
    "AudioFormat",
    "ClientListener",
    "DisconnectCallback",
    "GroupUpdateCallback",
    "MetadataCallback",
    "PCMFormat",
    "PairingSupport",
    "SendspinClient",
    "SendspinTimeFilter",
    "ServerInfo",
    "SourceCapture",
    "StreamEndCallback",
    "StreamStartCallback",
    "VisualizerCallback",
]
