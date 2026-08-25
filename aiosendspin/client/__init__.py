"""Public interface for the Sendspin client package."""

from .client import (
    AudioChunkCallback,
    DisconnectCallback,
    GroupUpdateCallback,
    MetadataCallback,
    ScheduledArtworkCallback,
    ScheduledColorCallback,
    ScheduledMetadataCallback,
    SendspinClient,
    StreamEndCallback,
    StreamStartCallback,
    VisualizerCallback,
)
from .listener import ClientListener
from .models import (
    SECRET_LOCATIONS,
    AudioFormat,
    PairingSupport,
    PCMFormat,
    PinDisplay,
    PinSpeaker,
    ServerInfo,
)
from .source import SourceCapture
from .time_sync import SendspinTimeFilter

__all__ = [
    "SECRET_LOCATIONS",
    "AudioChunkCallback",
    "AudioFormat",
    "ClientListener",
    "DisconnectCallback",
    "GroupUpdateCallback",
    "MetadataCallback",
    "PCMFormat",
    "PairingSupport",
    "PinDisplay",
    "PinSpeaker",
    "ScheduledArtworkCallback",
    "ScheduledColorCallback",
    "ScheduledMetadataCallback",
    "SendspinClient",
    "SendspinTimeFilter",
    "ServerInfo",
    "SourceCapture",
    "StreamEndCallback",
    "StreamStartCallback",
    "VisualizerCallback",
]
