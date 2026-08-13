"""Public interface for the Sendspin client package."""

from .client import (
    ArtworkTimestampCallback,
    AudioChunkCallback,
    DisconnectCallback,
    EffectiveArtworkCallback,
    EffectiveColorCallback,
    EffectiveMetadataCallback,
    GroupUpdateCallback,
    MetadataCallback,
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
    "ArtworkTimestampCallback",
    "AudioChunkCallback",
    "AudioFormat",
    "ClientListener",
    "DisconnectCallback",
    "EffectiveArtworkCallback",
    "EffectiveColorCallback",
    "EffectiveMetadataCallback",
    "GroupUpdateCallback",
    "MetadataCallback",
    "PCMFormat",
    "PairingSupport",
    "PinDisplay",
    "PinSpeaker",
    "SendspinClient",
    "SendspinTimeFilter",
    "ServerInfo",
    "SourceCapture",
    "StreamEndCallback",
    "StreamStartCallback",
    "VisualizerCallback",
]
