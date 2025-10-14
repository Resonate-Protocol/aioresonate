"""Resonate: Python implementation of the Resonate Protocol."""

from __future__ import annotations

# Re-export client library for easy import
from aioresonate.client import (
    AudioChunkCallback,
    GroupUpdateCallback,
    MetadataCallback,
    PCMFormat,
    ResonateClient,
    ResonateTimeFilter,
    ServerInfo,
    StreamEndCallback,
    StreamStartCallback,
)

__all__ = [
    "AudioChunkCallback",
    "GroupUpdateCallback",
    "MetadataCallback",
    "PCMFormat",
    "ResonateClient",
    "ResonateTimeFilter",
    "ServerInfo",
    "StreamEndCallback",
    "StreamStartCallback",
]
