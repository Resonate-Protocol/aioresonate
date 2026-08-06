"""Shared audio primitives: PCM format descriptor and codec encoders/decoders.

This package is server- and transport-independent so the client SDK can encode
captured audio without pulling in the server stack. PyAV is imported lazily by
the codec implementations, so importing this package stays cheap and dependency
free until a codec is actually used.
"""

from __future__ import annotations

from .bridge import AsrcSourceBridge, SourceBridge
from .format import AudioFormat

__all__ = ["AsrcSourceBridge", "AudioFormat", "SourceBridge"]
