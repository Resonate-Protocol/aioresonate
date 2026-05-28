"""Binary packing helper for the visualizer@v1 wire.

Each visualizer binary message carries exactly one frame of data and
follows the shared layout `[type:1][ts:8][data]`. The single helper
emits the `[type][ts]` header plus the caller-supplied data bytes.
Per-type value clipping and invariants live at the call site in
`v1.py`.
"""

from __future__ import annotations

import struct

from aiosendspin.models.types import BinaryMessageType

# Bit 0 of a beat frame's flags byte indicates a downbeat.
FLAG_DOWNBEAT = 0b0000_0001


def pack_visualizer_frame(msg_type: BinaryMessageType, timestamp_us: int, payload: bytes) -> bytes:
    """Pack one visualizer binary: `[type:1][ts:8][payload]`."""
    return bytes((msg_type.value,)) + struct.pack(">q", timestamp_us) + payload
