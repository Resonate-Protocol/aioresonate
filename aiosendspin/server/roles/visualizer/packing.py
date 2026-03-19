"""Binary packing helpers for draft visualizer role."""

from __future__ import annotations

import struct

import numpy as np

from aiosendspin.models.types import BinaryMessageType
from aiosendspin.models.visualizer import StreamStartVisualizer
from aiosendspin.server.roles.visualizer.features import ExtractedFrame


def pack_visualization_message(
    *,
    frames: list[ExtractedFrame],
    config: StreamStartVisualizer,
) -> bytes:
    """Pack a complete visualization data binary message (type 16).

    Wire format per spec:
        Byte 0: message type 16 (uint8)
        Byte 1: frame count (uint8)
        Remaining: frames [...] each: [timestamp:8][data per types]
    """
    if not frames:
        raise ValueError("cannot pack empty visualizer frame list")
    if len(frames) > 255:
        raise ValueError(f"max 255 frames per message, got {len(frames)}")

    output = bytearray()
    output.append(BinaryMessageType.VISUALIZATION_DATA.value)
    output.append(len(frames))

    for frame in frames:
        output.extend(struct.pack(">q", frame.timestamp_us))
        for typed in config.types:
            if typed == "loudness":
                value = 0 if frame.loudness is None else frame.loudness
                output.extend(struct.pack(">H", int(np.clip(value, 0, 65535))))
            elif typed == "f_peak":
                value = 0 if frame.f_peak is None else frame.f_peak
                output.extend(struct.pack(">H", int(np.clip(value, 0, 65535))))
            elif typed == "spectrum":
                if frame.spectrum is None:
                    if config.spectrum is None:
                        raise ValueError("spectrum in config.types but config.spectrum is None")
                    zeros = np.zeros(config.spectrum.n_disp_bins, dtype=np.uint16)
                    output.extend(zeros.astype(">u2").tobytes())
                else:
                    output.extend(frame.spectrum.astype(">u2", copy=False).tobytes())

    return bytes(output)
