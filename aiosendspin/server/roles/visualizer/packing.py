"""Binary packing helpers for draft visualizer role."""

from __future__ import annotations

import struct

import numpy as np

from aiosendspin.models.visualizer import StreamStartVisualizer
from aiosendspin.server.roles.visualizer.features import VisualizerFrame


def pack_visualizer_frames(
    *,
    frames: list[VisualizerFrame],
    config: StreamStartVisualizer,
) -> bytes:
    """Pack visualizer frames payload as [frame_count, frames...]."""
    if not frames:
        raise ValueError("cannot pack empty visualizer frame list")
    if len(frames) > 255:
        raise ValueError(f"max 255 frames per message, got {len(frames)}")

    output = bytearray()
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
                    assert config.spectrum is not None
                    zeros = np.zeros(config.spectrum.n_disp_bins, dtype=np.uint16)
                    output.extend(zeros.astype(">u2").tobytes())
                else:
                    output.extend(frame.spectrum.astype(">u2", copy=False).tobytes())

    return bytes(output)
