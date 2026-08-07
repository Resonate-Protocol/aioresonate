"""Decoding of captured source audio into PCM.

A source announces its input format in ``client_stream/start`` and then streams
encoded frames. ``SourceDecoder`` turns those frames into interleaved PCM in the
announced format, which the group role hands to the host application.
"""

from __future__ import annotations

import base64
from types import ModuleType
from typing import TYPE_CHECKING

from aiosendspin.models.types import AudioCodec
from aiosendspin.server.audio import AudioFormat, _get_av

if TYPE_CHECKING:
    import av

    from aiosendspin.models.source import ClientStreamStartSource


def _get_np() -> ModuleType:
    """Import numpy lazily (part of the optional ``server`` extra)."""
    import numpy as np  # noqa: PLC0415

    return np


class SourceDecoder:
    """Decode a source's captured audio frames into PCM matching its declared format."""

    def __init__(self, source: ClientStreamStartSource) -> None:
        """Create a decoder for the given announced stream format."""
        self._audio_format = AudioFormat(
            sample_rate=source.sample_rate,
            bit_depth=source.bit_depth,
            channels=source.channels,
        )
        self._wire_bytes, av_format, layout, self._av_bytes = self._audio_format.resolve_av_format()
        self._decoder: av.AudioCodecContext | None = None
        self._resampler: av.AudioResampler | None = None
        if source.codec == AudioCodec.PCM:
            return

        av_mod = _get_av()
        codec_name = "libopus" if source.codec == AudioCodec.OPUS else source.codec.value
        decoder = av_mod.AudioCodecContext.create(codec_name, "r")
        if source.codec_header:
            decoder.extradata = base64.b64decode(source.codec_header)
        self._decoder = decoder
        self._resampler = av_mod.AudioResampler(
            format=av_format, layout=layout, rate=source.sample_rate
        )

    @property
    def audio_format(self) -> AudioFormat:
        """Return the PCM format this decoder emits."""
        return self._audio_format

    @property
    def wire_bytes(self) -> int:
        """Return the number of wire bytes per sample of the emitted PCM."""
        return self._wire_bytes

    @property
    def frame_bytes(self) -> int:
        """Return the number of bytes per PCM frame (one sample across all channels)."""
        return self._audio_format.channels * self._wire_bytes

    def duration_us(self, pcm: bytes) -> int:
        """Return the playback duration of a PCM chunk in microseconds."""
        frame_bytes = self.frame_bytes
        if frame_bytes <= 0:
            return 0
        frames = len(pcm) // frame_bytes
        return round(frames * 1_000_000 / self._audio_format.sample_rate)

    def decode(self, data: bytes) -> list[bytes]:
        """Decode one captured frame into a list of interleaved PCM chunks."""
        if self._decoder is None or self._resampler is None:
            # PCM: the client already sends interleaved wire PCM.
            return [data] if data else []

        av_mod = _get_av()
        chunks: list[bytes] = []
        for frame in self._decoder.decode(av_mod.Packet(data)):
            chunks.extend(self._frame_to_pcm(out) for out in self._resampler.resample(frame))
        return chunks

    def _frame_to_pcm(self, frame: av.AudioFrame) -> bytes:
        """Convert a resampled (packed) PyAV frame into wire PCM bytes."""
        # After resampling to a packed (non-planar) format, all data is in planes[0].
        channels = self._audio_format.channels
        raw = bytes(frame.planes[0])[: frame.samples * channels * self._av_bytes]
        if self._wire_bytes == self._av_bytes:
            return raw
        # 24-bit wire format: PyAV produces s32 (4 bytes); pack to 3 little-endian bytes
        # by dropping the least-significant byte of each sample.
        np = _get_np()
        as_bytes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, self._av_bytes)
        return bytes(as_bytes[:, self._av_bytes - self._wire_bytes :].tobytes())
