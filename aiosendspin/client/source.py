"""Client-side source capture: encode local PCM and stream it to the server.

High-level helper for the source role. The caller feeds raw PCM matching
:attr:`SourceCapture.audio_format`; the SDK encodes to the chosen codec, builds
the codec header, stamps each frame in the server time domain, and streams
type-12 binary chunks.
"""

from __future__ import annotations

import base64
from collections import deque
from typing import TYPE_CHECKING

from aiosendspin.audio.codecs import create_encoder
from aiosendspin.audio.format import AudioFormat, _convert_s24_to_s32
from aiosendspin.models.types import AudioCodec

if TYPE_CHECKING:
    from aiosendspin.models.player import SupportedAudioFormat

    from .client import SendspinClient
    from .connection import SendspinConnection


class SourceCapture:
    """Encode and stream local PCM to the server for one input stream lifetime."""

    def __init__(
        self,
        client: SendspinClient,
        connection: SendspinConnection,
        audio_format: SupportedAudioFormat,
    ) -> None:
        """Create a capture streaming ``audio_format`` (codec + PCM shape)."""
        self._client = client
        self._connection = connection
        self._codec = audio_format.codec
        # The Opus encoder consumes s16 regardless of the declared depth, so any other
        # depth would be fed at the wrong stride and stream garbage.
        if audio_format.codec is AudioCodec.OPUS and audio_format.bit_depth != 16:
            msg = f"Opus capture requires 16-bit PCM, got {audio_format.bit_depth}-bit"
            raise ValueError(msg)
        self._format = AudioFormat(
            sample_rate=audio_format.sample_rate,
            bit_depth=audio_format.bit_depth,
            channels=audio_format.channels,
        )
        wire_bytes, _, _, _ = self._format.resolve_av_format()
        self._frame_stride = wire_bytes * self._format.channels
        self._encoder = create_encoder(
            self._codec.value,
            sample_rate=audio_format.sample_rate,
            bit_depth=audio_format.bit_depth,
            channels=audio_format.channels,
        )
        self._started = False
        self._capture_spans: deque[tuple[int, int]] = deque()

    @property
    def audio_format(self) -> AudioFormat:
        """PCM format the caller must feed (no client-side resampling)."""
        return self._format

    @property
    def codec(self) -> AudioCodec:
        """Codec being streamed."""
        return self._codec

    async def start(self) -> None:
        """Send ``client_stream/start`` to begin the capture stream."""
        if self._started:
            if self._connection.is_source_stream_active():
                return
            self._encoder.reset()
            self._capture_spans.clear()
            self._started = False
        if not self._connection.is_time_synchronized():
            raise RuntimeError("Source capture requires a synchronized clock")
        header = self._encoder.get_codec_header()
        header_b64 = base64.b64encode(header).decode("ascii") if header else None
        await self._connection.send_client_stream_start(
            codec=self._codec,
            sample_rate=self._format.sample_rate,
            channels=self._format.channels,
            bit_depth=self._format.bit_depth,
            codec_header=header_b64,
        )
        self._started = True

    async def feed(self, pcm: bytes, capture_timestamp_us: int | None = None) -> None:
        """Encode and stream PCM continuously.

        ``capture_timestamp_us`` is the client-clock time of the first sample in
        ``pcm``. Each emitted frame is converted independently to server time.
        """
        if not self._started:
            raise RuntimeError("SourceCapture.start() must be called before feed()")
        if not pcm:
            return
        anchor = capture_timestamp_us if capture_timestamp_us is not None else self._client.now_us()
        if len(pcm) % self._frame_stride:
            raise ValueError("pcm length must be a whole number of frames")
        self._capture_spans.append((len(pcm) // self._frame_stride, anchor))
        encoder_pcm = (
            _convert_s24_to_s32(pcm)
            if self._codec is AudioCodec.FLAC and self._format.bit_depth == 24
            else pcm
        )
        for frame, _frame_duration_us in self._encoder.process(encoder_pcm, anchor, 0):
            timestamp_us = self._connection.compute_source_timestamp(
                self._next_capture_timestamp() - self._encoder.lookahead_us
            )
            await self._connection.send_source_chunk(frame, timestamp_us=timestamp_us)
            self._consume_capture_samples(self._encoder.frame_samples)

    async def stop(self) -> None:
        """Flush the encoder and end the input stream."""
        if not self._started:
            return
        for frame, _frame_duration_us in self._encoder.flush():
            timestamp_us = self._connection.compute_source_timestamp(
                self._next_capture_timestamp() - self._encoder.lookahead_us
            )
            await self._connection.send_source_chunk(frame, timestamp_us=timestamp_us)
            self._consume_capture_samples(self._encoder.frame_samples)
        await self._connection.send_client_stream_end()
        self._encoder.reset()
        self._capture_spans.clear()
        self._started = False

    def _next_capture_timestamp(self) -> int:
        if self._capture_spans:
            return self._capture_spans[0][1]
        return self._client.now_us()

    def _consume_capture_samples(self, samples: int) -> None:
        while samples > 0 and self._capture_spans:
            span_samples, timestamp_us = self._capture_spans.popleft()
            if samples >= span_samples:
                samples -= span_samples
                continue
            advanced_us = samples * 1_000_000 // self._format.sample_rate
            self._capture_spans.appendleft((span_samples - samples, timestamp_us + advanced_us))
            return
