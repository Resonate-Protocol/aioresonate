"""Streaming helpers."""

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import av

from aioresonate.models import BinaryMessageType, pack_binary_header_raw

if TYPE_CHECKING:
    from aioresonate.server.client import ResonateClient

from .group import AudioFormat

BUFFER_HIGH_WATERMARK_RATIO = 0.9


def build_flac_stream_header(extradata: bytes) -> bytes:
    """Return a complete FLAC stream header for encoder ``extradata``."""
    if not extradata:
        return extradata
    return b"fLaC\x80" + len(extradata).to_bytes(3, "big") + extradata


def _samples_to_microseconds(sample_count: int, sample_rate: int) -> int:
    """Convert a number of samples to microseconds using ``sample_rate``."""
    return int(sample_count * 1_000_000 / sample_rate)


class _BufferedChunk(NamedTuple):
    """Represents compressed audio bytes scheduled for playback."""

    end_time_us: int
    """Absolute timestamp when these bytes should be fully consumed."""
    byte_count: int
    """Compressed byte count occupying the device buffer."""


class _BufferTracker:
    """Track buffered compressed audio and apply backpressure when needed."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        logger: logging.Logger,
        client_id: str,
        capacity_bytes: int,
    ) -> None:
        self._loop = loop
        self._logger = logger
        self.client_id = client_id
        self.capacity_bytes = capacity_bytes
        self.high_water_bytes = max(1, int(capacity_bytes * BUFFER_HIGH_WATERMARK_RATIO))
        self.buffered_chunks: deque[_BufferedChunk] = deque()
        self.buffered_bytes = 0
        self.max_usage_bytes = 0

    def prune_consumed(self, now_us: int | None = None) -> int:
        """Drop finished chunks and return the timestamp used for the calculation."""
        if now_us is None:
            now_us = int(self._loop.time() * 1_000_000)
        while self.buffered_chunks and self.buffered_chunks[0].end_time_us <= now_us:
            self.buffered_bytes -= self.buffered_chunks.popleft().byte_count
        self.buffered_bytes = max(self.buffered_bytes, 0)
        return now_us

    async def wait_for_capacity(self, bytes_needed: int, expected_end_us: int) -> None:
        """Block until the device buffer can accept ``bytes_needed`` more bytes."""
        if bytes_needed <= 0:
            return
        if bytes_needed >= self.capacity_bytes:
            self._logger.warning(
                "Chunk size %s exceeds reported buffer capacity %s for client %s",
                bytes_needed,
                self.capacity_bytes,
                self.client_id,
            )
            return

        while True:
            now_us = self.prune_consumed()
            projected_usage = self.buffered_bytes + bytes_needed
            if projected_usage <= self.capacity_bytes:
                # Returning here keeps the producer running because we are below capacity.
                return

            sleep_target_us = (
                self.buffered_chunks[0].end_time_us if self.buffered_chunks else expected_end_us
            )
            sleep_us = sleep_target_us - now_us
            if sleep_us <= 0:
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(sleep_us / 1_000_000)

    def register(self, end_time_us: int, byte_count: int) -> None:
        """Record bytes added to the buffer finishing at ``end_time_us``."""
        if byte_count <= 0:
            return
        self.buffered_chunks.append(_BufferedChunk(end_time_us, byte_count))
        self.buffered_bytes += byte_count
        self.max_usage_bytes = max(self.max_usage_bytes, self.buffered_bytes)


class _DirectStreamContext:
    """State container used while streaming audio directly to a client."""

    def __init__(
        self,
        *,
        client: "ResonateClient",
        audio_format: AudioFormat,
        input_audio_format: str,
        input_audio_layout: str,
        samples_per_chunk: int,
        buffer_tracker: _BufferTracker,
        play_start_time_us: int,
        encoder: av.AudioCodecContext | None,
        frame_stride_bytes: int,
        send_pcm: Callable[[bytes, int], None],
    ) -> None:
        self.client = client
        self.audio_format = audio_format
        self.input_audio_format = input_audio_format
        self.input_audio_layout = input_audio_layout
        self.samples_per_chunk = samples_per_chunk
        self._send_pcm = send_pcm
        self.buffer_tracker = buffer_tracker
        self.play_start_time_us = play_start_time_us
        self.encoder = encoder
        self.frame_stride_bytes = frame_stride_bytes
        self.samples_enqueued_total = 0
        self.samples_sent_total = 0
        self.pcm_bytes_consumed = 0
        self.compressed_bytes_sent = 0

    @property
    def sample_rate(self) -> int:
        return self.audio_format.sample_rate

    def _timeline_us(self, samples: int) -> int:
        """Translate ``samples`` from stream start into an absolute timestamp."""
        return self.play_start_time_us + _samples_to_microseconds(samples, self.sample_rate)

    async def send_pcm_chunk(self, chunk: bytes, sample_count: int) -> None:
        start_us = self._timeline_us(self.samples_sent_total)
        end_us = self._timeline_us(self.samples_sent_total + sample_count)
        await self.buffer_tracker.wait_for_capacity(len(chunk), end_us)
        self._send_pcm(chunk, start_us)
        self.buffer_tracker.register(end_us, len(chunk))
        self.samples_sent_total += sample_count
        self.compressed_bytes_sent += len(chunk)

    async def send_encoded_chunk(self, chunk: bytes, sample_count: int) -> None:
        encoder = self.encoder
        if encoder is None:
            raise RuntimeError("send_encoded_chunk called without encoder")

        frame = av.AudioFrame(
            format=self.input_audio_format,
            layout=self.input_audio_layout,
            samples=sample_count,
        )
        frame.sample_rate = encoder.sample_rate
        frame.planes[0].update(chunk)

        self.samples_enqueued_total += sample_count
        packets = encoder.encode(frame)

        if not packets:
            return

        for packet in packets:
            available_samples = max(self.samples_enqueued_total - self.samples_sent_total, 0)
            packet_samples = packet.duration or available_samples
            if available_samples and packet_samples > available_samples:
                packet_samples = available_samples
            if packet_samples <= 0:
                packet_samples = packet.duration or self.samples_per_chunk

            payload = bytes(packet)
            start_us = self._timeline_us(self.samples_sent_total)
            end_us = self._timeline_us(self.samples_sent_total + packet_samples)
            await self.buffer_tracker.wait_for_capacity(len(payload), end_us)
            header = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, start_us)
            self.client.send_message(header + payload)
            self.buffer_tracker.register(end_us, len(payload))
            self.compressed_bytes_sent += len(payload)
            self.samples_sent_total += packet_samples

    async def flush_encoder(self) -> None:
        encoder = self.encoder
        if encoder is None:
            return

        for packet in encoder.encode(None):
            available_samples = max(self.samples_enqueued_total - self.samples_sent_total, 0)
            packet_samples = packet.duration or available_samples or self.samples_per_chunk
            if available_samples and packet_samples > available_samples:
                packet_samples = available_samples

            payload = bytes(packet)
            start_us = self._timeline_us(self.samples_sent_total)
            end_us = self._timeline_us(self.samples_sent_total + packet_samples)
            await self.buffer_tracker.wait_for_capacity(len(payload), end_us)
            header = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, start_us)
            self.client.send_message(header + payload)
            self.buffer_tracker.register(end_us, len(payload))
            self.compressed_bytes_sent += len(payload)
            self.samples_sent_total += packet_samples

    async def drain_ready_chunks(self, input_buffer: bytearray, *, force_flush: bool) -> None:
        """Send chunks while avoiding buffer overflows unless flushing remaining data."""
        while True:
            available_samples = len(input_buffer) // self.frame_stride_bytes
            if available_samples <= 0:
                return

            if not force_flush and available_samples < self.samples_per_chunk:
                return
            sample_count = (
                available_samples if force_flush else min(available_samples, self.samples_per_chunk)
            )
            chunk_size = sample_count * self.frame_stride_bytes
            if chunk_size <= 0:
                return

            chunk = bytes(input_buffer[:chunk_size])
            del input_buffer[:chunk_size]
            self.pcm_bytes_consumed += chunk_size

            if self.encoder is None:
                await self.send_pcm_chunk(chunk, sample_count)
            else:
                await self.send_encoded_chunk(chunk, sample_count)

    async def _skip_initial_bytes(
        self, audio_stream: AsyncGenerator[bytes, None], bytes_to_skip_total: int
    ) -> bytearray:
        """Consume chunks until ``bytes_to_skip_total`` have been dropped.

        Returns the first buffered bytes that should be played after the skipped
        portion (possibly empty when the stream ended).
        """
        pending = bytearray()
        if bytes_to_skip_total <= 0:
            return pending
        async for buf in audio_stream:
            if not buf:
                continue
            if bytes_to_skip_total >= len(buf):
                bytes_to_skip_total -= len(buf)
                continue
            pending.extend(buf[bytes_to_skip_total:])
            bytes_to_skip_total = 0
            break
        while bytes_to_skip_total > 0:
            try:
                buf = await audio_stream.__anext__()
            except StopAsyncIteration:
                return pending
            if bytes_to_skip_total >= len(buf):
                bytes_to_skip_total -= len(buf)
                continue
            pending.extend(buf[bytes_to_skip_total:])
            bytes_to_skip_total = 0
        return pending
