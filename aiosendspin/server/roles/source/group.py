"""Group-level coordination for the source role.

A source client captures audio from a local input and streams it to the server.
The ``SourceGroupRole`` decodes those frames (per the format announced in
``client_stream/start``) and pushes the resulting PCM into the group's
``PushStream``, which resamples, encodes, and distributes it to the group's
players like any other audio.

Only one source feeds a group at a time: the first source to send audio becomes
the active source until it ends its stream or leaves the group.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

from aiosendspin.models.types import AudioCodec
from aiosendspin.server.audio import AudioFormat, _get_av
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.roles.base import GroupRole, Role
from aiosendspin.util import create_task

if TYPE_CHECKING:
    import av

    from aiosendspin.models.source import ClientStreamStartSource
    from aiosendspin.server.group import SendspinGroup
    from aiosendspin.server.push_stream import PushStream
    from aiosendspin.server.roles.source.v1 import SourceV1Role

logger = logging.getLogger(__name__)

_INGRESS_QUEUE_MAXSIZE = 128


def _get_np() -> ModuleType:
    """Import numpy lazily (part of the optional ``server`` extra)."""
    import numpy as np  # noqa: PLC0415

    return np


@dataclass(frozen=True, slots=True)
class SourceIngress:
    """A single captured audio frame awaiting decode and distribution."""

    role: SourceV1Role
    timestamp_us: int
    data: bytes


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


class SourceGroupRole(GroupRole):
    """Coordinates source clients feeding audio into the group."""

    role_family = "source"

    def __init__(self, group: SendspinGroup) -> None:
        """Initialize source coordination state for the group."""
        super().__init__(group)
        self._queue: asyncio.Queue[SourceIngress] = asyncio.Queue(maxsize=_INGRESS_QUEUE_MAXSIZE)
        self._worker_task: asyncio.Task[None] | None = None
        self._decoders: dict[str, SourceDecoder] = {}
        self._active_source_id: str | None = None
        self._push_stream: PushStream | None = None
        self._next_play_us: int | None = None

    # --- Membership / policy ---

    def on_member_join(self, role: Role) -> None:
        """Ask a newly connected source to start streaming (default server policy).

        The spec makes the server the sole initiator of source streaming; a source
        stays stopped until told to start. This reference policy starts a source as
        soon as it is available; a host application may override by managing
        start/stop itself.
        """
        if isinstance(role, _source_role_cls()):
            role.send_start_command()

    def on_member_leave(self, role: Role) -> None:
        """Clean up a departing source and stop the stream if it was active."""
        if not isinstance(role, _source_role_cls()):
            return
        self._decoders.pop(role.client_id, None)
        if role.client_id == self._active_source_id:
            self._stop_active_source()

    # --- Stream lifecycle (driven by SourceV1Role) ---

    def clear_decoder(self, role: SourceV1Role) -> None:
        """Drop a source's decoder so the next frame rebuilds it for the new format."""
        self._decoders.pop(role.client_id, None)

    def stop_source(self, role: SourceV1Role) -> None:
        """Handle a source ending its stream; stop the group stream if it was active."""
        self._decoders.pop(role.client_id, None)
        if role.client_id == self._active_source_id:
            self._stop_active_source()

    def enqueue(self, role: SourceV1Role, timestamp_us: int, data: bytes) -> None:
        """Queue a captured audio frame for decode and distribution."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = create_task(self._run_worker())
        try:
            self._queue.put_nowait(SourceIngress(role=role, timestamp_us=timestamp_us, data=data))
        except asyncio.QueueFull:
            logger.warning("Source ingress queue full; dropping captured frame")

    # --- Worker ---

    async def _run_worker(self) -> None:
        while True:
            ingress = await self._queue.get()
            try:
                await self._ingest_one(ingress)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to ingest source audio frame")
            finally:
                self._queue.task_done()

    async def _ingest_one(self, ingress: SourceIngress) -> None:
        role = ingress.role
        source_format = role.stream_source
        if source_format is None:
            return  # no active client_stream/start for this source

        # Only one source feeds the group at a time.
        if self._active_source_id is None:
            self._active_source_id = role.client_id
            self._next_play_us = None
        elif self._active_source_id != role.client_id:
            return

        decoder = self._decoders.get(role.client_id)
        if decoder is None:
            decoder = SourceDecoder(source_format)
            self._decoders[role.client_id] = decoder

        chunks = decoder.decode(ingress.data)
        if not chunks:
            return

        audio_format = decoder.audio_format
        frame_bytes = audio_format.channels * decoder.wire_bytes
        if self._next_play_us is None:
            self._next_play_us = ingress.timestamp_us

        for chunk in chunks:
            push_stream = self._ensure_push_stream()
            push_stream.prepare_audio(chunk, audio_format, channel_id=MAIN_CHANNEL)
            await push_stream.commit_audio(play_start_us=self._next_play_us)
            frame_count = len(chunk) // frame_bytes if frame_bytes else 0
            self._next_play_us += round(frame_count * 1_000_000 / audio_format.sample_rate)

    # --- Helpers ---

    def _ensure_push_stream(self) -> PushStream:
        if self._push_stream is None or self._push_stream.is_stopped:
            self._push_stream = self._group.start_stream()
        return self._push_stream

    def _stop_active_source(self) -> None:
        self._active_source_id = None
        self._next_play_us = None
        if self._push_stream is not None and not self._push_stream.is_stopped:
            self._group.stop_stream()
        self._push_stream = None


def _source_role_cls() -> type[SourceV1Role]:
    """Return the SourceV1Role class (imported lazily to avoid a cycle)."""
    from aiosendspin.server.roles.source.v1 import SourceV1Role  # noqa: PLC0415

    return SourceV1Role
