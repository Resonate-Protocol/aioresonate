"""SourceGroupRole - group-level source coordination."""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from aiosendspin.models.source import ControllerSourceItem
from aiosendspin.models.types import AudioCodec
from aiosendspin.server.audio import AudioFormat, _convert_s32_to_s24, _get_av
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.roles.base import GroupRole
from aiosendspin.util import create_task

if TYPE_CHECKING:
    from aiosendspin.server.group import SendspinGroup
    from aiosendspin.server.roles.base import Role
    from aiosendspin.server.roles.source.v1 import SourceV1Role

logger = logging.getLogger(__name__)


@dataclass
class SourceIngress:
    """Ingress metadata and payload for a source chunk."""

    role: SourceV1Role
    timestamp_us: int
    data: bytes


class SourceDecoder:
    """Decode compressed source frames into PCM bytes."""

    def __init__(
        self,
        *,
        codec: AudioCodec,
        sample_rate: int,
        channels: int,
        bit_depth: int,
        codec_header_b64: str | None,
    ) -> None:
        """Create a decoder for a specific source stream format."""
        codec_name = "opus" if codec == AudioCodec.OPUS else codec.value
        av = _get_av()
        self._decoder = av.AudioCodecContext.create(codec_name, "r")
        if codec_header_b64:
            self._decoder.extradata = base64.b64decode(codec_header_b64)
        self._decoder.open()
        av_format = "s16" if bit_depth == 16 else "s32"
        layout = "mono" if channels == 1 else "stereo"
        self._resampler = av.AudioResampler(format=av_format, layout=layout, rate=sample_rate)
        self._bit_depth = bit_depth
        self._channels = channels
        self._bytes_per_sample = 2 if bit_depth == 16 else 4

    def decode(self, data: bytes) -> list[bytes]:
        """Decode a single encoded payload into one or more PCM chunks."""
        av = _get_av()
        packet = av.Packet(data)
        output: list[bytes] = []
        for frame in self._decoder.decode(packet):
            for out_frame in self._resampler.resample(frame):
                expected = out_frame.samples * self._channels * self._bytes_per_sample
                pcm = bytes(out_frame.planes[0])[:expected]
                if self._bit_depth == 24:
                    pcm = _convert_s32_to_s24(pcm)
                output.append(pcm)
        return output


class SourceGroupRole(GroupRole):
    """Coordinates source roles inside a group."""

    role_family = "source"

    def __init__(self, group: SendspinGroup) -> None:
        """Initialize source group coordination state."""
        super().__init__(group)
        self._queue: asyncio.Queue[SourceIngress] = asyncio.Queue(maxsize=128)
        self._worker_task: asyncio.Task[None] | None = None
        self._active_source_id: str | None = None
        self._source_decoders: dict[str, SourceDecoder] = {}

    def on_member_join(self, _role: Role) -> None:
        """Refresh controller-visible source list when a source joins."""
        self.push_state()

    def on_member_leave(self, role: Role) -> None:
        """Cleanup source state and stop playback when active source leaves."""
        source_role = cast("SourceV1Role", role)
        client_id = source_role._client.client_id  # noqa: SLF001
        self._source_decoders.pop(client_id, None)
        was_active = self._active_source_id == client_id
        if was_active:
            self._active_source_id = None
            if self._group._push_stream is not None:  # noqa: SLF001
                create_task(self._group.stop())
        self.push_state()

    def list_sources(self) -> list[ControllerSourceItem]:
        """Return controller-facing source entries for all members."""
        items: list[ControllerSourceItem] = []
        for member in self._members:
            role = cast("SourceV1Role", member)
            items.append(
                ControllerSourceItem(
                    id=role._client.client_id,  # noqa: SLF001
                    name=role._client.name,  # noqa: SLF001
                    state=role.state,
                    signal=role.signal,
                    selected=role._client.client_id == self._active_source_id,  # noqa: SLF001
                    last_event=role.last_event,
                    last_event_ts_us=role.last_event_ts_us,
                )
            )
        return sorted(items, key=lambda x: x.id)

    def push_state(self) -> None:
        """Trigger a controller state refresh when source data changed."""
        controller_role = self._group.group_role("controller")
        if controller_role is None:
            return
        controller_push = getattr(controller_role, "push_state", None)
        if callable(controller_push):
            controller_push()

    def enqueue(self, role: SourceV1Role, timestamp_us: int, data: bytes) -> None:
        """Queue incoming source chunk for processing."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = create_task(self._run_worker())
        if self._queue.full():
            logger.warning("Dropping source chunk (queue full) from %s", role._client.client_id)  # noqa: SLF001
            return
        self._queue.put_nowait(SourceIngress(role=role, timestamp_us=timestamp_us, data=data))

    async def _run_worker(self) -> None:
        try:
            while True:
                ingress = await self._queue.get()
                try:
                    await self._ingest_one(ingress)
                except Exception:
                    logger.exception("Failed to ingest source chunk")
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            logger.debug("Source ingest worker cancelled")

    async def _ingest_one(self, ingress: SourceIngress) -> None:
        role = ingress.role
        stream_format = role.input_stream_format
        if stream_format is None:
            return
        audio_format = AudioFormat(
            sample_rate=stream_format.sample_rate,
            bit_depth=stream_format.bit_depth,
            channels=stream_format.channels,
        )
        chunks: list[bytes]
        if stream_format.codec == AudioCodec.PCM:
            chunks = [ingress.data]
        else:
            decoder = self._source_decoders.get(role._client.client_id)  # noqa: SLF001
            if decoder is None:
                decoder = SourceDecoder(
                    codec=stream_format.codec,
                    sample_rate=stream_format.sample_rate,
                    channels=stream_format.channels,
                    bit_depth=stream_format.bit_depth,
                    codec_header_b64=stream_format.codec_header,
                )
                self._source_decoders[role._client.client_id] = decoder  # noqa: SLF001
            chunks = decoder.decode(ingress.data)
            if not chunks:
                return

        stream = self._group._push_stream  # noqa: SLF001
        if stream is None or stream.is_stopped:
            stream = self._group.start_stream()
        if self._active_source_id != role._client.client_id:  # noqa: SLF001
            self._active_source_id = role._client.client_id  # noqa: SLF001
            self.push_state()

        bytes_per_sample = stream_format.bit_depth // 8
        play_start_us = ingress.timestamp_us
        for chunk in chunks:
            stream.prepare_audio(chunk, audio_format, channel_id=MAIN_CHANNEL)
            await stream.commit_audio(play_start_us=play_start_us)
            if bytes_per_sample > 0:
                frame_count = len(chunk) // (stream_format.channels * bytes_per_sample)
                if frame_count > 0:
                    play_start_us += round((frame_count * 1_000_000) / stream_format.sample_rate)

    def clear_decoder(self, role: SourceV1Role) -> None:
        """Drop decoder state when source format changed."""
        self._source_decoders.pop(role._client.client_id, None)  # noqa: SLF001

    def start_source(self, role: SourceV1Role) -> None:
        """Mark source as active for controller visibility."""
        if self._active_source_id != role._client.client_id:  # noqa: SLF001
            self._active_source_id = role._client.client_id  # noqa: SLF001
            self.push_state()

    async def stop_source(self, role: SourceV1Role) -> None:
        """Stop source playback for this group."""
        if self._active_source_id != role._client.client_id:  # noqa: SLF001
            return
        self._active_source_id = None
        if self._group._push_stream is not None:  # noqa: SLF001
            await self._group.stop()
        self.push_state()
