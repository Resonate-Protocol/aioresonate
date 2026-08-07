"""Relay a source-role stream into a player group through a drift bridge."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING, Literal

from aiosendspin.audio import AsrcSourceBridge, AudioFormat, SourceBridge

if TYPE_CHECKING:
    from aiosendspin.server.group import SendspinGroup
    from aiosendspin.server.push_stream import PushStream
    from aiosendspin.server.roles.source import SourceStreamStartedEvent


DEFAULT_OUTPUT_FORMAT = AudioFormat(sample_rate=48_000, bit_depth=16, channels=2)


async def relay_source_to_group(
    event: SourceStreamStartedEvent,
    group: SendspinGroup,
    *,
    bridge_kind: Literal["simple", "asrc"] = "asrc",
    output_format: AudioFormat = DEFAULT_OUTPUT_FORMAT,
    target_latency_ms: int = 250,
    max_latency_ms: int = 1_000,
    chunk_duration_ms: int = 25,
) -> None:
    """Own one group stream and relay a source event into it until capture ends."""
    bridge: SourceBridge
    if bridge_kind == "asrc":
        bridge = AsrcSourceBridge(
            input_format=event.audio_format,
            output_format=output_format,
            target_latency_ms=target_latency_ms,
            max_latency_ms=max_latency_ms,
        )
    else:
        bridge = SourceBridge(
            input_format=event.audio_format,
            output_format=output_format,
            target_latency_ms=target_latency_ms,
            max_latency_ms=max_latency_ms,
        )

    input_done = asyncio.Event()
    primed = asyncio.Event()

    async def ingest_source() -> None:
        try:
            async for pcm, capture_timestamp_us in event.handle:
                bridge.feed(pcm, capture_timestamp_us)
                if bridge.occupancy_us >= target_latency_ms * 1_000:
                    primed.set()
        finally:
            bridge.flush()
            input_done.set()
            primed.set()

    ingest_task = asyncio.create_task(ingest_source())
    push_stream: PushStream | None = None

    try:
        await primed.wait()

        if bridge.occupancy_us < target_latency_ms * 1_000:
            await ingest_task
            return

        push_stream = group.start_stream()
        push_stream.set_live_source(True)

        frames_per_chunk = round(output_format.sample_rate * chunk_duration_ms / 1_000)
        period_s = frames_per_chunk / output_format.sample_rate
        loop = asyncio.get_running_loop()
        next_tick = loop.time()

        while not input_done.is_set() or bridge.occupancy_us > 0:
            pcm = bridge.read(frames_per_chunk)
            push_stream.prepare_audio(pcm, output_format)
            await push_stream.commit_audio()

            next_tick += period_s
            await asyncio.sleep(max(0.0, next_tick - loop.time()))

        await ingest_task
    finally:
        if not ingest_task.done():
            ingest_task.cancel()
            with suppress(asyncio.CancelledError):
                await ingest_task
        if push_stream is not None:
            await group.stop()
