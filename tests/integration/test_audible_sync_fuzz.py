"""Seeded stress tests for audible sync invariants."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from uuid import UUID

import pytest

from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.models.types import AudioCodec
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import ManualClock
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.push_stream import PushStream
from tests.integration.sync_assertions import (
    DecodedSegment,
    assert_audible_sync,
    assert_pcm_chunks_continuous,
    decode_segments_from_events,
    first_audio_timestamp_after,
    pcm_s16le_stereo_for_range,
)
from tests.integration.sync_harness import (
    CaptureConnection,
    DummyServer,
    channel_resolver_for,
    make_player,
)

SEEDS = [3, 11, 23, 37, 41, 59, 67, 89]
# Optional ad-hoc deep sweep. Keep at 0 in normal development/CI.
# To expand coverage locally, set this to e.g. 250.
LARGE_SWEEP_SEED_COUNT = 0
LARGE_SWEEP_SEEDS = list(range(LARGE_SWEEP_SEED_COUNT))


@dataclass(slots=True)
class _PlayerCtx:
    player: SendspinClient
    conn: CaptureConnection
    channel_id: UUID


def _segment_duration_us(segments: list[DecodedSegment]) -> int:
    if not segments:
        return 0
    seg = segments[-1]
    frame_count = len(seg.pcm_s16le) // (2 * seg.channels)
    return int(frame_count * 1_000_000 / seg.sample_rate)


def _set_instant_join(player: SendspinClient) -> None:
    role = player.role("player@v1")
    assert role is not None
    role.get_join_delay_s = lambda: 0.0  # type: ignore[method-assign]


async def _commit_block(
    stream: PushStream,
    *,
    next_play_start_us: int,
    duration_us: int,
    channel_ids: set[UUID],
) -> int:
    frame_count = round((duration_us / 1_000_000) * 48_000)
    pcm = pcm_s16le_stereo_for_range(
        next_play_start_us,
        sample_rate=48_000,
        frame_count=frame_count,
    )
    fmt = AudioFormat(sample_rate=48_000, bit_depth=16, channels=2)

    for channel_id in channel_ids:
        stream.prepare_audio(pcm, fmt, channel_id=channel_id)

    return await stream.commit_audio(play_start_us=next_play_start_us)


async def _maybe_remove(group: SendspinGroup, player: SendspinClient) -> None:
    if player in group.clients and len(group.clients) > 2:
        await group.remove_client(player)


async def _run_seeded_fuzz(seed: int) -> None:  # noqa: PLR0915
    """Execute one deterministic seeded fuzz scenario and assert sync invariants."""
    rng = random.Random(seed)  # noqa: S311
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = DummyServer(loop=loop, clock=clock)

    custom_channel = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")

    anchor_player, group_a, conn_a = make_player(
        server,
        f"anchor-{seed}",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        buffer_capacity=2_000_000,
        channel_id=MAIN_CHANNEL,
    )

    codec_b = rng.choice([AudioCodec.PCM, AudioCodec.FLAC])
    codec_c = rng.choice([AudioCodec.PCM, AudioCodec.FLAC])
    channel_b = rng.choice([MAIN_CHANNEL, custom_channel])
    channel_c = rng.choice([MAIN_CHANNEL, custom_channel])

    player_b, _group_b, conn_b = make_player(
        server,
        f"b-{seed}",
        supported_formats=[
            SupportedAudioFormat(codec=codec_b, channels=2, sample_rate=44_100, bit_depth=16)
            if codec_b == AudioCodec.FLAC
            else SupportedAudioFormat(codec=codec_b, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        buffer_capacity=100_000,
        channel_id=channel_b,
    )
    player_c, _group_c, conn_c = make_player(
        server,
        f"c-{seed}",
        supported_formats=[
            SupportedAudioFormat(codec=codec_c, channels=2, sample_rate=44_100, bit_depth=16)
            if codec_c == AudioCodec.FLAC
            else SupportedAudioFormat(codec=codec_c, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        buffer_capacity=100_000,
        channel_id=channel_c,
    )

    ctx_by_id: dict[str, _PlayerCtx] = {
        f"anchor-{seed}": _PlayerCtx(
            player=anchor_player,
            conn=conn_a,
            channel_id=MAIN_CHANNEL,
        ),
        f"b-{seed}": _PlayerCtx(player=player_b, conn=conn_b, channel_id=channel_b),
        f"c-{seed}": _PlayerCtx(player=player_c, conn=conn_c, channel_id=channel_c),
    }

    for player in (player_b, player_c):
        _set_instant_join(player)

    channel_by_player = {
        f"anchor-{seed}": MAIN_CHANNEL,
        f"b-{seed}": channel_b,
        f"c-{seed}": channel_c,
    }

    stream = group_a.start_stream(channel_resolver=channel_resolver_for(channel_by_player))
    stream.enable_pcm_cache_for_channel(custom_channel)

    join_steps = {6: player_b, 12: player_c}
    regroup_enabled = rng.choice([True, False])
    remove_step = 22
    readd_step = 28

    custom_join_steps = [
        step
        for step, player in join_steps.items()
        if ctx_by_id[player.client_id].channel_id == custom_channel
    ]
    inject_step = min(custom_join_steps) - 1 if custom_join_steps and (seed % 2 == 0) else None

    join_indices: dict[str, int] = {}
    join_times_us: dict[str, int] = {}

    next_play_start_us = clock.now_us() + 250_000
    random_phase_steps = 36

    for step in range(1, random_phase_steps + 1):
        if inject_step is not None and step == inject_step:
            cached_main = stream.get_cached_pcm_chunks(MAIN_CHANNEL)
            if cached_main:
                for chunk in cached_main[-min(12, len(cached_main)) :]:
                    stream.prepare_historical_audio(
                        chunk.pcm_data,
                        AudioFormat(
                            sample_rate=chunk.sample_rate,
                            bit_depth=chunk.bit_depth,
                            channels=chunk.channels,
                        ),
                        channel_id=custom_channel,
                    )

        join_player = join_steps.get(step)
        if join_player is not None and join_player not in group_a.clients:
            join_indices[join_player.client_id] = len(ctx_by_id[join_player.client_id].conn.events)
            join_times_us[join_player.client_id] = clock.now_us()
            await group_a.add_client(join_player)

        if regroup_enabled and step == remove_step:
            await _maybe_remove(group_a, player_b)
        if regroup_enabled and step == readd_step and player_b not in group_a.clients:
            join_indices[player_b.client_id] = len(conn_b.events)
            join_times_us[player_b.client_id] = clock.now_us()
            await group_a.add_client(player_b)

        duration_us = rng.choice([20_000, 25_000, 30_000, 100_000])
        channel_ids: set[UUID] = {MAIN_CHANNEL}
        for ctx in ctx_by_id.values():
            if ctx.player is anchor_player or ctx.player not in group_a.clients:
                continue
            channel_ids.add(ctx.channel_id)

        play_start_us = await _commit_block(
            stream,
            next_play_start_us=next_play_start_us,
            duration_us=duration_us,
            channel_ids=channel_ids,
        )
        next_play_start_us = play_start_us + duration_us
        clock.advance_us(duration_us)

    for _ in range(24):
        active_channels = {MAIN_CHANNEL}
        for ctx in ctx_by_id.values():
            if ctx.player in group_a.clients:
                active_channels.add(ctx.channel_id)

        play_start_us = await _commit_block(
            stream,
            next_play_start_us=next_play_start_us,
            duration_us=25_000,
            channel_ids=active_channels,
        )
        next_play_start_us = play_start_us + 25_000
        clock.advance_us(25_000)

    for player_id, join_index in join_indices.items():
        conn = ctx_by_id[player_id].conn
        first_ts = first_audio_timestamp_after(conn.events, start_index=join_index)
        assert first_ts is not None, f"{player_id} never received post-join audio (seed={seed})"
        assert first_ts - join_times_us[player_id] <= 1_000_000, (
            f"{player_id} join-to-audio delay exceeded 1s (seed={seed})"
        )

    segments_by_player: dict[str, list[DecodedSegment]] = {}
    for player_id, ctx in ctx_by_id.items():
        segments = decode_segments_from_events(ctx.conn.events)
        if segments and _segment_duration_us(segments) >= 400_000:
            segments_by_player[player_id] = segments

    assert len(segments_by_player) >= 2
    assert_audible_sync(
        segments_by_player,
        max_skew_us=5_000,
        min_corr=0.35,
        enforce_corr=True,
        window_duration_us=250_000,
        warmup_us=0,
        window_anchor="tail",
        tail_padding_us=0,
        compare_to="reference",
    )

    for ctx in ctx_by_id.values():
        assert_pcm_chunks_continuous(ctx.conn.events, max_gap_us=6_000)

    assert all(task.done() for task in stream._catchup_tasks.values()), (  # noqa: SLF001
        f"pending catch-up tasks for seed={seed}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", SEEDS)
async def test_audible_sync_seeded_fuzz(seed: int) -> None:
    """PR-scope seeded stress set."""
    await _run_seeded_fuzz(seed)


@pytest.mark.asyncio
@pytest.mark.skipif(
    LARGE_SWEEP_SEED_COUNT <= 0,
    reason="Set LARGE_SWEEP_SEED_COUNT > 0 to run the large sweep.",
)
@pytest.mark.parametrize("seed", LARGE_SWEEP_SEEDS)
async def test_audible_sync_seeded_fuzz_large_sweep(seed: int) -> None:
    """Optional large seeded sweep run with one test item per seed."""
    await _run_seeded_fuzz(seed)
