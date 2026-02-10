"""Matrix-style integration tests for audible sync regressions."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from uuid import UUID

import pytest

from aiosendspin.models.core import StreamStartMessage
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
    best_lag_samples,
    choose_common_window,
    decode_segments_from_events,
    first_audio_timestamp_after,
    pcm_s16le_stereo_for_range,
    samples_for_window,
)
from tests.integration.sync_harness import (
    CaptureConnection,
    DummyServer,
    channel_resolver_for,
    make_player,
)


def _set_instant_join(player: SendspinClient) -> None:
    role = player.role("player@v1")
    assert role is not None
    role.get_join_delay_s = lambda: 0.0  # type: ignore[method-assign]


async def _commit_block(
    stream: PushStream,
    *,
    next_play_start_us: int,
    duration_us: int,
    channel_ids: list[UUID],
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


async def _scenario_existing_codec_main(
    server: DummyServer,
    clock: ManualClock,
) -> dict[str, CaptureConnection]:
    _player_a, group_a, conn_a = make_player(
        server,
        "pcm-a",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        buffer_capacity=2_000_000,
    )
    player_b, _group_b, conn_b = make_player(
        server,
        "pcm-b",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        buffer_capacity=150_000,
    )

    stream = group_a.start_stream(
        channel_resolver=channel_resolver_for(
            {
                "pcm-a": MAIN_CHANNEL,
                "pcm-b": MAIN_CHANNEL,
            }
        )
    )
    next_play_start_us = clock.now_us() + 250_000
    join_index = len(conn_b.events)

    for i in range(120):
        if i == 30:
            _set_instant_join(player_b)
            await group_a.add_client(player_b)
        play_start_us = await _commit_block(
            stream,
            next_play_start_us=next_play_start_us,
            duration_us=25_000,
            channel_ids=[MAIN_CHANNEL],
        )
        next_play_start_us = play_start_us + 25_000
        clock.advance_us(25_000)

    first_b = first_audio_timestamp_after(conn_b.events, start_index=join_index)
    assert first_b is not None
    return {"pcm-a": conn_a, "pcm-b": conn_b}


async def _scenario_new_codec_main(
    server: DummyServer,
    clock: ManualClock,
) -> dict[str, CaptureConnection]:
    _player_a, group_a, conn_a = make_player(
        server,
        "main-pcm",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        buffer_capacity=2_000_000,
    )
    player_b, _group_b, conn_b = make_player(
        server,
        "join-flac",
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.FLAC,
                channels=2,
                sample_rate=44_100,
                bit_depth=16,
            )
        ],
        buffer_capacity=80_000,
    )

    stream = group_a.start_stream(
        channel_resolver=channel_resolver_for(
            {
                "main-pcm": MAIN_CHANNEL,
                "join-flac": MAIN_CHANNEL,
            }
        )
    )
    next_play_start_us = clock.now_us() + 250_000
    join_index = len(conn_b.events)

    for i in range(70):
        if i == 12:
            _set_instant_join(player_b)
            await group_a.add_client(player_b)

        play_start_us = await _commit_block(
            stream,
            next_play_start_us=next_play_start_us,
            duration_us=100_000,
            channel_ids=[MAIN_CHANNEL],
        )
        next_play_start_us = play_start_us + 100_000
        clock.advance_us(100_000)

    first_b = first_audio_timestamp_after(conn_b.events, start_index=join_index)
    assert first_b is not None
    return {"main-pcm": conn_a, "join-flac": conn_b}


async def _scenario_custom_channel_no_history(
    server: DummyServer,
    clock: ManualClock,
) -> dict[str, CaptureConnection]:
    custom_channel = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    _player_a, group_a, conn_a = make_player(
        server,
        "main",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        buffer_capacity=500_000,
        channel_id=MAIN_CHANNEL,
    )
    player_b, _group_b, conn_b = make_player(
        server,
        "custom",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        buffer_capacity=120_000,
        channel_id=custom_channel,
    )

    stream = group_a.start_stream(
        channel_resolver=channel_resolver_for(
            {
                "main": MAIN_CHANNEL,
                "custom": custom_channel,
            }
        )
    )
    _set_instant_join(player_b)
    join_index = len(conn_b.events)
    await group_a.add_client(player_b)

    next_play_start_us = clock.now_us() + 250_000

    for _ in range(30):
        play_start_us = await _commit_block(
            stream,
            next_play_start_us=next_play_start_us,
            duration_us=25_000,
            channel_ids=[MAIN_CHANNEL],
        )
        next_play_start_us = play_start_us + 25_000
        clock.advance_us(25_000)

    for _ in range(80):
        play_start_us = await _commit_block(
            stream,
            next_play_start_us=next_play_start_us,
            duration_us=25_000,
            channel_ids=[MAIN_CHANNEL, custom_channel],
        )
        next_play_start_us = play_start_us + 25_000
        clock.advance_us(25_000)

    first_b = first_audio_timestamp_after(conn_b.events, start_index=join_index)
    assert first_b is not None
    return {"main": conn_a, "custom": conn_b}


async def _scenario_custom_channel_with_history(
    server: DummyServer,
    clock: ManualClock,
) -> dict[str, CaptureConnection]:
    custom_channel = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    _player_a, group_a, conn_a = make_player(
        server,
        "main-hist",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        buffer_capacity=500_000,
        channel_id=MAIN_CHANNEL,
    )

    stream = group_a.start_stream(
        channel_resolver=channel_resolver_for(
            {
                "main-hist": MAIN_CHANNEL,
                "custom-hist": custom_channel,
            }
        )
    )
    stream.enable_pcm_cache_for_channel(custom_channel)

    next_play_start_us = clock.now_us() + 250_000
    for _ in range(70):
        play_start_us = await _commit_block(
            stream,
            next_play_start_us=next_play_start_us,
            duration_us=25_000,
            channel_ids=[MAIN_CHANNEL],
        )
        next_play_start_us = play_start_us + 25_000
        clock.advance_us(25_000)

    cached_pcm = stream.get_cached_pcm_chunks(MAIN_CHANNEL)
    assert len(cached_pcm) >= 3
    # Inject a bounded historical window so the late-join custom channel
    # transitions back to live audio during this scenario.
    for chunk in cached_pcm[-12:]:
        stream.prepare_historical_audio(
            chunk.pcm_data,
            AudioFormat(
                sample_rate=chunk.sample_rate,
                bit_depth=chunk.bit_depth,
                channels=chunk.channels,
            ),
            channel_id=custom_channel,
        )

    player_b, _group_b, conn_b = make_player(
        server,
        "custom-hist",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        buffer_capacity=120_000,
        channel_id=custom_channel,
    )
    _set_instant_join(player_b)
    join_index = len(conn_b.events)
    await group_a.add_client(player_b)

    # First commit applies historical while main continues live.
    play_start_us = await _commit_block(
        stream,
        next_play_start_us=next_play_start_us,
        duration_us=25_000,
        channel_ids=[MAIN_CHANNEL],
    )
    next_play_start_us = play_start_us + 25_000
    clock.advance_us(25_000)

    # Stabilize both channels with live audio for comparison.
    for _ in range(40):
        play_start_us = await _commit_block(
            stream,
            next_play_start_us=next_play_start_us,
            duration_us=25_000,
            channel_ids=[MAIN_CHANNEL, custom_channel],
        )
        next_play_start_us = play_start_us + 25_000
        clock.advance_us(25_000)

    assert any(
        isinstance(event.payload, StreamStartMessage)
        for event in conn_b.events
        if event.kind == "json"
    )
    first_b = first_audio_timestamp_after(conn_b.events, start_index=join_index)
    assert first_b is not None

    # Historical replay should be correlated with main audio but intentionally delayed.
    main_segments = decode_segments_from_events(conn_a.events)
    custom_segments = decode_segments_from_events(conn_b.events)
    window_start_us = choose_common_window(
        {"main-hist": main_segments, "custom-hist": custom_segments},
        window_duration_us=500_000,
        warmup_us=500_000,
    )
    main_samples = samples_for_window(main_segments[-1], window_start_us, 500_000)
    custom_samples = samples_for_window(custom_segments[-1], window_start_us, 500_000)
    lag_samples, corr = best_lag_samples(
        custom_samples,
        main_samples,
        max_lag_samples=int(48_000 * 0.6),
    )
    assert corr >= 0.95
    assert abs(lag_samples) >= int(48_000 * 0.1)

    return {"main-hist": conn_a, "custom-hist": conn_b}


async def _maybe_remove(group: SendspinGroup, player: SendspinClient) -> None:
    if player in group.clients and len(group.clients) > 2:
        await group.remove_client(player)


async def _scenario_regroup_churn(
    server: DummyServer,
    clock: ManualClock,
) -> dict[str, CaptureConnection]:
    custom_channel = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    _player_a, group_a, conn_a = make_player(
        server,
        "churn-a",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=48_000, bit_depth=16)
        ],
        buffer_capacity=2_000_000,
        channel_id=MAIN_CHANNEL,
    )
    player_b, _group_b, conn_b = make_player(
        server,
        "churn-b",
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.FLAC,
                channels=2,
                sample_rate=44_100,
                bit_depth=16,
            )
        ],
        buffer_capacity=80_000,
        channel_id=MAIN_CHANNEL,
    )
    player_c, _group_c, conn_c = make_player(
        server,
        "churn-c",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=32_000, bit_depth=16)
        ],
        buffer_capacity=90_000,
        channel_id=custom_channel,
    )
    player_d, _group_d, conn_d = make_player(
        server,
        "churn-d",
        supported_formats=[
            SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=32_000, bit_depth=16)
        ],
        buffer_capacity=90_000,
        channel_id=MAIN_CHANNEL,
    )

    for player in (player_b, player_c, player_d):
        _set_instant_join(player)

    stream = group_a.start_stream(
        channel_resolver=channel_resolver_for(
            {
                "churn-a": MAIN_CHANNEL,
                "churn-b": MAIN_CHANNEL,
                "churn-c": custom_channel,
                "churn-d": MAIN_CHANNEL,
            }
        )
    )
    next_play_start_us = clock.now_us() + 250_000

    actions: dict[int, Callable[[], Awaitable[None]]] = {
        3: lambda: group_a.add_client(player_b),
        5: lambda: group_a.add_client(player_c),
        7: lambda: group_a.add_client(player_d),
        10: lambda: _maybe_remove(group_a, player_b),
        12: lambda: _maybe_remove(group_a, player_d),
        14: lambda: group_a.add_client(player_b),
        18: lambda: group_a.add_client(player_d),
    }

    for step in range(1, 45):
        channel_ids = [MAIN_CHANNEL]
        if player_c in group_a.clients:
            channel_ids.append(custom_channel)

        play_start_us = await _commit_block(
            stream,
            next_play_start_us=next_play_start_us,
            duration_us=100_000,
            channel_ids=channel_ids,
        )
        next_play_start_us = play_start_us + 100_000
        clock.advance_us(100_000)

        action = actions.get(step)
        if action is not None:
            await action()

    for _ in range(15):
        play_start_us = await _commit_block(
            stream,
            next_play_start_us=next_play_start_us,
            duration_us=100_000,
            channel_ids=[MAIN_CHANNEL, custom_channel],
        )
        next_play_start_us = play_start_us + 100_000
        clock.advance_us(100_000)

    return {
        "churn-a": conn_a,
        "churn-b": conn_b,
        "churn-c": conn_c,
        "churn-d": conn_d,
    }


SCENARIOS: dict[
    str,
    Callable[[DummyServer, ManualClock], Awaitable[dict[str, CaptureConnection]]],
] = {
    "existing_codec_main": _scenario_existing_codec_main,
    "new_codec_main": _scenario_new_codec_main,
    "custom_channel_no_history": _scenario_custom_channel_no_history,
    "custom_channel_with_history": _scenario_custom_channel_with_history,
    "regroup_churn": _scenario_regroup_churn,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
async def test_audible_sync_matrix(scenario_name: str) -> None:
    """Scenario matrix should detect regressions that are audibly out of sync."""
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = DummyServer(loop=loop, clock=clock)

    connections = await SCENARIOS[scenario_name](server, clock)

    segments_by_player: dict[str, list[DecodedSegment]] = {}
    for player_id, conn in connections.items():
        segments = decode_segments_from_events(conn.events)
        if segments:
            segments_by_player[player_id] = segments

    assert len(segments_by_player) >= 2

    if scenario_name != "custom_channel_with_history":
        min_corr = 0.90 if scenario_name == "existing_codec_main" else 0.85
        assert_audible_sync(
            segments_by_player,
            max_skew_us=5_000,
            min_corr=min_corr,
            window_duration_us=500_000,
            warmup_us=500_000,
            compare_to="reference",
        )

    for conn in connections.values():
        assert_pcm_chunks_continuous(conn.events, max_gap_us=6_000)
