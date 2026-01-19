"""Integration tests for multi-player timing and group changes."""

from __future__ import annotations

import asyncio
import math
from array import array
from dataclasses import dataclass
from typing import Literal

import pytest

from aiosendspin.models import unpack_binary_header
from aiosendspin.models.core import StreamClearMessage, StreamEndMessage, StreamStartMessage
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.channels import MAIN_CHANNEL, ChannelRouter
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import ManualClock
from aiosendspin.server.group import SendspinGroup


@dataclass(slots=True)
class _DummyServer:
    loop: asyncio.AbstractEventLoop
    clock: ManualClock
    id: str = "srv"
    name: str = "server"


EventKind = Literal["json", "bin"]


@dataclass(slots=True)
class _Event:
    kind: EventKind
    payload: object


class _CaptureConnection:
    """Capture connection that records JSON + binary messages in order."""

    def __init__(self) -> None:
        self.events: list[_Event] = []

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:
        self.events.append(_Event(kind="json", payload=message))

    def try_send_binary(self, data: bytes) -> bool:
        self.events.append(_Event(kind="bin", payload=data))
        return True

    def queue_high_water(self, threshold: float = 0.8) -> bool:  # noqa: ARG002
        return False


@dataclass(slots=True)
class _PcmSegment:
    sample_rate: int
    channels: int
    bit_depth: int
    start_timestamp_us: int
    pcm_s16le: bytes


def _chirp(t: float, *, f0: float, k: float) -> float:
    """Frequency-swept sine with time-varying instantaneous frequency."""
    return math.sin(2.0 * math.pi * (f0 * t + 0.5 * k * t * t))


def _signal_left(t: float) -> float:
    """Deterministic continuous-time test signal (left channel)."""
    # Two chirps with different slopes; unique enough for correlation.
    return 0.55 * _chirp(t, f0=233.0, k=1137.0) + 0.35 * _chirp(t, f0=911.0, k=271.0)


def _pcm_s16le_stereo_for_range(
    start_timestamp_us: int,
    *,
    sample_rate: int,
    frame_count: int,
) -> bytes:
    """Generate deterministic stereo PCM for a given absolute time range."""
    out = array("h")
    out_extend = out.extend

    for i in range(frame_count):
        t = (start_timestamp_us + int(i * 1_000_000 / sample_rate)) / 1_000_000.0
        left = max(-1.0, min(1.0, _signal_left(t)))
        # Right channel uses a phase-shifted variant (still deterministic).
        right = max(-1.0, min(1.0, _signal_left(t + 0.0013)))
        out_extend((int(left * 32767.0), int(right * 32767.0)))

    return out.tobytes()


def _extract_left_channel_s16le(pcm_s16le: bytes, channels: int) -> list[int]:
    """Extract the left channel samples from packed s16le PCM bytes."""
    samples = array("h")
    samples.frombytes(pcm_s16le)
    return list(samples[0::channels])


def _best_lag_samples(
    received: list[int],
    expected: list[int],
    *,
    max_lag_samples: int,
) -> tuple[int, float]:
    """Return the lag with the best normalized correlation score."""
    if not received or not expected:
        raise ValueError("signals must be non-empty")

    n = min(len(received), len(expected))
    rec = received[:n]
    exp = expected[:n]

    best_lag = 0
    best_score = -1.0

    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag >= 0:
            x = rec[: n - lag]
            y = exp[lag:n]
        else:
            x = rec[-lag:n]
            y = exp[: n + lag]

        if not x or not y:
            continue

        dot = 0.0
        norm_x = 0.0
        norm_y = 0.0
        for xi, yi in zip(x, y, strict=True):
            dot += xi * yi
            norm_x += xi * xi
            norm_y += yi * yi

        denom = math.sqrt(norm_x * norm_y)
        if denom <= 0.0:
            continue
        score = dot / denom
        if score > best_score:
            best_score = score
            best_lag = lag

    return best_lag, best_score


def _make_player(
    server: _DummyServer,
    client_id: str,
    *,
    preferred_pcm_rate: int,
) -> tuple[SendspinClient, SendspinGroup, _CaptureConnection]:
    """Create a connected player with a preferred PCM output sample rate."""
    client = SendspinClient(server, client_id=client_id)
    group = SendspinGroup(server, client)

    conn = _CaptureConnection()
    hello = type("Hello", (), {})()
    hello.client_id = client_id
    hello.name = client_id
    hello.player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.PCM,
                channels=2,
                sample_rate=preferred_pcm_rate,
                bit_depth=16,
            )
        ],
        buffer_capacity=2_000_000,
        supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
    )
    hello.artwork_support = None
    hello.visualizer_support = None

    client.attach_connection(conn, client_info=hello, active_roles=[Roles.PLAYER.value])
    client.mark_connected()
    return client, group, conn


def _pcm_segments_from_events(events: list[_Event]) -> list[_PcmSegment]:
    """Extract PCM segments (bounded by stream/clear and stream/end)."""
    segments: list[_PcmSegment] = []
    current_format: StreamStartMessage | None = None
    current_packets: list[bytes] = []
    current_timestamps: list[int] = []
    current_start_timestamp_us: int | None = None

    def _flush() -> None:
        nonlocal current_format, current_packets, current_timestamps, current_start_timestamp_us
        if current_format is None or not current_packets or current_start_timestamp_us is None:
            current_packets = []
            current_timestamps = []
            current_start_timestamp_us = None
            return

        fmt = current_format.payload.player
        payload = b"".join(current_packets)
        segments.append(
            _PcmSegment(
                sample_rate=int(fmt.sample_rate),
                channels=int(fmt.channels),
                bit_depth=int(fmt.bit_depth),
                start_timestamp_us=int(current_start_timestamp_us),
                pcm_s16le=payload,
            )
        )
        current_packets = []
        current_timestamps = []
        current_start_timestamp_us = None

    for ev in events:
        if ev.kind == "json":
            msg = ev.payload
            if isinstance(msg, StreamStartMessage):
                current_format = msg
                continue
            if isinstance(msg, StreamClearMessage | StreamEndMessage):
                _flush()
                continue
            continue

        data = ev.payload
        assert isinstance(data, (bytes, bytearray))
        header = unpack_binary_header(bytes(data))
        if current_format is None:
            continue
        if current_format.payload.player.codec != AudioCodec.PCM:
            continue
        if current_start_timestamp_us is None:
            current_start_timestamp_us = header.timestamp_us
        current_timestamps.append(header.timestamp_us)
        current_packets.append(bytes(data)[9:])

    _flush()
    return segments


def _choose_common_window(
    segments_by_player: list[list[_PcmSegment]],
    *,
    window_duration_us: int,
    warmup_us: int,
) -> int:
    """Pick a start timestamp present in all players' segments."""
    starts: list[int] = []
    ends: list[int] = []

    for segments in segments_by_player:
        if not segments:
            raise AssertionError("expected at least one segment per player")
        seg = segments[-1]
        frame_count = len(seg.pcm_s16le) // (2 * seg.channels)
        dur_us = int(frame_count * 1_000_000 / seg.sample_rate)
        starts.append(seg.start_timestamp_us + warmup_us)
        ends.append(seg.start_timestamp_us + dur_us)

    start_us = max(starts)
    end_us = min(ends)
    if end_us - start_us < window_duration_us:
        raise AssertionError("not enough common audio coverage for window")
    return start_us


def _samples_for_window(
    seg: _PcmSegment,
    window_start_us: int,
    window_duration_us: int,
) -> list[int]:
    """Extract left channel samples for a window based on timestamps."""
    frame_count_total = len(seg.pcm_s16le) // (2 * seg.channels)
    offset_frames = round((window_start_us - seg.start_timestamp_us) * seg.sample_rate / 1_000_000)
    window_frames = round(window_duration_us * seg.sample_rate / 1_000_000)
    offset_frames = max(0, min(offset_frames, frame_count_total))
    end_frames = max(0, min(offset_frames + window_frames, frame_count_total))

    # Extract raw frames from packed PCM.
    start_byte = offset_frames * seg.channels * 2
    end_byte = end_frames * seg.channels * 2
    return _extract_left_channel_s16le(seg.pcm_s16le[start_byte:end_byte], seg.channels)


def _expected_left_for_window(
    window_start_us: int,
    *,
    sample_rate: int,
    frame_count: int,
) -> list[int]:
    pcm = _pcm_s16le_stereo_for_range(
        window_start_us, sample_rate=sample_rate, frame_count=frame_count
    )
    return _extract_left_channel_s16le(pcm, 2)


@pytest.mark.asyncio
async def test_multi_player_group_join_sync_stable_source() -> None:
    """Stable source: late joiner stays within +/- 5ms of the global clock."""
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = _DummyServer(loop=loop, clock=clock)

    player_a, group_a, conn_a = _make_player(server, "pA", preferred_pcm_rate=48_000)
    player_b, _group_b, conn_b = _make_player(server, "pB", preferred_pcm_rate=32_000)

    router = ChannelRouter()
    router.set_channel(player_a.client_id, MAIN_CHANNEL)
    router.set_channel(player_b.client_id, MAIN_CHANNEL)

    stream = group_a.start_stream(channel_router=router)

    source_fmt = AudioFormat(sample_rate=48_000, bit_depth=16, channels=2, codec=AudioCodec.PCM)

    next_play_start_us = clock.now_us() + 250_000

    # Run 3 seconds virtual time; join B at t=1s.
    for i in range(120):  # 120 * 25ms = 3s
        if i == 40:
            await group_a.add_client(player_b)
        pcm = _pcm_s16le_stereo_for_range(
            next_play_start_us, sample_rate=source_fmt.sample_rate, frame_count=1200
        )
        stream.prepare_audio(pcm, source_fmt)
        play_start_us = await stream.commit_audio()
        assert abs(play_start_us - next_play_start_us) <= 1_000
        next_play_start_us = play_start_us + 25_000
        clock.advance_us(25_000)

    seg_a = _pcm_segments_from_events(conn_a.events)
    seg_b = _pcm_segments_from_events(conn_b.events)

    window_duration_us = 500_000  # 0.5s
    window_start_us = _choose_common_window(
        [seg_a, seg_b], window_duration_us=window_duration_us, warmup_us=250_000
    )

    a_last = seg_a[-1]
    b_last = seg_b[-1]
    frames_a = round(window_duration_us * a_last.sample_rate / 1_000_000)
    frames_b = round(window_duration_us * b_last.sample_rate / 1_000_000)

    received_a = _samples_for_window(a_last, window_start_us, window_duration_us)
    received_b = _samples_for_window(b_last, window_start_us, window_duration_us)

    expected_a = _expected_left_for_window(
        window_start_us, sample_rate=a_last.sample_rate, frame_count=frames_a
    )
    expected_b = _expected_left_for_window(
        window_start_us, sample_rate=b_last.sample_rate, frame_count=frames_b
    )

    lag_a, score_a = _best_lag_samples(
        received_a, expected_a, max_lag_samples=int(a_last.sample_rate * 0.005)
    )
    lag_b, score_b = _best_lag_samples(
        received_b, expected_b, max_lag_samples=int(b_last.sample_rate * 0.005)
    )

    lag_a_us = abs(lag_a) * 1_000_000 / a_last.sample_rate
    lag_b_us = abs(lag_b) * 1_000_000 / b_last.sample_rate
    assert lag_a_us <= 5_000
    assert lag_b_us <= 5_000
    assert abs(lag_a_us - lag_b_us) <= 5_000
    assert score_a >= 0.90
    assert score_b >= 0.90


@pytest.mark.asyncio
async def test_multi_player_sync_with_jittery_source_is_continuous() -> None:
    """Jittery chunk sizes: playback remains continuous and aligned."""
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = _DummyServer(loop=loop, clock=clock)

    player_a, group_a, conn_a = _make_player(server, "pA", preferred_pcm_rate=48_000)
    player_b, _group_b, conn_b = _make_player(server, "pB", preferred_pcm_rate=32_000)

    router = ChannelRouter()
    router.set_channel(player_a.client_id, MAIN_CHANNEL)
    router.set_channel(player_b.client_id, MAIN_CHANNEL)

    stream = group_a.start_stream(channel_router=router)
    await group_a.add_client(player_b)

    source_fmt = AudioFormat(sample_rate=48_000, bit_depth=16, channels=2, codec=AudioCodec.PCM)

    next_play_start_us = clock.now_us() + 250_000

    # Alternate 20ms and 30ms blocks (still "continuous" overall).
    pattern_frames = [960, 1440]  # 20ms, 30ms at 48kHz
    for i in range(120):  # 120 commits, ~3 seconds
        frame_count = pattern_frames[i % 2]
        pcm = _pcm_s16le_stereo_for_range(
            next_play_start_us, sample_rate=source_fmt.sample_rate, frame_count=frame_count
        )
        stream.prepare_audio(pcm, source_fmt)
        play_start_us = await stream.commit_audio()
        assert abs(play_start_us - next_play_start_us) <= 1_000
        duration_us = int(frame_count * 1_000_000 / source_fmt.sample_rate)
        next_play_start_us = play_start_us + duration_us
        clock.advance_us(duration_us)

    seg_a = _pcm_segments_from_events(conn_a.events)
    seg_b = _pcm_segments_from_events(conn_b.events)

    window_duration_us = 500_000
    window_start_us = _choose_common_window(
        [seg_a, seg_b], window_duration_us=window_duration_us, warmup_us=500_000
    )

    a_last = seg_a[-1]
    b_last = seg_b[-1]
    frames_a = round(window_duration_us * a_last.sample_rate / 1_000_000)
    frames_b = round(window_duration_us * b_last.sample_rate / 1_000_000)

    received_a = _samples_for_window(a_last, window_start_us, window_duration_us)
    received_b = _samples_for_window(b_last, window_start_us, window_duration_us)
    expected_a = _expected_left_for_window(
        window_start_us, sample_rate=a_last.sample_rate, frame_count=frames_a
    )
    expected_b = _expected_left_for_window(
        window_start_us, sample_rate=b_last.sample_rate, frame_count=frames_b
    )

    lag_a, score_a = _best_lag_samples(
        received_a, expected_a, max_lag_samples=int(a_last.sample_rate * 0.005)
    )
    lag_b, score_b = _best_lag_samples(
        received_b, expected_b, max_lag_samples=int(b_last.sample_rate * 0.005)
    )

    lag_a_us = abs(lag_a) * 1_000_000 / a_last.sample_rate
    lag_b_us = abs(lag_b) * 1_000_000 / b_last.sample_rate
    assert lag_a_us <= 5_000
    assert lag_b_us <= 5_000
    assert abs(lag_a_us - lag_b_us) <= 5_000
    assert score_a >= 0.85
    assert score_b >= 0.85


@pytest.mark.asyncio
async def test_unstable_source_creates_late_audio_after_gap() -> None:
    """Unstable source: a long production gap causes audio timestamps to fall behind now()."""
    loop = asyncio.get_running_loop()
    clock = ManualClock()
    server = _DummyServer(loop=loop, clock=clock)

    _player_a, group_a, conn_a = _make_player(server, "pA", preferred_pcm_rate=48_000)

    stream = group_a.start_stream(channel_router=ChannelRouter())
    source_fmt = AudioFormat(sample_rate=48_000, bit_depth=16, channels=2, codec=AudioCodec.PCM)

    next_play_start_us = clock.now_us() + 250_000

    # Produce 1s of audio.
    for _ in range(40):
        pcm = _pcm_s16le_stereo_for_range(
            next_play_start_us, sample_rate=source_fmt.sample_rate, frame_count=1200
        )
        stream.prepare_audio(pcm, source_fmt)
        play_start_us = await stream.commit_audio()
        assert abs(play_start_us - next_play_start_us) <= 1_000
        next_play_start_us = play_start_us + 25_000
        clock.advance_us(25_000)

    # Simulate a 2s gap with no audio production.
    events_before_gap = len(conn_a.events)
    clock.advance_us(2_000_000)
    resume_now_us = clock.now_us()

    pcm = _pcm_s16le_stereo_for_range(
        next_play_start_us, sample_rate=source_fmt.sample_rate, frame_count=1200
    )
    stream.prepare_audio(pcm, source_fmt)
    await stream.commit_audio()

    # Find the first audio chunk timestamp sent after the gap.
    first_after_gap_ts: int | None = None
    for ev in conn_a.events[events_before_gap:]:
        if ev.kind != "bin":
            continue
        header = unpack_binary_header(ev.payload)  # type: ignore[arg-type]
        first_after_gap_ts = header.timestamp_us
        break

    assert first_after_gap_ts is not None
    assert first_after_gap_ts < resume_now_us - 250_000
