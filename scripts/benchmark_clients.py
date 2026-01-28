"""Benchmark how many simultaneous player clients the server can stream to."""

# ruff: noqa

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import math
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass

from aiohttp import ClientSession, TCPConnector

from aiosendspin.client import SendspinClient
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.server import SendspinServer

logger = logging.getLogger(__name__)


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark simultaneous player clients streaming with clock sync. "
            "Runs a local server and connects N clients in-process."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host to bind")
    parser.add_argument("--port", type=int, default=19375, help="Server port to bind")
    parser.add_argument("--clients", type=int, default=10_000, help="Number of clients")
    parser.add_argument("--duration-s", type=int, default=60, help="Stream duration")
    parser.add_argument("--chunk-ms", type=int, default=100, help="PCM chunk size in ms")
    parser.add_argument("--sample-rate", type=int, default=48_000, help="PCM sample rate")
    parser.add_argument("--channels", type=int, default=2, help="PCM channel count")
    parser.add_argument("--bit-depth", type=int, default=16, help="PCM bit depth")
    parser.add_argument(
        "--ramp-rate",
        type=float,
        default=1_000.0,
        help="Client connect rate (clients/sec)",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=1_000,
        help="Max concurrent connect tasks",
    )
    parser.add_argument(
        "--connect-timeout-s",
        type=float,
        default=10.0,
        help="Per-client connect timeout",
    )
    parser.add_argument(
        "--connect-phase-timeout-s",
        type=float,
        default=30.0,
        help="Total timeout for the connect phase (0 to disable)",
    )
    parser.add_argument(
        "--sync-sample-interval-s",
        type=float,
        default=5.0,
        help="How often to sample time sync stats",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args(argv)


def _make_player_support(
    sample_rate: int,
    channels: int,
    bit_depth: int,
) -> ClientHelloPlayerSupport:
    return ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.PCM,
                channels=channels,
                sample_rate=sample_rate,
                bit_depth=bit_depth,
            )
        ],
        buffer_capacity=512 * 1024,
        supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
    )


def _make_audio_format(sample_rate: int, channels: int, bit_depth: int) -> AudioFormat:
    return AudioFormat(sample_rate=sample_rate, bit_depth=bit_depth, channels=channels)


def _make_silence_chunk(fmt: AudioFormat, chunk_ms: int) -> bytes:
    frames = int(fmt.sample_rate * (chunk_ms / 1000.0))
    frame_bytes = fmt.channels * (fmt.bit_depth // 8)
    return b"\x00" * (frames * frame_bytes)


@dataclass(slots=True)
class AudioStats:
    chunks: int = 0
    bytes: int = 0
    empty_chunks: int = 0
    bad_sizes: int = 0
    out_of_order: int = 0
    gaps: int = 0
    gap_us_total: int = 0
    gap_us_max: int = 0
    last_ts_us: int | None = None
    last_duration_us: int | None = None


@dataclass(slots=True)
class SyncStats:
    synchronized: int = 0
    total: int = 0
    error_us: list[int] | None = None
    offset_us: list[int] | None = None


def _summarize(values: list[int]) -> str:
    if not values:
        return "n=0"
    values_sorted = sorted(values)
    n = len(values_sorted)
    p50 = values_sorted[n // 2]
    p95 = values_sorted[min(n - 1, math.ceil(n * 0.95) - 1)]
    p99 = values_sorted[min(n - 1, math.ceil(n * 0.99) - 1)]
    vmin = values_sorted[0]
    vmax = values_sorted[-1]
    return f"n={n} min={vmin} p50={p50} p95={p95} p99={p99} max={vmax}"


async def _connect_clients(
    *,
    url: str,
    count: int,
    ramp_rate: float,
    max_in_flight: int,
    connect_timeout_s: float,
    connect_phase_timeout_s: float,
    session: ClientSession,
    sample_rate: int,
    channels: int,
    bit_depth: int,
) -> tuple[list[SendspinClient], dict[str, AudioStats]]:
    player_support = _make_player_support(sample_rate, channels, bit_depth)

    semaphore = asyncio.Semaphore(max_in_flight)
    tasks: list[asyncio.Task[SendspinClient | None]] = []

    audio_stats: dict[str, AudioStats] = {}

    async def _connect_one(index: int) -> SendspinClient | None:
        client_id = f"bench-{index:05d}"
        client = SendspinClient(
            client_id=client_id,
            client_name=client_id,
            roles=[Roles.PLAYER],
            player_support=player_support,
            session=session,
        )
        stats = AudioStats()
        audio_stats[client_id] = stats

        def _on_audio_chunk(timestamp_us: int, payload: bytes, fmt: object) -> None:
            stats.chunks += 1
            stats.bytes += len(payload)
            if not payload:
                stats.empty_chunks += 1
                return
            fmt_channels = getattr(fmt, "channels", channels)
            fmt_bit_depth = getattr(fmt, "bit_depth", bit_depth)
            fmt_sample_rate = getattr(fmt, "sample_rate", sample_rate)
            frame_size = fmt_channels * (fmt_bit_depth // 8)
            if frame_size == 0 or len(payload) % frame_size != 0:
                stats.bad_sizes += 1
                return

            frames = len(payload) // frame_size
            duration_us = round(frames * 1_000_000 / fmt_sample_rate)
            if stats.last_ts_us is not None:
                if timestamp_us <= stats.last_ts_us:
                    stats.out_of_order += 1
                elif stats.last_duration_us is not None:
                    expected = stats.last_ts_us + stats.last_duration_us
                    gap = timestamp_us - expected
                    if gap > 2_000:  # allow 2ms jitter
                        stats.gaps += 1
                        stats.gap_us_total += gap
                        stats.gap_us_max = max(stats.gap_us_max, gap)
            stats.last_ts_us = timestamp_us
            stats.last_duration_us = duration_us

        client.add_audio_chunk_listener(_on_audio_chunk)
        async with semaphore:
            try:
                await asyncio.wait_for(client.connect(url), timeout=connect_timeout_s)
                return client
            except Exception:
                logger.exception("Failed to connect client %s", client_id)
                with contextlib.suppress(Exception):
                    if client.connected:
                        await client.disconnect()
                return None

    ramp_interval = 0.0 if ramp_rate <= 0 else 1.0 / ramp_rate
    for i in range(count):
        tasks.append(asyncio.create_task(_connect_one(i)))
        if ramp_interval:
            await asyncio.sleep(ramp_interval)

    if connect_phase_timeout_s > 0:
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=connect_phase_timeout_s,
            )
        except TimeoutError:
            for task in tasks:
                task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            logger.warning(
                "Connect phase timed out after %.1fs (continuing with connected clients)",
                connect_phase_timeout_s,
            )
    else:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    clients: list[SendspinClient] = []
    failures = 0
    for result in results:
        if isinstance(result, SendspinClient):
            clients.append(result)
        else:
            failures += 1

    logger.warning("Connected %d/%d clients (%d failures)", len(clients), count, failures)
    return clients, audio_stats


async def _group_clients(server: SendspinServer, clients: list[SendspinClient]) -> None:
    if not clients:
        return

    leader_id = clients[0]._client_id
    leader = server.get_client(leader_id)
    if leader is None:
        raise RuntimeError("Leader client not found on server")
    group = leader.group

    for client in clients[1:]:
        server_client = server.get_client(client._client_id)
        if server_client is None:
            continue
        await group.add_client(server_client)


def _collect_sync_stats(clients: list[SendspinClient]) -> SyncStats:
    stats = SyncStats(synchronized=0, total=len(clients), error_us=[], offset_us=[])
    for client in clients:
        if not client.is_time_synchronized():
            continue
        stats.synchronized += 1
        # Access internal filter for accuracy stats (benchmark-only).
        filt = client._time_filter  # noqa: SLF001
        stats.error_us.append(filt.error)
        stats.offset_us.append(abs(int(filt.offset)))
    return stats


def _log_sync_stats(stats: SyncStats) -> None:
    prefix = "ALL GOOD"
    if stats.total == 0:
        logger.warning("%s time sync: n=0", prefix)
        return
    if stats.synchronized != stats.total:
        prefix = "CHECK"
    ratio = f"{stats.synchronized}/{stats.total}"
    err_summary = _summarize(stats.error_us or [])
    off_summary = _summarize(stats.offset_us or [])
    logger.warning(
        "%s time sync: %s | error_us(%s) offset_us(%s)",
        prefix,
        ratio,
        err_summary,
        off_summary,
    )


async def _run_stream(
    *,
    server: SendspinServer,
    clients: list[SendspinClient],
    audio_stats: dict[str, AudioStats],
    duration_s: int,
    chunk_ms: int,
    sample_rate: int,
    channels: int,
    bit_depth: int,
) -> None:
    if not clients:
        return

    leader = server.get_client(clients[0]._client_id)
    if leader is None:
        raise RuntimeError("Leader client not found on server")

    group = leader.group
    stream = group.start_stream()
    fmt = _make_audio_format(sample_rate, channels, bit_depth)
    chunk = _make_silence_chunk(fmt, chunk_ms)

    start = time.monotonic()
    next_tick = start

    logger.warning(
        "ALL GOOD streaming: starting %ds with %dms chunks @ %dHz %dch %dbit",
        duration_s,
        chunk_ms,
        sample_rate,
        channels,
        bit_depth,
    )
    sampled_10s = False
    while True:
        now = time.monotonic()
        if now - start >= duration_s:
            break

        stream.prepare_audio(chunk, fmt)
        await stream.commit_audio()
        await stream.wait_for_buffer_space()

        if not sampled_10s and now - start >= 10.0:
            _log_sync_stats(_collect_sync_stats(clients))
            sampled_10s = True

        next_tick += chunk_ms / 1000.0
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    stream.stop()
    _log_sync_stats(_collect_sync_stats(clients))
    _log_audio_stats(audio_stats)


def _log_audio_stats(audio_stats: dict[str, AudioStats]) -> None:
    if not audio_stats:
        logger.warning("CHECK streaming: audio stats n=0")
        return

    totals = AudioStats()
    for stats in audio_stats.values():
        totals.chunks += stats.chunks
        totals.bytes += stats.bytes
        totals.empty_chunks += stats.empty_chunks
        totals.bad_sizes += stats.bad_sizes
        totals.out_of_order += stats.out_of_order
        totals.gaps += stats.gaps
        totals.gap_us_total += stats.gap_us_total
        totals.gap_us_max = max(totals.gap_us_max, stats.gap_us_max)

    avg_gap = 0
    if totals.gaps:
        avg_gap = totals.gap_us_total // totals.gaps

    prefix = "ALL GOOD"
    if totals.empty_chunks or totals.bad_sizes or totals.out_of_order or totals.gaps:
        prefix = "CHECK"

    logger.warning(
        "%s streaming: audio stats chunks=%d bytes=%d empty=%d bad_sizes=%d "
        "out_of_order=%d gaps=%d gap_us_avg=%d gap_us_max=%d",
        prefix,
        totals.chunks,
        totals.bytes,
        totals.empty_chunks,
        totals.bad_sizes,
        totals.out_of_order,
        totals.gaps,
        avg_gap,
        totals.gap_us_max,
    )


async def _main_async(argv: Iterable[str]) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    loop = asyncio.get_running_loop()
    server = SendspinServer(loop=loop, server_id="bench", server_name="bench")
    clients: list[SendspinClient] = []
    audio_stats: dict[str, AudioStats] = {}
    session: ClientSession | None = None
    started = False

    try:
        await server.start_server(
            port=args.port,
            host=args.host,
            advertise_addresses=[],
            discover_clients=False,
        )
        started = True

        url = f"ws://{args.host}:{args.port}{server.API_PATH}"
        connector = TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=0)
        session = ClientSession(connector=connector)

        logger.warning("Connecting %d clients to %s", args.clients, url)
        clients, audio_stats = await _connect_clients(
            url=url,
            count=args.clients,
            ramp_rate=args.ramp_rate,
            max_in_flight=args.max_in_flight,
            connect_timeout_s=args.connect_timeout_s,
            connect_phase_timeout_s=args.connect_phase_timeout_s,
            session=session,
            sample_rate=args.sample_rate,
            channels=args.channels,
            bit_depth=args.bit_depth,
        )
        await _group_clients(server, clients)
        await _run_stream(
            server=server,
            clients=clients,
            audio_stats=audio_stats,
            duration_s=args.duration_s,
            chunk_ms=args.chunk_ms,
            sample_rate=args.sample_rate,
            channels=args.channels,
            bit_depth=args.bit_depth,
        )
    finally:
        for client in clients:
            with contextlib.suppress(Exception):
                await client.disconnect()
        if session is not None:
            with contextlib.suppress(Exception):
                await session.close()
        if started:
            with contextlib.suppress(Exception):
                await server.close()
        else:
            with contextlib.suppress(Exception):
                if server._owns_session and not server._client_session.closed:  # noqa: SLF001
                    await server._client_session.close()  # noqa: SLF001

    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(_main_async(sys.argv[1:])))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
