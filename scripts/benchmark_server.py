"""Benchmark server performance with clients in separate processes."""

# ruff: noqa

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.server import SendspinServer

logger = logging.getLogger(__name__)

# Path to the client worker script
CLIENT_WORKER = Path(__file__).parent / "benchmark_client_worker.py"


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark server with clients in separate processes. "
            "Profile the server process to see server-only hotspots."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host to bind")
    parser.add_argument("--port", type=int, default=19375, help="Server port to bind")
    parser.add_argument("--clients", type=int, default=100, help="Number of clients")
    parser.add_argument(
        "--client-processes",
        type=int,
        default=4,
        help="Number of client worker processes",
    )
    parser.add_argument("--duration-s", type=int, default=10, help="Stream duration")
    parser.add_argument("--chunk-ms", type=int, default=25, help="PCM chunk size in ms")
    parser.add_argument("--sample-rate", type=int, default=48_000, help="PCM sample rate")
    parser.add_argument("--channels", type=int, default=2, help="PCM channel count")
    parser.add_argument("--bit-depth", type=int, default=16, help="PCM bit depth")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args(argv)


def _make_audio_format(sample_rate: int, channels: int, bit_depth: int) -> AudioFormat:
    return AudioFormat(sample_rate=sample_rate, bit_depth=bit_depth, channels=channels)


def _make_silence_chunk(fmt: AudioFormat, chunk_ms: int) -> bytes:
    frames = int(fmt.sample_rate * (chunk_ms / 1000.0))
    frame_bytes = fmt.channels * (fmt.bit_depth // 8)
    return b"\x00" * (frames * frame_bytes)


async def _run_server(args: argparse.Namespace) -> None:
    """Run server and stream audio."""
    loop = asyncio.get_running_loop()
    server = SendspinServer(loop=loop, server_id="bench", server_name="bench")

    await server.start_server(
        port=args.port,
        host=args.host,
        advertise_addresses=[],
        discover_clients=False,
    )

    url = f"ws://{args.host}:{args.port}{server.API_PATH}"
    logger.warning("Server started at %s", url)

    # Calculate client distribution across processes
    clients_per_worker = args.clients // args.client_processes
    remainder = args.clients % args.client_processes

    # Spawn client worker subprocesses
    workers: list[asyncio.subprocess.Process] = []
    start_idx = 0

    logger.warning(
        "Spawning %d client processes for %d total clients...",
        args.client_processes,
        args.clients,
    )

    for i in range(args.client_processes):
        count = clients_per_worker + (1 if i < remainder else 0)
        if count == 0:
            continue

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(CLIENT_WORKER),
            "--url",
            url,
            "--start-index",
            str(start_idx),
            "--count",
            str(count),
            "--sample-rate",
            str(args.sample_rate),
            "--channels",
            str(args.channels),
            "--bit-depth",
            str(args.bit_depth),
            "--log-level",
            args.log_level,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        workers.append(proc)
        start_idx += count

    # Wait for all workers to report READY
    total_connected = 0
    for proc in workers:
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=30.0)
            if line.startswith(b"READY"):
                parts = line.decode().strip().split()
                if len(parts) >= 2:
                    total_connected += int(parts[1])
        except TimeoutError:
            logger.error("Timeout waiting for client worker")

    logger.warning("Client workers report %d total connected", total_connected)

    # Wait for server to see all clients
    await asyncio.sleep(0.5)

    connected_clients = list(server.connected_clients)
    logger.warning("Server sees %d connected clients", len(connected_clients))

    if not connected_clients:
        logger.error("No clients connected, aborting")
        for proc in workers:
            proc.terminate()
        await server.close()
        return

    # Group all clients together
    leader = connected_clients[0]
    group = leader.group
    for client in connected_clients[1:]:
        await group.add_client(client)

    # Start streaming
    stream = group.start_stream()
    fmt = _make_audio_format(args.sample_rate, args.channels, args.bit_depth)
    chunk = _make_silence_chunk(fmt, args.chunk_ms)

    logger.warning(
        "Streaming for %ds with %dms chunks @ %dHz %dch %dbit to %d clients",
        args.duration_s,
        args.chunk_ms,
        args.sample_rate,
        args.channels,
        args.bit_depth,
        len(connected_clients),
    )

    start = time.monotonic()
    next_tick = start
    chunks_sent = 0

    while True:
        now = time.monotonic()
        if now - start >= args.duration_s:
            break

        stream.prepare_audio(chunk, fmt)
        await stream.commit_audio()
        chunks_sent += 1

        next_tick += args.chunk_ms / 1000.0
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    stream.stop()
    elapsed = time.monotonic() - start

    logger.warning(
        "Streaming complete: %d chunks in %.2fs (%.1f chunks/sec)",
        chunks_sent,
        elapsed,
        chunks_sent / elapsed,
    )

    # Terminate workers and collect their output
    for proc in workers:
        proc.terminate()

    for proc in workers:
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            for line in stdout.decode().splitlines():
                if line.startswith("DONE"):
                    logger.warning("Worker: %s", line)
        except TimeoutError:
            proc.kill()

    await server.close()


def main() -> None:
    args = _parse_args(sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [server] %(levelname)s %(message)s",
    )

    try:
        asyncio.run(_run_server(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
