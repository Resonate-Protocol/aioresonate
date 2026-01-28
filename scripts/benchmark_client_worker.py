"""Client worker process for benchmark_server.py - runs in separate process."""

# ruff: noqa

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
from collections.abc import Iterable

from aiohttp import ClientSession, TCPConnector

from aiosendspin.client import SendspinClient
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles

logger = logging.getLogger(__name__)


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Client worker for server benchmark")
    parser.add_argument("--url", required=True, help="WebSocket URL to connect to")
    parser.add_argument("--start-index", type=int, required=True, help="Starting client index")
    parser.add_argument("--count", type=int, required=True, help="Number of clients")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--bit-depth", type=int, default=16)
    parser.add_argument("--log-level", default="WARNING")
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


async def _run_clients(args: argparse.Namespace) -> None:
    """Connect clients and keep them running until SIGTERM/SIGINT."""
    stop_event = asyncio.Event()

    def _signal_handler():
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    player_support = _make_player_support(args.sample_rate, args.channels, args.bit_depth)
    clients: list[SendspinClient] = []
    chunks_received = 0
    bytes_received = 0

    def _on_chunk(ts: int, data: bytes, fmt: object) -> None:
        nonlocal chunks_received, bytes_received
        chunks_received += 1
        bytes_received += len(data)

    connector = TCPConnector(limit=0, limit_per_host=0)
    async with ClientSession(connector=connector) as session:
        # Connect all clients
        for i in range(args.count):
            idx = args.start_index + i
            client_id = f"bench-{idx:05d}"
            client = SendspinClient(
                client_id=client_id,
                client_name=client_id,
                roles=[Roles.PLAYER],
                player_support=player_support,
                session=session,
            )
            client.add_audio_chunk_listener(_on_chunk)

            try:
                await asyncio.wait_for(client.connect(args.url), timeout=10.0)
                clients.append(client)
            except Exception as e:
                logger.warning("Failed to connect %s: %s", client_id, e)

        logger.warning("Connected %d/%d clients", len(clients), args.count)

        # Signal ready by printing to stdout
        print(f"READY {len(clients)}", flush=True)

        # Wait for stop signal
        await stop_event.wait()

        # Disconnect
        for client in clients:
            with contextlib.suppress(Exception):
                await client.disconnect()

    print(f"DONE chunks={chunks_received} bytes={bytes_received}", flush=True)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [client] %(levelname)s %(message)s",
    )
    asyncio.run(_run_clients(args))


if __name__ == "__main__":
    main()
