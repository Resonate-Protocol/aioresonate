"""Command-line interface for running a Resonate client."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf

from aioresonate.client import ResonateClient
from aioresonate.models.controller import GroupUpdateServerPayload
from aioresonate.models.core import SessionUpdatePayload
from aioresonate.models.metadata import SessionUpdateMetadata
from aioresonate.models.types import MediaCommand, PlaybackStateType, UndefinedField

logger = logging.getLogger(__name__)


SERVICE_TYPE = "_resonate-server._tcp.local."
DEFAULT_PATH = "/resonate"


@dataclass
class CLIState:
    """Holds state mirrored from the server for CLI presentation."""

    playback_state: PlaybackStateType | None = None
    supported_commands: set[MediaCommand] = field(default_factory=set)
    volume: int | None = None
    muted: bool | None = None
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    track_progress: int | None = None
    track_duration: int | None = None

    def update_metadata(self, metadata: SessionUpdateMetadata) -> bool:
        """Merge new metadata into the state and report if anything changed."""
        changed = False
        for attr in ("title", "artist", "album", "track_progress", "track_duration"):
            value = getattr(metadata, attr)
            if isinstance(value, UndefinedField):
                continue
            if getattr(self, attr) != value:
                setattr(self, attr, value)
                changed = True
        return changed

    def describe(self) -> str:
        """Return a human-friendly description of the current state."""
        lines: list[str] = []
        if self.title:
            lines.append(f"Now playing: {self.title}")
        if self.artist:
            lines.append(f"Artist: {self.artist}")
        if self.album:
            lines.append(f"Album: {self.album}")
        if self.track_duration:
            progress = self.track_progress or 0
            lines.append(f"Progress: {progress:>2} / {self.track_duration:>2} s")
        if self.volume is not None:
            vol_line = f"Volume: {self.volume}%"
            if self.muted:
                vol_line += " (muted)"
            lines.append(vol_line)
        if self.playback_state is not None:
            lines.append(f"State: {self.playback_state.value}")
        return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the resonate client."""
    parser = argparse.ArgumentParser(description="Run a Resonate CLI client")
    parser.add_argument(
        "--url",
        default=None,
        help=("WebSocket URL of the Resonate server. If omitted, discover via mDNS."),
    )
    parser.add_argument(
        "--name",
        default="Resonate CLI",
        help="Friendly name for this client",
    )
    parser.add_argument(
        "--id",
        default="resonate-cli",
        help="Unique identifier for this client",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level to use",
    )
    parser.add_argument(
        "--discover-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for mDNS discovery before giving up",
    )
    parser.add_argument(
        "--static-delay-ms",
        type=float,
        default=0.0,
        help="Extra playback delay in milliseconds applied after clock sync",
    )
    return parser.parse_args(argv)


async def _discover_server(timeout: float) -> str | None:
    """Discover a Resonate server via mDNS."""
    loop = asyncio.get_running_loop()

    class _Listener:
        def __init__(self) -> None:
            self.result: asyncio.Future[str] = loop.create_future()

        def _schedule(
            self, zeroconf: AsyncZeroconf, service_type: str, name: str
        ) -> None:
            if self.result.done():
                return

            async def process() -> None:
                info = await zeroconf.async_get_service_info(service_type, name)
                if info is None:
                    return
                addresses = info.parsed_addresses()
                if not addresses:
                    return
                host = addresses[0]
                path_raw = info.properties.get(b"path")
                path = path_raw.decode("utf-8", "ignore") if isinstance(path_raw, bytes) else DEFAULT_PATH
                if not path:
                    path = DEFAULT_PATH
                if not path.startswith("/"):
                    path = '/' + path
                host_fmt = f"[{host}]" if ":" in host else host
                url = f"ws://{host_fmt}:{info.port}{path}"
                if not self.result.done():
                    self.result.set_result(url)

            loop.create_task(process())

        def add_service(
            self, zeroconf: AsyncZeroconf, service_type: str, name: str
        ) -> None:
            self._schedule(zeroconf, service_type, name)

        def update_service(
            self, zeroconf: AsyncZeroconf, service_type: str, name: str
        ) -> None:
            self._schedule(zeroconf, service_type, name)

        def remove_service(
            self, zeroconf: AsyncZeroconf, service_type: str, name: str
        ) -> None:
            return

    listener = _Listener()
    async with AsyncZeroconf() as zeroconf:
        browser = AsyncServiceBrowser(zeroconf.zeroconf, SERVICE_TYPE, cast(Any, listener))
        try:
            return await asyncio.wait_for(listener.result, timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            await browser.async_cancel()


async def main_async(argv: Sequence[str] | None = None) -> int:
    """Entry point executing the asynchronous CLI workflow."""
    args = parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=getattr(logging, args.log_level))

    state = CLIState()
    client = ResonateClient(
        client_id=args.id,
        client_name=args.name,
        static_delay_ms=args.static_delay_ms,
    )

    client.add_metadata_listener(lambda payload: _handle_session_update(state, payload))
    client.add_group_update_listener(lambda payload: _handle_group_update(state, payload))
    client.add_stream_start_listener(lambda _message: _print_event("Stream started"))
    client.add_stream_end_listener(lambda: _print_event("Stream ended"))

    url = args.url
    if url is None:
        url = await _discover_server(args.discover_timeout)
        if url is None:
            logger.error("Failed to discover a Resonate server via mDNS")
            return 1
        logger.info("Discovered Resonate server at %s", url)

    try:
        await client.connect(url)
    except Exception:  # pragma: no cover - network failure path
        logger.exception("Failed to connect to %s", url)
        await client.disconnect()
        return 1

    if not client.audio_available:
        _print_event("Audio playback disabled (sounddevice not installed)")

    _print_instructions()

    keyboard_task = asyncio.create_task(_keyboard_loop(client, state))

    try:
        await keyboard_task
    except asyncio.CancelledError:  # pragma: no cover - cancellation path
        pass
    finally:
        await client.disconnect()

    return 0


async def _handle_session_update(state: CLIState, payload: SessionUpdatePayload) -> None:
    if payload.playback_state is not None and payload.playback_state != state.playback_state:
        state.playback_state = payload.playback_state
        _print_event(f"Playback state: {payload.playback_state.value}")

    if payload.metadata is not None and state.update_metadata(payload.metadata):
        _print_event(state.describe())


async def _handle_group_update(state: CLIState, payload: GroupUpdateServerPayload) -> None:
    supported: set[MediaCommand] = set()
    for command in payload.supported_commands:
        try:
            supported.add(command if isinstance(command, MediaCommand) else MediaCommand(command))
        except ValueError:
            continue
    state.supported_commands = supported

    if payload.volume != state.volume:
        state.volume = payload.volume
        _print_event(f"Volume: {payload.volume}%")
    if payload.muted != state.muted:
        state.muted = payload.muted
        _print_event("Muted" if payload.muted else "Unmuted")


async def _keyboard_loop(client: ResonateClient, state: CLIState) -> None:
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if line == "":  # EOF
            break
        raw_line = line.strip()
        if not raw_line:
            continue
        parts = raw_line.split()
        command_lower = raw_line.lower()
        keyword = parts[0].lower()
        if command_lower in {"quit", "exit", "q"}:
            break
        if command_lower in {"play", "p"}:
            await _send_media_command(client, state, MediaCommand.PLAY)
        elif command_lower in {"pause", "space"}:
            await _send_media_command(client, state, MediaCommand.PAUSE)
        elif command_lower in {"stop", "s"}:
            await _send_media_command(client, state, MediaCommand.STOP)
        elif command_lower in {"next", "n"}:
            await _send_media_command(client, state, MediaCommand.NEXT)
        elif command_lower in {"previous", "prev", "b"}:
            await _send_media_command(client, state, MediaCommand.PREVIOUS)
        elif command_lower in {"vol+", "volume+", "+"}:
            await _change_volume(client, state, 5)
        elif command_lower in {"vol-", "volume-", "-"}:
            await _change_volume(client, state, -5)
        elif command_lower in {"mute", "m"}:
            await _toggle_mute(client, state)
        elif command_lower == "toggle":
            await _toggle_play_pause(client, state)
        elif keyword == "delay":
            _handle_delay_command(client, parts)
        else:
            _print_event("Unknown command")


async def _send_media_command(
    client: ResonateClient, state: CLIState, command: MediaCommand
) -> None:
    if command not in state.supported_commands:
        _print_event(f"Server does not support {command.value}")
        return
    await client.send_group_command(command)


async def _toggle_play_pause(client: ResonateClient, state: CLIState) -> None:
    if state.playback_state == PlaybackStateType.PLAYING:
        await _send_media_command(client, state, MediaCommand.PAUSE)
    else:
        await _send_media_command(client, state, MediaCommand.PLAY)


async def _change_volume(client: ResonateClient, state: CLIState, delta: int) -> None:
    if MediaCommand.VOLUME not in state.supported_commands:
        _print_event("Server does not support volume control")
        return
    current = state.volume if state.volume is not None else 50
    target = max(0, min(100, current + delta))
    await client.send_group_command(MediaCommand.VOLUME, volume=target)


async def _toggle_mute(client: ResonateClient, state: CLIState) -> None:
    if MediaCommand.MUTE not in state.supported_commands:
        _print_event("Server does not support mute control")
        return
    target = not bool(state.muted)
    await client.send_group_command(MediaCommand.MUTE, mute=target)

def _handle_delay_command(client: ResonateClient, parts: list[str]) -> None:
    """Process delay commands from the keyboard loop."""
    if not parts or parts[0].lower() != "delay":
        return
    if len(parts) == 1:
        _print_event(f"Static delay: {client.static_delay_ms:.1f} ms")
        return
    if len(parts) == 3 and parts[1] in {"+", "-"}:
        try:
            delta = float(parts[2])
        except ValueError:
            _print_event("Invalid delay value")
            return
        if parts[1] == "-":
            delta = -delta
        client.set_static_delay_ms(client.static_delay_ms + delta)
        _print_event(f"Static delay: {client.static_delay_ms:.1f} ms")
        return
    if len(parts) == 2:
        try:
            value = float(parts[1])
        except ValueError:
            _print_event("Invalid delay value")
            return
        client.set_static_delay_ms(value)
        _print_event(f"Static delay: {client.static_delay_ms:.1f} ms")
        return
    _print_event("Usage: delay [<ms>|+ <ms>|- <ms>]")


def _print_event(message: str) -> None:
    print(message, flush=True)  # noqa: T201


def _print_instructions() -> None:
    print(  # noqa: T201
        ("Commands: play(p), pause, stop(s), next(n), prev(b), vol+/-, mute, toggle, delay, quit(q)\n"
        "  delay [<ms>|+ <ms>|- <ms>] shows or adjusts the static delay"),
        flush=True,
    )


def main() -> int:
    """Run the CLI client."""
    try:
        return asyncio.run(main_async(sys.argv[1:]))
    except KeyboardInterrupt:  # pragma: no cover - signal handling
        print()  # noqa: T201
        return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
