"""Command-line interface for running a Resonate client."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from zeroconf import ServiceListener

import aioconsole
from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf

from aioresonate.cli_audio import AudioPlayer
from aioresonate.client import PCMFormat, ResonateClient
from aioresonate.models.controller import GroupUpdateServerPayload
from aioresonate.models.core import SessionUpdatePayload, StreamStartMessage
from aioresonate.models.metadata import ClientHelloMetadataSupport, SessionUpdateMetadata
from aioresonate.models.player import ClientHelloPlayerSupport
from aioresonate.models.types import MediaCommand, PlaybackStateType, Roles, UndefinedField

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


def _build_service_url(host: str, port: int, properties: dict[bytes, bytes | None]) -> str:
    """Construct WebSocket URL from mDNS service info."""
    path_raw = properties.get(b"path")
    path = path_raw.decode("utf-8", "ignore") if isinstance(path_raw, bytes) else DEFAULT_PATH
    if not path:
        path = DEFAULT_PATH
    if not path.startswith("/"):
        path = "/" + path
    host_fmt = f"[{host}]" if ":" in host else host
    return f"ws://{host_fmt}:{port}{path}"


class _ServiceDiscoveryListener:
    """Listens for Resonate server advertisements via mDNS."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.result: asyncio.Future[str] = loop.create_future()
        self.tasks: set[asyncio.Task[None]] = set()
        self._loop = loop

    async def _process_service_info(
        self, zeroconf: AsyncZeroconf, service_type: str, name: str
    ) -> None:
        """Extract and construct WebSocket URL from service info."""
        info = await zeroconf.async_get_service_info(service_type, name)
        if info is None or info.port is None:
            return
        addresses = info.parsed_addresses()
        if not addresses:
            return
        host = addresses[0]
        url = _build_service_url(host, info.port, info.properties)
        if not self.result.done():
            self.result.set_result(url)

    def _schedule(self, zeroconf: AsyncZeroconf, service_type: str, name: str) -> None:
        if self.result.done():
            return
        task = self._loop.create_task(self._process_service_info(zeroconf, service_type, name))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def add_service(self, zeroconf: AsyncZeroconf, service_type: str, name: str) -> None:
        self._schedule(zeroconf, service_type, name)

    def update_service(self, zeroconf: AsyncZeroconf, service_type: str, name: str) -> None:
        self._schedule(zeroconf, service_type, name)

    def remove_service(self, _zeroconf: AsyncZeroconf, _service_type: str, _name: str) -> None:
        return


class ServiceDiscovery:
    """Manages continuous discovery of Resonate servers via mDNS."""

    def __init__(self) -> None:
        """Initialize the service discovery manager."""
        self._listener: _ServiceDiscoveryListener | None = None
        self._browser: AsyncServiceBrowser | None = None
        self._zeroconf: AsyncZeroconf | None = None

    async def discover_first(self, discovery_timeout: float) -> str | None:
        """
        Discover the first available Resonate server.

        Args:
            discovery_timeout: Seconds to wait for discovery.

        Returns:
            WebSocket URL of the server, or None if discovery timed out.
        """
        loop = asyncio.get_running_loop()
        self._listener = _ServiceDiscoveryListener(loop)
        self._zeroconf = AsyncZeroconf()
        await self._zeroconf.__aenter__()

        try:
            self._browser = AsyncServiceBrowser(
                self._zeroconf.zeroconf, SERVICE_TYPE, cast("ServiceListener", self._listener)
            )
            return await asyncio.wait_for(self._listener.result, discovery_timeout)
        except TimeoutError:
            return None
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop discovery and clean up resources."""
        if self._browser:
            await self._browser.async_cancel()
            self._browser = None
        if self._zeroconf:
            await self._zeroconf.__aexit__(None, None, None)
            self._zeroconf = None
        self._listener = None


async def _discover_server(discovery_timeout: float) -> str | None:
    """Discover a Resonate server via mDNS."""
    discovery = ServiceDiscovery()
    try:
        return await discovery.discover_first(discovery_timeout)
    finally:
        await discovery.stop()


def _create_audio_chunk_handler(
    client: ResonateClient,
) -> tuple[
    Callable[[int, bytes, PCMFormat], None],
    Callable[[], AudioPlayer | None],
]:
    """
    Create an audio chunk handler and accessor for the audio player.

    Returns:
        A tuple of (handler_function, get_audio_player_function).
    """
    audio_player: AudioPlayer | None = None
    current_format: PCMFormat | None = None
    sync_ready: bool = False
    first_sync_message_printed: bool = False
    dropped_chunks: int = 0

    def handle_audio_chunk(server_timestamp_us: int, audio_data: bytes, fmt: PCMFormat) -> None:
        """Handle incoming audio chunks."""
        nonlocal audio_player, current_format, sync_ready, first_sync_message_printed
        nonlocal dropped_chunks

        # Check if time sync is ready - critical for accurate playback timing
        was_sync_ready = sync_ready
        sync_ready = client.is_time_synchronized()

        # Print message when sync becomes ready
        if sync_ready and not was_sync_ready:
            if dropped_chunks > 0:
                logger.info("Time sync ready after dropping %d early chunks", dropped_chunks)
                _print_event(
                    f"Time sync ready (dropped {dropped_chunks} early chunks), starting playback"
                )
            else:
                logger.info("Time sync ready")
                _print_event("Time sync ready, starting playback")
            dropped_chunks = 0

        # Drop chunks if sync is not ready - prevents desync on initial connection
        if not sync_ready:
            dropped_chunks += 1
            if not first_sync_message_printed:
                logger.debug("Waiting for time sync to converge before playing audio...")
                _print_event("Waiting for time synchronization...")
                first_sync_message_printed = True
            return

        # Initialize or reconfigure audio player if format changed
        if audio_player is None or current_format != fmt:
            if audio_player is not None:
                audio_player.clear()

            loop = asyncio.get_running_loop()
            # Use client's public time conversion methods (based on monotonic loop time)
            audio_player = AudioPlayer(loop, client.compute_play_time, client.compute_server_time)
            audio_player.set_format(fmt)
            current_format = fmt

        # Submit audio chunk with server timestamp (AudioPlayer will compute client play time)
        if audio_player is not None:
            audio_player.submit(server_timestamp_us, audio_data)

    def get_audio_player() -> AudioPlayer | None:
        return audio_player

    return handle_audio_chunk, get_audio_player


def _create_stream_handlers(
    get_audio_player: Callable[[], AudioPlayer | None],
) -> tuple[Callable[[StreamStartMessage], None], Callable[[], None]]:
    """
    Create stream start/end handlers that clear audio queue.

    Returns:
        A tuple of (stream_start_handler, stream_end_handler).
    """

    def handle_stream_start(_message: StreamStartMessage) -> None:
        """Handle stream start by clearing stale audio chunks."""
        audio_player = get_audio_player()
        if audio_player is not None:
            audio_player.clear()
            logger.debug("Cleared audio queue on stream start")
        _print_event("Stream started")

    def handle_stream_end() -> None:
        """Handle stream end by clearing audio queue to prevent desync on resume."""
        audio_player = get_audio_player()
        if audio_player is not None:
            audio_player.clear()
            logger.debug("Cleared audio queue on stream end")
        _print_event("Stream ended")

    return handle_stream_start, handle_stream_end


async def main_async(argv: Sequence[str] | None = None) -> int:
    """Entry point executing the asynchronous CLI workflow."""
    args = parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=getattr(logging, args.log_level))

    state = CLIState()
    client = ResonateClient(
        client_id=args.id,
        client_name=args.name,
        roles=[Roles.CONTROLLER, Roles.PLAYER, Roles.METADATA],
        player_support=ClientHelloPlayerSupport(
            support_codecs=["pcm"],
            support_channels=[2, 1],
            support_sample_rates=[44_100],
            support_bit_depth=[16],
            buffer_capacity=1_000_000,
        ),
        metadata_support=ClientHelloMetadataSupport(
            support_picture_formats=[],
            media_width=None,
            media_height=None,
        ),
        static_delay_ms=args.static_delay_ms,
    )

    # Create audio and stream handlers
    handle_audio_chunk, get_audio_player = _create_audio_chunk_handler(client)
    handle_stream_start, handle_stream_end = _create_stream_handlers(get_audio_player)

    client.set_metadata_listener(lambda payload: _handle_session_update(state, payload))
    client.set_group_update_listener(lambda payload: _handle_group_update(state, payload))
    client.set_stream_start_listener(handle_stream_start)
    client.set_stream_end_listener(handle_stream_end)
    client.set_audio_chunk_listener(handle_audio_chunk)

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

    # Audio player will be created when first audio chunk arrives

    _print_instructions()

    keyboard_task = asyncio.create_task(_keyboard_loop(client, state, get_audio_player))

    # Set up signal handler for graceful shutdown on Ctrl+C
    loop = asyncio.get_running_loop()

    def signal_handler() -> None:
        logger.debug("Received interrupt signal, shutting down...")
        keyboard_task.cancel()

    loop.add_signal_handler(signal.SIGINT, signal_handler)

    try:
        await keyboard_task
    except asyncio.CancelledError:  # pragma: no cover - cancellation path
        logger.debug("Keyboard task cancelled")
    finally:
        # Remove signal handler
        loop.remove_signal_handler(signal.SIGINT)
        audio_player = get_audio_player()
        if audio_player is not None:
            await audio_player.stop()
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


async def _keyboard_loop(
    client: ResonateClient,
    state: CLIState,
    get_audio_player: Callable[[], AudioPlayer | None],
) -> None:
    try:
        while True:
            try:
                line = await aioconsole.ainput()
            except EOFError:
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
                _handle_delay_command(client, parts, get_audio_player)
            else:
                _print_event("Unknown command")
    except asyncio.CancelledError:
        # Graceful shutdown on Ctrl+C
        logger.debug("Keyboard loop cancelled, exiting gracefully")
        raise


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


def _handle_delay_command(
    client: ResonateClient,
    parts: list[str],
    get_audio_player: Callable[[], AudioPlayer | None],
) -> None:
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
        # Clear audio queue to prevent desync from chunks with stale timing
        audio_player = get_audio_player()
        if audio_player is not None:
            audio_player.clear()
            logger.debug("Cleared audio queue after delay change")
        _print_event(f"Static delay: {client.static_delay_ms:.1f} ms")
        return
    if len(parts) == 2:
        try:
            value = float(parts[1])
        except ValueError:
            _print_event("Invalid delay value")
            return
        client.set_static_delay_ms(value)
        # Clear audio queue to prevent desync from chunks with stale timing
        audio_player = get_audio_player()
        if audio_player is not None:
            audio_player.clear()
            logger.debug("Cleared audio queue after delay change")
        _print_event(f"Static delay: {client.static_delay_ms:.1f} ms")
        return
    _print_event("Usage: delay [<ms>|+ <ms>|- <ms>]")


def _print_event(message: str) -> None:
    print(message, flush=True)  # noqa: T201


def _print_instructions() -> None:
    print(  # noqa: T201
        (
            "Commands: play(p), pause, stop(s), next(n), prev(b), vol+/-, mute, toggle, delay, "
            "quit(q)\n  delay [<ms>|+ <ms>|- <ms>] shows or adjusts the static delay"
        ),
        flush=True,
    )


def main() -> int:
    """Run the CLI client."""
    return asyncio.run(main_async(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
