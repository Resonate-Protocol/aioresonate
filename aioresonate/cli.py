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
from aiohttp import ClientError
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
        self._loop = loop
        self._current_url: str | None = None
        self._first_result: asyncio.Future[str] = loop.create_future()
        self.tasks: set[asyncio.Task[None]] = set()

    @property
    def current_url(self) -> str | None:
        """Get the current discovered server URL, or None if no servers."""
        return self._current_url

    async def wait_for_first(self) -> str:
        """Wait for the first server to be discovered."""
        return await self._first_result

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
        self._current_url = url

        # Signal first server discovery
        if not self._first_result.done():
            self._first_result.set_result(url)

    def _schedule(self, zeroconf: AsyncZeroconf, service_type: str, name: str) -> None:
        task = self._loop.create_task(self._process_service_info(zeroconf, service_type, name))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def add_service(self, zeroconf: AsyncZeroconf, service_type: str, name: str) -> None:
        self._schedule(zeroconf, service_type, name)

    def update_service(self, zeroconf: AsyncZeroconf, service_type: str, name: str) -> None:
        self._schedule(zeroconf, service_type, name)

    def remove_service(self, _zeroconf: AsyncZeroconf, _service_type: str, _name: str) -> None:
        """Handle service removal (server offline)."""
        self._current_url = None


class ServiceDiscovery:
    """Manages continuous discovery of Resonate servers via mDNS."""

    def __init__(self) -> None:
        """Initialize the service discovery manager."""
        self._listener: _ServiceDiscoveryListener | None = None
        self._browser: AsyncServiceBrowser | None = None
        self._zeroconf: AsyncZeroconf | None = None

    async def start(self) -> None:
        """Start continuous discovery (keeps running until stop() is called)."""
        loop = asyncio.get_running_loop()
        self._listener = _ServiceDiscoveryListener(loop)
        self._zeroconf = AsyncZeroconf()
        await self._zeroconf.__aenter__()

        try:
            self._browser = AsyncServiceBrowser(
                self._zeroconf.zeroconf, SERVICE_TYPE, cast("ServiceListener", self._listener)
            )
        except Exception:
            await self.stop()
            raise

    async def wait_for_first_server(self) -> str:
        """Wait indefinitely for the first server to be discovered."""
        if self._listener is None:
            raise RuntimeError("Discovery not started. Call start() first.")
        return await self._listener.wait_for_first()

    def current_url(self) -> str | None:
        """Get the current discovered server URL, or None if no servers."""
        return self._listener.current_url if self._listener else None

    async def stop(self) -> None:
        """Stop discovery and clean up resources."""
        if self._browser:
            await self._browser.async_cancel()
            self._browser = None
        if self._zeroconf:
            await self._zeroconf.__aexit__(None, None, None)
            self._zeroconf = None
        self._listener = None


async def _sleep_interruptible(duration: float, keyboard_task: asyncio.Task[None]) -> bool:
    """Sleep with keyboard interrupt support. Return True if interrupted."""
    remaining = duration
    while remaining > 0 and not keyboard_task.done():
        await asyncio.sleep(min(0.5, remaining))
        remaining -= 0.5
    return keyboard_task.done()


def _should_reset_backoff(current_url: str | None, last_attempted_url: str) -> bool:
    """Check if URL changed, indicating server came back online."""
    return bool(current_url and current_url != last_attempted_url)


def _update_backoff_and_url(
    current_url: str | None,
    last_attempted_url: str,
    error_backoff: float,
    max_backoff: float,
) -> tuple[str | None, str, float]:
    """Update URL and backoff based on discovery.

    Returns (new_url, new_last_attempted, new_backoff).
    """
    if _should_reset_backoff(current_url, last_attempted_url):
        logger.info("Server URL changed to %s, reconnecting immediately", current_url)
        assert current_url is not None
        return current_url, current_url, 1.0
    return None, last_attempted_url, min(error_backoff * 2, max_backoff)


async def _handle_error_backoff(error_backoff: float, keyboard_task: asyncio.Task[None]) -> bool:
    """Sleep for error backoff with keyboard interrupt support.

    Returns True if interrupted by keyboard, False if completed normally.
    """
    _print_event(f"Connection error, retrying in {error_backoff:.0f}s...")
    return await _sleep_interruptible(error_backoff, keyboard_task)


async def _wait_for_server_reappear(
    discovery: ServiceDiscovery, keyboard_task: asyncio.Task[None]
) -> str | None:
    """Wait for server to reappear on the network.

    Returns the new URL if server reappears, None if interrupted.
    """
    logger.info("Server offline, waiting for rediscovery...")
    _print_event("Waiting for server...")

    while not (new_url := discovery.current_url()) and not keyboard_task.done():  # noqa: ASYNC110
        await asyncio.sleep(1.0)

    return new_url


async def _connection_loop(
    client: ResonateClient,
    discovery: ServiceDiscovery,
    audio_handler: AudioStreamHandler,
    initial_url: str,
    keyboard_task: asyncio.Task[None],
) -> None:
    """
    Run the connection loop with automatic reconnection on disconnect.

    Connects to the server, waits for disconnect, cleans up, then retries
    only if the server is visible via mDNS. Reconnects immediately when
    server reappears. Uses exponential backoff (up to 5 min) for errors.

    Args:
        client: Resonate client instance.
        discovery: Service discovery manager.
        audio_handler: Audio stream handler.
        initial_url: Initial server URL.
        keyboard_task: Keyboard input task to monitor.
    """
    url = initial_url
    last_attempted_url = url
    error_backoff = 1.0
    max_backoff = 300.0  # 5 minutes

    while not keyboard_task.done():
        try:
            await client.connect(url)
            logger.info("Connected to %s", url)
            _print_event(f"Connected to {url}")
            error_backoff = 1.0  # Reset backoff on successful connect
            last_attempted_url = url

            # Wait for disconnect or keyboard exit
            while client.connected and not keyboard_task.done():  # noqa: ASYNC110
                await asyncio.sleep(0.5)

            if keyboard_task.done():
                break

            # Connection dropped
            logger.info("Connection lost")
            _print_event("Connection lost")

            # Clean up audio state
            if audio_handler.audio_player is not None:
                await audio_handler.audio_player.stop()
                audio_handler.audio_player = None
            audio_handler.reset_sync_state()

            # Update URL from discovery
            new_url = discovery.current_url()

            # Wait for server to reappear if it's gone
            if not new_url:
                new_url = await _wait_for_server_reappear(discovery, keyboard_task)
                if keyboard_task.done():
                    break

            # Use the discovered URL
            if new_url:
                url = new_url
            _print_event(f"Reconnecting to {url}...")

        except (TimeoutError, OSError, ClientError) as e:
            # Network-related errors - log cleanly
            logger.debug(
                "Connection error (%s), retrying in %.0fs", type(e).__name__, error_backoff
            )

            if await _handle_error_backoff(error_backoff, keyboard_task):
                break

            # Check if URL changed while sleeping
            current_url = discovery.current_url()
            new_url, last_attempted_url, error_backoff = _update_backoff_and_url(
                current_url, last_attempted_url, error_backoff, max_backoff
            )
            if new_url:
                url = new_url
        except Exception:
            # Unexpected errors - log with full traceback
            logger.exception("Unexpected error during connection")
            _print_event("Unexpected error occurred")
            await asyncio.sleep(error_backoff)
            error_backoff = min(error_backoff * 2, max_backoff)


class AudioStreamHandler:
    """Manages audio playback state and stream lifecycle."""

    def __init__(self, client: ResonateClient) -> None:
        """Initialize the audio stream handler."""
        self._client = client
        self.audio_player: AudioPlayer | None = None
        self._current_format: PCMFormat | None = None
        self._sync_ready = False
        self._first_sync_message_printed = False
        self._dropped_chunks = 0

    def on_audio_chunk(self, server_timestamp_us: int, audio_data: bytes, fmt: PCMFormat) -> None:
        """Handle incoming audio chunks with time sync validation."""
        # Check if time sync is ready - critical for accurate playback timing
        was_sync_ready = self._sync_ready
        self._sync_ready = self._client.is_time_synchronized()

        # Print message when sync becomes ready
        if self._sync_ready and not was_sync_ready:
            if self._dropped_chunks > 0:
                logger.info("Time sync ready after dropping %d early chunks", self._dropped_chunks)
                _print_event(
                    f"Time sync ready (dropped {self._dropped_chunks} early chunks), "
                    "starting playback"
                )
            else:
                logger.info("Time sync ready")
                _print_event("Time sync ready, starting playback")
            self._dropped_chunks = 0

        # Drop chunks if sync is not ready - prevents desync on initial connection
        if not self._sync_ready:
            self._dropped_chunks += 1
            if not self._first_sync_message_printed:
                logger.debug("Waiting for time sync to converge before playing audio...")
                _print_event("Waiting for time synchronization...")
                self._first_sync_message_printed = True
            return

        # Initialize or reconfigure audio player if format changed
        if self.audio_player is None or self._current_format != fmt:
            if self.audio_player is not None:
                self.audio_player.clear()

            loop = asyncio.get_running_loop()
            # Use client's public time conversion methods (based on monotonic loop time)
            self.audio_player = AudioPlayer(
                loop, self._client.compute_play_time, self._client.compute_server_time
            )
            self.audio_player.set_format(fmt)
            self._current_format = fmt

        # Submit audio chunk with server timestamp (AudioPlayer will compute client play time)
        if self.audio_player is not None:
            self.audio_player.submit(server_timestamp_us, audio_data)

    def on_stream_start(self, _message: StreamStartMessage) -> None:
        """Handle stream start by clearing stale audio chunks."""
        if self.audio_player is not None:
            self.audio_player.clear()
            logger.debug("Cleared audio queue on stream start")
        _print_event("Stream started")

    def on_stream_end(self) -> None:
        """Handle stream end by clearing audio queue to prevent desync on resume."""
        if self.audio_player is not None:
            self.audio_player.clear()
            logger.debug("Cleared audio queue on stream end")
        _print_event("Stream ended")

    def get_audio_player(self) -> AudioPlayer | None:
        """Get the current audio player instance."""
        return self.audio_player

    def reset_sync_state(self) -> None:
        """Reset time synchronization state."""
        self._sync_ready = False
        self._first_sync_message_printed = False
        self._dropped_chunks = 0


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

    # Start service discovery
    discovery = ServiceDiscovery()
    await discovery.start()

    try:
        # Get initial server URL
        url = args.url
        if url is None:
            logger.info("Waiting for mDNS discovery of Resonate server...")
            _print_event("Searching for Resonate server...")
            try:
                url = await discovery.wait_for_first_server()
                logger.info("Discovered Resonate server at %s", url)
                _print_event(f"Found server at {url}")
            except Exception:
                logger.exception("Failed to discover server")
                return 1

        # Create audio and stream handlers
        audio_handler = AudioStreamHandler(client)

        client.set_metadata_listener(lambda payload: _handle_session_update(state, payload))
        client.set_group_update_listener(lambda payload: _handle_group_update(state, payload))
        client.set_stream_start_listener(audio_handler.on_stream_start)
        client.set_stream_end_listener(audio_handler.on_stream_end)
        client.set_audio_chunk_listener(audio_handler.on_audio_chunk)

        # Audio player will be created when first audio chunk arrives

        _print_instructions()

        # Create and start keyboard task
        keyboard_task = asyncio.create_task(
            _keyboard_loop(client, state, audio_handler.get_audio_player)
        )

        # Set up signal handler for graceful shutdown on Ctrl+C
        loop = asyncio.get_running_loop()

        def signal_handler() -> None:
            logger.debug("Received interrupt signal, shutting down...")
            keyboard_task.cancel()

        loop.add_signal_handler(signal.SIGINT, signal_handler)

        try:
            # Run connection loop with auto-reconnect
            await _connection_loop(client, discovery, audio_handler, url, keyboard_task)
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            logger.debug("Connection loop cancelled")
        finally:
            # Remove signal handler
            loop.remove_signal_handler(signal.SIGINT)
            if audio_handler.audio_player is not None:
                await audio_handler.audio_player.stop()
            await client.disconnect()

    finally:
        # Stop discovery
        await discovery.stop()

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
