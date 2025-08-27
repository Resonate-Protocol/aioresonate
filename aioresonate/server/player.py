"""Represents a single player device connected to the server."""

import asyncio
import logging
import struct
from collections.abc import Callable, Coroutine
from contextlib import suppress
from typing import TYPE_CHECKING

from aiohttp import ClientWebSocketResponse, WSMessage, WSMsgType, web
from attr import dataclass

from aioresonate import models
from aioresonate.models import client_messages, server_messages
from aioresonate.models.types import MediaCommand

from .group import PlayerGroup

MAX_PENDING_MSG = 512

logger = logging.getLogger(__name__)

# The cyclic import is not an issue during runtime, so hide it
# pyright: reportImportCycles=none
if TYPE_CHECKING:
    from .server import ResonateServer


class PlayerEvent:
    """Base event type used by Player.add_event_listener()."""


@dataclass
class VolumeChangedEvent(PlayerEvent):
    """The volume or mute status of the player was changed."""

    volume: int
    muted: bool


@dataclass
class StreamStartEvent(PlayerEvent):
    """The player issued a start/play stream command event."""


@dataclass
class StreamStopEvent(PlayerEvent):
    """The player issued a stop stream command event."""


@dataclass
class StreamPauseEvent(PlayerEvent):
    """The player issued a pause stream command event."""


class Player:
    """A Player that is connected to a ResonateServer.

    Playback is handled through groups, use Player.group to get the
    assigned group.
    """

    _server: "ResonateServer"
    wsock_client: ClientWebSocketResponse | None = None
    wsock_server: web.WebSocketResponse | None = None
    request: web.Request | None = None
    url: str | None = None
    _player_id: str | None = None
    player_info: client_messages.ClientHelloPayload | None = None
    # Task responsible for sending audio and other data
    _writer_task: asyncio.Task[None] | None = None
    _to_write: asyncio.Queue[server_messages.ServerMessage | bytes]
    session_info: server_messages.SessionStartPayload | None = None
    _group: PlayerGroup
    _event_cbs: list[Callable[[PlayerEvent], Coroutine[None, None, None]]]
    _volume: int = 100
    _muted: bool = False
    _on_player_add: Callable[["Player"], None]
    _on_player_remove: Callable[["Player"], None]
    _logger: logging.Logger

    def __init__(
        self,
        server: "ResonateServer",
        on_player_add: Callable[["Player"], None],
        on_player_remove: Callable[["Player"], None],
        request: web.Request | None = None,
        wsock_client: ClientWebSocketResponse | None = None,
    ) -> None:
        """Do not call this constructor.

        Use ResonateServer.on_player_connect or ResonateServer.connect_to_player instead.
        """
        self._server = server
        self._on_player_add = on_player_add
        self._on_player_remove = on_player_remove
        if request is not None:
            self.request = request
            self.wsock_server = web.WebSocketResponse(heartbeat=55)
            self._logger = logger.getChild(f"unknown-{self.request.remote}")
            self._logger.debug("Player initialized")
        elif wsock_client is not None:
            self._logger = logger.getChild("unknown-client")
            self.wsock_client = wsock_client
        else:
            raise ValueError("Either request or wsock_client must be provided")
        self._to_write = asyncio.Queue(maxsize=MAX_PENDING_MSG)
        self._group = PlayerGroup(server, self)
        self._event_cbs = []

    async def disconnect(self) -> None:
        """Disconnect client and cancel tasks."""
        self._logger.debug("Disconnecting client")

        # Cancel running tasks
        if self._writer_task and not self._writer_task.done():
            self._logger.debug("Cancelling writer task")
            _ = self._writer_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._writer_task
        # Handle task is cancelled implicitly when wsock closes or externally

        # Close websocket
        if self.wsock_client is not None and not self.wsock_client.closed:
            _ = await self.wsock_client.close()
        elif self.wsock_server is not None and not self.wsock_server.closed:
            _ = await self.wsock_server.close()

        if self._player_id is not None:
            self._on_player_remove(self)

        self._logger.info("Client disconnected")

    @property
    def group(self) -> PlayerGroup:
        """Get the group assigned to this player."""
        return self._group

    @property
    def player_id(self) -> str:
        """The unique identifier of this Player."""
        # This should only be called once the player was correctly initialized
        assert self._player_id
        return self._player_id

    @property
    def name(self) -> str:
        """The human-readable name of this Player."""
        assert self.player_info  # Player should be fully initialized by now
        return self.player_info.name

    @property
    def info(self) -> client_messages.ClientHelloPayload:
        """List of information and capabilities reported by this player."""
        assert self.player_info  # Player should be fully initialized by now
        return self.player_info

    def set_volume(self, volume: int) -> None:
        """Set the volume of this player."""
        if self._volume == volume:
            return
        self._logger.debug("Setting volume from %d to %d", self._volume, volume)
        self.send_message(
            server_messages.VolumeSetMessage(server_messages.VolumeSetPayload(volume))
        )

    def mute(self) -> None:
        """Mute this player."""
        if self._muted:
            return
        self._logger.debug("Muting player")
        self.send_message(server_messages.MuteSetMessage(server_messages.MuteSetPayload(mute=True)))

    def unmute(self) -> None:
        """Unmute this player."""
        if not self._muted:
            return
        self._logger.debug("Unmuting player")
        self.send_message(
            server_messages.MuteSetMessage(server_messages.MuteSetPayload(mute=False))
        )

    @property
    def muted(self) -> bool:
        """Mute state of this player."""
        return self._muted

    @property
    def volume(self) -> int:
        """Volume of this player."""
        return self._volume

    def ungroup(self) -> None:
        """Remove the player from the group.

        If the player is already alone, this function does nothing.
        """
        if len(self._group.players) > 1:
            self._logger.debug("Ungrouping player from group")
            self._group.remove_player(self)
        else:
            self._logger.debug("Player already alone in group, no ungrouping needed")

    async def _setup_connection(self) -> None:
        """Establish WebSocket connection and return remote address."""
        if self.wsock_server is not None:
            assert self.request is not None
            try:
                async with asyncio.timeout(10):
                    _ = await self.wsock_server.prepare(self.request)
            except TimeoutError:
                self._logger.warning("Timeout preparing request")
                raise

        self._logger.info("Connection established")

        self._logger.debug("Creating writer task")
        self._writer_task = self._server.loop.create_task(self._writer())

        # Send Server Hello
        self._logger.debug("Sending server hello")
        self.send_message(
            server_messages.ServerHelloMessage(
                payload=server_messages.ServerHelloPayload(
                    name=self._server.name,
                    server_id=self._server.id,
                )
            )
        )

    async def _run_message_loop(self) -> None:
        """Run the main message processing loop."""
        wsock = self.wsock_server or self.wsock_client
        assert wsock is not None
        receive_task: asyncio.Task[WSMessage] | None = None
        # Listen for all incoming messages
        try:
            while not wsock.closed:
                # Wait for either a message or the writer task to complete (meaning the player
                # disconnected or errored)
                receive_task = self._server.loop.create_task(wsock.receive())
                assert self._writer_task is not None  # for type checking
                done, pending = await asyncio.wait(
                    [receive_task, self._writer_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if self._writer_task in done:
                    self._logger.warning("Writer task ended, closing connection")
                    # Cancel the receive task if it's still pending
                    if receive_task in pending:
                        _ = receive_task.cancel()
                    break

                # Get the message from the completed receive task
                try:
                    msg = await receive_task
                except (ConnectionError, asyncio.CancelledError, TimeoutError) as e:
                    self._logger.error("Error receiving message: %s", e)
                    break

                timestamp = int(self._server.loop.time() * 1_000_000)

                if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
                    break

                if msg.type != WSMsgType.TEXT:
                    continue

                try:
                    await self._handle_message(
                        client_messages.ClientMessage.from_json(msg.data), timestamp
                    )
                except Exception:
                    self._logger.exception("error parsing message")
            self._logger.debug("wsock was closed")

        except asyncio.CancelledError:
            self._logger.debug("Connection closed by client")
        except Exception:
            self._logger.exception("Unexpected error inside websocket API")
        finally:
            if receive_task and not receive_task.done():
                _ = receive_task.cancel()

    async def _cleanup_connection(self) -> None:
        """Clean up WebSocket connection and tasks."""
        wsock = self.wsock_client or self.wsock_server
        try:
            if wsock and not wsock.closed:
                _ = await wsock.close()
        except Exception:
            self._logger.exception("Failed to close websocket")
        await self.disconnect()

    async def handle_client(self) -> None:
        """Handle the websocket connection."""
        try:
            # Establish connection and setup
            await self._setup_connection()

            # Run the main message loop
            await self._run_message_loop()
        finally:
            # Clean up connection and tasks
            await self._cleanup_connection()

    async def _handle_message(self, message: client_messages.ClientMessage, timestamp: int) -> None:
        """Handle incoming commands from the client."""
        match message:
            case client_messages.ClientHelloMessage(player_info):
                self._logger.info("Received session/hello")
                self.player_info = player_info
                self._player_id = player_info.client_id
                self._logger.info("Player ID set to %s", self._player_id)
                self._logger = logger.getChild(self._player_id)
                self._on_player_add(self)
            case client_messages.PlayerStateMessage(state):
                if not self._player_id:
                    self._logger.warning("Received player/state before session/hello")
                    return
                self._logger.debug(
                    "Received player state: volume=%d, muted=%s", state.volume, state.muted
                )
                if self._muted != state.muted or self._volume != state.volume:
                    self._volume = state.volume
                    self._muted = state.muted
                    self._signal_event(VolumeChangedEvent(volume=self._volume, muted=self._muted))
                # TODO: handle state.state changes, but how?
            case client_messages.ClientTimeMessage(player_time):
                self.send_message(
                    server_messages.ServerTimeMessage(
                        server_messages.ServerTimePayload(
                            client_transmitted=player_time.client_transmitted,
                            server_received=timestamp,
                            server_transmitted=int(self._server.loop.time() * 1_000_000),
                        )
                    )
                )
            case client_messages.StreamCommandMessage(stream_command):
                match stream_command.command:
                    case MediaCommand.PLAY:
                        self._signal_event(StreamStartEvent())
                    case MediaCommand.STOP:
                        self._signal_event(StreamStopEvent())
                    case MediaCommand.PAUSE:
                        self._signal_event(StreamPauseEvent())
                        raise NotImplementedError(
                            f"MediaCommand {stream_command.command} is not supported"
                        )
            case client_messages.ClientMessage():
                pass  # unused base type

    async def _writer(self) -> None:
        """Write outgoing messages from the queue."""
        # Exceptions if Socket disconnected or cancelled by connection handler
        wsock = self.wsock_server or self.wsock_client
        assert wsock is not None
        try:
            while not wsock.closed:
                item = await self._to_write.get()

                if isinstance(item, bytes):
                    _, timestamp_us, _ = struct.unpack(models.BINARY_HEADER_FORMAT, item[:13])
                    now = int(self._server.loop.time() * 1_000_000)
                    if timestamp_us - now < 0:
                        self._logger.error(
                            "Audio chunk after should have played already, skipping it"
                        )
                        continue
                    if timestamp_us - now < 500_000:
                        self._logger.warning(
                            "sending audio chunk that needs to be played very soon (in %d us)",
                            (timestamp_us - now),
                        )
                    await wsock.send_bytes(item)
                else:
                    assert isinstance(item, server_messages.ServerMessage)  # for type checking
                    if isinstance(item, server_messages.ServerTimeMessage):
                        item.payload.server_transmitted = int(self._server.loop.time() * 1_000_000)
                    await wsock.send_str(item.to_json())
            self._logger.debug("WebSocket Connection was closed for the player, ending writer task")
        except Exception:
            self._logger.exception("Error in writer task for player")

    def send_message(self, message: server_messages.ServerMessage | bytes) -> None:
        """Enqueue a JSON or binary message to be sent to the client."""
        # TODO: handle full queue
        if isinstance(message, bytes):
            # Only log binary messages occasionally to reduce spam
            pass
        elif not isinstance(message, server_messages.ServerTimeMessage):
            # Only log important non-time messages
            self._logger.debug("Enqueueing message: %s", type(message).__name__)
        self._to_write.put_nowait(message)

    def add_event_listener(
        self, callback: Callable[[PlayerEvent], Coroutine[None, None, None]]
    ) -> Callable[[], None]:
        """Register a callback to listen for state changes of this player.

        State changes include:
        - The volume was changed
        - The player joined a group

        Returns a function to remove the listener.
        """
        self._event_cbs.append(callback)
        return lambda: self._event_cbs.remove(callback)

    def _signal_event(self, event: PlayerEvent) -> None:
        for cb in self._event_cbs:
            _ = self._server.loop.create_task(cb(event))
