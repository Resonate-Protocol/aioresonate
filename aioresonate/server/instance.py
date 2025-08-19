"""Represents a single player device connected to the server."""

import asyncio
import logging
import struct
from collections.abc import Callable, Coroutine
from contextlib import suppress
from typing import TYPE_CHECKING

from aiohttp import ClientWebSocketResponse, WSMsgType, web
from attr import dataclass

from aioresonate import models
from aioresonate.models import client_messages, server_messages

from .group import PlayerGroup

MAX_PENDING_MSG = 512

logger = logging.getLogger(__name__)

# The cyclic import is not an issue during runtime, so hide it
# pyright: reportImportCycles=none
if TYPE_CHECKING:
    from .server import ResonateServer


class PlayerInstanceEvent:
    """Base event type used by PlayerInstance.add_event_listener()."""


@dataclass
class VolumeChangedEvent(PlayerInstanceEvent):
    """The volume or mute status of the player was changed."""

    volume: int
    muted: bool


class PlayerInstance:
    """A Player that is connected to a ResonateServer.

    Playback is handled through groups, use PlayerInstance.group to get the
    assigned group.
    """

    _server: "ResonateServer"
    request: web.Request
    wsock: web.WebSocketResponse | ClientWebSocketResponse
    url: str | None
    _player_id: str | None = None
    player_info: client_messages.PlayerHelloPayload | None = None
    # Task responsible for handling messages from the player
    _handle_task: asyncio.Task[str] | None = None
    # Task responsible for sending audio and other data
    _writer_task: asyncio.Task[None] | None = None
    # Task responsible for processing the audio stream
    stream_task: asyncio.Task[None] | None = None
    _to_write: asyncio.Queue[server_messages.ServerMessage | bytes]
    session_info: server_messages.SessionStartPayload | None = None
    _group: PlayerGroup
    _event_cbs: list[Callable[[PlayerInstanceEvent], Coroutine[None, None, None]]]
    _volume: int = 100
    _muted: bool = False

    def __init__(
        self,
        server: "ResonateServer",
        request: web.Request | None,
        url: str | None,
        wsock_client: ClientWebSocketResponse | None,
    ) -> None:
        """Do not call this constructor.

        Use ResonateServer.on_player_connect or ResonateServer.connect_to_player instead.
        """
        self._server = server
        if request is not None:
            self.request = request
            self.wsock = web.WebSocketResponse(heartbeat=55)
        elif url is not None:
            assert wsock_client is not None
            self.url = url
            self.wsock = wsock_client
        self._to_write = asyncio.Queue(maxsize=MAX_PENDING_MSG)
        self._group = PlayerGroup(server, self)
        self._event_cbs = []

    async def disconnect(self) -> None:
        """Disconnect client and cancel tasks."""
        logger.debug("Disconnecting client %s", self.player_id or self.request.remote)

        # Cancel running tasks
        if self.stream_task and not self.stream_task.done():
            _ = self.stream_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.stream_task
        if self._writer_task and not self._writer_task.done():
            _ = self._writer_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._writer_task
        # Handle task is cancelled implicitly when wsock closes or externally

        # Close websocket
        if not self.wsock.closed:
            _ = await self.wsock.close()

        logger.info("Client %s disconnected", self.player_id or self.request.remote)

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
    def info(self) -> client_messages.PlayerHelloPayload:
        """List of information and capabilities reported by this player."""
        assert self.player_info  # Player should be fully initialized by now
        return self.player_info

    def set_volume(self, volume: int) -> None:
        """Set the volume of this player."""
        if self._volume == volume:
            return
        self._volume = volume
        self.send_message(
            server_messages.VolumeSetMessage(server_messages.VolumeSetPayload(volume))
        )
        self._signal_event(VolumeChangedEvent(volume=self._volume, muted=self._muted))

    def mute(self) -> None:
        """Mute this player."""
        if self._muted:
            return
        self._muted = True
        self.send_message(
            server_messages.MuteSetMessage(server_messages.MuteSetPayload(self._muted))
        )
        self._signal_event(VolumeChangedEvent(volume=self._volume, muted=self._muted))

    def unmute(self) -> None:
        """Unmute this player."""
        if not self._muted:
            return
        self._muted = False
        self.send_message(
            server_messages.MuteSetMessage(server_messages.MuteSetPayload(self._muted))
        )
        self._signal_event(VolumeChangedEvent(volume=self._volume, muted=self._muted))

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
            self._group.remove_player(self)

    async def handle_client(self) -> web.WebSocketResponse | ClientWebSocketResponse:
        """Handle the websocket connection."""
        # Establish a WebSocket connection to the player
        wsock = self.wsock
        if self.url is None:
            assert isinstance(wsock, web.WebSocketResponse)
            remote_addr = self.request.remote or "Unknown"
            try:
                async with asyncio.timeout(10):
                    _ = await wsock.prepare(self.request)
            except TimeoutError:
                logger.warning("Timeout preparing request from %s", remote_addr)
                return self.wsock
        else:
            remote_addr = self.url

        logger.info("Connection established with %s", remote_addr)

        self._writer_task = self._server.loop.create_task(self._writer())

        # Send Server Hello
        self.send_message(
            server_messages.ServerHelloMessage(
                payload=server_messages.ServerHelloPayload(
                    name=self._server.name,
                    server_id=self._server.id,
                )
            )
        )

        # Listen for all incoming messages
        try:
            while not wsock.closed:
                msg = await wsock.receive()
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
                    logger.exception("error parsing message")
            logger.debug("wsock was closed for %s", remote_addr)

        except asyncio.CancelledError:
            logger.debug("Connection closed by client")
        except Exception:
            logger.exception("Unexpected error inside websocket API")
        finally:
            # TODO: run disconnect here?
            try:
                # Make sure all error messages are written before closing
                await self._writer_task
                _ = await wsock.close()
            except asyncio.QueueFull:  # can be raised by put_nowait
                _ = self._writer_task.cancel()

        return self.wsock

    async def _handle_message(self, message: client_messages.ClientMessage, timestamp: int) -> None:
        """Handle incoming commands from the client."""
        match message:
            case client_messages.PlayerHelloMessage(player_info):
                logger.info(
                    "Received player/hello from %s (%s)", player_info.player_id, player_info.name
                )
                self.player_info = player_info
                self._player_id = player_info.player_id
                self._server._on_player_add(self)  # noqa: SLF001
            case client_messages.PlayerStateMessage(state):
                if not self.player_id:
                    logger.warning("Received player/state before player/hello")
                    return
                if self.muted != state.muted or self.volume != state.volume:
                    self._volume = state.volume
                    self._muted = state.muted
                    self._signal_event(VolumeChangedEvent(volume=self._volume, muted=self._muted))
                # TODO: handle state.state changes, but how?
            case client_messages.PlayerTimeMessage(player_time):
                self.send_message(
                    server_messages.ServerTimeMessage(
                        server_messages.ServerTimePayload(
                            player_transmitted=player_time.player_transmitted,
                            server_received=timestamp,
                            server_transmitted=int(self._server.loop.time() * 1_000_000),
                        )
                    )
                )
            case client_messages.StreamCommandMessage():
                raise NotImplementedError
            case client_messages.ClientMessage:
                pass  # unused base type

    async def _writer(self) -> None:
        """Write outgoing messages from the queue."""
        # Exceptions if Socket disconnected or cancelled by connection handler
        with suppress(
            RuntimeError,
            ConnectionResetError,
            asyncio.CancelledError,
        ):
            while not self.wsock.closed:
                item = await self._to_write.get()

                if isinstance(item, bytes):
                    _, timestamp_us, _ = struct.unpack(models.BINARY_HEADER_FORMAT, item[:13])
                    now = int(self._server.loop.time() * 1_000_000)
                    if timestamp_us - now < 0:
                        logger.error("Audio chunk after should have played already, skipping it")
                        continue
                    if timestamp_us - now < 500_000:
                        logger.warning(
                            "sending audio chunk that needs to be played very soon (in %d us)",
                            (timestamp_us - now),
                        )
                    await self.wsock.send_bytes(item)
                else:
                    if isinstance(item, server_messages.ServerTimeMessage):
                        item.payload.server_transmitted = int(self._server.loop.time() * 1_000_000)
                    await self.wsock.send_str(item.to_json())

    def send_message(self, message: server_messages.ServerMessage) -> None:
        """Enqueue a JSON message to be sent to the client."""
        # TODO: handle full queue
        self._to_write.put_nowait(message)

    def send_audio_chunk(self, timestamp_us: int, sample_count: int, audio_data: bytes) -> None:
        """Pack audio data the audio header."""
        # TODO: do any encoding here if needed
        header = struct.pack(
            models.BINARY_HEADER_FORMAT,
            models.BinaryMessageType.PlayAudioChunk.value,
            timestamp_us,
            sample_count,
        )
        self._to_write.put_nowait(header + audio_data)

    def add_event_listener(
        self, callback: Callable[[PlayerInstanceEvent], Coroutine[None, None, None]]
    ) -> Callable[[], None]:
        """Register a callback to listen for state changes of this player.

        State changes include:
        - The volume was changed
        - The player joined a group

        Returns a function to remove the listener.
        """
        self._event_cbs.append(callback)
        return lambda: self._event_cbs.remove(callback)

    def _signal_event(self, event: PlayerInstanceEvent) -> None:
        for cb in self._event_cbs:
            _ = self._server.loop.create_task(cb(event))
