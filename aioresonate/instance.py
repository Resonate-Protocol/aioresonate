"""Represents a single player device connected to the server."""

import asyncio
import logging
import struct
from contextlib import suppress
from typing import TYPE_CHECKING

from aiohttp import ClientWebSocketResponse, WSMsgType, web

from aioresonate import models

from .group import PlayerGroup

MAX_PENDING_MSG = 512

logger = logging.getLogger(__name__)

# The cyclic import is not an issue during runtime, so hide it
# pyright: reportImportCycles=none
if TYPE_CHECKING:
    from .server import ResonateServer


class PlayerInstance:
    """A Player that is connected to a ResonateServer.

    Playback is handled through groups, use PlayerInstance.group to get the
    assigned group.
    """

    _state: models.PlayerState = models.PlayerState(
        state=models.PlayerStateType.IDLE, volume=100, muted=False
    )
    _server: "ResonateServer"
    request: web.Request
    wsock: web.WebSocketResponse | ClientWebSocketResponse
    url: str | None
    _player_id: str | None = None
    player_info: models.PlayerInfo | None = None
    # Task responsible for handling messages from the player
    _handle_task: asyncio.Task[str] | None = None
    # Task responsible for sending audio and other data
    _writer_task: asyncio.Task[None] | None = None
    # Task responsible for processing the audio stream
    stream_task: asyncio.Task[None] | None = None
    _to_write: asyncio.Queue[models.ServerMessages | str | bytes]
    session_info: models.SessionInfo | None = None
    _group: PlayerGroup

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

    @property
    def state(self) -> models.PlayerState:
        """The state of the player."""
        return self._state

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
    def capabilities(self) -> dict[str, str]:
        """List of capabilities supported by this player."""
        raise NotImplementedError

    def set_volume(self, volume: int) -> None:
        """Set the volume of this player."""
        raise NotImplementedError

    def mute(self) -> None:
        """Mute this player."""
        raise NotImplementedError

    def unmute(self) -> None:
        """Unmute this player."""
        raise NotImplementedError

    @property
    def muted(self) -> bool:
        """Mute state of this player."""
        raise NotImplementedError

    @property
    def volume(self) -> int:
        """Volume of this player."""
        raise NotImplementedError

    def ungroup(self) -> None:
        """Remove the player from the group.

        If the player is already alone, this function does nothing.
        """
        if len(self._group.players) > 1:
            self._group.remove_player(self)

    async def handle_client(self) -> web.WebSocketResponse | ClientWebSocketResponse:
        """Handle the websocket connection."""
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

        # 1. Send Source Hello
        self.send_message(
            models.SourceHelloMessage(
                payload=models.SourceInfo(
                    name="Music Assistant",
                    # TODO: will this make problems with multiple MA instances?
                    source_id="ma",  # TODO: make this configurable
                )
            )
        )

        try:
            while not wsock.closed:
                msg = await wsock.receive()
                timestamp = int(self._server.loop.time() * 1_000_000)

                if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
                    break

                if msg.type != WSMsgType.TEXT:
                    continue

                try:
                    await self._handle_message(models.Message.from_json(msg.data), timestamp)
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
                self._to_write.put_nowait("")
                # Make sure all error messages are written before closing
                await self._writer_task
                _ = await wsock.close()
            except asyncio.QueueFull:  # can be raised by put_nowait
                _ = self._writer_task.cancel()

        return self.wsock

    async def _handle_message(self, message: models.Message, timestamp: int) -> None:
        """Handle incoming commands from the client."""
        msg_type = message.type
        payload = message.payload
        if msg_type == "player/hello":
            # TODO: reject if player_id is already connected
            player_info = models.PlayerInfo.from_dict(payload)
            logger.info(
                "Received player/hello from %s (%s)", player_info.player_id, player_info.name
            )
            self.player_info = player_info
            self._player_id = player_info.player_id
            self._server._on_player_add(self)  # noqa: SLF001

        elif msg_type == "player/state":
            if not self.player_id:
                logger.warning("Received player/state before player/hello")
                return
            state_info = models.PlayerState.from_dict(payload)
            self._state = state_info

        elif msg_type == "player/time":
            payload["source_received"] = timestamp
            payload["source_transmitted"] = int(self._server.loop.time() * 1_000_000)
            self.send_message(
                models.SourceTimeMessage(payload=models.SourceTimeInfo.from_dict(payload))
            )

        else:
            logger.debug("%s received unhandled command type: %", self.player_id, msg_type)

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
                elif isinstance(item, models.ServerMessages):
                    if isinstance(item, models.SourceTimeMessage):
                        item.payload.source_transmitted = int(self._server.loop.time() * 1_000_000)
                    await self.wsock.send_str(item.to_json())
                else:
                    await self.wsock.send_str(item)

    def send_message(self, message: models.ServerMessages) -> None:
        """Enqueue a JSON message to be sent to the client."""
        # TODO: handle full queue
        self._to_write.put_nowait(message)

    def send_binary(self, data: bytes) -> None:
        """Enqueue a binary message to be sent to the client."""
        self._to_write.put_nowait(data)

    def send_audio_chunk(self, timestamp_us: int, sample_count: int, audio_data: bytes) -> None:
        """Pack audio data the audio header."""
        # TODO: do any encoding here if needed
        binary_chunk = self.pack_audio_chunk(timestamp_us, sample_count, audio_data)
        self.send_binary(binary_chunk)

    def pack_audio_chunk(self, timestamp_us: int, sample_count: int, audio_data: bytes) -> bytes:
        """Pack audio data the audio header."""
        header = struct.pack(
            models.BINARY_HEADER_FORMAT,
            models.BinaryMessageType.PlayAudioChunk.value,
            timestamp_us,
            sample_count,
        )
        return header + audio_data
