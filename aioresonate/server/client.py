"""Represents a single client device connected to the server."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from contextlib import suppress
from enum import Enum
from typing import TYPE_CHECKING, cast

from aiohttp import ClientWebSocketResponse, WSMessage, WSMsgType, web
from attr import dataclass

from aioresonate.models import unpack_binary_header
from aioresonate.models.controller import (
    GroupCommandClientMessage,
    GroupGetListClientMessage,
    GroupJoinClientMessage,
    GroupUnjoinClientMessage,
)
from aioresonate.models.core import (
    ClientHelloMessage,
    ClientHelloPayload,
    ClientTimeMessage,
    ServerHelloMessage,
    ServerHelloPayload,
    ServerTimeMessage,
    ServerTimePayload,
)
from aioresonate.models.player import PlayerUpdateMessage, StreamRequestFormatMessage
from aioresonate.models.types import ClientMessage, Roles, ServerMessage

from .group import ClientGroup

MAX_PENDING_MSG = 512

logger = logging.getLogger(__name__)

# The cyclic import is not an issue during runtime, so hide it
# pyright: reportImportCycles=none
if TYPE_CHECKING:
    from .server import ResonateServer


class DisconnectBehaviour(Enum):
    """Enum for disconnect behaviour options."""

    UNGROUP = "ungroup"
    """
    The client will ungroup itself from its current group when it gets disconnected.

    Playback will continue on the remaining group members.
    """
    STOP = "stop"
    """
    The client will stop playback of the whole group when it gets disconnected.
    """


class ClientEvent:
    """Base event type used by Client.add_event_listener()."""


@dataclass
class VolumeChangedEvent(ClientEvent):
    """The volume or mute status of the player was changed."""

    volume: int
    muted: bool


@dataclass
class ClientGroupChangedEvent(ClientEvent):
    """The client was moved to a different group."""

    new_group: "ClientGroup"
    """The new group the client is now part of."""


class Client:
    """
    A Client that is connected to a ResonateServer.

    Playback is handled through groups, use Client.group to get the
    assigned group.
    """

    _server: "ResonateServer"
    """Reference to the ResonateServer instance this client belongs to."""
    _wsock_client: ClientWebSocketResponse | None = None
    """
    WebSocket connection from the server to the client.

    This is only set for server-initiated connections.
    """
    _wsock_server: web.WebSocketResponse | None = None
    """
    WebSocket connection from the client to the server.

    This is only set for client-initiated connections.
    """
    _request: web.Request | None = None
    """
    Web Request used for client-initiated connections.

    This is only set for client-initiated connections.
    """
    _client_id: str | None = None
    _client_info: ClientHelloPayload | None = None
    _writer_task: asyncio.Task[None] | None = None
    """Task responsible for sending JSON and binary data."""
    _to_write: asyncio.Queue[ServerMessage | bytes]
    """Queue for messages to be sent to the client through the WebSocket."""
    _group: ClientGroup
    _event_cbs: list[Callable[[ClientEvent], Coroutine[None, None, None]]]
    _volume: int = 100
    _muted: bool = False
    _closing: bool = False
    disconnect_behaviour: DisconnectBehaviour
    """
    Controls the disconnect behavior for this client.

    UNGROUP (default): Client leaves its current group but playback continues
        on remaining group members.
    STOP: Client stops playback for the entire group when disconnecting.
    """
    _handle_client_connect: Callable[["Client"], None]
    _handle_client_disconnect: Callable[["Client"], None]
    _logger: logging.Logger
    _roles: list[Roles]

    def __init__(
        self,
        server: "ResonateServer",
        handle_client_connect: Callable[["Client"], None],
        handle_client_disconnect: Callable[["Client"], None],
        request: web.Request | None = None,
        wsock_client: ClientWebSocketResponse | None = None,
    ) -> None:
        """
        DO NOT CALL THIS CONSTRUCTOR. INTERNAL USE ONLY.

        Use ResonateServer.on_client_connect or ResonateServer.connect_to_client instead.

        Args:
            server: The ResonateServer instance this client belongs to.
            handle_client_connect: Callback function called when the client's handshake is complete.
            handle_client_disconnect: Callback function called when the client disconnects.
            request: Optional web request object for client-initiated connections.
                Only one of request or wsock_client must be provided.
            wsock_client: Optional client WebSocket response for server-initiated connections.
                Only one of request or wsock_client must be provided.
        """
        self._server = server
        self._handle_client_connect = handle_client_connect
        self._handle_client_disconnect = handle_client_disconnect
        if request is not None:
            assert wsock_client is None
            self._request = request
            self._wsock_server = web.WebSocketResponse(heartbeat=55)
            self._logger = logger.getChild(f"unknown-{self._request.remote}")
            self._logger.debug("Client initialized")
        elif wsock_client is not None:
            assert request is None
            self._logger = logger.getChild("unknown-client")
            self._wsock_client = wsock_client
        else:
            raise ValueError("Either request or wsock_client must be provided")
        self._to_write = asyncio.Queue(maxsize=MAX_PENDING_MSG)
        self._group = ClientGroup(server, self)
        self._event_cbs = []
        self._closing = False
        self._roles = []
        self.disconnect_behaviour = DisconnectBehaviour.UNGROUP

    async def disconnect(self, *, retry_connection: bool = True) -> None:
        """Disconnect this client from the server."""
        if not retry_connection:
            self._closing = True
        self._logger.debug("Disconnecting client")

        if self.disconnect_behaviour == DisconnectBehaviour.UNGROUP:
            self.ungroup()
            # Try to stop playback if we were playing alone before disconnecting
            _ = self.group.stop()
        elif self.disconnect_behaviour == DisconnectBehaviour.STOP:
            _ = self.group.stop()
            self.ungroup()

        # Cancel running tasks
        if self._writer_task and not self._writer_task.done():
            self._logger.debug("Cancelling writer task")
            _ = self._writer_task.cancel()  # Don't care about cancellation result
            with suppress(asyncio.CancelledError):
                await self._writer_task
        # Handle task is cancelled implicitly when wsock closes or externally

        # Close WebSocket
        if self._wsock_client is not None and not self._wsock_client.closed:
            _ = await self._wsock_client.close()  # Don't care about close result
        elif self._wsock_server is not None and not self._wsock_server.closed:
            _ = await self._wsock_server.close()  # Don't care about close result

        if self._client_id is not None:
            self._handle_client_disconnect(self)

        self._logger.info("Client disconnected")

    @property
    def group(self) -> ClientGroup:
        """Get the group assigned to this client."""
        return self._group

    @property
    def client_id(self) -> str:
        """The unique identifier of this Client."""
        # This should only be called once the client was correctly initialized
        assert self._client_id
        return self._client_id

    @property
    def name(self) -> str:
        """The human-readable name of this Client."""
        assert self._client_info  # Client should be fully initialized by now
        return self._client_info.name

    @property
    def info(self) -> ClientHelloPayload:
        """List of information and capabilities reported by this client."""
        assert self._client_info  # Client should be fully initialized by now
        return self._client_info

    @property
    def websocket_connection(self) -> web.WebSocketResponse | ClientWebSocketResponse:
        """
        Returns the active WebSocket connection for this client.

        This provides access to the underlying WebSocket connection, which can be
        either a server-side WebSocketResponse (for client-initiated connections)
        or a ClientWebSocketResponse (for server-initiated connections).
        """
        wsock = self._wsock_server or self._wsock_client
        assert wsock is not None
        return wsock

    def set_volume(self, volume: int) -> None:
        """Set the volume of this player."""
        self._logger.debug("Setting volume from %d to %d", self._volume, volume)
        self._logger.error("NOT SUPPORTED BY SPEC YET")

    def mute(self) -> None:
        """Mute this player."""
        self._logger.debug("Muting player")
        self._logger.error("NOT SUPPORTED BY SPEC YET")

    def unmute(self) -> None:
        """Unmute this player."""
        self._logger.debug("Unmuting player")
        self._logger.error("NOT SUPPORTED BY SPEC YET")

    @property
    def muted(self) -> bool:
        """Mute state of this player."""
        return self._muted

    @property
    def volume(self) -> int:
        """Volume of this player."""
        return self._volume

    @property
    def closing(self) -> bool:
        """Whether this player is in the process of closing/disconnecting."""
        return self._closing

    @property
    def roles(self) -> list[Roles]:
        """List of roles this client supports."""
        return self._roles

    def check_role(self, role: Roles) -> bool:
        """Check if the client supports a specific role."""
        return role in self._roles

    def _ensure_role(self, role: Roles) -> None:
        """Raise a ValueError if the client does not support a specific role."""
        if role not in self._roles:
            raise ValueError(f"Client does not support role: {role}")

    def _set_group(self, group: "ClientGroup") -> None:
        """
        Set the group for this client. For internal use by ClientGroup only.

        NOTE: this does not update the group's client list

        Args:
            group: The ClientGroup to assign this client to.
        """
        self._group = group

        # Emit event for group change
        self._signal_event(ClientGroupChangedEvent(group))

    def ungroup(self) -> None:
        """
        Remove the client from the group.

        If the client is already alone, this function does nothing.
        """
        if len(self._group.clients) > 1:
            self._logger.debug("Ungrouping client from group")
            self._group.remove_client(self)
        else:
            self._logger.debug("Client already alone in group, no ungrouping needed")

    async def _setup_connection(self) -> None:
        """Establish WebSocket connection."""
        if self._wsock_server is not None:
            assert self._request is not None
            try:
                async with asyncio.timeout(10):
                    # Prepare response, writer not needed
                    _ = await self._wsock_server.prepare(self._request)
            except TimeoutError:
                self._logger.warning("Timeout preparing request")
                raise

        self._logger.info("Connection established")

        self._logger.debug("Creating writer task")
        self._writer_task = self._server.loop.create_task(self._writer())
        # server/hello will be sent after receiving client/hello

    async def _run_message_loop(self) -> None:
        """Run the main message processing loop."""
        wsock = self._wsock_server or self._wsock_client
        assert wsock is not None
        receive_task: asyncio.Task[WSMessage] | None = None
        # Listen for all incoming messages
        try:
            while not wsock.closed:
                # Wait for either a message or the writer task to complete (meaning the client
                # disconnected or errored)
                receive_task = self._server.loop.create_task(wsock.receive())
                assert self._writer_task is not None  # for type checking
                done, pending = await asyncio.wait(
                    [receive_task, self._writer_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if self._writer_task in done:
                    self._logger.debug("Writer task ended, closing connection")
                    # Cancel the receive task if it's still pending
                    if receive_task in pending:
                        _ = receive_task.cancel()  # Don't care about cancellation result
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
                        ClientMessage.from_json(cast("str", msg.data)), timestamp
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
                _ = receive_task.cancel()  # Don't care about cancellation result

    async def _cleanup_connection(self) -> None:
        """Clean up WebSocket connection and tasks."""
        wsock = self._wsock_client or self._wsock_server
        try:
            if wsock and not wsock.closed:
                _ = await wsock.close()  # Don't care about close result
        except Exception:
            self._logger.exception("Failed to close websocket")
        await self.disconnect()

    async def _handle_client(self) -> None:
        """
        Handle the complete websocket connection lifecycle.

        This method is private and should only be called by ResonateServer
        during client connection handling.
        """
        try:
            # Establish connection and setup
            await self._setup_connection()

            # Run the main message loop
            await self._run_message_loop()
        finally:
            # Clean up connection and tasks
            await self._cleanup_connection()

    async def _handle_message(self, message: ClientMessage, timestamp: int) -> None:
        """Handle incoming commands from the client."""
        if self._client_info is None and not isinstance(message, ClientHelloMessage):
            raise ValueError("First message must be client/hello")
        match message:
            # Core messages
            case ClientHelloMessage(client_info):
                self._logger.info("Received client/hello")
                self._client_info = client_info
                self._roles = client_info.supported_roles
                self._client_id = client_info.client_id
                self._logger.info("Client ID set to %s", self._client_id)
                self._logger = logger.getChild(self._client_id)
                self._handle_client_connect(self)
                self._logger.debug("Sending server/hello in response to client/hello")
                self.send_message(
                    ServerHelloMessage(
                        payload=ServerHelloPayload(
                            server_id=self._server.id, name=self._server.name, version=1
                        )
                    )
                )
            case ClientTimeMessage(client_time):
                self.send_message(
                    ServerTimeMessage(
                        ServerTimePayload(
                            client_transmitted=client_time.client_transmitted,
                            server_received=timestamp,
                            server_transmitted=int(self._server.loop.time() * 1_000_000),
                        )
                    )
                )
            # Player messages
            case PlayerUpdateMessage(state):
                self._ensure_role(Roles.PLAYER)
                self._logger.debug(
                    "Received player state: volume=%d, muted=%s", state.volume, state.muted
                )
                if self._muted != state.muted or self._volume != state.volume:
                    self._volume = state.volume
                    self._muted = state.muted
                    self._signal_event(VolumeChangedEvent(volume=self._volume, muted=self._muted))
            case StreamRequestFormatMessage(payload):
                self._ensure_role(Roles.PLAYER)
                self.group.handle_stream_format_request(self, payload)
            # Controller messages
            case GroupGetListClientMessage():
                self._ensure_role(Roles.CONTROLLER)
                raise NotImplementedError("Group listing is not supported yet")
            case GroupJoinClientMessage(_):
                self._ensure_role(Roles.CONTROLLER)
                raise NotImplementedError("Joining groups is not supported yet")
            case GroupUnjoinClientMessage(_):
                self._ensure_role(Roles.CONTROLLER)
                raise NotImplementedError("Leaving groups is not supported yet")
            case GroupCommandClientMessage(group_command):
                self._ensure_role(Roles.CONTROLLER)
                self.group._handle_group_command(group_command)  # noqa: SLF001

    async def _writer(self) -> None:
        """Write outgoing messages from the queue."""
        # Exceptions if socket disconnected or cancelled by connection handler
        wsock = self._wsock_server or self._wsock_client
        assert wsock is not None
        try:
            while not wsock.closed and not self._closing:
                item = await self._to_write.get()

                if isinstance(item, bytes):
                    # Unpack binary header using helper function
                    header = unpack_binary_header(item)
                    now = int(self._server.loop.time() * 1_000_000)
                    if header.timestamp_us - now < 0:
                        self._logger.error("Audio chunk should have played already, skipping it")
                        continue
                    if header.timestamp_us - now < 500_000:
                        self._logger.warning(
                            "sending audio chunk that needs to be played very soon (in %d us)",
                            (header.timestamp_us - now),
                        )
                    try:
                        await wsock.send_bytes(item)
                    except ConnectionError:
                        self._logger.warning(
                            "Connection error sending binary data, ending writer task"
                        )
                        break
                else:
                    assert isinstance(item, ServerMessage)  # for type checking
                    if isinstance(item, ServerTimeMessage):
                        item.payload.server_transmitted = int(self._server.loop.time() * 1_000_000)
                    try:
                        await wsock.send_str(item.to_json())
                    except ConnectionError:
                        self._logger.warning(
                            "Connection error sending JSON data, ending writer task"
                        )
                        break
            self._logger.debug("WebSocket Connection was closed for the client, ending writer task")
        except Exception:
            self._logger.exception("Error in writer task for client")

    def send_message(self, message: ServerMessage | bytes) -> None:
        """
        Enqueue a JSON or binary message to be sent directly to the client.

        It is recommended to not use this method, but to use the higher-level
        API of this library instead.

        NOTE: Binary messages are directly sent to the client, you need to add the
        header yourself using pack_binary_header().
        """
        # TODO: handle full queue
        if isinstance(message, bytes):
            # Only log binary messages occasionally to reduce spam
            pass
        elif not isinstance(message, ServerTimeMessage):
            # Only log important non-time messages
            self._logger.debug("Enqueueing message: %s", type(message).__name__)
        self._to_write.put_nowait(message)

    def add_event_listener(
        self, callback: Callable[[ClientEvent], Coroutine[None, None, None]]
    ) -> Callable[[], None]:
        """
        Register a callback to listen for state changes of this client.

        State changes include:
        - The volume was changed
        - The client joined a group

        Returns a function to remove the listener.
        """
        self._event_cbs.append(callback)
        return lambda: self._event_cbs.remove(callback)

    def _signal_event(self, event: ClientEvent) -> None:
        for cb in self._event_cbs:
            _ = self._server.loop.create_task(cb(event))  # Fire and forget event callback
