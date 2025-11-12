"""Represents a single client device connected to the server."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from contextlib import suppress
from enum import Enum
from typing import TYPE_CHECKING, cast

from aiohttp import ClientWebSocketResponse, WSMessage, WSMsgType, web

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
from aioresonate.models.player import (
    PlayerUpdateMessage,
    StreamRequestFormatMessage,
)
from aioresonate.models.types import BinaryMessageType, ClientMessage, Roles, ServerMessage

from .controller import ControllerClient
from .events import ClientEvent, ClientGroupChangedEvent
from .group import ResonateGroup
from .metadata import MetadataClient
from .player import PlayerClient
from .visualizer import VisualizerClient

MAX_PENDING_MSG = 4096


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


class ResonateClient:
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
    _group: ResonateGroup
    _event_cbs: list[Callable[[ClientEvent], Coroutine[None, None, None]]]
    _closing: bool = False
    _disconnecting: bool = False
    """Flag to prevent multiple concurrent disconnect tasks."""
    disconnect_behaviour: DisconnectBehaviour
    """
    Controls the disconnect behavior for this client.

    UNGROUP (default): Client leaves its current group but playback continues
        on remaining group members.
    STOP: Client stops playback for the entire group when disconnecting.
    """
    _handle_client_connect: Callable[["ResonateClient"], None]
    _handle_client_disconnect: Callable[["ResonateClient"], None]
    _logger: logging.Logger
    _roles: list[Roles]
    _player: PlayerClient | None = None
    _controller: ControllerClient | None = None
    _metadata_client: MetadataClient | None = None
    _visualizer: VisualizerClient | None = None

    def __init__(
        self,
        server: "ResonateServer",
        handle_client_connect: Callable[["ResonateClient"], None],
        handle_client_disconnect: Callable[["ResonateClient"], None],
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
        self._group = ResonateGroup(server, self)
        self._event_cbs = []
        self._closing = False
        self._disconnecting = False
        self._roles = []
        self.disconnect_behaviour = DisconnectBehaviour.UNGROUP

    async def disconnect(self, *, retry_connection: bool = True) -> None:
        """Disconnect this client from the server."""
        if not retry_connection:
            self._closing = True
        self._disconnecting = True
        self._logger.debug("Disconnecting client")

        if self.disconnect_behaviour == DisconnectBehaviour.UNGROUP:
            await self.ungroup()
            # Try to stop playback if we were playing alone before disconnecting
            await self.group.stop()
        elif self.disconnect_behaviour == DisconnectBehaviour.STOP:
            await self.group.stop()
            await self.ungroup()

        # Cancel running tasks
        if self._writer_task and not self._writer_task.done():
            self._logger.debug("Cancelling writer task")
            self._writer_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._writer_task
        # Handle task is cancelled implicitly when wsock closes or externally

        # Close WebSocket
        if self._wsock_client is not None and not self._wsock_client.closed:
            await self._wsock_client.close()
        elif self._wsock_server is not None and not self._wsock_server.closed:
            await self._wsock_server.close()

        if self._client_id is not None:
            self._handle_client_disconnect(self)

        self._logger.info("Client disconnected")

    @property
    def group(self) -> ResonateGroup:
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

    @property
    def player(self) -> PlayerClient | None:
        """Return the attached player instance, if available."""
        if self._player is None and Roles.PLAYER in self._roles:
            self._player = PlayerClient(self)
        return self._player

    @property
    def require_player(self) -> PlayerClient:
        """Return the player or raise if the role is unsupported."""
        if self._player is None:
            raise ValueError(f"Client does not support role: {Roles.PLAYER}")
        return self._player

    @property
    def controller(self) -> ControllerClient | None:
        """Return the controller role helper, if initialized."""
        return self._controller

    @property
    def require_controller(self) -> ControllerClient:
        """Return controller helper or raise if role unsupported."""
        if self._controller is None:
            raise ValueError(f"Client does not support role: {Roles.CONTROLLER}")
        return self._controller

    @property
    def metadata(self) -> MetadataClient | None:
        """Return the metadata role helper, if initialized."""
        return self._metadata_client

    @property
    def require_metadata(self) -> MetadataClient:
        """Return metadata helper or raise if role unsupported."""
        if self._metadata_client is None:
            raise ValueError(f"Client does not support role: {Roles.METADATA}")
        return self._metadata_client

    @property
    def visualizer(self) -> VisualizerClient | None:
        """Return the visualizer role helper, if initialized."""
        return self._visualizer

    @property
    def require_visualizer(self) -> VisualizerClient:
        """Return visualizer helper or raise if role unsupported."""
        if self._visualizer is None:
            raise ValueError(f"Client does not support role: {Roles.VISUALIZER}")
        return self._visualizer

    def _set_group(self, group: "ResonateGroup") -> None:
        """
        Set the group for this client. For internal use by ResonateGroup only.

        NOTE: this does not update the group's client list

        Args:
            group: The ResonateGroup to assign this client to.
        """
        self._group = group

        # Emit event for group change
        self._signal_event(ClientGroupChangedEvent(group))

    async def ungroup(self) -> None:
        """
        Remove the client from the group.

        If the client is already alone, this function does nothing.
        """
        if len(self._group.clients) > 1:
            self._logger.debug("Ungrouping client from group")
            await self._group.remove_client(self)
        else:
            self._logger.debug("Client already alone in group, no ungrouping needed")

    async def _setup_connection(self) -> None:
        """Establish WebSocket connection."""
        if self._wsock_server is not None:
            assert self._request is not None
            try:
                async with asyncio.timeout(10):
                    # Prepare response, writer not needed
                    await self._wsock_server.prepare(self._request)
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
                        receive_task.cancel()
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
                receive_task.cancel()

    async def _cleanup_connection(self) -> None:
        """Clean up WebSocket connection and tasks."""
        wsock = self._wsock_client or self._wsock_server
        try:
            if wsock and not wsock.closed:
                await wsock.close()
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

                # Initialize role helpers based on supported roles
                if Roles.PLAYER in self._roles:
                    self._player = PlayerClient(self)
                if Roles.CONTROLLER in self._roles:
                    self._controller = ControllerClient(self)
                if Roles.METADATA in self._roles:
                    self._metadata_client = MetadataClient(self)
                if Roles.VISUALIZER in self._roles:
                    self._visualizer = VisualizerClient(self)

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
                self.require_player.handle_player_update(state)
            case StreamRequestFormatMessage(payload):
                self._ensure_role(Roles.PLAYER)
                self.group.handle_stream_format_request(self, payload)
            # Controller messages
            case GroupGetListClientMessage() as group_get_list:
                self.require_controller.handle_get_list(group_get_list)
            case GroupJoinClientMessage(payload):
                self.require_controller.handle_join(payload)
            case GroupUnjoinClientMessage() as group_unjoin:
                self.require_controller.handle_unjoin(group_unjoin)
            case GroupCommandClientMessage(group_command):
                self.require_controller.handle_command(group_command)

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

                    # Only validate timestamps for audio chunks, since they are time-sensitive
                    if header.message_type == BinaryMessageType.AUDIO_CHUNK.value:
                        now = int(self._server.loop.time() * 1_000_000)
                        if header.timestamp_us - now < 0:
                            self._logger.error(
                                "Audio chunk should have played already, skipping it"
                            )
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
        try:
            self._to_write.put_nowait(message)
        except asyncio.QueueFull:
            # Only trigger disconnect once, even if queue fills repeatedly
            if not self._disconnecting:
                self._logger.error("Message queue full, client too slow - disconnecting")
                task = self._server.loop.create_task(self.disconnect(retry_connection=True))
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            return

        if isinstance(message, bytes):
            pass
        elif not isinstance(message, ServerTimeMessage):
            self._logger.debug("Enqueueing message: %s", type(message).__name__)

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

        def _remove() -> None:
            with suppress(ValueError):
                self._event_cbs.remove(callback)

        return _remove

    def _signal_event(self, event: ClientEvent) -> None:
        for cb in self._event_cbs:
            task = self._server.loop.create_task(cb(event))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
