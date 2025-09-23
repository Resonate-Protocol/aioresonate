"""Represents a single client device connected to the server."""

import asyncio
import base64
import logging
from collections import deque
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import suppress
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple, cast

import av
from aiohttp import ClientWebSocketResponse, WSMessage, WSMsgType, web
from attr import dataclass
from av import logging as av_logging

from aioresonate.models import BinaryMessageType, pack_binary_header_raw, unpack_binary_header
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
    StreamStartMessage,
    StreamStartPayload,
)
from aioresonate.models.player import (
    PlayerUpdateMessage,
    StreamRequestFormatMessage,
    StreamStartPlayer,
)
from aioresonate.models.types import ClientMessage, Roles, ServerMessage

from .group import AudioCodec, AudioFormat, ClientGroup

MAX_PENDING_MSG = 512
BUFFER_HIGH_WATERMARK_RATIO = 0.9
BUFFER_SLEEP_MIN_US = 5_000


class _BufferedChunk(NamedTuple):
    """Represents compressed audio bytes scheduled for playback."""

    end_time_us: int
    """Absolute timestamp when these bytes should be fully consumed."""
    byte_count: int
    """Compressed byte count occupying the device buffer."""


def _samples_to_microseconds(sample_count: int, sample_rate: int) -> int:
    """Convert a number of samples to microseconds using ``sample_rate``."""
    return int(sample_count * 1_000_000 / sample_rate)


class _BufferTracker:
    """Tracks compressed bytes queued for playback and applies backpressure."""

    __slots__ = (
        "_logger",
        "_loop",
        "buffered_bytes",
        "buffered_chunks",
        "capacity_bytes",
        "client_id",
        "high_water_bytes",
        "max_usage_bytes",
    )

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        logger: logging.Logger,
        client_id: str,
        capacity_bytes: int,
    ) -> None:
        self._loop = loop
        self._logger = logger
        self.client_id = client_id
        self.capacity_bytes = capacity_bytes
        self.high_water_bytes = max(1, int(capacity_bytes * BUFFER_HIGH_WATERMARK_RATIO))
        self.buffered_chunks: deque[_BufferedChunk] = deque()
        self.buffered_bytes = 0
        self.max_usage_bytes = 0

    def prune_consumed(self, now_us: int | None = None) -> int:
        """Drop chunks that should have finished playback by ``now_us``."""
        if now_us is None:
            now_us = int(self._loop.time() * 1_000_000)
        while self.buffered_chunks and self.buffered_chunks[0].end_time_us <= now_us:
            self.buffered_bytes -= self.buffered_chunks.popleft().byte_count
        self.buffered_bytes = max(self.buffered_bytes, 0)
        return now_us

    async def wait_for_capacity(self, bytes_needed: int, expected_end_us: int) -> None:
        """Block until the device buffer can accept ``bytes_needed`` more bytes."""
        if bytes_needed <= 0:
            return
        if bytes_needed >= self.capacity_bytes:
            self._logger.warning(
                "Chunk size %s exceeds reported buffer capacity %s for client %s",
                bytes_needed,
                self.capacity_bytes,
                self.client_id,
            )
            return

        while True:
            now_us = self.prune_consumed()
            projected_usage = self.buffered_bytes + bytes_needed
            if projected_usage <= self.capacity_bytes:
                if projected_usage > self.high_water_bytes and self.capacity_bytes:
                    fill = 100 * projected_usage / self.capacity_bytes
                    self._logger.debug(
                        "Buffer at %.1f%% for client %s (%s/%s bytes)",
                        fill,
                        self.client_id,
                        projected_usage,
                        self.capacity_bytes,
                    )
                return

            sleep_target_us = (
                self.buffered_chunks[0].end_time_us if self.buffered_chunks else expected_end_us
            )
            sleep_us = max(BUFFER_SLEEP_MIN_US, sleep_target_us - now_us)
            await asyncio.sleep(sleep_us / 1_000_000)

    def register(self, end_time_us: int, byte_count: int) -> None:
        """Record bytes added to the buffer finishing at ``end_time_us``."""
        if byte_count <= 0:
            return
        self.buffered_chunks.append(_BufferedChunk(end_time_us, byte_count))
        self.buffered_bytes += byte_count
        self.max_usage_bytes = max(self.max_usage_bytes, self.buffered_bytes)


class _DirectStreamContext:
    """Holds state while streaming audio directly to a client."""

    __slots__ = (
        "_send_pcm",
        "audio_format",
        "buffer_tracker",
        "client",
        "compressed_bytes_sent",
        "encoder",
        "frame_stride_bytes",
        "input_audio_format",
        "input_audio_layout",
        "pcm_bytes_consumed",
        "play_start_time_us",
        "samples_enqueued_total",
        "samples_per_chunk",
        "samples_sent_total",
    )

    def __init__(
        self,
        *,
        client: "Client",
        audio_format: AudioFormat,
        input_audio_format: str,
        input_audio_layout: str,
        samples_per_chunk: int,
        buffer_tracker: _BufferTracker,
        play_start_time_us: int,
        encoder: av.AudioCodecContext | None,
        frame_stride_bytes: int,
        send_pcm: Callable[[bytes, int], None],
    ) -> None:
        self.client = client
        self.audio_format = audio_format
        self.input_audio_format = input_audio_format
        self.input_audio_layout = input_audio_layout
        self.samples_per_chunk = samples_per_chunk
        self._send_pcm = send_pcm
        self.buffer_tracker = buffer_tracker
        self.play_start_time_us = play_start_time_us
        self.encoder = encoder
        self.frame_stride_bytes = frame_stride_bytes
        self.samples_enqueued_total = 0
        self.samples_sent_total = 0
        self.pcm_bytes_consumed = 0
        self.compressed_bytes_sent = 0

    @property
    def sample_rate(self) -> int:
        return self.audio_format.sample_rate

    def _timeline_us(self, samples: int) -> int:
        return self.play_start_time_us + _samples_to_microseconds(samples, self.sample_rate)

    async def send_pcm_chunk(self, chunk: bytes, sample_count: int) -> None:
        start_us = self._timeline_us(self.samples_sent_total)
        end_us = self._timeline_us(self.samples_sent_total + sample_count)
        await self.buffer_tracker.wait_for_capacity(len(chunk), end_us)
        self._send_pcm(chunk, start_us)
        self.buffer_tracker.register(end_us, len(chunk))
        self.samples_sent_total += sample_count
        self.compressed_bytes_sent += len(chunk)

    async def send_encoded_chunk(self, chunk: bytes, sample_count: int) -> None:
        encoder = self.encoder
        if encoder is None:
            raise RuntimeError("send_encoded_chunk called without encoder")

        frame = av.AudioFrame(
            format=self.input_audio_format,
            layout=self.input_audio_layout,
            samples=sample_count,
        )
        frame.sample_rate = encoder.sample_rate
        frame.planes[0].update(chunk)

        self.samples_enqueued_total += sample_count
        packets = encoder.encode(frame)

        if not packets:
            return

        for packet in packets:
            available_samples = max(self.samples_enqueued_total - self.samples_sent_total, 0)
            packet_samples = packet.duration or available_samples
            if available_samples and packet_samples > available_samples:
                packet_samples = available_samples
            if packet_samples <= 0:
                packet_samples = packet.duration or self.samples_per_chunk

            payload = bytes(packet)
            start_us = self._timeline_us(self.samples_sent_total)
            end_us = self._timeline_us(self.samples_sent_total + packet_samples)
            await self.buffer_tracker.wait_for_capacity(len(payload), end_us)
            header = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, start_us)
            self.client.send_message(header + payload)
            self.buffer_tracker.register(end_us, len(payload))
            self.compressed_bytes_sent += len(payload)
            self.samples_sent_total += packet_samples

    async def flush_encoder(self) -> None:
        encoder = self.encoder
        if encoder is None:
            return

        for packet in encoder.encode(None):
            available_samples = max(self.samples_enqueued_total - self.samples_sent_total, 0)
            packet_samples = packet.duration or available_samples or self.samples_per_chunk
            if available_samples and packet_samples > available_samples:
                packet_samples = available_samples

            payload = bytes(packet)
            start_us = self._timeline_us(self.samples_sent_total)
            end_us = self._timeline_us(self.samples_sent_total + packet_samples)
            await self.buffer_tracker.wait_for_capacity(len(payload), end_us)
            header = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, start_us)
            self.client.send_message(header + payload)
            self.buffer_tracker.register(end_us, len(payload))
            self.compressed_bytes_sent += len(payload)
            self.samples_sent_total += packet_samples

    async def drain_ready_chunks(self, input_buffer: bytearray, *, force_flush: bool) -> None:
        while True:
            available_samples = len(input_buffer) // self.frame_stride_bytes
            if available_samples <= 0:
                return

            if not force_flush and available_samples < self.samples_per_chunk:
                return
            sample_count = (
                available_samples if force_flush else min(available_samples, self.samples_per_chunk)
            )
            chunk_size = sample_count * self.frame_stride_bytes
            if chunk_size <= 0:
                return

            chunk = bytes(input_buffer[:chunk_size])
            del input_buffer[:chunk_size]
            self.pcm_bytes_consumed += chunk_size

            if self.encoder is None:
                await self.send_pcm_chunk(chunk, sample_count)
            else:
                await self.send_encoded_chunk(chunk, sample_count)


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

    def determine_optimal_format(
        self,
        source_format: AudioFormat,
        preferred_codec: AudioCodec = AudioCodec.OPUS,
    ) -> AudioFormat:
        """
        Determine the optimal audio format for this client given a source format.

        Prefers higher quality within the client's capabilities and falls back gracefully.

        Args:
            source_format: The source audio format to match against.
            preferred_codec: Preferred audio codec (e.g., Opus). Falls back when unsupported.

        Returns:
            AudioFormat: The optimal format for this client.
        """
        player_info = self.info

        # Determine optimal sample rate
        sample_rate = source_format.sample_rate
        if (
            player_info.player_support
            and sample_rate not in player_info.player_support.support_sample_rates
        ):
            # Prefer lower rates that are closest to source, fallback to minimum
            lower_rates = [
                r for r in player_info.player_support.support_sample_rates if r < sample_rate
            ]
            sample_rate = (
                max(lower_rates)
                if lower_rates
                else min(player_info.player_support.support_sample_rates)
            )
            self._logger.debug(
                "Adjusted sample_rate for client %s: %s", self.client_id, sample_rate
            )

        # Determine optimal bit depth
        bit_depth = source_format.bit_depth
        if (
            player_info.player_support
            and bit_depth not in player_info.player_support.support_bit_depth
        ):
            if 16 in player_info.player_support.support_bit_depth:
                bit_depth = 16
            else:
                raise NotImplementedError("Only 16bit is supported for now")
            self._logger.debug("Adjusted bit_depth for client %s: %s", self.client_id, bit_depth)

        # Determine optimal channel count
        channels = source_format.channels
        if (
            player_info.player_support
            and channels not in player_info.player_support.support_channels
        ):
            # Prefer stereo, then mono
            if 2 in player_info.player_support.support_channels:
                channels = 2
            elif 1 in player_info.player_support.support_channels:
                channels = 1
            else:
                raise NotImplementedError("Only mono and stereo are supported")
            self._logger.debug("Adjusted channels for client %s: %s", self.client_id, channels)

        # Determine optimal codec with fallback chain
        codec_fallbacks = [preferred_codec, AudioCodec.FLAC, AudioCodec.OPUS, AudioCodec.PCM]
        codec = None
        for candidate_codec in codec_fallbacks:
            if (
                player_info.player_support
                and candidate_codec.value in player_info.player_support.support_codecs
            ):
                # Special handling for Opus - check if sample rates are compatible
                if candidate_codec == AudioCodec.OPUS:
                    opus_rate_candidates = [
                        (8000, sample_rate <= 8000),
                        (12000, sample_rate <= 12000),
                        (16000, sample_rate <= 16000),
                        (24000, sample_rate <= 24000),
                        (48000, True),  # Default fallback
                    ]

                    opus_sample_rate = None
                    for candidate_rate, condition in opus_rate_candidates:
                        if (
                            condition
                            and player_info.player_support
                            and candidate_rate in player_info.player_support.support_sample_rates
                        ):
                            opus_sample_rate = candidate_rate
                            break

                    if opus_sample_rate is None:
                        self._logger.error(
                            "Client %s does not support any Opus sample rates, trying next codec",
                            self.client_id,
                        )
                        continue  # Try next codec in fallback chain

                    # Opus is viable, adjust sample rate and use it
                    if sample_rate != opus_sample_rate:
                        self._logger.debug(
                            "Adjusted sample_rate for Opus on client %s: %s -> %s",
                            self.client_id,
                            sample_rate,
                            opus_sample_rate,
                        )
                    sample_rate = opus_sample_rate

                codec = candidate_codec
                break

        if codec is None:
            raise ValueError(f"Client {self.client_id} does not support any known codec")

        if codec != preferred_codec:
            self._logger.info(
                "Falling back from preferred codec %s to %s for client %s",
                preferred_codec,
                codec,
                self.client_id,
            )

        # FLAC and PCM support any sample rate, no adjustment needed
        return AudioFormat(sample_rate, bit_depth, channels, codec)

    def _build_encoder(
        self,
        audio_format: AudioFormat,
        input_audio_layout: str,
        input_audio_format: str,
    ) -> tuple[av.AudioCodecContext | None, str | None, int]:
        """
        Create and open an encoder if needed.

        Returns:
            tuple of (encoder, header_b64, samples_per_chunk).
            For PCM, returns (None, None, default_samples_per_chunk).
        """
        if audio_format.codec == AudioCodec.PCM:
            # Default to ~25ms chunks for PCM
            samples_per_chunk = int(audio_format.sample_rate * 0.025)
            return None, None, samples_per_chunk

        encoder = cast(
            "av.AudioCodecContext", av.AudioCodecContext.create(audio_format.codec.value, "w")
        )
        encoder.sample_rate = audio_format.sample_rate
        encoder.layout = input_audio_layout
        encoder.format = input_audio_format
        if audio_format.codec == AudioCodec.FLAC:
            # Only default compression level for now
            encoder.options = {"compression_level": "5"}
        with av_logging.Capture() as logs:
            encoder.open()
        for log in logs:
            self._logger.debug("Opening AudioCodecContext log from av: %s", log)
        header = bytes(encoder.extradata) if encoder.extradata else b""
        # For FLAC, we need to construct a proper FLAC stream header ourselves
        # since ffmpeg only provides the StreamInfo metadata block in extradata:
        # See https://datatracker.ietf.org/doc/rfc9639/ Section 8.1
        if audio_format.codec == AudioCodec.FLAC and header:
            # FLAC stream signature (4 bytes): "fLaC"
            # Metadata block header (4 bytes):
            # - Bit 0: last metadata block (1 since we only have one)
            # - Bits 1-7: block type (0 for StreamInfo)
            # - Next 3 bytes: block length of the next metadata block in bytes
            # StreamInfo block (34 bytes): as provided by ffmpeg
            header = b"fLaC\x80" + (len(header)).to_bytes(3, "big") + header
        codec_header_b64 = base64.b64encode(header).decode()
        samples_per_chunk = (
            int(encoder.frame_size) if encoder.frame_size else int(audio_format.sample_rate * 0.025)
        )
        return encoder, codec_header_b64, samples_per_chunk

    async def _skip_initial_bytes(
        self, audio_stream: AsyncGenerator[bytes, None], bytes_to_skip_total: int
    ) -> bytearray:
        """Consume from audio_stream until bytes_to_skip_total are skipped; return remainder."""
        pending = bytearray()
        if bytes_to_skip_total <= 0:
            return pending
        async for buf in audio_stream:
            if not buf:
                continue
            if bytes_to_skip_total >= len(buf):
                bytes_to_skip_total -= len(buf)
                continue
            pending.extend(buf[bytes_to_skip_total:])
            bytes_to_skip_total = 0
            break
        while bytes_to_skip_total > 0:
            try:
                buf = await audio_stream.__anext__()
            except StopAsyncIteration:
                return pending
            if bytes_to_skip_total >= len(buf):
                bytes_to_skip_total -= len(buf)
                continue
            pending.extend(buf[bytes_to_skip_total:])
            bytes_to_skip_total = 0
        return pending

    def _send_pcm(self, chunk: bytes, timestamp_us: int) -> None:
        header = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, timestamp_us)
        self.send_message(header + chunk)

    async def play_media_direct(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        audio_format: AudioFormat,
        *,
        play_start_time_us: int,
        stream_start_time_us: int = 0,
    ) -> int:
        """
        Stream pre-resampled PCM to this client with optional compression and precise timing.

        Prefer the group-level method for simple cases. This low-level API targets advanced
        scenarios like per-client DSP. It does not stop the client/group; the caller manages
        lifecycle.

        Mid-stream behavior: Skips PCM samples equal to `stream_start_time_us` before sending,
        enabling seeking and late joins.

        Args:
            audio_stream: Async generator yielding PCM bytes already in `audio_format`.
            audio_format: Target format for this client (rate, depth, channels, codec).
            play_start_time_us: Absolute timestamp when playback should begin for the first chunk.
            stream_start_time_us: Offset within the stream to start from (microseconds).

        Returns:
            Absolute timestamp (microseconds) when this stream will end.
        """
        self._ensure_role(Roles.PLAYER)
        player_info = self.info
        assert player_info.player_support is not None, "Player support info required"

        # Validate input format
        if audio_format.bit_depth not in (16, 24):
            raise ValueError("Only 16 or 24 bit PCM is supported")
        if audio_format.channels not in (1, 2):
            raise ValueError("Only mono or stereo are supported")

        bytes_per_sample = 2 if audio_format.bit_depth == 16 else 3
        input_audio_format = "s16" if audio_format.bit_depth == 16 else "s24"
        input_audio_layout = "stereo" if audio_format.channels == 2 else "mono"

        # Setup encoder if needed and prepare codec header
        encoder, codec_header_b64, samples_per_chunk = self._build_encoder(
            audio_format, input_audio_layout, input_audio_format
        )

        # Send stream start to this client
        player_stream_info = StreamStartPlayer(
            codec=audio_format.codec.value,
            sample_rate=audio_format.sample_rate,
            channels=audio_format.channels,
            bit_depth=audio_format.bit_depth,
            codec_header=codec_header_b64,
        )
        self.send_message(StreamStartMessage(StreamStartPayload(player=player_stream_info)))

        frame_stride_bytes = audio_format.channels * bytes_per_sample
        buffer_capacity_bytes = player_info.player_support.buffer_capacity
        buffer_tracker = _BufferTracker(
            loop=self._server.loop,
            logger=self._logger,
            client_id=self.client_id,
            capacity_bytes=buffer_capacity_bytes,
        )
        context = _DirectStreamContext(
            client=self,
            audio_format=audio_format,
            input_audio_format=input_audio_format,
            input_audio_layout=input_audio_layout,
            samples_per_chunk=samples_per_chunk,
            buffer_tracker=buffer_tracker,
            play_start_time_us=play_start_time_us,
            encoder=encoder,
            frame_stride_bytes=frame_stride_bytes,
            send_pcm=self._send_pcm,
        )

        # Skip initial offset within the stream
        bytes_to_skip_total = (
            int((stream_start_time_us * audio_format.sample_rate) / 1_000_000) * frame_stride_bytes
        )
        pending = await self._skip_initial_bytes(audio_stream, bytes_to_skip_total)
        if bytes_to_skip_total > 0 and not pending:
            return play_start_time_us

        input_buffer = pending

        async for chunk in audio_stream:
            if not chunk:
                continue
            input_buffer.extend(chunk)
            await context.drain_ready_chunks(input_buffer, force_flush=False)

        await context.drain_ready_chunks(input_buffer, force_flush=True)
        await context.flush_encoder()

        buffer_tracker.prune_consumed()
        end_timestamp_us = play_start_time_us + _samples_to_microseconds(
            context.samples_sent_total,
            audio_format.sample_rate,
        )

        self._logger.debug(
            "Completed direct stream for client %s: pcm=%sB compressed=%sB max_buffer=%s/%sB",
            self.client_id,
            context.pcm_bytes_consumed,
            context.compressed_bytes_sent,
            buffer_tracker.max_usage_bytes,
            buffer_capacity_bytes,
        )

        return end_timestamp_us
