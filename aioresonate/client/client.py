"""Resonate Client implementation to connect to a Resonate Server."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from aiohttp import ClientSession, ClientWebSocketResponse, WSMessage, WSMsgType

from aioresonate.models import BINARY_HEADER_SIZE, BinaryMessageType, unpack_binary_header
from aioresonate.models.controller import (
    GroupCommandClientMessage,
    GroupCommandClientPayload,
    GroupUpdateServerMessage,
    GroupUpdateServerPayload,
)
from aioresonate.models.core import (
    ClientHelloMessage,
    ClientHelloPayload,
    ClientTimeMessage,
    ClientTimePayload,
    ServerHelloMessage,
    ServerHelloPayload,
    ServerTimeMessage,
    ServerTimePayload,
    SessionUpdateMessage,
    SessionUpdatePayload,
    StreamEndMessage,
    StreamStartMessage,
    StreamUpdateMessage,
)
from aioresonate.models.metadata import ClientHelloMetadataSupport
from aioresonate.models.player import (
    ClientHelloPlayerSupport,
    PlayerUpdateMessage,
    PlayerUpdatePayload,
    StreamStartPlayer,
)
from aioresonate.models.types import MediaCommand, PlayerStateType, Roles, ServerMessage

from .time_sync import ResonateTimeFilter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PCMFormat:
    """PCM audio format description."""

    sample_rate: int
    """Sample rate in Hz (e.g., 48000, 44100)."""
    channels: int
    """Number of audio channels (1=mono, 2=stereo)."""
    bit_depth: int
    """Bits per sample (e.g., 16, 24, 32)."""

    def __post_init__(self) -> None:
        """Validate the provided PCM audio format."""
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if self.bit_depth not in (16, 24, 32):
            raise ValueError("bit_depth must be 16, 24, or 32")

    @property
    def frame_size(self) -> int:
        """Return bytes per PCM frame."""
        return self.channels * (self.bit_depth // 8)


MetadataCallback = Callable[[SessionUpdatePayload], Awaitable[None] | None]
GroupUpdateCallback = Callable[[GroupUpdateServerPayload], Awaitable[None] | None]
StreamStartCallback = Callable[[StreamStartMessage], Awaitable[None] | None]
StreamEndCallback = Callable[[], Awaitable[None] | None]
# Callback for audio chunks: (server_timestamp_us, audio_data, format)
AudioChunkCallback = Callable[[int, bytes, PCMFormat], Awaitable[None] | None]


@dataclass(slots=True)
class ServerInfo:
    """Information about the connected server."""

    server_id: str
    name: str
    version: int


class ResonateClient:
    """
    Async Resonate client capable of handling playback and metadata.

    Attributes:
        _client_id: Unique identifier for this client.
        _client_name: Human-readable name for this client.
        _explicit_roles: Optional list of roles this client supports (CONTROLLER,
            PLAYER, METADATA). If None, defaults to all roles.
        _explicit_player_support: Optional custom player capabilities. If None,
            defaults to PCM support with standard sample rates.
        _explicit_metadata_support: Optional custom metadata capabilities. If None,
            defaults to basic metadata support.
        _session: Optional aiohttp ClientSession for WebSocket connection. If None,
            a session is created and owned by this client.
    """

    _client_id: str
    """Unique identifier for this client."""
    _client_name: str
    """Human-readable name for this client."""
    _roles: list[Roles]
    """List of roles this client supports."""
    _player_support: ClientHelloPlayerSupport | None
    """Player capabilities (only set if PLAYER role is supported)."""
    _metadata_support: ClientHelloMetadataSupport | None
    """Metadata capabilities (only set if METADATA role is supported)."""
    _session: ClientSession | None
    """Optional aiohttp ClientSession for WebSocket connection."""

    _loop: asyncio.AbstractEventLoop | None = None
    """Event loop for this client."""
    _ws: ClientWebSocketResponse | None = None
    """WebSocket connection to the server."""
    _owns_session: bool
    """Whether this client owns and should close the session."""
    _connected: bool = False
    """Whether the client is currently connected."""
    _server_info: ServerInfo | None = None
    """Information about the connected server."""
    _server_hello_event: asyncio.Event | None = None
    """Event signaled when server hello is received."""

    _reader_task: asyncio.Task[None] | None = None
    """Background task reading messages from server."""
    _time_task: asyncio.Task[None] | None = None
    """Background task for time synchronization."""

    _static_delay_us: int = 0
    """Static playback delay in microseconds."""
    _pending_time_message: bool = False
    """Whether a time sync message is awaiting response."""

    _current_player: StreamStartPlayer | None = None
    """Current active player configuration."""
    _current_pcm_format: PCMFormat | None = None
    """Current PCM audio format for active stream."""

    _group_state: GroupUpdateServerPayload | None = None
    """Latest group state received from server."""
    _session_state: SessionUpdatePayload | None = None
    """Latest session state received from server."""

    _metadata_callback: MetadataCallback | None = None
    """Callback invoked on session/update messages."""
    _group_callback: GroupUpdateCallback | None = None
    """Callback invoked on group/update messages."""
    _stream_start_callback: StreamStartCallback | None = None
    """Callback invoked when a stream starts."""
    _stream_end_callback: StreamEndCallback | None = None
    """Callback invoked when a stream ends."""
    _audio_chunk_callback: AudioChunkCallback | None = None
    """Callback invoked when audio chunks are received."""

    def __init__(
        self,
        client_id: str,
        client_name: str,
        roles: Sequence[Roles],
        *,
        player_support: ClientHelloPlayerSupport | None = None,
        metadata_support: ClientHelloMetadataSupport | None = None,
        session: ClientSession | None = None,
        static_delay_ms: float = 0.0,
    ) -> None:
        """
        Create a new Resonate client instance.

        Args:
            client_id: Unique identifier for this client.
            client_name: Human-readable name for this client.
            roles: Sequence of roles this client supports. Must include PLAYER
                if player_support is provided; must include METADATA if
                metadata_support is provided.
            player_support: Custom player capabilities. Required if PLAYER role
                is specified; raises ValueError if missing.
            metadata_support: Custom metadata capabilities. Required if METADATA
                role is specified; raises ValueError if missing.
            session: Optional aiohttp ClientSession. If None, a session is created
                and managed by this client.
            static_delay_ms: Static playback delay in milliseconds applied after
                clock synchronization. Defaults to 0.0.

        Raises:
            ValueError: If PLAYER in roles but player_support is None, or if
                METADATA in roles but metadata_support is None.
        """
        self._client_id = client_id
        self._client_name = client_name
        self._roles = list(roles)

        # Validate and store player support
        if Roles.PLAYER in self._roles:
            if player_support is None:
                raise ValueError("player_support is required when PLAYER role is specified")
            self._player_support = player_support
        else:
            self._player_support = None

        # Validate and store metadata support
        if Roles.METADATA in self._roles:
            if metadata_support is None:
                raise ValueError("metadata_support is required when METADATA role is specified")
            self._metadata_support = metadata_support
        else:
            self._metadata_support = None
        self._session = session
        self._owns_session = session is None
        self._send_lock = asyncio.Lock()
        self._time_filter = ResonateTimeFilter()
        self.set_static_delay_ms(static_delay_ms)

    @property
    def server_info(self) -> ServerInfo | None:
        """Return information about the connected server, if available."""
        return self._server_info

    @property
    def connected(self) -> bool:
        """Return True if the client currently has an active connection."""
        return self._connected and self._ws is not None and not self._ws.closed

    @property
    def static_delay_ms(self) -> float:
        """Return the currently configured static playback delay in milliseconds."""
        return self._static_delay_us / 1_000.0

    def set_static_delay_ms(self, delay_ms: float) -> None:
        """Update the static playback delay applied after clock synchronisation."""
        delay_us = round(delay_ms * 1_000.0)
        if delay_us == self._static_delay_us:
            return
        self._static_delay_us = delay_us
        logger.info("Set static playback delay to %.1f ms", self.static_delay_ms)

    async def connect(self, url: str) -> None:
        """Connect to a Resonate server via WebSocket."""
        if self.connected:
            logger.debug("Already connected")
            return

        self._loop = asyncio.get_running_loop()
        if self._session is None:
            self._session = ClientSession()
        self._server_hello_event = asyncio.Event()

        logger.info("Connecting to Resonate server at %s", url)
        self._ws = await self._session.ws_connect(url, heartbeat=30)
        self._connected = True

        self._reader_task = self._loop.create_task(self._reader_loop())
        await self._send_client_hello()

        try:
            await asyncio.wait_for(self._server_hello_event.wait(), timeout=10)
        except TimeoutError as err:
            await self.disconnect()
            raise TimeoutError("Timed out waiting for server/hello response") from err

        await self._send_time_message()
        self._time_task = self._loop.create_task(self._time_sync_loop())
        logger.info("Handshake with server complete")

    async def disconnect(self) -> None:
        """Disconnect from the server and release resources."""
        self._connected = False
        current_task = asyncio.current_task(loop=self._loop) if self._loop else None

        if self._time_task is not None and self._time_task is not current_task:
            self._time_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._time_task
            self._time_task = None
        if self._reader_task is not None:
            if self._reader_task is not current_task:
                self._reader_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._reader_task
            self._reader_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None
        self._time_filter.reset()
        self._server_info = None
        self._server_hello_event = None
        self._group_state = None
        self._session_state = None
        self._current_pcm_format = None
        self._current_player = None
        self._pending_time_message = False

    async def send_player_state(
        self,
        *,
        state: PlayerStateType,
        volume: int,
        muted: bool,
    ) -> None:
        """Send the current player state to the server."""
        if not self.connected:
            raise RuntimeError("Client is not connected")
        message = PlayerUpdateMessage(
            payload=PlayerUpdatePayload(state=state, volume=volume, muted=muted)
        )
        await self._send_message(message.to_json())

    async def send_group_command(
        self,
        command: MediaCommand,
        *,
        volume: int | None = None,
        mute: bool | None = None,
    ) -> None:
        """Send a group command (playback control) to the server."""
        if not self.connected:
            raise RuntimeError("Client is not connected")
        payload = GroupCommandClientPayload(command=command, volume=volume, mute=mute)
        message = GroupCommandClientMessage(payload=payload)
        await self._send_message(message.to_json())

    def set_metadata_listener(self, callback: MetadataCallback | None) -> None:
        """Set or clear (if None) the callback invoked on session/update messages."""
        self._metadata_callback = callback

    def set_group_update_listener(self, callback: GroupUpdateCallback | None) -> None:
        """Set or clear (if None) the callback invoked on group/update messages."""
        self._group_callback = callback

    def set_stream_start_listener(self, callback: StreamStartCallback | None) -> None:
        """Set or clear (if None) the callback invoked when a stream starts."""
        self._stream_start_callback = callback

    def set_stream_end_listener(self, callback: StreamEndCallback | None) -> None:
        """Set or clear (if None) the callback invoked when a stream ends."""
        self._stream_end_callback = callback

    def set_audio_chunk_listener(self, callback: AudioChunkCallback | None) -> None:
        """
        Set or clear (if None) the callback invoked when audio chunks are received.

        The callback receives:
        - server_timestamp_us: Server timestamp when this audio should play
        - audio_data: Raw PCM audio bytes
        - format: PCMFormat describing the audio format

        To convert server timestamps to client play time (monotonic loop time),
        use the compute_play_time() and compute_server_time() methods provided
        by this client instance. These handle time synchronization and static delay
        automatically.
        """
        self._audio_chunk_callback = callback

    def _build_client_hello(self) -> ClientHelloMessage:
        payload = ClientHelloPayload(
            client_id=self._client_id,
            name=self._client_name,
            version=1,
            supported_roles=self._roles,
            player_support=self._player_support,
            metadata_support=self._metadata_support,
        )
        return ClientHelloMessage(payload=payload)

    async def _send_client_hello(self) -> None:
        hello = self._build_client_hello()
        await self._send_message(hello.to_json())

    async def _send_time_message(self) -> None:
        if self._pending_time_message or not self.connected:
            return
        now_us = self._now_us()
        message = ClientTimeMessage(payload=ClientTimePayload(client_transmitted=now_us))
        self._pending_time_message = True
        try:
            await self._send_message(message.to_json())
        except Exception:
            self._pending_time_message = False
            raise

    async def _send_message(self, payload: str) -> None:
        if not self._ws:
            raise RuntimeError("WebSocket is not connected")
        async with self._send_lock:
            await self._ws.send_str(payload)

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                await self._handle_ws_message(msg)
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            pass
        except Exception:
            logger.exception("WebSocket reader encountered an error")
        finally:
            if self._connected:
                await self.disconnect()

    async def _handle_ws_message(self, msg: WSMessage) -> None:
        if msg.type is WSMsgType.TEXT:
            await self._handle_json_message(msg.data)
        elif msg.type is WSMsgType.BINARY:
            await self._handle_binary_message(msg.data)
        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
            logger.info("WebSocket closed by server")
            await self.disconnect()
        elif msg.type is WSMsgType.ERROR:
            logger.error("WebSocket error: %s", self._ws.exception() if self._ws else "unknown")
            await self.disconnect()

    async def _handle_json_message(self, data: str) -> None:
        try:
            message = ServerMessage.from_json(data)
        except Exception:
            logger.exception("Failed to parse server message: %s", data)
            return

        match message:
            case ServerHelloMessage(payload=payload):
                self._handle_server_hello(payload)
            case ServerTimeMessage(payload=payload):
                self._handle_server_time(payload)
            case StreamStartMessage():
                await self._handle_stream_start(message)
            case StreamUpdateMessage():
                await self._handle_stream_update(message)
            case StreamEndMessage():
                await self._handle_stream_end()
            case SessionUpdateMessage(payload=payload):
                await self._handle_session_update(payload)
            case GroupUpdateServerMessage(payload=payload):
                await self._handle_group_update(payload)
            case _:
                logger.debug("Unhandled server message type: %s", type(message).__name__)

    async def _handle_binary_message(self, payload: bytes) -> None:
        try:
            header = unpack_binary_header(payload)
        except Exception:
            logger.exception("Failed to unpack binary header")
            return

        try:
            message_type = BinaryMessageType(header.message_type)
        except ValueError:
            logger.warning("Unknown binary message type: %s", header.message_type)
            return

        if message_type is BinaryMessageType.AUDIO_CHUNK:
            await self._handle_audio_chunk(header.timestamp_us, payload[BINARY_HEADER_SIZE:])
        else:
            logger.debug("Ignoring unsupported binary message type: %s", message_type)

    def _handle_server_hello(self, payload: ServerHelloPayload) -> None:
        self._server_info = ServerInfo(
            server_id=payload.server_id,
            name=payload.name,
            version=payload.version,
        )
        if self._server_hello_event:
            self._server_hello_event.set()
        logger.info(
            "Connected to server '%s' (%s) version %s",
            payload.name,
            payload.server_id,
            payload.version,
        )

    def _handle_server_time(self, payload: ServerTimePayload) -> None:
        now_us = self._now_us()
        offset = (
            (payload.server_received - payload.client_transmitted)
            + (payload.server_transmitted - now_us)
        ) / 2
        delay = (
            (now_us - payload.client_transmitted)
            - (payload.server_transmitted - payload.server_received)
        ) / 2
        self._time_filter.update(round(offset), round(delay), now_us)
        self._pending_time_message = False

    async def _handle_stream_start(self, message: StreamStartMessage) -> None:
        logger.info("Stream started")
        player = message.payload.player
        if player is None:
            logger.warning("Stream start message missing player payload")
            return

        if player.codec != "pcm":
            logger.error("Unsupported codec '%s' - only PCM is supported", player.codec)
            return

        pcm_format = PCMFormat(
            sample_rate=player.sample_rate,
            channels=player.channels,
            bit_depth=player.bit_depth,
        )
        self._configure_audio_output(pcm_format)
        self._current_player = StreamStartPlayer(
            codec=player.codec,
            sample_rate=player.sample_rate,
            channels=player.channels,
            bit_depth=player.bit_depth,
            codec_header=player.codec_header,
        )
        await self._notify_stream_start(message)
        await self._send_time_message()

    async def _handle_stream_update(self, message: StreamUpdateMessage) -> None:
        player_update = message.payload.player
        if player_update is None:
            return
        if self._current_player is None:
            logger.debug("Ignoring stream update without active stream")
            return

        codec = player_update.codec or self._current_player.codec
        if codec != "pcm":
            logger.error("Unsupported codec update '%s' - only PCM is supported", codec)
            return

        sample_rate = player_update.sample_rate or self._current_player.sample_rate
        channels = player_update.channels or self._current_player.channels
        bit_depth = player_update.bit_depth or self._current_player.bit_depth

        pcm_format = PCMFormat(sample_rate=sample_rate, channels=channels, bit_depth=bit_depth)
        self._configure_audio_output(pcm_format)
        self._current_player.codec = codec
        self._current_player.sample_rate = sample_rate
        self._current_player.channels = channels
        self._current_player.bit_depth = bit_depth
        if player_update.codec_header is not None:
            self._current_player.codec_header = player_update.codec_header

    async def _handle_stream_end(self) -> None:
        logger.info("Stream ended")
        self._current_player = None
        self._current_pcm_format = None
        await self._notify_stream_end()

    async def _handle_session_update(self, payload: SessionUpdatePayload) -> None:
        self._session_state = payload
        await self._notify_metadata_callback(payload)

    async def _handle_group_update(self, payload: GroupUpdateServerPayload) -> None:
        self._group_state = payload
        await self._notify_group_callback(payload)

    def _configure_audio_output(self, pcm_format: PCMFormat) -> None:
        """Store the current audio format for use in callbacks."""
        self._current_pcm_format = pcm_format

    async def _handle_audio_chunk(self, timestamp_us: int, payload: bytes) -> None:
        """Handle incoming audio chunk and notify callback."""
        if self._audio_chunk_callback is None:
            return
        if self._current_pcm_format is None:
            logger.debug("Dropping audio chunk without format")
            return

        # Pass server timestamp directly to callback - it handles time conversion
        # to allow for dynamic time base updates
        try:
            result = self._audio_chunk_callback(timestamp_us, payload, self._current_pcm_format)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("Error in audio chunk callback %s", self._audio_chunk_callback)

    def compute_play_time(self, server_timestamp_us: int) -> int:
        """
        Convert server timestamp to client play time with static delay applied.

        This method converts a server timestamp to the equivalent client timestamp
        (based on monotonic loop time) and adds the configured static delay.
        Use this to determine when audio should be played on the client.

        Args:
            server_timestamp_us: Server timestamp in microseconds.

        Returns:
            Client play time in microseconds (monotonic loop time + static delay).
        """
        if self._time_filter.is_synchronized:
            client_time = self._time_filter.compute_client_time(server_timestamp_us)
            return client_time + self._static_delay_us
        # Fallback: add a conservative delay if time sync isn't ready yet
        return self._now_us() + 500_000 + self._static_delay_us

    def compute_server_time(self, client_timestamp_us: int) -> int:
        """
        Convert client timestamp to server timestamp with static delay removed.

        This is the inverse of compute_play_time. It converts a client timestamp
        (monotonic loop time) to the equivalent server timestamp, removing the
        static delay first.

        Args:
            client_timestamp_us: Client timestamp in microseconds (monotonic loop time).

        Returns:
            Server timestamp in microseconds.
        """
        # Remove static delay first, then convert to server time
        adjusted_client_time = client_timestamp_us - self._static_delay_us
        return self._time_filter.compute_server_time(adjusted_client_time)

    async def _notify_metadata_callback(self, payload: SessionUpdatePayload) -> None:
        if self._metadata_callback is None:
            return
        try:
            result = self._metadata_callback(payload)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("Error in metadata callback %s", self._metadata_callback)

    async def _notify_group_callback(self, payload: GroupUpdateServerPayload) -> None:
        if self._group_callback is None:
            return
        try:
            result = self._group_callback(payload)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("Error in group callback %s", self._group_callback)

    async def _notify_stream_start(self, message: StreamStartMessage) -> None:
        if self._stream_start_callback is None:
            return
        try:
            result = self._stream_start_callback(message)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("Error in stream start callback %s", self._stream_start_callback)

    async def _notify_stream_end(self) -> None:
        if self._stream_end_callback is None:
            return
        try:
            result = self._stream_end_callback()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("Error in stream end callback %s", self._stream_end_callback)

    async def _time_sync_loop(self) -> None:
        try:
            while self.connected:
                try:
                    await self._send_time_message()
                except Exception:
                    logger.exception("Failed to send time sync message")
                await asyncio.sleep(self._compute_time_sync_interval())
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            pass

    def _compute_time_sync_interval(self) -> float:
        if not self._time_filter.is_synchronized:
            return 0.2
        error = self._time_filter.error
        if error < 1_000:
            return 3.0
        if error < 2_000:
            return 1.0
        if error < 5_000:
            return 0.5
        return 0.2

    def _now_us(self) -> int:
        loop = self._loop or asyncio.get_running_loop()
        return int(loop.time() * 1_000_000)

    async def __aenter__(self) -> Self:
        """Enter the async context manager returning this instance."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Disconnect when leaving the async context manager."""
        await self.disconnect()
