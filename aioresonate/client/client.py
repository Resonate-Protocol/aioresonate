"""Resonate Client implementation to connect to a Resonate Server."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

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
    StreamRequestFormatMessage,
    StreamRequestFormatPayload,
    StreamStartPlayer,
)
from aioresonate.models.types import MediaCommand, PlayerStateType, Roles, ServerMessage

from .audio import AudioPlayer, PCMFormat
from .time_sync import ResonateTimeFilter

logger = logging.getLogger(__name__)

MetadataCallback = Callable[[SessionUpdatePayload], Awaitable[None] | None]
GroupUpdateCallback = Callable[[GroupUpdateServerPayload], Awaitable[None] | None]
StreamStartCallback = Callable[[StreamStartMessage], Awaitable[None] | None]
StreamEndCallback = Callable[[], Awaitable[None] | None]


@dataclass(slots=True)
class ServerInfo:
    """Information about the connected server."""

    server_id: str
    name: str
    version: int


class ResonateClient:
    """Async Resonate client capable of handling playback and metadata."""

    def __init__(
        self,
        client_id: str,
        client_name: str,
        *,
        roles: Sequence[Roles] | None = None,
        player_support: ClientHelloPlayerSupport | None = None,
        metadata_support: ClientHelloMetadataSupport | None = None,
        session: ClientSession | None = None,
        static_delay_ms: float = 0.0,
    ) -> None:
        """Create a new Resonate client instance."""
        self._client_id = client_id
        self._client_name = client_name
        self._explicit_roles = list(roles) if roles is not None else None
        self._explicit_player_support = player_support
        self._explicit_metadata_support = metadata_support
        self._session = session
        self._owns_session = session is None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws: ClientWebSocketResponse | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._time_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()
        self._time_filter = ResonateTimeFilter()
        self._static_delay_us = 0
        self.set_static_delay_ms(static_delay_ms)
        self._audio_player: AudioPlayer | None = None
        self._server_info: ServerInfo | None = None
        self._metadata_callbacks: list[MetadataCallback] = []
        self._group_callbacks: list[GroupUpdateCallback] = []
        self._stream_start_callbacks: list[StreamStartCallback] = []
        self._stream_end_callbacks: list[StreamEndCallback] = []
        self._server_hello_event: asyncio.Event | None = None
        self._connected = False
        self._current_player: StreamStartPlayer | None = None
        self._current_pcm_format: PCMFormat | None = None
        self._group_state: GroupUpdateServerPayload | None = None
        self._session_state: SessionUpdatePayload | None = None
        self._pending_time_message = False

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    @property
    def server_info(self) -> ServerInfo | None:
        """Return information about the connected server, if available."""
        return self._server_info

    @property
    def connected(self) -> bool:
        """Return True if the client currently has an active connection."""
        return self._connected and self._ws is not None and not self._ws.closed

    @property
    def audio_available(self) -> bool:
        """Return True if audio playback is available."""
        return bool(self._audio_player and self._audio_player.audio_available)

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
        self._audio_player = AudioPlayer(self._loop)

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
        if self._audio_player is not None:
            await self._audio_player.stop()
            self._audio_player = None
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None
        self._time_filter.reset()
        self._server_info = None
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
        await self._send_json(message)

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
        await self._send_json(message)

    async def request_stream_format_change(
        self,
        *,
        codec: str | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
        bit_depth: int | None = None,
    ) -> None:
        """Request a different stream format from the server."""
        if not self.connected:
            raise RuntimeError("Client is not connected")
        payload = StreamRequestFormatPayload(
            codec=codec,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
        )
        message = StreamRequestFormatMessage(payload=payload)
        await self._send_json(message)

    def add_metadata_listener(self, callback: MetadataCallback) -> None:
        """Register a callback invoked on session/update messages."""
        self._metadata_callbacks.append(callback)

    def add_group_update_listener(self, callback: GroupUpdateCallback) -> None:
        """Register a callback invoked on group/update messages."""
        self._group_callbacks.append(callback)

    def add_stream_start_listener(self, callback: StreamStartCallback) -> None:
        """Register a callback invoked when a stream starts."""
        self._stream_start_callbacks.append(callback)

    def add_stream_end_listener(self, callback: StreamEndCallback) -> None:
        """Register a callback invoked when a stream ends."""
        self._stream_end_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_client_hello(self) -> ClientHelloMessage:
        roles = self._explicit_roles or [Roles.CONTROLLER, Roles.PLAYER, Roles.METADATA]

        player_support = None
        if Roles.PLAYER in roles:
            player_support = self._explicit_player_support or ClientHelloPlayerSupport(
                support_codecs=["pcm"],
                support_channels=[2, 1],
                support_sample_rates=[48_000, 44_100],
                support_bit_depth=[16],
                buffer_capacity=1_000_000,
            )

        metadata_support = None
        if Roles.METADATA in roles:
            metadata_support = self._explicit_metadata_support or ClientHelloMetadataSupport(
                support_picture_formats=[],
                media_width=None,
                media_height=None,
            )

        payload = ClientHelloPayload(
            client_id=self._client_id,
            name=self._client_name,
            version=1,
            supported_roles=list(roles),
            player_support=player_support,
            metadata_support=metadata_support,
        )
        return ClientHelloMessage(payload=payload)

    async def _send_client_hello(self) -> None:
        hello = self._build_client_hello()
        await self._send_json(hello)

    async def _send_time_message(self) -> None:
        if self._pending_time_message or not self.connected:
            return
        now_us = self._now_us()
        message = ClientTimeMessage(payload=ClientTimePayload(client_transmitted=now_us))
        self._pending_time_message = True
        try:
            await self._send_json(message)
        except Exception:
            self._pending_time_message = False
            raise

    async def _send_json(self, message: Any) -> None:
        if not self._ws:
            raise RuntimeError("WebSocket is not connected")
        payload = message.to_json() if hasattr(message, "to_json") else str(message)
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
        self._time_filter.update(offset, delay, now_us)
        self._pending_time_message = False

    async def _handle_stream_start(self, message: StreamStartMessage) -> None:
        logger.info("Stream started")
        player = message.payload.player
        if player is None:
            logger.warning("Stream start message missing player payload")
            return

        if player.codec != "pcm":
            logger.error("Unsupported codec '%s' - requesting PCM fallback", player.codec)
            await self.request_stream_format_change(codec="pcm")
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
            logger.error("Unsupported codec update '%s'", codec)
            await self.request_stream_format_change(codec="pcm")
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
        if self._audio_player is not None:
            self._audio_player.clear()
        self._current_player = None
        self._current_pcm_format = None
        await self._notify_stream_end()

    async def _handle_session_update(self, payload: SessionUpdatePayload) -> None:
        self._session_state = payload
        await self._notify_callbacks(self._metadata_callbacks, payload)

    async def _handle_group_update(self, payload: GroupUpdateServerPayload) -> None:
        self._group_state = payload
        await self._notify_callbacks(self._group_callbacks, payload)

    def _configure_audio_output(self, pcm_format: PCMFormat) -> None:
        self._current_pcm_format = pcm_format
        if self._audio_player is None:
            return
        self._audio_player.clear()
        self._audio_player.set_format(pcm_format)
        self._audio_player.start()

    async def _handle_audio_chunk(self, timestamp_us: int, payload: bytes) -> None:
        if self._audio_player is None or not self._audio_player.audio_available:
            return
        if self._current_pcm_format is None:
            logger.debug("Dropping audio chunk without format")
            return

        play_at_us = self._compute_play_time(timestamp_us)
        now_us = self._now_us()
        if play_at_us < now_us - 200_000:
            logger.debug("Dropping stale audio chunk (late by %d us)", now_us - play_at_us)
            return
        self._audio_player.submit(play_at_us, payload)

    def _compute_play_time(self, server_timestamp_us: int) -> int:
        if self._time_filter.ready:
            play_time = self._time_filter.compute_client_time(server_timestamp_us)
            return play_time + self._static_delay_us
        # Fallback: add a conservative delay if time sync isn't ready yet
        return self._now_us() + 500_000 + self._static_delay_us

    async def _notify_callbacks(
        self,
        callbacks: list[Callable[[Any], Awaitable[None] | None]],
        payload: Any,
    ) -> None:
        for callback in callbacks:
            try:
                result = callback(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Error in client callback %s", callback)

    async def _notify_stream_start(self, message: StreamStartMessage) -> None:
        await self._notify_callbacks(self._stream_start_callbacks, message)

    async def _notify_stream_end(self) -> None:
        for callback in self._stream_end_callbacks:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Error in stream end callback %s", callback)

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
        if not self._time_filter.ready:
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
