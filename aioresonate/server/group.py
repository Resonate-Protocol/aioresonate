"""Manages and synchronizes playback for a group of one or more clients."""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from asyncio import Task
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, cast

import av
from av import logging as av_logging
from PIL import Image

from aioresonate.models import (
    BinaryMessageType,
    pack_binary_header_raw,
)
from aioresonate.models.controller import GroupCommandClientPayload
from aioresonate.models.core import (
    SessionUpdateMessage,
    SessionUpdatePayload,
    StreamEndMessage,
    StreamStartMessage,
    StreamStartPayload,
)
from aioresonate.models.metadata import (
    StreamStartMetadata,
)
from aioresonate.models.player import (
    StreamRequestFormatPayload,
    StreamStartPlayer,
)
from aioresonate.models.types import (
    MediaCommand,
    PictureFormat,
    PlaybackStateType,
    Roles,
)
from aioresonate.models.visualizer import StreamStartVisualizer
from aioresonate.server.stream import build_flac_stream_header

from .metadata import Metadata
from .stream import AudioCodec, AudioFormat, ClientStreamConfig, MediaStream, Streamer

# The cyclic import is not an issue during runtime, so hide it
# pyright: reportImportCycles=none
if TYPE_CHECKING:
    from .client import ResonateClient
    from .player import PlayerClient
    from .server import ResonateServer

INITIAL_PLAYBACK_DELAY_US = 1_000_000

logger = logging.getLogger(__name__)


class GroupEvent:
    """Base event type used by ResonateGroup.add_event_listener()."""


# TODO: make types more fancy
@dataclass
class GroupCommandEvent(GroupEvent):
    """A command was sent to the group."""

    command: MediaCommand
    """The command that was sent."""
    volume: int | None = None
    """For MediaCommand.VOLUME, the target volume (0-100)."""
    mute: bool | None = None
    """For MediaCommand.MUTE, the target mute status."""


@dataclass
class GroupStateChangedEvent(GroupEvent):
    """Group state has changed."""

    state: PlaybackStateType
    """The new group state."""


@dataclass
class GroupMemberAddedEvent(GroupEvent):
    """A client was added to the group."""

    client_id: str
    """The ID of the client that was added."""


@dataclass
class GroupMemberRemovedEvent(GroupEvent):
    """A client was removed from the group."""

    client_id: str
    """The ID of the client that was removed."""


@dataclass
class GroupDeletedEvent(GroupEvent):
    """This group has no more members and has been deleted."""


@dataclass
class _StreamerCommand:
    """Base class for commands sent to the shared Streamer task."""


@dataclass
class _StreamerReconfigureCommand(_StreamerCommand):
    """Request to reconfigure the running streamer with new client topology."""

    channel_formats: dict[str, AudioFormat]
    client_configs: list[ClientStreamConfig]
    new_client_ids: set[str]
    removed_client_ids: set[str]
    result: asyncio.Future[dict[str, StreamStartPlayer]]


class ResonateGroup:
    """
    A group of one or more clients for synchronized playback.

    Handles synchronized audio streaming across multiple clients with automatic
    format conversion and buffer management. Every client is always assigned to
    a group to simplify grouping requests.
    """

    _clients: list[ResonateClient]
    """List of all clients in this group."""
    _player_formats: dict[str, AudioFormat]
    """Mapping of client IDs (with the player role) to their selected audio formats."""
    _client_art_formats: dict[str, PictureFormat]
    """Mapping of client IDs (with the metadata role) to their selected artwork formats."""
    _server: ResonateServer
    """Reference to the ResonateServer instance."""
    _stream_task: Task[int] | None = None
    """Task handling the audio streaming loop, None when not streaming."""
    _stream_audio_format: AudioFormat | None = None
    """The source audio format for the current stream, None when not streaming."""
    _current_metadata: Metadata | None = None
    """Current metadata for the group, None if no metadata set."""
    _current_media_art: Image.Image | None = None
    """Current media art image for the group, None if no image set."""
    _audio_encoders: dict[AudioFormat, av.AudioCodecContext]
    """Mapping of audio formats to their av encoder contexts."""
    _audio_headers: dict[AudioFormat, str]
    """Mapping of audio formats to their base64 encoded headers."""
    _preferred_stream_codec: AudioCodec = AudioCodec.OPUS
    """Preferred codec used by the current stream."""
    _event_cbs: list[Callable[[GroupEvent], Coroutine[None, None, None]]]
    """List of event callbacks for this group."""
    _current_state: PlaybackStateType = PlaybackStateType.STOPPED
    """Current playback state of the group."""
    _group_id: str
    """Unique identifier for this group."""
    _streamer: Streamer | None
    """Active Streamer instance for the current stream, None when not streaming."""
    _media_stream: MediaStream | None
    """Current MediaStream being played, None when not streaming."""
    _channel_formats: dict[str, AudioFormat]
    """Mapping of channel names to their audio formats for the current stream."""
    _channel_generators: dict[str, AsyncGenerator[bytes, None]]
    """Mapping of channel names to their audio data generators for the current stream."""
    _player_channels: dict[str, str]
    """Mapping of player IDs to their assigned channel names."""
    _stream_commands: asyncio.Queue[_StreamerCommand] | None
    """Command queue for the active streamer task, None when not streaming."""
    _play_start_time_us: int | None
    """Absolute timestamp in microseconds when playback started, None when not streaming."""

    def __init__(self, server: ResonateServer, *args: ResonateClient) -> None:
        """
        DO NOT CALL THIS CONSTRUCTOR. INTERNAL USE ONLY.

        Groups are managed automatically by the server.

        Initialize a new ResonateGroup.

        Args:
            server: The ResonateServer instance this group belongs to.
            *args: Clients to add to this group.
        """
        self._clients = list(args)
        self._player_formats = {}
        self._client_art_formats = {}
        self._server = server
        self._stream_task: Task[int] | None = None
        self._stream_audio_format: AudioFormat | None = None
        self._current_metadata = None
        self._current_media_art = None
        self._audio_encoders = {}
        self._audio_headers = {}
        self._event_cbs = []
        self._group_id = str(uuid.uuid4())
        self._streamer: Streamer | None = None
        self._media_stream: MediaStream | None = None
        self._channel_formats: dict[str, AudioFormat] = {}
        self._channel_generators: dict[str, AsyncGenerator[bytes, None]] = {}
        self._player_channels: dict[str, str] = {}
        self._stream_commands: asyncio.Queue[_StreamerCommand] | None = None
        self._play_start_time_us: int | None = None
        logger.debug(
            "ResonateGroup initialized with %d client(s): %s",
            len(self._clients),
            [type(c).__name__ for c in self._clients],
        )

    async def play_media(  # noqa: PLR0915
        self,
        media_stream: MediaStream,
        *,
        play_start_time_us: int | None = None,
        stream_start_time_us: int = 0,
    ) -> int:
        """Start synchronized playback for the current group using a MediaStream."""
        logger.debug(
            "Starting play_media with play_start_time_us=%s, stream_start_time_us=%s",
            play_start_time_us,
            stream_start_time_us,
        )

        self._media_stream = media_stream
        self._streamer = None
        self._channel_generators.clear()
        self._player_channels.clear()

        default_format = media_stream.default_channel_format()
        self._stream_audio_format = default_format

        start_time_us = (
            play_start_time_us
            if play_start_time_us is not None
            else int(self._server.loop.time() * 1_000_000) + INITIAL_PLAYBACK_DELAY_US
        )
        self._play_start_time_us = start_time_us

        group_players = self.players()
        if not group_players:
            logger.info("No player clients in group; skipping playback")
            self._current_state = PlaybackStateType.STOPPED
            return start_time_us

        self._player_formats.clear()
        self._channel_formats = dict(media_stream.available_channels())
        self._channel_generators = {
            name: media_stream.iter_channel(name) for name in self._channel_formats
        }

        # Configure each player with their format and channel
        # All players get the default channel unless MediaStream provides additional channels
        for player in group_players:
            client = player.client
            player_format = player.determine_optimal_format(default_format)
            self._player_formats[client.client_id] = player_format
            # All players use default channel for now (additional channels can be added later)
            self._player_channels[client.client_id] = media_stream.default_channel_name

        channel_formats = dict(self._channel_formats)
        streamer = Streamer(
            loop=self._server.loop,
            logger=logger,
            play_start_time_us=start_time_us,
        )

        # Build client configurations for all players
        client_configs: list[ClientStreamConfig] = []
        for player in group_players:
            support = player.support
            if support is None:
                raise ValueError(f"Player {player.client.client_id} lacks support payload")
            channel_name = self._player_channels[player.client.client_id]
            client_configs.append(
                ClientStreamConfig(
                    client_id=player.client.client_id,
                    target_format=self._player_formats[player.client.client_id],
                    buffer_capacity_bytes=support.buffer_capacity,
                    channel=channel_name,
                    send=player.client.send_message,
                    logger=None,
                )
            )

        start_payloads = streamer.configure(
            channels=channel_formats,
            clients=client_configs,
        )
        self._channel_formats = channel_formats
        self._streamer = streamer
        self._stream_commands = asyncio.Queue()
        self._stream_task = self._server.loop.create_task(
            self._run_streamer(streamer, self._channel_generators, self._stream_commands)
        )

        # Notify clients about the upcoming stream configuration
        for player in group_players:
            player_payload = start_payloads.get(player.client.client_id)
            self._send_stream_start_msg(
                player.client,
                None,
                player_info=player_payload,
            )

        for client in self._clients:
            if client.check_role(Roles.PLAYER):
                continue
            if client.check_role(Roles.METADATA) or client.check_role(Roles.VISUALIZER):
                self._send_stream_start_msg(client, None, include_player=False)

        self._current_state = PlaybackStateType.PLAYING
        self._signal_event(GroupStateChangedEvent(PlaybackStateType.PLAYING))

        end_time_us = start_time_us
        if self._stream_task is not None:
            end_time_us = await self._stream_task
            self._stream_task = None

        self._channel_generators.clear()
        self._channel_formats.clear()
        self._player_channels.clear()
        self._streamer = None
        self._media_stream = None
        self._stream_commands = None

        return end_time_us

    async def _run_streamer(  # noqa: PLR0915
        self,
        streamer: Streamer,
        channel_generators: dict[str, AsyncGenerator[bytes, None]],
        command_queue: asyncio.Queue[_StreamerCommand] | None,
    ) -> int:
        """Consume media channels, distribute via streamer, and return end timestamp."""
        last_end_us = self._play_start_time_us or int(self._server.loop.time() * 1_000_000)
        cancelled = False
        pending_chunks: dict[str, asyncio.Task[bytes]] = {
            name: self._server.loop.create_task(generator.__anext__())
            for name, generator in channel_generators.items()
        }
        command_task: asyncio.Task[_StreamerCommand] | None = None
        if command_queue is not None:
            command_task = self._server.loop.create_task(command_queue.get())

        try:
            while pending_chunks:
                for name in list(pending_chunks):
                    if name not in channel_generators:
                        task = pending_chunks.pop(name)
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                for name, generator in channel_generators.items():
                    if name not in pending_chunks:
                        pending_chunks[name] = self._server.loop.create_task(generator.__anext__())

                wait_set: set[asyncio.Task[object]] = set(pending_chunks.values())
                if command_task is not None:
                    wait_set.add(command_task)
                done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

                if command_task is not None and command_task in done:
                    command = command_task.result()
                    if isinstance(command, _StreamerReconfigureCommand):
                        try:
                            streamer.flush()
                            await streamer.send()
                            streamer.reset()
                            start_payloads = streamer.configure(
                                channels=command.channel_formats,
                                clients=command.client_configs,
                            )
                        except Exception as exc:
                            if not command.result.done():
                                command.result.set_exception(exc)
                            logger.exception("Failed to reconfigure streamer")
                        else:
                            if not command.result.done():
                                command.result.set_result(start_payloads)
                    if command_queue is not None:
                        command_task = self._server.loop.create_task(command_queue.get())
                    continue

                completed_channels = [name for name, task in pending_chunks.items() if task in done]
                chunk_map: dict[str, bytes] = {}
                for name in completed_channels:
                    task = pending_chunks.pop(name)
                    try:
                        chunk = task.result()
                    except StopAsyncIteration:
                        channel_generators.pop(name, None)
                        continue
                    except Exception:
                        logger.exception("Channel %s raised during streaming", name)
                        channel_generators.pop(name, None)
                        continue
                    chunk_map[name] = chunk
                    pending_chunks[name] = self._server.loop.create_task(
                        channel_generators[name].__anext__()
                    )

                if not chunk_map:
                    continue

                streamer.prepare(chunk_map)
                await streamer.send()

            streamer.flush()
            await streamer.send()
            if streamer.last_chunk_end_time_us is not None:
                last_end_us = streamer.last_chunk_end_time_us
        except asyncio.CancelledError:
            cancelled = True
            streamer.flush()
            await streamer.send()
            raise
        else:
            return last_end_us
        finally:
            for generator in channel_generators.values():
                with suppress(Exception):
                    await generator.aclose()
            if command_task is not None:
                command_task.cancel()
                with suppress(asyncio.CancelledError):
                    await command_task
            if cancelled and streamer.last_chunk_end_time_us is not None:
                last_end_us = streamer.last_chunk_end_time_us

    async def _reconfigure_streamer(
        self,
        *,
        media_stream: MediaStream,
        new_client_ids: set[str],
        removed_client_ids: set[str],
    ) -> dict[str, StreamStartPlayer]:
        """Reconfigure the running streamer and return start payloads for new clients."""
        if self._streamer is None or self._stream_commands is None or self._stream_task is None:
            raise RuntimeError("Streamer is not running")

        channel_formats = dict(self._channel_formats)
        client_configs: list[ClientStreamConfig] = []

        for player in self.players():
            support = player.support
            if support is None:
                raise ValueError(f"Player {player.client.client_id} lacks support payload")
            client_id = player.client.client_id
            target_format = self._player_formats[client_id]
            channel_name = self._player_channels.get(client_id, media_stream.default_channel_name)
            client_configs.append(
                ClientStreamConfig(
                    client_id=client_id,
                    target_format=target_format,
                    buffer_capacity_bytes=support.buffer_capacity,
                    channel=channel_name,
                    send=player.client.send_message,
                    logger=None,
                )
            )
        future: asyncio.Future[dict[str, StreamStartPlayer]] = self._server.loop.create_future()
        await self._stream_commands.put(
            _StreamerReconfigureCommand(
                channel_formats=channel_formats,
                client_configs=client_configs,
                new_client_ids=new_client_ids,
                removed_client_ids=removed_client_ids,
                result=future,
            )
        )
        start_payloads = await future
        self._channel_formats = channel_formats
        return start_payloads

    def suggest_optimal_sample_rate(self, source_sample_rate: int) -> int:
        """
        Suggest an optimal sample rate for the next track.

        Analyzes all player clients in this group and returns the best sample rate that
        minimizes resampling across group members. Preference order:
        - If there is a common supported rate across all players, choose the one closest
          to the source sample rate (tie-breaker: higher rate).
        - Otherwise, choose the rate supported by the most players; among those, pick the
          closest to the source (tie-breaker: higher rate).

        Args:
            source_sample_rate: The sample rate of the upcoming source media.

        Returns:
            The recommended sample rate in Hz.
        """
        supported_sets: list[set[int]] = [
            set(client.info.player_support.support_sample_rates)
            for client in self._clients
            if client.check_role(Roles.PLAYER) and client.info.player_support
        ]

        if not supported_sets:
            return source_sample_rate

        # Helper for choosing the closest candidate, biasing towards higher rates on ties
        def choose(candidates: set[int]) -> int:
            # Compute the minimal absolute distance to the source sample rate
            best_distance = min(abs(r - source_sample_rate) for r in candidates)
            # Keep all candidates at that distance and pick the highest rate on a tie
            best_rates = [r for r in candidates if abs(r - source_sample_rate) == best_distance]
            return max(best_rates)

        # 1) Intersection across all players
        if (supported_sets) and (intersection := set.intersection(*supported_sets)):
            return choose(intersection)

        # 2) No common rate; pick the rate supported by the most players, then closest to source
        counts: dict[int, int] = {}
        for s in supported_sets:
            for r in s:
                counts[r] = counts.get(r, 0) + 1
        max_count = max(counts.values())
        top_rates = {r for r, c in counts.items() if c == max_count}
        return choose(top_rates)

    def _get_or_create_audio_encoder(self, audio_format: AudioFormat) -> av.AudioCodecContext:
        """
        Get or create an audio encoder for the given audio format.

        Args:
            audio_format: The audio format to create an encoder for.
                The sample rate and bit depth will be shared for both the input and output streams.
                The input stream must be in a s16 or s24 format. The output stream will be in the
                specified codec.

        Returns:
            av.AudioCodecContext: The audio encoder context.
        """
        if audio_format in self._audio_encoders:
            return self._audio_encoders[audio_format]

        # Create audio encoder context
        ctx = cast(
            "av.AudioCodecContext", av.AudioCodecContext.create(audio_format.codec.value, "w")
        )
        ctx.sample_rate = audio_format.sample_rate
        ctx.layout = "stereo" if audio_format.channels == 2 else "mono"
        assert audio_format.bit_depth in (16, 24)
        ctx.format = "s16" if audio_format.bit_depth == 16 else "s24"

        if audio_format.codec == AudioCodec.FLAC:
            # Default compression level for now
            ctx.options = {"compression_level": "5"}

        with av_logging.Capture() as logs:
            ctx.open()
        for log in logs:
            logger.debug("Opening AudioCodecContext log from av: %s", log)

        # Store the encoder and extract the header
        self._audio_encoders[audio_format] = ctx
        header = bytes(ctx.extradata) if ctx.extradata else b""

        if audio_format.codec == AudioCodec.FLAC and header:
            header = build_flac_stream_header(header)

        self._audio_headers[audio_format] = base64.b64encode(header).decode()

        logger.debug(
            "Created audio encoder: frame_size=%d, header_length=%d",
            ctx.frame_size,
            len(header),
        )

        return ctx

    def _get_audio_header(self, audio_format: AudioFormat) -> str | None:
        """
        Get the codec header for the given audio format.

        Args:
            audio_format: The audio format to get the header for.

        Returns:
            str: Base64 encoded codec header.
        """
        if audio_format.codec == AudioCodec.PCM:
            return None
        if audio_format not in self._audio_headers:
            # Create encoder to generate header
            self._get_or_create_audio_encoder(audio_format)

        return self._audio_headers[audio_format]

    def _send_stream_start_msg(
        self,
        client: ResonateClient,
        audio_format: AudioFormat | None = None,
        *,
        include_player: bool = True,
        player_info: StreamStartPlayer | None = None,
    ) -> None:
        """Send a stream start message to a client with the specified audio format for players."""
        logger.debug(
            "_send_stream_start_msg: client=%s, format=%s",
            client.client_id,
            audio_format,
        )
        if include_player and client.check_role(Roles.PLAYER):
            if player_info is not None:
                player_stream_info = player_info
            else:
                if audio_format is None:
                    raise ValueError("audio_format must be provided for player clients")
                player_stream_info = StreamStartPlayer(
                    codec=audio_format.codec.value,
                    sample_rate=audio_format.sample_rate,
                    channels=audio_format.channels,
                    bit_depth=audio_format.bit_depth,
                    codec_header=self._get_audio_header(audio_format),
                )
        else:
            player_stream_info = None
        if client.check_role(Roles.METADATA) and client.info.metadata_support:
            # Choose the first supported picture format as a simple strategy
            supported = client.info.metadata_support.support_picture_formats
            art_format: PictureFormat | None = None
            for fmt in (PictureFormat.JPEG, PictureFormat.PNG, PictureFormat.BMP):
                if fmt.value in supported:
                    art_format = fmt
                    self._client_art_formats[client.client_id] = art_format
                    break
            if art_format is not None:
                metadata_stream_info = StreamStartMetadata(art_format=art_format)
            else:
                metadata_stream_info = None
        else:
            metadata_stream_info = None

        # TODO: finish once spec is finalized
        visualizer_stream_info = (
            StreamStartVisualizer() if client.check_role(Roles.VISUALIZER) else None
        )

        stream_info = StreamStartPayload(
            player=player_stream_info,
            metadata=metadata_stream_info,
            visualizer=visualizer_stream_info,
        )
        if player_stream_info or metadata_stream_info or visualizer_stream_info:
            logger.debug(
                "Sending stream start message to client %s: %s",
                client.client_id,
                stream_info,
            )
            client.send_message(StreamStartMessage(stream_info))

    def _send_stream_end_msg(self, client: ResonateClient) -> None:
        """Send a stream end message to a client to stop playback."""
        logger.debug("ending stream for %s (%s)", client.name, client.client_id)
        # Lifetime of album artwork is bound to the stream
        self._client_art_formats.pop(client.client_id, None)
        client.send_message(StreamEndMessage())

    async def stop(self, stop_time_us: int | None = None) -> bool:  # noqa: PLR0915
        """
        Stop playback for the group and clean up resources.

        Compared to pause(), this also:
        - Cancels the audio streaming task
        - Sends stream end messages to all clients
        - Clears all buffers and format mappings
        - Cleans up all audio encoders

        Args:
            stop_time_us: Optional absolute timestamp (microseconds) when playback should
                stop. When provided and in the future, the stop request is scheduled and
                this method returns immediately.

        Returns:
            bool: True if an active or scheduled stream was stopped (or scheduled to stop),
            False if no stream was active.
        """
        active = self._stream_task is not None

        if stop_time_us is not None:
            now_us = int(self._server.loop.time() * 1_000_000)
            if stop_time_us > now_us:
                delay = (stop_time_us - now_us) / 1_000_000

                async def _delayed_stop() -> None:
                    try:
                        await self.stop()
                    except Exception:
                        logger.exception("Scheduled stop failed")

                self._server.loop.call_later(
                    delay, lambda: self._server.loop.create_task(_delayed_stop())
                )
                return active

        if not active:
            logger.debug("stop called but no active stream task")
            return False

        logger.debug(
            "Stopping playback for group with clients: %s",
            [c.client_id for c in self._clients],
        )

        if self._stream_task is not None:
            stream_task = self._stream_task
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Unhandled exception while stopping stream task")
            self._stream_task = None

        if self._streamer is not None:
            self._streamer.reset()
            self._streamer = None

        for generator in self._channel_generators.values():
            with suppress(Exception):
                await generator.aclose()
        self._channel_generators.clear()
        self._channel_formats.clear()
        self._player_channels.clear()
        self._media_stream = None
        self._stream_commands = None

        for client in self._clients:
            self._send_stream_end_msg(client)
            if client.check_role(Roles.PLAYER):
                self._player_formats.pop(client.client_id, None)

        self._audio_encoders.clear()
        self._audio_headers.clear()
        self._current_media_art = None
        self._stream_audio_format = None
        self._play_start_time_us = None

        if self._current_state != PlaybackStateType.STOPPED:
            self._signal_event(GroupStateChangedEvent(PlaybackStateType.STOPPED))
            self._current_state = PlaybackStateType.STOPPED

        timestamp = int(self._server.loop.time() * 1_000_000)
        cleared_metadata = Metadata.cleared_update(timestamp)
        for client in self._clients:
            playback_state = (
                PlaybackStateType.STOPPED
                if (client.check_role(Roles.CONTROLLER) or client.check_role(Roles.METADATA))
                else None
            )
            metadata_payload = cleared_metadata if client.check_role(Roles.METADATA) else None
            message = SessionUpdateMessage(
                SessionUpdatePayload(
                    group_id=self._group_id,
                    playback_state=playback_state,
                    metadata=metadata_payload,
                )
            )
            client.send_message(message)
        return True

    def set_metadata(self, metadata: Metadata | None, timestamp: int | None = None) -> None:
        """
        Set metadata for the group and send to all clients.

        Only sends updates for fields that have changed since the last call.

        Args:
            metadata: The new metadata to send to clients.
            timestamp: Optional timestamp in microseconds for the metadata update.
                If None, uses the current server time.
        """
        # TODO: integrate this more closely with play_media?
        # Check if metadata has actually changed
        if self._current_metadata == metadata:
            return
        last_metadata = self._current_metadata

        if timestamp is None:
            timestamp = int(self._server.loop.time() * 1_000_000)
        if metadata is None:
            # Clear all metadata fields when metadata is None
            metadata_update = Metadata.cleared_update(timestamp)
        else:
            # Only include fields that have changed since the last metadata update
            metadata_update = metadata.diff_update(last_metadata, timestamp)

        # Send the update to all clients in the group
        message = SessionUpdateMessage(
            SessionUpdatePayload(
                group_id=self._group_id,
            )
        )
        for client in self._clients:
            if client.check_role(Roles.METADATA):
                message.payload.metadata = metadata_update
            else:
                message.payload.metadata = None
            if client.check_role(Roles.CONTROLLER) or client.check_role(Roles.METADATA):
                message.payload.playback_state = (
                    PlaybackStateType.PLAYING
                    if self._current_state == PlaybackStateType.PLAYING
                    else PlaybackStateType.PAUSED
                )
            else:
                message.payload.playback_state = None
            logger.debug(
                "Sending session update to client %s",
                client.client_id,
            )
            client.send_message(message)

        # Update current metadata
        self._current_metadata = metadata

    def set_media_art(self, image: Image.Image) -> None:
        """Set the artwork image for the current media."""
        # Store the current media art for new clients that join later
        self._current_media_art = image

        for client in self._clients:
            self._send_media_art_to_client(client, image)

    def _letterbox_image(
        self, image: Image.Image, target_width: int, target_height: int
    ) -> Image.Image:
        """
        Resize image to fit within target dimensions while preserving aspect ratio.

        Uses letterboxing (black bars) to fill any remaining space.

        Args:
            image: Source image to resize
            target_width: Target width in pixels
            target_height: Target height in pixels

        Returns:
            Resized image with letterboxing if needed
        """
        # Calculate aspect ratios
        image_aspect = image.width / image.height
        target_aspect = target_width / target_height

        if image_aspect > target_aspect:
            # Image is wider than target - fit by width, letterbox on top/bottom
            new_width = target_width
            new_height = int(target_width / image_aspect)
        else:
            # Image is taller than target - fit by height, letterbox on left/right
            new_height = target_height
            new_width = int(target_height * image_aspect)

        # Resize the image to the calculated size
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a new image with the target size and black background
        letterboxed = Image.new("RGB", (target_width, target_height), (0, 0, 0))

        # Calculate position to center the resized image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        # Paste the resized image onto the letterboxed background
        letterboxed.paste(resized, (x_offset, y_offset))

        return letterboxed

    def _send_media_art_to_client(self, client: ResonateClient, image: Image.Image) -> None:
        """Send media art to a specific client with appropriate format and sizing."""
        if not client.check_role(Roles.METADATA) or not client.info.metadata_support:
            return

        art_format = self._client_art_formats.get(client.client_id)
        if art_format is None:
            # Do nothing if we are not in an active session or this client doesn't support artwork
            return
        metadata_support = client.info.metadata_support
        width = metadata_support.media_width
        height = metadata_support.media_height

        if width is None and height is None:
            # No size constraints, use original image size
            resized_image = image
        elif width is not None and height is None:
            # Only width constraint, scale height to maintain aspect ratio
            aspect_ratio = image.height / image.width
            height = int(width * aspect_ratio)
            resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
        elif width is None and height is not None:
            # Only height constraint, scale width to maintain aspect ratio
            aspect_ratio = image.width / image.height
            width = int(height * aspect_ratio)
            resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
        else:
            # Both width and height constraints - use letterboxing to preserve aspect ratio
            resized_image = self._letterbox_image(image, cast("int", width), cast("int", height))

        with BytesIO() as img_bytes:
            if art_format == PictureFormat.JPEG:
                resized_image.save(img_bytes, format="JPEG", quality=85)
            elif art_format == PictureFormat.PNG:
                resized_image.save(img_bytes, format="PNG", compress_level=6)
            elif art_format == PictureFormat.BMP:
                resized_image.save(img_bytes, format="BMP")
            else:
                raise NotImplementedError(f"Unsupported artwork format: {art_format}")
            img_bytes.seek(0)
            img_data = img_bytes.read()
            header = pack_binary_header_raw(
                BinaryMessageType.MEDIA_ART.value, int(self._server.loop.time() * 1_000_000)
            )
            client.send_message(header + img_data)

    @property
    def clients(self) -> list[ResonateClient]:
        """All clients that are part of this group."""
        return self._clients

    def players(self) -> list[PlayerClient]:
        """Return player helpers for all members that support the role."""
        return [client.player for client in self._clients if client.player is not None]

    def _handle_group_command(self, cmd: GroupCommandClientPayload) -> None:
        # TODO: verify that this command is actually supported for the current state
        event = GroupCommandEvent(
            command=cmd.command,
            volume=cmd.volume,
            mute=cmd.mute,
        )
        self._signal_event(event)

    def add_event_listener(
        self, callback: Callable[[GroupEvent], Coroutine[None, None, None]]
    ) -> Callable[[], None]:
        """
        Register a callback to listen for state changes of this group.

        State changes include:
        - The group started playing
        - The group stopped/finished playing

        Returns a function to remove the listener.
        """
        self._event_cbs.append(callback)
        return lambda: self._event_cbs.remove(callback)

    def _signal_event(self, event: GroupEvent) -> None:
        for cb in self._event_cbs:
            self._server.loop.create_task(cb(event))

    @property
    def state(self) -> PlaybackStateType:
        """Current playback state of the group."""
        return self._current_state

    async def remove_client(self, client: ResonateClient) -> None:
        """
        Remove a client from this group.

        If a stream is active, the client receives a stream end message.
        The client is automatically moved to its own new group since every
        client must belong to a group.
        If the client is not part of this group, this will have no effect.

        Args:
            client: The client to remove from this group.
        """
        if client not in self._clients:
            logger.debug("client %s not in group, skipping removal", client.client_id)
            return
        logger.debug("removing %s from group with members: %s", client.client_id, self._clients)
        if len(self._clients) == 1:
            # Delete this group if that was the last client
            await self.stop()
            self._clients = []
        else:
            self._clients.remove(client)
            if client.check_role(Roles.PLAYER):
                self._player_formats.pop(client.client_id, None)
                self._player_channels.pop(client.client_id, None)
            self._send_stream_end_msg(client)

            # Reconfigure streamer if actively streaming
            if (
                self._stream_task is not None
                and self._media_stream is not None
                and client.check_role(Roles.PLAYER)
            ):
                removed_ids = {client.client_id}
                try:
                    await self._reconfigure_streamer(
                        media_stream=self._media_stream,
                        new_client_ids=set(),
                        removed_client_ids=removed_ids,
                    )
                except RuntimeError:
                    logger.info(
                        "Stopping playback to rebuild streamer after removing %s",
                        client.client_id,
                    )
                    await self.stop()
                except Exception:
                    logger.exception(
                        "Failed to reconfigure streamer after removing %s",
                        client.client_id,
                    )
                    await self.stop()
        if not self._clients:
            # Emit event for group deletion, no clients left
            self._signal_event(GroupDeletedEvent())
        else:
            # Emit event for client removal
            self._signal_event(GroupMemberRemovedEvent(client.client_id))
        # Each client needs to be in a group, add it to a new one
        client._set_group(ResonateGroup(self._server, client))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    async def add_client(self, client: ResonateClient) -> None:  # noqa: PLR0915
        """
        Add a client to this group.

        The client is first removed from any existing group. If a session is
        currently active, players are immediately joined to the session with
        an appropriate audio format.

        Args:
            client: The client to add to this group.
        """
        logger.debug("adding %s to group with members: %s", client.client_id, self._clients)
        await client.group.stop()
        if client in self._clients:
            return
        # Remove it from any existing group first
        await client.ungroup()

        # Add client to this group's client list
        self._clients.append(client)

        # Emit event for client addition
        self._signal_event(GroupMemberAddedEvent(client.client_id))

        # Then set the group (which will emit ClientGroupChangedEvent)
        client._set_group(self)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        if self._stream_task is not None and self._stream_audio_format is not None:
            logger.debug("Joining client %s to current stream", client.client_id)
            if client.check_role(Roles.PLAYER):
                player_format = client.require_player.determine_optimal_format(
                    self._stream_audio_format
                )
                self._player_formats[client.client_id] = player_format
                # Assign to default channel (additional channels can be added later if needed)
                if self._media_stream:
                    self._player_channels[client.client_id] = (
                        self._media_stream.default_channel_name
                    )

                if (
                    self._streamer is None
                    or self._stream_commands is None
                    or self._stream_task is None
                ):
                    logger.info(
                        "Stopping playback to add player %s (streamer inactive)",
                        client.client_id,
                    )
                    await self.stop()
                else:
                    new_ids = {client.client_id}
                    if self._media_stream is None:
                        logger.info("No media stream available, stopping playback")
                        await self.stop()
                    else:
                        try:
                            start_payloads = await self._reconfigure_streamer(
                                media_stream=self._media_stream,
                                new_client_ids=new_ids,
                                removed_client_ids=set(),
                            )
                        except RuntimeError:
                            logger.info(
                                "Stopping playback to restart streamer for new client %s",
                                client.client_id,
                            )
                            await self.stop()
                        except Exception:
                            logger.exception(
                                "Failed to reconfigure streamer for new client %s; stopping stream",
                                client.client_id,
                            )
                            await self.stop()
                        else:
                            player_lookup = {
                                player.client.client_id: player for player in self.players()
                            }
                            for added_id in new_ids:
                                payload = start_payloads.get(added_id)
                                player_obj = player_lookup.get(added_id)
                                if payload is not None and player_obj is not None:
                                    self._send_stream_start_msg(
                                        player_obj.client,
                                        None,
                                        player_info=payload,
                                    )
            elif client.check_role(Roles.METADATA) or client.check_role(Roles.VISUALIZER):
                self._send_stream_start_msg(client, None, include_player=False)

        # Send current metadata to the new player if available
        if self._current_metadata is not None:
            if client.check_role(Roles.METADATA):
                metadata_update = self._current_metadata.snapshot_update(
                    int(self._server.loop.time() * 1_000_000)
                )
            else:
                metadata_update = None
            if client.check_role(Roles.CONTROLLER) or client.check_role(Roles.METADATA):
                playback_state = (
                    PlaybackStateType.PLAYING
                    if self._current_state == PlaybackStateType.PLAYING
                    else PlaybackStateType.PAUSED
                )
            else:
                playback_state = None
            message = SessionUpdateMessage(
                SessionUpdatePayload(
                    group_id=self._group_id,
                    playback_state=playback_state,
                    metadata=metadata_update,
                )
            )

            logger.debug("Sending session update to new client %s", client.client_id)
            client.send_message(message)

        # Send current media art to the new client if available
        if self._current_media_art is not None:
            self._send_media_art_to_client(client, self._current_media_art)

    def handle_stream_format_request(
        self,
        player: ResonateClient,
        request: StreamRequestFormatPayload,
    ) -> None:
        """Handle stream/request-format from a player and send stream/update."""
        raise NotImplementedError("Dynamic format changes are not yet implemented")
