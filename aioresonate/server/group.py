"""Manages and synchronizes playback for a group of one or more clients."""

from __future__ import annotations

import asyncio
import logging
import uuid
from asyncio import Task
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, cast
from uuid import UUID

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

from .metadata import Metadata
from .stream import AudioCodec, AudioFormat, ClientStreamConfig, MediaStream, Streamer

# The cyclic import is not an issue during runtime, so hide it
# pyright: reportImportCycles=none
if TYPE_CHECKING:
    import av

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
class _StreamerReconfigureCommand:
    """Signal to reconfigure the running streamer with new player topology."""

    all_player_configs: list[ClientStreamConfig]
    """List of ClientStreamConfig for all players (existing and new)."""


class ResonateGroup:
    """
    A group of one or more clients for synchronized playback.

    Handles synchronized audio streaming across multiple clients with automatic
    format conversion and buffer management. Every client is always assigned to
    a group to simplify grouping requests.
    """

    _clients: list[ResonateClient]
    """List of all clients in this group."""
    _client_art_formats: dict[str, PictureFormat]
    """Mapping of client IDs (with the metadata role) to their selected artwork formats."""
    _server: ResonateServer
    """Reference to the ResonateServer instance."""
    _stream_task: Task[int] | None = None
    """Task handling the audio streaming loop, None when not streaming."""
    _current_metadata: Metadata | None = None
    """Current metadata for the group, None if no metadata set."""
    _current_media_art: Image.Image | None = None
    """Current media art image for the group, None if no image set."""
    _audio_encoders: dict[AudioFormat, av.AudioCodecContext]
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
    _stream_commands: asyncio.Queue[_StreamerReconfigureCommand] | None
    """Command queue for the active streamer task, None when not streaming."""
    _play_start_time_us: int | None
    """Absolute timestamp in microseconds when playback started, None when not streaming."""
    _scheduled_stop_handle: asyncio.TimerHandle | None
    """Timer handle for scheduled stop, None when no stop is scheduled."""

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
        self._client_art_formats = {}
        self._server = server
        self._stream_task: Task[int] | None = None
        self._current_metadata = None
        self._current_media_art = None
        self._audio_encoders = {}
        self._event_cbs = []
        self._group_id = str(uuid.uuid4())
        self._streamer: Streamer | None = None
        self._media_stream: MediaStream | None = None
        self._stream_commands: asyncio.Queue[_StreamerReconfigureCommand] | None = None
        self._play_start_time_us: int | None = None
        self._scheduled_stop_handle: asyncio.TimerHandle | None = None
        logger.debug(
            "ResonateGroup initialized with %d client(s): %s",
            len(self._clients),
            [type(c).__name__ for c in self._clients],
        )

    async def play_media(
        self,
        media_stream: MediaStream,
        *,
        play_start_time_us: int | None = None,
    ) -> int:
        """Start synchronized playback for the current group using a MediaStream."""
        logger.debug(
            "Starting play_media with play_start_time_us=%s",
            play_start_time_us,
        )

        # Cancel any previously scheduled stop to prevent race conditions
        if self._scheduled_stop_handle is not None:
            logger.debug("Canceling previously scheduled stop")
            self._scheduled_stop_handle.cancel()
            self._scheduled_stop_handle = None

        self._media_stream = media_stream
        self._streamer = None

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

        streamer = Streamer(
            loop=self._server.loop,
            play_start_time_us=start_time_us,
        )
        self._streamer = streamer
        self._media_stream = media_stream

        # Build configs for all players (all are new for initial setup)
        all_player_configs: list[ClientStreamConfig] = []
        for player in group_players:
            assert player.support
            target_format = player.determine_optimal_format(media_stream.main_channel[1])
            all_player_configs.append(
                ClientStreamConfig(
                    client_id=player.client.client_id,
                    target_format=target_format,
                    buffer_capacity_bytes=player.support.buffer_capacity,
                    send=player.client.send_message,
                )
            )

        start_payloads, channel_sources = await streamer.configure(all_player_configs, media_stream)
        self._stream_commands = asyncio.Queue()
        self._stream_task = self._server.loop.create_task(
            self._run_streamer(streamer, media_stream, channel_sources)
        )

        # Notify clients about the upcoming stream configuration
        for player in group_players:
            player_payload = start_payloads.get(player.client.client_id)
            assert player_payload is not None
            self._send_stream_start_msg(
                player.client,
                player_payload,
            )

        for client in self._clients:
            if client.check_role(Roles.PLAYER):
                continue
            if client.check_role(Roles.METADATA) or client.check_role(Roles.VISUALIZER):
                self._send_stream_start_msg(client, None)

        self._current_state = PlaybackStateType.PLAYING
        self._signal_event(GroupStateChangedEvent(PlaybackStateType.PLAYING))

        end_time_us = start_time_us
        if self._stream_task is not None:
            end_time_us = await self._stream_task
            self._stream_task = None

        self._streamer = None
        self._media_stream = None
        self._stream_commands = None

        return end_time_us

    def _send_session_update_to_clients(self) -> None:
        """Send session/update messages to all clients with current playback state and metadata."""
        for client in self._clients:
            if client.check_role(Roles.METADATA):
                metadata_update = (
                    self._current_metadata.snapshot_update(
                        int(self._server.loop.time() * 1_000_000)
                    )
                    if self._current_metadata is not None
                    else None
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
            client.send_message(message)

    async def _handle_reconfiguration_command(
        self,
        command: _StreamerReconfigureCommand,
        streamer: Streamer,
        media_stream: MediaStream,
        active_channels: dict[UUID, AsyncGenerator[bytes, None]],
        just_started_channels: set[UUID],
    ) -> None:
        """Handle a streamer reconfiguration command by updating topology and notifying clients."""
        # Reconfigure with current player topology
        start_payloads, new_sources = await streamer.configure(
            command.all_player_configs, media_stream
        )

        # Add new channel sources to active channels
        for channel_id, source in new_sources.items():
            if channel_id not in active_channels:
                active_channels[channel_id] = source
                just_started_channels.add(channel_id)

        # Drop channel sources that were removed by configure()
        removed_channel_ids = set(active_channels) - streamer.get_channel_ids()
        for removed_id in removed_channel_ids:
            source = active_channels.pop(removed_id)
            just_started_channels.discard(removed_id)
            with suppress(Exception):
                await source.aclose()

        # Send stream/start messages to affected players
        player_lookup = {player.client.client_id: player for player in self.players()}
        for client_id, player_payload in start_payloads.items():
            player_obj = player_lookup.get(client_id)
            if player_obj is not None:
                self._send_stream_start_msg(
                    player_obj.client,
                    player_stream_info=player_payload,
                )
        # Send session/update to all clients
        # TODO: only send to clients that were affected by the change!
        self._send_session_update_to_clients()
        logger.debug("streamer reconfigured")

    async def _prefill_channel_buffers(
        self,
        streamer: Streamer,
        active_channels: dict[UUID, AsyncGenerator[bytes, None]],
        just_started_channels: set[UUID],
    ) -> None:
        """Pre-fill buffers for channels that just started before beginning playback."""
        channels_to_check = list(just_started_channels)
        for channel_id in channels_to_check:
            if channel_id not in active_channels:
                just_started_channels.discard(channel_id)
                continue
            # Pre-fill this channel's buffer before starting playback
            while streamer.channel_needs_data(channel_id):
                source = active_channels[channel_id]
                try:
                    chunk = await asyncio.wait_for(anext(source), timeout=30.0)

                    # Skip preparing chunks that are already in the past to avoid wasted work
                    channel_state = streamer._channels.get(channel_id)  # noqa: SLF001
                    if channel_state:
                        # Calculate what this chunk's start timestamp would be
                        sample_count = len(chunk) // channel_state.source_format_params.frame_stride
                        start_samples = channel_state.samples_produced
                        start_us = streamer._play_start_time_us + int(  # noqa: SLF001
                            start_samples
                            * 1_000_000
                            / channel_state.source_format_params.audio_format.sample_rate
                        )

                        # Check if chunk is already outdated
                        now_us = int(self._server.loop.time() * 1_000_000)

                        if start_us < now_us:
                            # Skip outdated chunk but update sample counter to maintain sync
                            channel_state.samples_produced += sample_count
                            logger.debug(
                                "Skipping outdated chunk for channel %s (start: %d us, now: %d us)",
                                channel_id,
                                start_us,
                                now_us,
                            )
                            continue

                    streamer.prepare(channel_id, chunk, during_initial_buffering=True)
                    continue  # Continue filling buffer
                except StopAsyncIteration:
                    pass  # Channel exhausted (normal completion)
                except TimeoutError:
                    logger.error("Channel %s timed out during prefill, removing", channel_id)
                except Exception:
                    logger.exception("Channel %s failed during prefill, removing", channel_id)
                # Channel done (exhausted, timed out, or failed) - clean up and exit
                del active_channels[channel_id]
                with suppress(Exception):
                    await source.aclose()
                break
            # Channel is now pre-filled, remove from just_started set
            just_started_channels.discard(channel_id)

    async def _read_pending_chunks(
        self,
        streamer: Streamer,
        active_channels: dict[UUID, AsyncGenerator[bytes, None]],
    ) -> bool:
        """Read chunks from channels that need data until buffers are full.

        Returns:
            True if there are still active channels, False if all channels are exhausted.
        """
        any_channel_needs_data = True
        while any_channel_needs_data and active_channels:
            any_channel_needs_data = False
            for channel_id in list(active_channels.keys()):
                if not streamer.channel_needs_data(channel_id):
                    continue
                any_channel_needs_data = True
                source = active_channels[channel_id]
                try:
                    chunk = await asyncio.wait_for(anext(source), timeout=30.0)
                    streamer.prepare(channel_id, chunk)
                    continue  # Done, continue with next channel
                except StopAsyncIteration:
                    pass  # Channel exhausted (normal completion)
                except TimeoutError:
                    logger.error("Channel %s timed out during read, removing", channel_id)
                except Exception:
                    logger.exception("Channel %s failed during read, removing", channel_id)
                # Channel done (exhausted, timed out, or failed) - clean up
                del active_channels[channel_id]
                with suppress(Exception):
                    await source.aclose()

        return bool(active_channels)

    async def _run_streamer(
        self,
        streamer: Streamer,
        media_stream: MediaStream,
        active_channels: dict[UUID, AsyncGenerator[bytes, None]],
    ) -> int:
        """Consume media channels, distribute via streamer, and return end timestamp."""
        last_end_us = self._play_start_time_us or int(self._server.loop.time() * 1_000_000)
        just_started_channels: set[UUID] = set(active_channels.keys())

        try:
            while True:
                # Check for commands before processing chunks
                if self._stream_commands is not None and not self._stream_commands.empty():
                    command = self._stream_commands.get_nowait()
                    await self._handle_reconfiguration_command(
                        command, streamer, media_stream, active_channels, just_started_channels
                    )
                    continue

                # Pre-fill buffers for channels that just started
                if just_started_channels:
                    await self._prefill_channel_buffers(
                        streamer, active_channels, just_started_channels
                    )

                # Read chunks from channels that need data until buffers are full
                if not await self._read_pending_chunks(streamer, active_channels):
                    break  # All channels exhausted

                # Send prepared chunks after buffers are full
                await streamer.send()

            # Normal completion - flush and send remaining chunks
            streamer.flush()
            await streamer.send()
            if streamer.last_chunk_end_time_us is not None:
                last_end_us = streamer.last_chunk_end_time_us
        except asyncio.CancelledError:
            # Cancellation - flush and send remaining chunks before cleanup
            streamer.flush()
            await streamer.send()
            raise
        else:
            return last_end_us
        finally:
            # Always close all remaining active channels to prevent resource leaks
            for source in list(active_channels.values()):
                with suppress(Exception):
                    await source.aclose()
            active_channels.clear()

    def _reconfigure_streamer(self) -> None:
        """Reconfigure the running streamer with current client topology."""
        if (
            self._streamer is None
            or self._stream_commands is None
            or self._stream_task is None
            or self._media_stream is None
        ):
            raise RuntimeError("Streamer is not running")

        # Build configs for all current players
        all_player_configs: list[ClientStreamConfig] = []
        for player in self.players():
            assert player.support
            target_format = player.determine_optimal_format(self._media_stream.main_channel[1])
            all_player_configs.append(
                ClientStreamConfig(
                    client_id=player.client.client_id,
                    target_format=target_format,
                    buffer_capacity_bytes=player.support.buffer_capacity,
                    send=player.client.send_message,
                )
            )

        # Signal the streamer to reconfigure on next iteration with the new topology
        self._stream_commands.put_nowait(
            _StreamerReconfigureCommand(
                all_player_configs=all_player_configs,
            )
        )

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

    def _send_stream_start_msg(
        self,
        client: ResonateClient,
        player_stream_info: StreamStartPlayer | None = None,
    ) -> None:
        """Send a stream start message to a client with the specified audio format for players."""
        assert client.check_role(Roles.PLAYER) == (player_stream_info is not None)
        if client.check_role(Roles.METADATA) and client.info.metadata_support:
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

    def _schedule_delayed_stop(self, stop_time_us: int, active: bool, needs_cleanup: bool) -> bool:  # noqa: FBT001
        """Schedule a delayed stop at the specified timestamp.

        Args:
            stop_time_us: Absolute timestamp when stop should occur
            active: Whether stream task is currently active
            needs_cleanup: Whether cleanup is needed

        Returns:
            True if stop was scheduled, False if nothing to do
        """
        now_us = int(self._server.loop.time() * 1_000_000)
        if stop_time_us <= now_us:
            return False

        # Only schedule if there's something to stop or cleanup
        if not active and not needs_cleanup:
            return False

        delay = (stop_time_us - now_us) / 1_000_000

        async def _delayed_stop() -> None:
            # Store handle locally to detect if it's been replaced
            handle = self._scheduled_stop_handle
            try:
                await self.stop()  # This will clear _scheduled_stop_handle
            except Exception:
                logger.exception("Scheduled stop failed")
            finally:
                # Only clear if this handle is still current (e.g., stop() was interrupted
                # or a new stop was scheduled during the stop() call)
                if self._scheduled_stop_handle is handle:
                    self._scheduled_stop_handle = None

        def _schedule_stop() -> None:
            task = self._server.loop.create_task(_delayed_stop())
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

        self._scheduled_stop_handle = self._server.loop.call_later(delay, _schedule_stop)
        return True

    async def _cancel_stream_task(self) -> None:
        """Cancel the active stream task and wait for it to complete."""
        if self._stream_task is None:
            return

        stream_task = self._stream_task
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Unhandled exception while stopping stream task")
        self._stream_task = None

    async def _cleanup_streaming_resources(self) -> None:
        """Clean up all streaming-related resources."""
        if self._streamer is not None:
            self._streamer.reset()
            self._streamer = None

        if self._media_stream is not None:
            with suppress(Exception):
                await self._media_stream.main_channel[0].aclose()
        self._media_stream = None
        self._stream_commands = None

        for client in self._clients:
            self._send_stream_end_msg(client)

        self._audio_encoders.clear()
        self._current_media_art = None
        self._play_start_time_us = None

    def _send_stopped_state_to_clients(self) -> None:
        """Send stopped state and cleared metadata to all clients."""
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

    async def stop(self, stop_time_us: int | None = None) -> bool:
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
            bool: True if an active stream was stopped (or scheduled to stop),
            False if no stream was active and no cleanup was required.
        """
        # Cancel any existing scheduled stop first to prevent race conditions
        if self._scheduled_stop_handle is not None:
            logger.debug("Canceling previously scheduled stop in stop()")
            self._scheduled_stop_handle.cancel()
            self._scheduled_stop_handle = None

        active = self._stream_task is not None
        needs_cleanup = self._current_state != PlaybackStateType.STOPPED

        # Handle delayed stop if requested
        if stop_time_us is not None and self._schedule_delayed_stop(
            stop_time_us, active, needs_cleanup
        ):
            return active or needs_cleanup

        if not active and not needs_cleanup:
            return False

        logger.debug(
            "Stopping playback for group with clients: %s",
            [c.client_id for c in self._clients],
        )

        try:
            await self._cancel_stream_task()
        finally:
            await self._cleanup_streaming_resources()

        if self._current_state != PlaybackStateType.STOPPED:
            self._signal_event(GroupStateChangedEvent(PlaybackStateType.STOPPED))
            self._current_state = PlaybackStateType.STOPPED

        self._send_stopped_state_to_clients()
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

    async def set_media_art(self, image: Image.Image) -> None:
        """Set the artwork image for the current media."""
        # Store the current media art for new clients that join later
        self._current_media_art = image

        await asyncio.gather(
            *[self._send_media_art_to_client(client, image) for client in self._clients],
            return_exceptions=True,
        )

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

    async def _send_media_art_to_client(self, client: ResonateClient, image: Image.Image) -> None:
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

        # Process and encode image in thread to avoid blocking event loop
        img_data = await asyncio.to_thread(
            self._process_and_encode_image,
            image,
            width,
            height,
            art_format,
        )

        header = pack_binary_header_raw(
            BinaryMessageType.MEDIA_ART.value, int(self._server.loop.time() * 1_000_000)
        )
        client.send_message(header + img_data)

    def _process_and_encode_image(
        self,
        image: Image.Image,
        width: int | None,
        height: int | None,
        art_format: PictureFormat,
    ) -> bytes:
        """
        Process and encode image for client.

        NOTE: This method is not async friendly.
        """
        # Process image with appropriate resizing/letterboxing
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
            return img_bytes.read()

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

        def _remove() -> None:
            with suppress(ValueError):
                self._event_cbs.remove(callback)

        return _remove

    def _signal_event(self, event: GroupEvent) -> None:
        for cb in self._event_cbs:
            task = self._server.loop.create_task(cb(event))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

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
            return
        logger.debug("removing %s from group with members: %s", client.client_id, self._clients)
        if len(self._clients) == 1:
            # Delete this group if that was the last client
            await self.stop()
            self._clients = []
        else:
            self._clients.remove(client)
            self._send_stream_end_msg(client)

            # Reconfigure streamer if actively streaming
            if (
                self._stream_task is not None
                and self._media_stream is not None
                and client.check_role(Roles.PLAYER)
            ):
                self._reconfigure_streamer()
        if not self._clients:
            # Emit event for group deletion, no clients left
            self._signal_event(GroupDeletedEvent())
        else:
            # Emit event for client removal
            self._signal_event(GroupMemberRemovedEvent(client.client_id))
        # Each client needs to be in a group, add it to a new one
        client._set_group(ResonateGroup(self._server, client))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    async def add_client(self, client: ResonateClient) -> None:
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
        if self._stream_task is not None and self._media_stream:
            logger.debug("Joining client %s to current stream", client.client_id)
            if client.check_role(Roles.PLAYER):
                self._reconfigure_streamer()
            elif client.check_role(Roles.METADATA) or client.check_role(Roles.VISUALIZER):
                self._send_stream_start_msg(client, None)

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
            await self._send_media_art_to_client(client, self._current_media_art)

    def handle_stream_format_request(
        self,
        player: ResonateClient,
        request: StreamRequestFormatPayload,
    ) -> None:
        """Handle stream/request-format from a player and send stream/update."""
        raise NotImplementedError("Dynamic format changes are not yet implemented")
