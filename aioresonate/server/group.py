"""Manages and synchronizes playback for a group of one or more clients."""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from abc import ABC, abstractmethod
from asyncio import Queue, QueueFull, Task
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
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
    StreamUpdateMessage,
    StreamUpdatePayload,
)
from aioresonate.models.metadata import (
    StreamStartMetadata,
)
from aioresonate.models.player import (
    StreamRequestFormatPayload,
    StreamStartPlayer,
    StreamUpdatePlayer,
)
from aioresonate.models.types import (
    MediaCommand,
    PictureFormat,
    PlaybackStateType,
    Roles,
)
from aioresonate.models.visualizer import StreamStartVisualizer
from aioresonate.server.streaming import build_flac_stream_header

from .metadata import Metadata

# The cyclic import is not an issue during runtime, so hide it
# pyright: reportImportCycles=none
if TYPE_CHECKING:
    from .client import ResonateClient
    from .player import PlayerClient
    from .server import ResonateServer

INITIAL_PLAYBACK_DELAY_US = 1_000_000

logger = logging.getLogger(__name__)


class AudioCodec(Enum):
    """Supported audio codecs."""

    PCM = "pcm"
    FLAC = "flac"
    OPUS = "opus"


class GroupEvent:
    """Base event type used by ClientGroup.add_event_listener()."""


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


@dataclass(frozen=True)
class AudioFormat:
    """Audio format of a stream."""

    sample_rate: int
    """Sample rate in Hz (e.g., 44100, 48000)."""
    bit_depth: int
    """Bit depth in bits per sample (16 or 24)."""
    channels: int
    """Number of audio channels (1 for mono, 2 for stereo)."""
    codec: AudioCodec = AudioCodec.PCM
    """Audio codec of the stream."""


class DirectStreamSession(ABC):
    """
    Interface resonate servers can implement to supply per-client PCM streams.

    Implementations are responsible for ensuring returned generators stay
    aligned with the shared playback clock. Generators may be cancelled at any
    time when clients leave or playback stops.
    """

    supports_seek: bool = False
    """
    True when the session can start streams at arbitrary offsets.

    If False offset_time_us passed to get_stream() will always be 0.
    """

    def handles_client(self, _client: ResonateClient) -> bool:
        """Return True when this session replaces the shared stream for client."""
        return True

    @abstractmethod
    async def get_stream(
        self,
        client: ResonateClient,
        audio_format: AudioFormat,
        start_time_us: int,
        offset_time_us: int,
    ) -> AsyncGenerator[bytes, None]:
        """
        Return PCM audio stream for a client.

        Will only be called for clients where handles_client() returned True.

        The returned generator may be cancelled at any time if the client leaves
        or playback stops.

        The generator must yield raw PCM audio chunks in the specified format.
        """


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
    _stream_task: Task[None] | None = None
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
    _scheduled_format_changes: dict[str, tuple[StreamUpdateMessage, AudioFormat]]
    """Mapping of client IDs to upcoming stream updates requested by the player."""

    def __init__(self, server: ResonateServer, *args: ResonateClient) -> None:
        """
        DO NOT CALL THIS CONSTRUCTOR. INTERNAL USE ONLY.

        Groups are managed automatically by the server.

        Initialize a new ClientGroup.

        Args:
            server: The ResonateServer instance this group belongs to.
            *args: Clients to add to this group.
        """
        self._server = server
        self._clients = list(args)
        self._player_formats = {}
        self._current_metadata = None
        self._current_media_art = None
        self._audio_encoders = {}
        self._audio_headers = {}
        self._event_cbs = []
        self._group_id = str(uuid.uuid4())
        self._scheduled_format_changes = {}
        self._client_art_formats = {}
        self._client_stream_tasks: dict[str, asyncio.Task[int]] = {}
        self._fanout_queues: dict[str, Queue[bytes | None]] = {}
        self._fanout_subscribers: dict[AudioFormat, list[Queue[bytes | None]]] = {}
        self._direct_session: DirectStreamSession | None = None
        self._play_start_time_us: int | None = None
        logger.debug(
            "ClientGroup initialized with %d client(s): %s",
            len(self._clients),
            [type(c).__name__ for c in self._clients],
        )

    def _group_players(self) -> list[PlayerClient]:
        """Return player helpers for all members that support the role."""
        players: list[PlayerClient] = []
        for client in self._clients:
            player = client.player
            if player is not None:
                players.append(player)
        return players

    async def play_media(  # noqa: PLR0915
        self,
        audio_stream: AsyncGenerator[bytes, None],
        audio_stream_format: AudioFormat,
        *,
        play_start_time_us: int | None = None,
        stream_start_time_us: int = 0,
        direct_session: DirectStreamSession | None = None,
    ) -> int:
        """
        Start synchronized playback for the current group.

        TODO: caller is responsible to play new media or stop stream after this is done

        Args:
            audio_stream: Async generator yielding PCM audio chunks as bytes.
            audio_stream_format: Format specification for the input audio data.
            play_start_time_us: Absolute timestamp when playback should begin. If None,
                the group schedules playback as soon as possible.
                Use this to schedule the next track in advance for gapless playback.
            stream_start_time_us: Offset within the source stream (in microseconds) that all
                players should start from. Mid-stream joins are handled later when per-client
                streaming is delegated.
            direct_session: Optional DirectStreamSession instance that can provide per-client
                PCM streams for any client. (For example, to handle DSP independently per client.)

        Returns:
            Absolute timestamp (microseconds) when the stream is expected to finish once
            the audio_stream generator is exhausted.
        """
        logger.debug(
            "Starting play_media with audio_stream_format=%s, "
            "play_start_time_us=%s, stream_start_time_us=%s",
            audio_stream_format,
            play_start_time_us,
            stream_start_time_us,
        )

        self._stream_audio_format = audio_stream_format
        start_time_us = (
            play_start_time_us
            if play_start_time_us is not None
            else int(self._server.loop.time() * 1_000_000) + INITIAL_PLAYBACK_DELAY_US
        )
        group_players = self._group_players()

        format_to_player: dict[AudioFormat, list[PlayerClient]] = {}

        self._player_formats.clear()

        for player in group_players:
            client = player.client
            player_format = player.determine_optimal_format(audio_stream_format)
            self._player_formats[client.client_id] = player_format
            format_to_player[player_format].append(player)
            logger.debug("Selected format %s for player %s", player_format, client.client_id)

        for client in self._clients:
            if client.check_role(Roles.PLAYER):
                continue
            if client.check_role(Roles.METADATA) or client.check_role(Roles.VISUALIZER):
                self._send_stream_start_msg(client, None)

        self._fanout_queues.clear()
        self._fanout_subscribers.clear()
        self._client_stream_tasks.clear()
        self._direct_session = direct_session

        # Partition players into those handled by the direct session and those
        # that should use the shared fan-out pipeline.
        shared_clients_map: dict[AudioFormat, list[PlayerClient]] = {}
        direct_players: list[PlayerClient] = []

        for player in group_players:
            client = player.client
            player_format = self._player_formats[client.client_id]
            if direct_session is not None and direct_session.handles_client(client):
                direct_players.append(player)
            else:
                shared_clients_map[player_format].append(player)

        # Launch direct-session streams first so that the shared fan-out only
        # has to service the remaining players.
        for player in direct_players:
            player_format = self._player_formats[player.client.client_id]
            await self._start_direct_stream_for_client(
                player.client,
                player_format,
                start_time_us=start_time_us,
                stream_start_time_us=stream_start_time_us,
            )

        sentinel: None = None
        if shared_clients_map:
            # Build and start the shared fan-out pipeline for all remaining
            # players that still need group-managed resampling.
            format_subscribers: dict[AudioFormat, list[Queue[bytes | None]]] = {}

            for audio_format, players_for_format in format_to_player.items():
                queues: list[Queue[bytes | None]] = []
                for player in players_for_format:
                    queue: Queue[bytes | None] = Queue()
                    self._fanout_queues[player.client.client_id] = queue
                    queues.append(queue)
                format_subscribers[audio_format] = queues

            self._fanout_subscribers = format_subscribers
            self._play_start_time_us = start_time_us
            self._stream_task = self._start_fanout_task(
                sentinel=sentinel,
                format_subscribers=format_subscribers,
                audio_stream=audio_stream,
                audio_stream_format=audio_stream_format,
            )
            for players_for_format in shared_clients_map.values():
                for player in players_for_format:
                    self._send_stream_start_msg(
                        player.client, self._player_formats[player.client.client_id]
                    )
                    queue = self._fanout_queues[player.client.client_id]

                    async def stream_gen(
                        queue_ref: Queue[bytes | None] = queue,
                        sentinel_ref: None = sentinel,
                    ) -> AsyncGenerator[bytes, None]:
                        while True:
                            chunk = await queue_ref.get()
                            if chunk is sentinel_ref:
                                break
                            assert isinstance(chunk, bytes)
                            yield chunk

                    self._client_stream_tasks[player.client.client_id] = (
                        self._server.loop.create_task(
                            player._play_media_direct(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                                stream_gen(),
                                self._player_formats[player.client.client_id],
                                play_start_time_us=start_time_us,
                                stream_start_time_us=stream_start_time_us,
                            )
                        )
                    )
        else:
            self._play_start_time_us = start_time_us
            self._stream_task = None

        self._current_state = PlaybackStateType.PLAYING
        self._signal_event(GroupStateChangedEvent(PlaybackStateType.PLAYING))

        end_times = await asyncio.gather(*self._client_stream_tasks.values())
        if self._stream_task is not None:
            await self._stream_task

        return max(end_times) if end_times else start_time_us

    async def _start_direct_stream_for_client(
        self,
        client: ResonateClient,
        player_format: AudioFormat,
        *,
        start_time_us: int,
        stream_start_time_us: int,
    ) -> None:
        """Start a direct-session stream for ``client`` when available."""
        session = self._direct_session
        assert session is not None

        supports_seek = session.supports_seek
        offset_us = stream_start_time_us if supports_seek else 0
        if stream_start_time_us and not supports_seek:
            logger.debug(
                "Direct session for %s does not support seeking; starting from beginning",
                client.client_id,
            )

        stream = await session.get_stream(client, player_format, start_time_us, offset_us)

        task = self._server.loop.create_task(
            client.player_throw._play_media_direct(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                stream,
                player_format,
                play_start_time_us=start_time_us,
                stream_start_time_us=offset_us,
            )
        )
        self._client_stream_tasks[client.client_id] = task

        if client.check_role(Roles.METADATA) or client.check_role(Roles.VISUALIZER):
            self._send_stream_start_msg(client, None, include_player=False)

    async def _handle_late_join_direct(
        self, client: ResonateClient, player_format: AudioFormat
    ) -> None:
        """Attempt to launch a direct stream for a client that joined mid-session."""
        session = self._direct_session
        if session is None or not session.handles_client(client):
            return

        if self._play_start_time_us is None:
            logger.debug(
                "Cannot start direct stream for %s without known group start time",
                client.client_id,
            )
            return

        offset_us = max(
            0,
            int(self._server.loop.time() * 1_000_000) - self._play_start_time_us,
        )
        await self._start_direct_stream_for_client(
            client,
            player_format,
            start_time_us=self._play_start_time_us,
            stream_start_time_us=offset_us,
        )

    def _start_fanout_task(
        self,
        *,
        sentinel: None,
        format_subscribers: dict[AudioFormat, list[Queue[bytes | None]]],
        audio_stream: AsyncGenerator[bytes, None],
        audio_stream_format: AudioFormat,
    ) -> asyncio.Task[None]:
        """Start background fan-out task that resamples and distributes PCM chunks."""
        # The fan-out task consumes the shared PCM source, resamples it for
        # every unique output format, and pushes chunks into per-client queues.

        async def fanout_audio() -> None:
            format_result = self._validate_audio_format(audio_stream_format)
            if format_result is None:
                raise ValueError("Unsupported source audio format")
            input_bytes_per_sample, input_audio_format, input_audio_layout = format_result
            bytes_per_input_sample = audio_stream_format.channels * input_bytes_per_sample
            samples_per_chunk = self._calculate_optimal_chunk_samples(audio_stream_format)
            resamplers: dict[AudioFormat, av.AudioResampler] = {}
            input_buffer = bytearray()
            minimum_chunk_bytes = samples_per_chunk * bytes_per_input_sample

            async def dispatch_frame(frame: av.AudioFrame) -> None:
                for target_format, subscriber_queues in format_subscribers.items():
                    resampler = resamplers.get(target_format)
                    if resampler is None:
                        resampler = av.AudioResampler(
                            format="s16" if target_format.bit_depth == 16 else "s24",
                            layout="stereo" if target_format.channels == 2 else "mono",
                            rate=target_format.sample_rate,
                        )
                        resamplers[target_format] = resampler

                    out_frames = resampler.resample(frame)
                    for out_frame in out_frames:
                        bytes_per_sample = 2 if target_format.bit_depth == 16 else 3
                        expected_bytes = (
                            bytes_per_sample * target_format.channels * out_frame.samples
                        )
                        pcm_bytes = bytes(out_frame.planes[0])[:expected_bytes]
                        await asyncio.gather(
                            *(queue.put(pcm_bytes) for queue in subscriber_queues),
                            return_exceptions=True,
                        )

            async def process_available(*, force_flush: bool) -> None:
                while len(input_buffer) >= bytes_per_input_sample:
                    if not force_flush and len(input_buffer) < minimum_chunk_bytes:
                        break

                    available_samples = len(input_buffer) // bytes_per_input_sample
                    sample_count = (
                        available_samples
                        if force_flush
                        else min(available_samples, samples_per_chunk)
                    )
                    chunk_bytes = bytes(input_buffer[: sample_count * bytes_per_input_sample])
                    del input_buffer[: sample_count * bytes_per_input_sample]

                    frame = av.AudioFrame(
                        format=input_audio_format,
                        layout=input_audio_layout,
                        samples=sample_count,
                    )
                    frame.sample_rate = audio_stream_format.sample_rate
                    frame.planes[0].update(chunk_bytes)
                    await dispatch_frame(frame)

            try:
                async for chunk in audio_stream:
                    if chunk:
                        input_buffer.extend(chunk)
                    await process_available(force_flush=False)
                await process_available(force_flush=True)
            finally:
                await asyncio.gather(
                    *(
                        queue.put(sentinel)
                        for queues in format_subscribers.values()
                        for queue in queues
                    ),
                    return_exceptions=True,
                )

        return self._server.loop.create_task(fanout_audio())

    def suggest_optimal_sample_rate(self, source_sample_rate: int) -> int:
        """Suggest an optimal sample rate for the next track.

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

        # For FLAC, we need to construct a proper FLAC stream header ourselves
        # since ffmpeg only provides the StreamInfo metadata block in extradata:
        # See https://datatracker.ietf.org/doc/rfc9639/ Section 8.1
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

    def _calculate_optimal_chunk_samples(self, source_format: AudioFormat) -> int:
        compressed_players = [
            player
            for player in self._clients
            if self._player_formats.get(player.client_id, AudioFormat(0, 0, 0)).codec
            != AudioCodec.PCM
        ]

        if not compressed_players:
            # All players use PCM, use 25ms chunks
            return int(source_format.sample_rate * 0.025)

        # TODO: replace this logic by allowing each device to have their own preferred chunk size,
        # does this even work in cases with different codecs?
        max_frame_size = 0
        for player in compressed_players:
            player_format = self._player_formats[player.client_id]
            encoder = self._get_or_create_audio_encoder(player_format)

            # Scale frame size to source sample rate
            scaled_frame_size = int(
                encoder.frame_size * source_format.sample_rate / player_format.sample_rate
            )
            max_frame_size = max(max_frame_size, scaled_frame_size)

        return max_frame_size if max_frame_size > 0 else int(source_format.sample_rate * 0.025)

    def _send_stream_start_msg(
        self,
        client: ResonateClient,
        audio_format: AudioFormat | None = None,
        *,
        include_player: bool = True,
    ) -> None:
        """Send a stream start message to a client with the specified audio format for players."""
        logger.debug(
            "_send_stream_start_msg: client=%s, format=%s",
            client.client_id,
            audio_format,
        )
        if include_player and client.check_role(Roles.PLAYER):
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
        _ = self._client_art_formats.pop(client.client_id, None)
        client.send_message(StreamEndMessage())

    async def stop(self, stop_time_us: int | None = None) -> bool:  # noqa: PLR0915
        """
        Stop playback for the group and clean up resources.

        Compared to pause(), this also:
        - Cancels the audio streaming task
        - Sends stream end messages to all clients
        - Clears all buffers and format mappings
        - Cleans up all audio encoders
        - Clears active client stream tasks

        Args:
            stop_time_us: Optional absolute timestamp (microseconds) when playback should
                stop. When provided and in the future, the stop request is scheduled and
                this method returns immediately.

        Returns:
            bool: True if an active or scheduled stream was stopped (or scheduled to stop),
            False if no stream was active.
        """
        active = self._stream_task is not None or bool(self._client_stream_tasks)

        if stop_time_us is not None:
            now_us = int(self._server.loop.time() * 1_000_000)
            if stop_time_us > now_us:
                delay = (stop_time_us - now_us) / 1_000_000

                async def _delayed_stop() -> None:
                    try:
                        await self.stop()
                    except Exception:
                        logger.exception("Scheduled stop failed")

                self._server.loop.call_later(delay, _delayed_stop)
                return active

        if not active:
            logger.debug("stop called but no active stream task")
            return False

        logger.debug(
            "Stopping playback for group with clients: %s",
            [c.client_id for c in self._clients],
        )

        active_client_tasks = list(self._client_stream_tasks.values())
        for queue in list(self._fanout_queues.values()):
            with suppress(QueueFull):
                queue.put_nowait(None)

        for task in active_client_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Unhandled exception while stopping client task")
        self._client_stream_tasks.clear()

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

        for client in self._clients:
            self._send_stream_end_msg(client)
            if client.check_role(Roles.PLAYER):
                self._player_formats.pop(client.client_id, None)

        self._direct_session = None

        self._audio_encoders.clear()
        self._audio_headers.clear()
        self._fanout_queues.clear()
        self._fanout_subscribers.clear()
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

    def set_metadata(self, metadata: Metadata | None) -> None:
        """
        Set metadata for the group and send to all clients.

        Only sends updates for fields that have changed since the last call.

        Args:
            metadata: The new metadata to send to clients.
        """
        # TODO: integrate this more closely with play_media?
        # Check if metadata has actually changed
        if self._current_metadata == metadata:
            return
        last_metadata = self._current_metadata

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
            _ = self._server.loop.create_task(cb(event))  # Fire and forget event callback

    @property
    def state(self) -> PlaybackStateType:
        """Current playback state of the group."""
        return self._current_state

    def remove_client(self, client: ResonateClient) -> None:
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
            _ = self.stop()
            self._clients = []
        else:
            self._clients.remove(client)
            if self._stream_task is not None:
                # Notify the client that the stream ended
                try:
                    self._send_stream_end_msg(client)
                except QueueFull:
                    logger.warning("Failed to send stream end message to %s", client.client_id)
                if client.check_role(Roles.PLAYER):
                    player_format = self._player_formats.pop(client.client_id, None)
                    queue = self._fanout_queues.pop(client.client_id, None)
                    if (
                        queue is not None
                        and player_format is not None
                        and (subscribers := self._fanout_subscribers.get(player_format))
                    ):
                        with suppress(ValueError):
                            subscribers.remove(queue)
                    if queue is not None:
                        with suppress(QueueFull):
                            queue.put_nowait(None)
                    if task := self._client_stream_tasks.pop(client.client_id, None):
                        task.cancel()
        if not self._clients:
            # Emit event for group deletion, no clients left
            self._signal_event(GroupDeletedEvent())
        else:
            # Emit event for client removal
            self._signal_event(GroupMemberRemovedEvent(client.client_id))
        # Each client needs to be in a group, add it to a new one
        client._set_group(ResonateGroup(self._server, client))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    def add_client(self, client: ResonateClient) -> None:  # noqa: PLR0915
        """
        Add a client to this group.

        The client is first removed from any existing group. If a session is
        currently active, players are immediately joined to the session with
        an appropriate audio format.

        Args:
            client: The client to add to this group.
        """
        logger.debug("adding %s to group with members: %s", client.client_id, self._clients)
        _ = client.group.stop()
        if client in self._clients:
            return
        # Remove it from any existing group first
        client.ungroup()

        # Add client to this group's client list
        self._clients.append(client)

        # Emit event for client addition
        self._signal_event(GroupMemberAddedEvent(client.client_id))

        # Then set the group (which will emit ClientGroupChangedEvent)
        client._set_group(self)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        if self._stream_task is not None and self._stream_audio_format is not None:
            logger.debug("Joining client %s to current stream", client.client_id)
            if client.check_role(Roles.PLAYER):
                player_format = client.player_throw.determine_optimal_format(
                    self._stream_audio_format
                )
                self._player_formats[client.client_id] = player_format

                if self._direct_session is not None:
                    self._server.loop.create_task(
                        self._handle_late_join_direct(client, player_format)
                    )
                elif self._fanout_subscribers and self._play_start_time_us is not None:
                    queue: Queue[bytes | None] = Queue()
                    self._fanout_queues[client.client_id] = queue
                    subscribers = self._fanout_subscribers.setdefault(player_format, [])
                    subscribers.append(queue)

                    stream_offset_us = max(
                        0,
                        int(self._server.loop.time() * 1_000_000) - self._play_start_time_us,
                    )

                    async def stream_gen(
                        queue_ref: Queue[bytes | None] = queue,
                    ) -> AsyncGenerator[bytes, None]:
                        while True:
                            chunk = await queue_ref.get()
                            if chunk is None:
                                break
                            assert isinstance(chunk, bytes)
                            yield chunk

                    self._client_stream_tasks[client.client_id] = self._server.loop.create_task(
                        client.player_throw._play_media_direct(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                            stream_gen(),
                            player_format,
                            play_start_time_us=self._play_start_time_us,
                            stream_start_time_us=stream_offset_us,
                        )
                    )
                    self._send_stream_start_msg(client, player_format)
                    if client.check_role(Roles.METADATA) or client.check_role(Roles.VISUALIZER):
                        self._send_stream_start_msg(client, None, include_player=False)
                else:
                    logger.warning(
                        "Active stream missing fan-out context; "
                        "falling back to stream/start for %s",
                        client.client_id,
                    )
                    self._send_stream_start_msg(client, player_format)
            else:
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
            self._send_media_art_to_client(client, self._current_media_art)

    def _validate_audio_format(self, audio_format: AudioFormat) -> tuple[int, str, str] | None:
        """
        Validate audio format and return format parameters.

        Args:
            audio_format: The source audio format to validate.

        Returns:
            Tuple of (bytes_per_sample, audio_format_str, layout_str) or None if invalid.
        """
        if audio_format.bit_depth == 16:
            input_bytes_per_sample = 2
            input_audio_format = "s16"
        elif audio_format.bit_depth == 24:
            input_bytes_per_sample = 3
            input_audio_format = "s24"
        else:
            logger.error("Only 16bit and 24bit audio is supported")
            return None

        if audio_format.channels == 1:
            input_audio_layout = "mono"
        elif audio_format.channels == 2:
            input_audio_layout = "stereo"
        else:
            logger.error("Only 1 and 2 channel audio is supported")
            return None

        return input_bytes_per_sample, input_audio_format, input_audio_layout

    def _resample_and_encode_to_player(
        self,
        player: ResonateClient,
        player_format: AudioFormat,
        in_frame: av.AudioFrame,
        resamplers: dict[AudioFormat, av.AudioResampler],
        chunk_timestamp_us: int,
    ) -> tuple[int, int]:
        """
        Resample audio for a specific player and encode/send the data.

        Args:
            player: The player to send audio data to.
            player_format: The target audio format for the player.
            in_frame: The input audio frame to resample.
            resamplers: Dictionary of existing resamplers for reuse.
            chunk_timestamp_us: Timestamp for the audio chunk in microseconds.

        Returns:
            Tuple of (sample_count, duration_of_chunk_us).
        """
        resampler = resamplers.get(player_format)
        if resampler is None:
            resampler = av.AudioResampler(
                format="s16" if player_format.bit_depth == 16 else "s24",
                layout="stereo" if player_format.channels == 2 else "mono",
                rate=player_format.sample_rate,
            )
            resamplers[player_format] = resampler

        out_frames = resampler.resample(in_frame)
        if len(out_frames) != 1:
            logger.warning("resampling resulted in %s frames", len(out_frames))

        sample_count = out_frames[0].samples
        if player_format.codec in (AudioCodec.OPUS, AudioCodec.FLAC):
            encoder = self._get_or_create_audio_encoder(player_format)
            packets = encoder.encode(out_frames[0])

            for packet in packets:
                header = pack_binary_header_raw(
                    BinaryMessageType.AUDIO_CHUNK.value,
                    chunk_timestamp_us,
                )
                player.send_message(header + bytes(packet))
        elif player_format.codec == AudioCodec.PCM:
            # Send as raw PCM
            # We need to manually slice the audio data since the buffer may be
            # larger than than the expected size
            audio_data = bytes(out_frames[0].planes[0])[
                : (2 if player_format.bit_depth == 16 else 3)
                * player_format.channels
                * sample_count
            ]
            if len(out_frames[0].planes) != 1:
                logger.warning("resampling resulted in %s planes", len(out_frames[0].planes))

            header = pack_binary_header_raw(
                BinaryMessageType.AUDIO_CHUNK.value,
                chunk_timestamp_us,
            )
            player.send_message(header + audio_data)
        else:
            raise NotImplementedError(f"Codec {player_format.codec} is not supported yet")

        duration_of_chunk_us = int((sample_count / player_format.sample_rate) * 1_000_000)
        return sample_count, duration_of_chunk_us

    def handle_stream_format_request(
        self,
        player: ResonateClient,
        request: StreamRequestFormatPayload,
    ) -> None:
        """Handle stream/request-format from a player and send stream/update."""
        # Only applicable if there is an active stream
        if self._stream_task is None or self._stream_audio_format is None:
            logger.debug(
                "Ignoring stream/request-format from %s without active stream",
                player.client_id,
            )
            return

        # Start from the current player format or determine from source
        current = self._player_formats.get(player.client_id)
        assert current is not None, "Player must have a current format if streaming"

        # Apply requested overrides
        codec = current.codec
        if request.codec is not None:
            try:
                codec = AudioCodec(request.codec)
            except ValueError:
                logger.warning(
                    "Player %s requested switch to unsupported codec %s, ignoring",
                    player.client_id,
                    request.codec,
                )
                codec = current.codec
            # Ensure requested codec is supported by player
            if (
                player.info.player_support
                and codec.value not in player.info.player_support.support_codecs
            ):
                raise ValueError(
                    f"Player {player.client_id} does not support requested codec {codec}"
                )

        sample_rate = request.sample_rate or current.sample_rate
        if (
            player.info.player_support
            and sample_rate not in player.info.player_support.support_sample_rates
        ):
            raise ValueError(
                f"Player {player.client_id} does not support requested sample rate {sample_rate}"
            )

        bit_depth = request.bit_depth or current.bit_depth
        if (
            player.info.player_support
            and bit_depth not in player.info.player_support.support_bit_depth
        ):
            raise ValueError(
                f"Player {player.client_id} does not support requested bit depth {bit_depth}"
            )
        if bit_depth != 16:
            raise NotImplementedError("Only 16bit audio is supported for now")

        channels = request.channels or current.channels
        if (
            player.info.player_support
            and channels not in player.info.player_support.support_channels
        ):
            raise ValueError(
                f"Player {player.client_id} does not support requested channel count {channels}"
            )
        if channels not in (1, 2):
            raise NotImplementedError("Only mono and stereo audio is supported for now")

        new_format = AudioFormat(
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            channels=channels,
            codec=codec,
        )

        # Do not send the update yet, so the sending of this message and the actual format
        # change during streaming happen in the correct order
        header = self._get_audio_header(new_format)

        update = StreamUpdatePlayer(
            codec=new_format.codec.value,
            sample_rate=new_format.sample_rate,
            channels=new_format.channels,
            bit_depth=new_format.bit_depth,
            codec_header=header,
        )
        self._scheduled_format_changes[player.client_id] = (
            StreamUpdateMessage(StreamUpdatePayload(player=update)),
            new_format,
        )

    def _update_player_format(self, player: ResonateClient) -> None:
        """Apply any scheduled format changes for a player if needed."""
        if change := self._scheduled_format_changes.pop(player.client_id, None):
            format_change_message, new_format = change
            logger.debug(
                "Switching format for %s from %s to %s",
                player.client_id,
                self._player_formats.get(player.client_id, None),
                new_format,
            )
            player.send_message(format_change_message)
            self._player_formats[player.client_id] = new_format
