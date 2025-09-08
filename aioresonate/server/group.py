"""Manages and synchronizes playback for a group of one or more players."""

import asyncio
import base64
import logging
from asyncio import QueueFull, Task
from collections.abc import AsyncGenerator, Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import av
from av import logging as av_logging

from aioresonate.models import BinaryMessageType, pack_binary_header_raw, server_messages
from aioresonate.models.types import RepeatMode

# The cyclic import is not an issue during runtime, so hide it
# pyright: reportImportCycles=none
if TYPE_CHECKING:
    from .player import Player
    from .server import ResonateServer

INITIAL_PLAYBACK_DELAY_US = 1_000_000

logger = logging.getLogger(__name__)


class AudioCodec(Enum):
    """Supported audio codecs."""

    PCM = "pcm"
    FLAC = "flac"
    OPUS = "opus"


class GroupState(Enum):
    """Player group playback state."""

    IDLE = "idle"
    """Group is not currently playing any media."""
    PLAYING = "playing"
    """Group is actively playing media."""


class GroupEvent:
    """Base event type used by PlayerGroup.add_event_listener()."""


@dataclass
class GroupStateChangedEvent(GroupEvent):
    """Group state has changed."""

    state: GroupState
    """The new group state."""


@dataclass
class GroupMemberAddedEvent(GroupEvent):
    """A player was added to the group."""

    player_id: str
    """The ID of the player that was added."""


@dataclass
class GroupMemberRemovedEvent(GroupEvent):
    """A player was removed from the group."""

    player_id: str
    """The ID of the player that was removed."""


@dataclass
class GroupDeletedEvent(GroupEvent):
    """This group has no more members and has been deleted."""


@dataclass(frozen=True)
class AudioFormat:
    """
    Audio format specification.

    Represents the audio format parameters for both compressed and uncompressed audio.
    """

    sample_rate: int
    """Sample rate in Hz (e.g., 44100, 48000)."""
    bit_depth: int
    """Bit depth in bits per sample (16 or 24)."""
    channels: int
    """Number of audio channels (1 for mono, 2 for stereo)."""
    codec: AudioCodec = AudioCodec.PCM
    """Audio codec to use."""


@dataclass
class Metadata:
    """Metadata for media playback."""

    # TODO: finish this once the spec is finalized

    title: str = ""
    """Title of the current media."""
    artist: str = ""
    """Artist of the current media."""
    album: str = ""
    """Album of the current media."""
    year: int = 0
    """Release year of the current media."""
    track: int = 0
    """Track number of the current media."""
    repeat: RepeatMode = RepeatMode.OFF
    """Current repeat mode."""
    shuffle: bool = False
    """Whether shuffle is enabled."""


class PlayerGroup:
    """
    A group of one or more players for synchronized playback.

    Handles synchronized audio streaming across multiple players with automatic
    format conversion and buffer management. Every player is always assigned to
    a group to simplify grouping requests.
    """

    _players: list["Player"]
    """List of all players in this group."""
    _player_formats: dict[str, AudioFormat]
    """Mapping of player IDs to their selected audio formats."""
    _server: "ResonateServer"
    """Reference to the ResonateServer instance."""
    _stream_task: Task[None] | None = None
    """Task handling the audio streaming loop, None when not streaming."""
    _stream_audio_format: AudioFormat | None = None
    """The source audio format for the current stream, None when not streaming."""
    _current_metadata: Metadata | None = None
    """Current metadata for the group, None if no metadata set."""
    _audio_encoders: dict[AudioFormat, av.AudioCodecContext]
    """Mapping of audio formats to their av encoder contexts."""
    _audio_headers: dict[AudioFormat, str]
    """Mapping of audio formats to their base64 encoded headers."""
    _preferred_stream_codec: AudioCodec = AudioCodec.OPUS
    """Preferred codec used by the current stream."""
    _event_cbs: list[Callable[[GroupEvent], Coroutine[None, None, None]]]
    """List of event callbacks for this group."""
    _current_state: GroupState = GroupState.IDLE
    """Current playback state of the group."""

    def __init__(self, server: "ResonateServer", *args: "Player") -> None:
        """
        DO NOT CALL THIS CONSTRUCTOR. INTERNAL USE ONLY.

        Groups are managed automatically by the server.

        Initialize a new PlayerGroup.

        Args:
            server: The ResonateServer instance this group belongs to.
            *args: Players to add to this group.
        """
        self._server = server
        self._players = list(args)
        self._player_formats = {}
        self._current_metadata = None
        self._audio_encoders = {}
        self._audio_headers = {}
        self._event_cbs = []
        logger.debug(
            "PlayerGroup initialized with %d player(s): %s",
            len(self._players),
            [type(p).__name__ for p in self._players],
        )

    async def play_media(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        audio_stream_format: AudioFormat,
        preferred_stream_codec: AudioCodec = AudioCodec.OPUS,
    ) -> None:
        """
        Start playback of a new media stream.

        Stops any current stream and starts a new one with the given audio source.
        The audio source should provide uncompressed PCM audio data.
        Format conversion and synchronization for all players will be handled automatically.

        Args:
            audio_stream: Async generator yielding PCM audio chunks as bytes.
            audio_stream_format: Format specification for the input audio data.
        """
        logger.debug("Starting play_media with audio_stream_format: %s", audio_stream_format)
        stopped = self.stop()
        if stopped:
            # Wait a bit to allow players to process the session end
            await asyncio.sleep(0.5)
        # TODO: open questions:
        # - how to communicate to the caller what audio_format is preferred,
        #   especially on topology changes
        # - how to sync metadata/media_art with this audio stream?

        self._stream_audio_format = audio_stream_format
        self._preferred_stream_codec = preferred_stream_codec

        for player in self._players:
            logger.debug("Selecting format for player %s", player.player_id)
            player_format = self.determine_player_format(
                player, audio_stream_format, preferred_stream_codec
            )
            self._player_formats[player.player_id] = player_format
            logger.debug(
                "Sending session start to player %s with format %s",
                player.player_id,
                player_format,
            )
            self._send_session_start_msg(player, player_format)

        self._stream_task = self._server.loop.create_task(
            self._stream_audio(
                int(self._server.loop.time() * 1_000_000) + INITIAL_PLAYBACK_DELAY_US,
                audio_stream,
                audio_stream_format,
            )
        )

        self._current_state = GroupState.PLAYING
        self._signal_event(GroupStateChangedEvent(GroupState.PLAYING))

    def determine_player_format(
        self,
        player: "Player",
        source_format: AudioFormat,
        preferred_codec: AudioCodec = AudioCodec.OPUS,
    ) -> AudioFormat:
        """
        Determine the optimal audio format for the given player and source.

        Analyzes the player's capabilities and returns the best matching format,
        preferring higher quality when available and falling back gracefully.

        Args:
            player: The player to determine a format for.
            source_format: The source audio format to match against.
            preferred_codec: Preferred audio codec (e.g., Opus).
                In case the player doesn't support it, falls back to another codec.

        Returns:
            AudioFormat: The optimal format for the player.
        """
        # TODO: move this to player instead
        player_info = player.info

        # Determine optimal sample rate
        sample_rate = source_format.sample_rate
        if sample_rate not in player_info.support_sample_rates:
            # Prefer lower rates that are closest to source, fallback to minimum
            lower_rates = [r for r in player_info.support_sample_rates if r < sample_rate]
            sample_rate = max(lower_rates) if lower_rates else min(player_info.support_sample_rates)
            logger.debug("Adjusted sample_rate for player %s: %s", player.player_id, sample_rate)

        # Determine optimal bit depth
        bit_depth = source_format.bit_depth
        if bit_depth not in player_info.support_bit_depth:
            # Prefer 16-bit, then 24-bit
            if 16 in player_info.support_bit_depth:
                bit_depth = 16
            elif 24 in player_info.support_bit_depth:
                bit_depth = 24
            else:
                raise NotImplementedError("Only 16bit and 24bit are supported")
            logger.debug("Adjusted bit_depth for player %s: %s", player.player_id, bit_depth)

        # Determine optimal channel count
        channels = source_format.channels
        if channels not in player_info.support_channels:
            # Prefer stereo, then mono
            if 2 in player_info.support_channels:
                channels = 2
            elif 1 in player_info.support_channels:
                channels = 1
            else:
                raise NotImplementedError("Only mono and stereo are supported")
            logger.debug("Adjusted channels for player %s: %s", player.player_id, channels)

        # Determine optimal codec with fallback chain
        codec_fallbacks = [preferred_codec, AudioCodec.FLAC, AudioCodec.OPUS, AudioCodec.PCM]
        codec = None
        for candidate_codec in codec_fallbacks:
            if candidate_codec.value in player_info.support_codecs:
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
                        if condition and candidate_rate in player_info.support_sample_rates:
                            opus_sample_rate = candidate_rate
                            break

                    if opus_sample_rate is None:
                        logger.error(
                            "Player %s does not support any Opus sample rates, trying next codec",
                            player.player_id,
                        )
                        continue  # Try next codec in fallback chain

                    # Opus is viable, adjust sample rate and use it
                    if sample_rate != opus_sample_rate:
                        logger.debug(
                            "Adjusted sample_rate for Opus on player %s: %s -> %s",
                            player.player_id,
                            sample_rate,
                            opus_sample_rate,
                        )
                    sample_rate = opus_sample_rate

                codec = candidate_codec
                break

        if codec is None:
            raise ValueError(f"Player {player.player_id} does not support any known codec")

        if codec != preferred_codec:
            logger.info(
                "Falling back from preferred codec %s to %s for player %s",
                preferred_codec,
                codec,
                player.player_id,
            )

        # FLAC and PCM support any sample rate, no adjustment needed
        return AudioFormat(sample_rate, bit_depth, channels, codec)

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
            # FLAC stream signature (4 bytes): "fLaC"
            # Metadata block header (4 bytes):
            # - Bit 0: last metadata block (1 since we only have one)
            # - Bits 1-7: block type (0 for StreamInfo)
            # - Next 3 bytes: block length of the next metadata block in bytes
            # StreamInfo block (34 bytes): as provided by ffmpeg
            header = b"fLaC\x80" + (len(header)).to_bytes(3, "big") + header

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
            for player in self._players
            if self._player_formats.get(player.player_id, AudioFormat(0, 0, 0)).codec
            != AudioCodec.PCM
        ]

        if not compressed_players:
            # All players use PCM, use 25ms chunks
            return int(source_format.sample_rate * 0.025)

        # TODO: replace this logic by allowing each device to have their own preferred chunk size,
        # does this even work in cases with different codecs?
        max_frame_size = 0
        for player in compressed_players:
            player_format = self._player_formats[player.player_id]
            encoder = self._get_or_create_audio_encoder(player_format)

            # Scale frame size to source sample rate
            scaled_frame_size = int(
                encoder.frame_size * source_format.sample_rate / player_format.sample_rate
            )
            max_frame_size = max(max_frame_size, scaled_frame_size)

        return max_frame_size if max_frame_size > 0 else int(source_format.sample_rate * 0.025)

    def _send_session_start_msg(self, player: "Player", audio_format: AudioFormat) -> None:
        """Send a session start message to a player with the specified audio format."""
        logger.debug(
            "_send_session_start_msg: player=%s, format=%s",
            player.player_id,
            audio_format,
        )
        session_info = server_messages.SessionStartPayload(
            session_id=str(uuid4()),
            codec=audio_format.codec.value,
            sample_rate=audio_format.sample_rate,
            channels=audio_format.channels,
            bit_depth=audio_format.bit_depth,
            now=int(self._server.loop.time() * 1_000_000),
            codec_header=self._get_audio_header(audio_format),
        )
        logger.debug(
            "Sending session start message to player %s: %s", player.player_id, session_info
        )
        player.send_message(server_messages.SessionStartMessage(session_info))

    def _send_session_end_msg(self, player: "Player") -> None:
        """Send a session end message to a player to stop playback."""
        logger.debug("ending session for %s (%s)", player.name, player.player_id)
        player.send_message(server_messages.SessionEndMessage(server_messages.SessionEndPayload()))

    def stop(self) -> bool:
        """
        Stop playback for the group and clean up resources.

        Compared to pause(), this also:
        - Cancels the audio streaming task
        - Sends session end messages to all players
        - Clears all buffers and format mappings
        - Cleans up all audio encoders

        Returns:
            bool: True if an active stream was stopped, False if no stream was active.
        """
        if self._stream_task is None:
            logger.debug("stop called but no active stream task")
            return False
        logger.debug(
            "Stopping playback for group with players: %s",
            [p.player_id for p in self._players],
        )
        _ = self._stream_task.cancel()  # Don't care about cancellation result
        for player in self._players:
            self._send_session_end_msg(player)
            del self._player_formats[player.player_id]

        self._audio_encoders.clear()
        self._audio_headers.clear()
        self._stream_task = None

        if self._current_state != GroupState.IDLE:
            self._signal_event(GroupStateChangedEvent(GroupState.IDLE))
            self._current_state = GroupState.IDLE
        return True

    def set_metadata(self, metadata: Metadata) -> None:
        """
        Set metadata for the group and send to all players.

        Only sends updates for fields that have changed since the last call.

        Args:
            metadata: The new metadata to send to players.
        """
        # TODO: integrate this more closely with play_media?
        # Check if metadata has actually changed
        if self._current_metadata == metadata:
            return

        # Create partial update payload with only changed fields
        update_payload = server_messages.MetadataUpdatePayload()

        if self._current_metadata is None:
            # First time setting metadata, send all fields
            update_payload.title = metadata.title
            update_payload.artist = metadata.artist
            update_payload.album = metadata.album
            update_payload.year = metadata.year
            update_payload.track = metadata.track
            update_payload.repeat = metadata.repeat
            update_payload.shuffle = metadata.shuffle
        else:
            # Only send changed fields
            if self._current_metadata.title != metadata.title:
                update_payload.title = metadata.title
            if self._current_metadata.artist != metadata.artist:
                update_payload.artist = metadata.artist
            if self._current_metadata.album != metadata.album:
                update_payload.album = metadata.album
            if self._current_metadata.year != metadata.year:
                update_payload.year = metadata.year
            if self._current_metadata.track != metadata.track:
                update_payload.track = metadata.track
            if self._current_metadata.repeat != metadata.repeat:
                update_payload.repeat = metadata.repeat
            if self._current_metadata.shuffle != metadata.shuffle:
                update_payload.shuffle = metadata.shuffle

        # TODO: finish this once the spec is finalized, include group_members and support_commands

        # Send to all players in the group
        message = server_messages.MetadataUpdateMessage(update_payload)
        for player in self._players:
            logger.debug(
                "Sending metadata update message to player %s: %s",
                player.player_id,
                message.to_json(),
            )
            player.send_message(message)

        # Update current metadata
        self._current_metadata = metadata

    @property
    def players(self) -> list["Player"]:
        """All players that are part of this group."""
        return self._players

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
    def state(self) -> GroupState:
        """Current playback state of the group."""
        return self._current_state

    def remove_player(self, player: "Player") -> None:
        """
        Remove a player from this group.

        If a stream is active, the player receives a session end message.
        The player is automatically moved to its own new group since every
        player must belong to a group.
        If the player is not part of this group, this will have no effect.

        Args:
            player: The player to remove from this group.
        """
        if player not in self._players:
            logger.debug("player %s not in group, skipping removal", player.player_id)
            return
        logger.debug("removing %s from group with members: %s", player.player_id, self._players)
        if len(self._players) == 1:
            # Delete this group if that was the last player
            _ = self.stop()
            self._players = []
        else:
            self._players.remove(player)
            if self._stream_task is not None:
                # Notify the player that the session ended
                try:
                    self._send_session_end_msg(player)
                except QueueFull:
                    logger.warning("Failed to send session end message to %s", player.player_id)
                del self._player_formats[player.player_id]
        if not self._players:
            # Emit event for group deletion, no players left
            self._signal_event(GroupDeletedEvent())
        else:
            # Emit event for player removal
            self._signal_event(GroupMemberRemovedEvent(player.player_id))
        # Each player needs to be in a group, add it to a new one
        player._set_group(PlayerGroup(self._server, player))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    def add_player(self, player: "Player") -> None:
        """
        Add a player to this group.

        The player is first removed from any existing group. If a stream is
        currently active, the player is immediately joined to the stream with
        an appropriate audio format.

        Args:
            player: The player to add to this group.
        """
        logger.debug("adding %s to group with members: %s", player.player_id, self._players)
        _ = player.group.stop()
        if player in self._players:
            return
        # Remove it from any existing group first
        player.ungroup()
        player._set_group(self)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        if self._stream_task is not None and self._stream_audio_format is not None:
            logger.debug("Joining player %s to current stream", player.player_id)
            # Join it to the current stream
            player_format = self.determine_player_format(
                player, self._stream_audio_format, self._preferred_stream_codec
            )
            self._player_formats[player.player_id] = player_format
            self._send_session_start_msg(player, player_format)

        # Send current metadata to the new player if available
        if self._current_metadata is not None:
            update_payload = server_messages.MetadataUpdatePayload(
                title=self._current_metadata.title,
                artist=self._current_metadata.artist,
                album=self._current_metadata.album,
                year=self._current_metadata.year,
                track=self._current_metadata.track,
                repeat=self._current_metadata.repeat,
                shuffle=self._current_metadata.shuffle,
            )
            message = server_messages.MetadataUpdateMessage(update_payload)

            logger.debug(
                "Sending current metadata to new player %s: %s", player.player_id, message.to_json()
            )
            player.send_message(message)

        self._players.append(player)

        # Emit event for player addition
        self._signal_event(GroupMemberAddedEvent(player.player_id))

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
        player: "Player",
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
                    BinaryMessageType.PlayAudioChunk.value,
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
                BinaryMessageType.PlayAudioChunk.value,
                chunk_timestamp_us,
            )
            player.send_message(header + audio_data)
        else:
            raise NotImplementedError(f"Codec {player_format.codec} is not supported yet")

        duration_of_chunk_us = int((sample_count / player_format.sample_rate) * 1_000_000)
        return sample_count, duration_of_chunk_us

    async def _calculate_timing_and_sleep(
        self,
        chunk_timestamp_us: int,
        buffer_duration_us: int,
    ) -> None:
        """
        Calculate timing and sleep if needed to maintain buffer levels.

        Args:
            chunk_timestamp_us: Current chunk timestamp in microseconds.
            buffer_duration_us: Maximum buffer duration in microseconds.
        """
        time_until_next_chunk = chunk_timestamp_us - int(self._server.loop.time() * 1_000_000)

        # TODO: I think this may exclude the burst at startup?
        if time_until_next_chunk > buffer_duration_us:
            await asyncio.sleep((time_until_next_chunk - buffer_duration_us) / 1_000_000)

    async def _stream_audio(
        self,
        start_time_us: int,
        audio_source: AsyncGenerator[bytes, None],
        audio_format: AudioFormat,
    ) -> None:
        """
        Handle the audio streaming loop for all players in the group.

        This method processes the audio source, converts formats as needed for each
        player, maintains synchronization via timestamps, and manages buffer levels
        to prevent overflows.

        Args:
            start_time_us: Initial playback timestamp in microseconds.
            audio_source: Generator providing PCM audio chunks.
            audio_format: Format specification for the source audio.
        """
        # TODO: Complete resampling
        # -  deduplicate conversion when multiple players use the same rate
        # - Maybe notify the library user that play_media should be restarted with
        #   a better format?
        # - Support other formats than pcm
        # - Optimize this

        try:
            logger.debug(
                "_stream_audio started: start_time_us=%d, audio_format=%s",
                start_time_us,
                audio_format,
            )

            # Validate and set up audio format
            format_result = self._validate_audio_format(audio_format)
            if format_result is None:
                return
            input_bytes_per_sample, input_audio_format, input_audio_layout = format_result

            # Initialize streaming context variables
            input_sample_size = audio_format.channels * input_bytes_per_sample
            input_sample_rate = audio_format.sample_rate
            input_samples_per_chunk = self._calculate_optimal_chunk_samples(audio_format)
            chunk_timestamp_us = start_time_us

            resamplers: dict[AudioFormat, av.AudioResampler] = {}

            in_frame = av.AudioFrame(
                format=input_audio_format,
                layout=input_audio_layout,
                samples=input_samples_per_chunk,
            )
            in_frame.sample_rate = input_sample_rate
            input_buffer = bytearray()

            logger.debug("Entering audio streaming loop")
            async for chunk in audio_source:
                input_buffer += bytes(chunk)
                while len(input_buffer) >= (input_samples_per_chunk * input_sample_size):
                    chunk_to_encode = input_buffer[: (input_samples_per_chunk * input_sample_size)]
                    del input_buffer[: (input_samples_per_chunk * input_sample_size)]

                    in_frame.planes[0].update(bytes(chunk_to_encode))

                    sample_count = None
                    # TODO: to what should we set this?
                    buffer_duration_us = 2_000_000
                    duration_of_samples_in_chunk: list[int] = []

                    for player in self._players:
                        player_format = self._player_formats[player.player_id]
                        try:
                            sample_count, duration_us = self._resample_and_encode_to_player(
                                player, player_format, in_frame, resamplers, chunk_timestamp_us
                            )
                            duration_of_samples_in_chunk.append(duration_us)
                        except QueueFull:
                            logger.warning(
                                "Error sending audio chunk to %s, disconnecting player",
                                player.player_id,
                            )
                            await player.disconnect()

                        # Calculate buffer duration for this player
                        player_buffer_capacity_samples = player.info.buffer_capacity // (
                            (player_format.bit_depth // 8) * player_format.channels
                        )
                        player_buffer_duration = int(
                            1_000_000 * player_buffer_capacity_samples / player_format.sample_rate
                        )
                        buffer_duration_us = min(buffer_duration_us, player_buffer_duration)

                    if sample_count is None:
                        logger.error("No players in group, stopping stream")
                        return

                    # TODO: Is mean the correct approach here?
                    # Or just make it based on the input stream
                    chunk_timestamp_us += int(
                        sum(duration_of_samples_in_chunk) / len(duration_of_samples_in_chunk)
                    )

                    await self._calculate_timing_and_sleep(chunk_timestamp_us, buffer_duration_us)

            # TODO: flush buffer
            logger.debug("Audio streaming loop ended")
        except Exception:
            logger.exception("failed to stream audio")
        finally:
            self.stop()
