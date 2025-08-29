"""Manages and synchronizes playback for a group of one or more players."""

import asyncio
import logging
from asyncio import Task
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

import av

from aioresonate.models import BinaryMessageType, pack_binary_header_raw, server_messages
from aioresonate.models.types import RepeatMode

# The cyclic import is not an issue during runtime, so hide it
# pyright: reportImportCycles=none
if TYPE_CHECKING:
    from .player import Player
    from .server import ResonateServer

INITIAL_PLAYBACK_DELAY_US = 1_000_000
CHUNK_DURATION_US = 25_000

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioFormat:
    """LPCM audio format specification.

    Represents the audio format parameters for uncompressed PCM audio.
    """

    sample_rate: int
    """Sample rate in Hz (e.g., 44100, 48000)."""
    bit_depth: int
    """Bit depth in bits per sample (16 or 24)."""
    channels: int
    """Number of audio channels (1 for mono, 2 for stereo)."""


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
    """A group of one or more players for synchronized playback.

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

    def __init__(self, server: "ResonateServer", *args: "Player") -> None:
        """DO NOT CALL THIS CONSTRUCTOR. INTERNAL USE ONLY.

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
        logger.debug(
            "PlayerGroup initialized with %d player(s): %s",
            len(self._players),
            [type(p).__name__ for p in self._players],
        )

    async def play_media(
        self, audio_source: AsyncGenerator[bytes, None], audio_format: AudioFormat
    ) -> None:
        """Start playback of a new media stream.

        Stops any current stream and starts a new one with the given audio source.
        The audio source should provide uncompressed PCM audio data.
        Format conversion and synchronization for all players will be handled automatically.

        Args:
            audio_source: Async generator yielding PCM audio chunks as bytes.
            audio_format: Format specification for the input audio data.
        """
        logger.debug("Starting play_media with audio_format: %s", audio_format)
        stopped = self.stop()
        if stopped:
            # Wait a bit to allow players to process the session end
            await asyncio.sleep(0.5)
        # TODO: open questions:
        # - how to communicate to the caller what audio_format is preferred,
        #   especially on topology changes
        # - how to sync metadata/media_art with this audio stream?

        self._stream_audio_format = audio_format

        for player in self._players:
            logger.debug("Selecting format for player %s", player.player_id)
            player_format = self.determine_player_format(player, audio_format)
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
                audio_source,
                audio_format,
            )
        )

    def determine_player_format(self, player: "Player", source_format: AudioFormat) -> AudioFormat:
        """Determine the optimal audio format for the given player and source.

        Analyzes the player's capabilities and returns the best matching format,
        preferring higher quality when available and falling back gracefully.

        Args:
            player: The player to determine a format for.
            source_format: The source audio format to match against.

        Returns:
            AudioFormat: The optimal format for the player.
        """
        # TODO: move this to player instead
        support_sample_rates = player.info.support_sample_rates
        support_bit_depth = player.info.support_bit_depth
        support_channels = player.info.support_channels

        sample_rate = source_format.sample_rate
        if sample_rate not in support_sample_rates:
            lower_rates = [r for r in support_sample_rates if r < sample_rate]
            sample_rate = max(lower_rates) if lower_rates else min(support_sample_rates)
            logger.debug("Adjusted sample_rate for player %s: %s", player.player_id, sample_rate)

        bit_depth = source_format.bit_depth
        if bit_depth not in support_bit_depth:
            if 16 in support_bit_depth:
                bit_depth = 16
            elif 24 in support_bit_depth:
                bit_depth = 24
            else:
                raise NotImplementedError("Only 16bit and 24bit are supported")
            logger.debug("Adjusted bit_depth for player %s: %s", player.player_id, bit_depth)

        channels = source_format.channels
        if channels not in support_channels:
            if 2 in support_channels:
                channels = 2
            elif 1 in support_channels:
                channels = 1
            else:
                raise NotImplementedError("Only mono and stereo are supported")
            logger.debug("Adjusted channels for player %s: %s", player.player_id, channels)

        if "pcm" not in player.info.support_codecs:
            raise NotImplementedError("Only pcm is supported for now")

        return AudioFormat(sample_rate, bit_depth, channels)

    def _send_session_start_msg(self, player: "Player", audio_format: AudioFormat) -> None:
        """Send a session start message to a player with the specified audio format."""
        logger.debug(
            "_send_session_start_msg: player=%s, format=%s",
            player.player_id,
            audio_format,
        )
        session_info = server_messages.SessionStartPayload(
            session_id=str(uuid4()),
            codec="pcm",
            sample_rate=audio_format.sample_rate,
            channels=audio_format.channels,
            bit_depth=audio_format.bit_depth,
            now=int(self._server.loop.time() * 1_000_000),
            codec_header=None,
        )
        player.send_message(server_messages.SessionStartMessage(session_info))

    def _send_session_end_msg(self, player: "Player") -> None:
        """Send a session end message to a player to stop playback."""
        logger.debug("ending session for %s (%s)", player.name, player.player_id)
        player.send_message(server_messages.SessionEndMessage(server_messages.SessionEndPayload()))

    def stop(self) -> bool:
        """Stop playback for the group and clean up resources.

        Compared to pause(), this also:
        - Cancels the audio streaming task
        - Sends session end messages to all players
        - Clears all buffers and format mappings

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
        self._stream_task = None
        return True

    def set_metadata(self, metadata: Metadata) -> None:
        """Set metadata for the group and send to all players.

        Only sends updates for fields that have changed since the last call.
        Also includes group_members and support_commands in the update.

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
            update_payload.title = metadata.title or None
            update_payload.artist = metadata.artist or None
            update_payload.album = metadata.album or None
            update_payload.year = metadata.year if metadata.year != 0 else None
            update_payload.track = metadata.track if metadata.track != 0 else None
            update_payload.repeat = metadata.repeat
            update_payload.shuffle = metadata.shuffle
        else:
            # Only send changed fields
            if self._current_metadata.title != metadata.title:
                update_payload.title = metadata.title or None
            if self._current_metadata.artist != metadata.artist:
                update_payload.artist = metadata.artist or None
            if self._current_metadata.album != metadata.album:
                update_payload.album = metadata.album or None
            if self._current_metadata.year != metadata.year:
                update_payload.year = metadata.year if metadata.year != 0 else None
            if self._current_metadata.track != metadata.track:
                update_payload.track = metadata.track if metadata.track != 0 else None
            if self._current_metadata.repeat != metadata.repeat:
                update_payload.repeat = metadata.repeat
            if self._current_metadata.shuffle != metadata.shuffle:
                update_payload.shuffle = metadata.shuffle

        # TODO: finish this once the spec is finalized
        update_payload.group_members = [player.player_id for player in self._players]

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

    def remove_player(self, player: "Player") -> None:
        """Remove a player from this group.

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
        self._players.remove(player)
        if self._stream_task is not None:
            # Notify the player that the session ended
            self._send_session_end_msg(player)
            del self._player_formats[player.player_id]
        # Each player needs to be in a group, add it to a new one
        player._set_group(PlayerGroup(self._server, player))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    def add_player(self, player: "Player") -> None:
        """Add a player to this group.

        The player is first removed from any existing group. If a stream is
        currently active, the player is immediately joined to the stream with
        an appropriate audio format.

        Args:
            player: The player to add to this group.
        """
        logger.debug("adding %s to group with members: %s", player.player_id, self._players)
        if player in self._players:
            return
        # Remove it from any existing group first
        player.ungroup()
        player._set_group(self)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        if self._stream_task is not None and self._stream_audio_format is not None:
            logger.debug("Joining player %s to current stream", player.player_id)
            # Join it to the current stream
            player_format = self.determine_player_format(player, self._stream_audio_format)
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
                group_members=[p.player_id for p in self._players] + [player.player_id],
            )
            # TODO: send update to other players, since group_members changed
            message = server_messages.MetadataUpdateMessage(update_payload)

            logger.debug(
                "Sending current metadata to new player %s: %s", player.player_id, message.to_json()
            )
            player.send_message(message)

        self._players.append(player)

    def _validate_audio_format(self, audio_format: AudioFormat) -> tuple[int, str, str] | None:
        """Validate audio format and return format parameters.

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

    def _resample_and_send_to_player(
        self,
        player: "Player",
        player_format: AudioFormat,
        in_frame: av.AudioFrame,
        resamplers: dict[AudioFormat, av.AudioResampler],
        chunk_timestamp_us: int,
    ) -> tuple[int, int]:
        """Resample audio for a specific player and send the data.

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
        # TODO: ESPHome should probably be cutting the audio_data,
        # this only works with pcm
        audio_data = bytes(out_frames[0].planes[0])[: (sample_count * 4)]
        if len(out_frames[0].planes) != 1:
            logger.warning("resampling resulted in %s planes", len(out_frames[0].planes))

        header = pack_binary_header_raw(
            BinaryMessageType.PlayAudioChunk.value,
            chunk_timestamp_us,
            sample_count,
        )
        player.send_message(header + audio_data)

        duration_of_chunk_us = int((sample_count / player_format.sample_rate) * 1_000_000)
        return sample_count, duration_of_chunk_us

    async def _calculate_timing_and_sleep(
        self,
        chunk_timestamp_us: int,
        buffer_duration_us: int,
    ) -> None:
        """Calculate timing and sleep if needed to maintain buffer levels.

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
        """Handle the audio streaming loop for all players in the group.

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
            chunk_length = CHUNK_DURATION_US / 1_000_000
            input_samples_per_chunk = int(input_sample_rate * chunk_length)
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
                        sample_count, duration_us = self._resample_and_send_to_player(
                            player, player_format, in_frame, resamplers, chunk_timestamp_us
                        )
                        duration_of_samples_in_chunk.append(duration_us)

                        # Calculate buffer duration for this player
                        player_buffer_capacity_samples = player.info.buffer_capacity // (
                            (player_format.bit_depth // 8) * player_format.channels
                        )
                        player_buffer_duration = int(
                            1_000_000 * player_buffer_capacity_samples / player_format.sample_rate
                        )
                        buffer_duration_us = min(buffer_duration_us, player_buffer_duration)

                    # We have at least one player playing
                    assert sample_count is not None

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
