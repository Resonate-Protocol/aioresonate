"""Manages and synchronizes playback for a group of one or more players."""

import asyncio
import logging
import struct
from asyncio import Task
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

import av

from aioresonate.models import BINARY_HEADER_FORMAT, BinaryMessageType, server_messages

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
    """LPCM audio format."""

    sample_rate: int
    bit_depth: int
    channels: int


class PlayerGroup:
    """A group of one or more players."""

    # In this implementation, every player is always assigned to a group.
    # This simplifies grouping requests initiated by the player.

    _players: list["Player"]
    _player_formats: dict[str, AudioFormat]
    _server: "ResonateServer"
    _stream_task: Task[None] | None = None
    _stream_audio_format: AudioFormat | None = None

    def __init__(self, server: "ResonateServer", *args: "Player") -> None:
        """Do not call this constructor."""
        self._server = server
        self._players = list(args)
        self._player_formats = {}
        logger.debug(
            "PlayerGroup initialized with %d player(s): %s",
            len(self._players),
            [type(p).__name__ for p in self._players],
        )

    async def play_media(
        self, audio_source: AsyncGenerator[bytes, None], audio_format: AudioFormat
    ) -> None:
        """Start playback of a new media stream.

        The library expects uncompressed PCM audio and will handle encoding.
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
        # TODO: port _stream_audio

        # TODO: Stop any prior stream

        # TODO: dynamic session info

        self._stream_audio_format = audio_format

        for player in self._players:
            logger.debug("Selecting format for player %s", player.player_id)
            player_format = self.select_player_format(player, audio_format)
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

    def select_player_format(self, player: "Player", source_format: AudioFormat) -> AudioFormat:
        """Select the most optimal audio format for the given source."""
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
            now=int(self._server.loop.time() * 1_000_000),  # TODO: maybe remove from spec?
            codec_header=None,
        )
        player.send_message(server_messages.SessionStartMessage(session_info))

    def _send_session_end_msg(self, player: "Player") -> None:
        logger.debug("ending session for %s (%s)", player.name, player.player_id)
        player.send_message(
            server_messages.SessionEndMessage(server_messages.SessionEndPayload(player.player_id))
        )

    async def set_metadata(self, metadata: dict[str, str]) -> None:
        """Send a metadata/update message to all players in the group."""
        raise NotImplementedError

    async def set_media_art(self, art_data: bytes, art_format: str) -> None:
        """Send a binary media art message to all players in the group."""
        raise NotImplementedError

    def pause(self) -> None:
        """Pause the playback of all players in this group."""
        raise NotImplementedError

    def resume(self) -> None:
        """Resume playback after a pause."""
        raise NotImplementedError

    def stop(self) -> bool:
        """Stop playback of the group.

        Compared to pause, this also:
        - clears the audio source stream
        - clears metadata
        - and clears all buffers

        Returns false if there was no active stream to stop.
        """
        if self._stream_task is None:
            logger.debug("stop called but no active stream task")
            return False
        logger.debug(
            "Stopping playback for group with players: %s",
            [p.player_id for p in self._players],
        )
        _ = self._stream_task.cancel()
        for player in self._players:
            self._send_session_end_msg(player)
            del self._player_formats[player.player_id]
        self._stream_task = None
        return True

    @property
    def players(self) -> list["Player"]:
        """List of all players that are part of this group."""
        return self._players

    def remove_player(self, player: "Player") -> None:
        """Remove a player from this group."""
        assert player in self._players  # TODO: better error
        logger.debug("removing %s from group with members: %s", player.player_id, self._players)
        self._players.remove(player)
        if self._stream_task is not None:
            # Notify the player that the session ended
            self._send_session_end_msg(player)
            del self._player_formats[player.player_id]
        # Each player needs to be in a group, add it to a new one
        player._group = PlayerGroup(self._server, player)  # noqa: SLF001

    def add_player(self, player: "Player") -> None:
        """Add a player to this group."""
        logger.debug("adding %s to group with members: %s", player.player_id, self._players)
        if player in self._players:
            return
        # Remove it from any existing group first
        player.ungroup()
        if self._stream_task is not None and self._stream_audio_format is not None:
            logger.debug("Joining player %s to current stream", player.player_id)
            # Join it to the current stream
            player_format = self.select_player_format(player, self._stream_audio_format)
            self._player_formats[player.player_id] = player_format
            self._send_session_start_msg(player, player_format)
        self._players.append(player)

    async def _stream_audio(  # noqa: PLR0915 # TODO: split
        self,
        start_time_us: int,
        audio_source: AsyncGenerator[bytes, None],
        audio_format: AudioFormat,
    ) -> None:
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
            if audio_format.bit_depth == 16:
                input_bytes_per_sample = 2
                input_audio_format = "s16"
            elif audio_format.bit_depth == 24:
                input_bytes_per_sample = 3
                input_audio_format = "s24"
            else:
                logger.error("Only 16bit and 24bit audio is supported")
                return

            if audio_format.channels == 1:
                input_audio_layout = "mono"
            elif audio_format.channels == 2:
                input_audio_layout = "stereo"
            else:
                logger.error("Only 1 and 2 channel audio is supported")
                return
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
                            logger.warning(
                                "resampling resulted in %s planes", len(out_frames[0].planes)
                            )
                        header = struct.pack(
                            BINARY_HEADER_FORMAT,
                            BinaryMessageType.PlayAudioChunk.value,
                            chunk_timestamp_us,
                            sample_count,
                        )
                        player.send_message(header + audio_data)
                        duration_of_samples_in_chunk.append(
                            int((sample_count / player_format.sample_rate) * 1_000_000)
                        )

                        player_buffer_capacity_samples = player.info.buffer_capacity // (
                            (player_format.bit_depth // 8) * player_format.channels
                        )
                        # For now the buffer duration is limited by the smallest player
                        buffer_duration_us = min(
                            buffer_duration_us,
                            int(
                                1_000_000
                                * player_buffer_capacity_samples
                                / player_format.sample_rate
                            ),
                        )

                    assert sample_count is not None

                    # TODO: Is mean the correct approach here?
                    # Or just make it based on the input stream
                    chunk_timestamp_us += int(
                        sum(duration_of_samples_in_chunk) / len(duration_of_samples_in_chunk)
                    )

                    time_until_next_chunk = chunk_timestamp_us - int(
                        self._server.loop.time() * 1_000_000
                    )

                    # TODO: I think this may exclude the burst at startup?
                    if time_until_next_chunk > buffer_duration_us:
                        await asyncio.sleep(
                            (time_until_next_chunk - buffer_duration_us) / 1_000_000
                        )
            # TODO: flush buffer
            logger.debug("Audio streaming loop ended")
        except Exception:
            logger.exception("failed to stream audio")
