"""Manages and synchronizes playback for a group of one or more players."""

import asyncio
from asyncio import Task
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import models

if TYPE_CHECKING:
    from .instance import PlayerInstance
    from .server import ResonateServer

INITIAL_PLAYBACK_DELAY_US = 1_000_000
BUFFER_DURATION_US = 2_000_000
MAX_PENDING_MSG = 512
TARGET_CHUNK_DURATION_MS = 25
STREAM_CODEC = "pcm"
STREAM_SAMPLE_RATE = 48000
STREAM_CHANNELS = 2
STREAM_BIT_DEPTH = 16
STREAM_SAMPLE_SIZE = STREAM_CHANNELS * (STREAM_BIT_DEPTH // 8)
TARGET_CHUNK_BYTES = STREAM_SAMPLE_SIZE * STREAM_SAMPLE_RATE // 1000 * TARGET_CHUNK_DURATION_MS
TARGET_CHUNK_SAMPLES = STREAM_SAMPLE_RATE // 1000 * TARGET_CHUNK_DURATION_MS


@dataclass
class AudioFormat:
    """LPCM audio format."""

    sample_rate: int
    bit_depth: int
    channels: int


class PlayerGroup:
    """A group of one or more players."""

    # In this implementation, every player is always assigned to a group.
    # This simplifies grouping requests initiated by the player.

    _players: list["PlayerInstance"]
    _server: "ResonateServer"
    _stream_task: Task[None] | None = None

    def __init__(self, server: "ResonateServer", *args: "PlayerInstance") -> None:
        """Do not call this constructor."""
        self._server = server
        self._players = list(args)

    async def play_media(
        self, audio_source: AsyncGenerator[bytes, None], audio_format: AudioFormat
    ) -> None:
        """Start playback of a new media stream.

        The library expects uncompressed PCM audio and will handle encoding.
        """
        self.stop()
        # TODO: open questions:
        # - how to communicate to the caller what audio_format is preferred,
        #   especially on topology changes
        # - how to sync metadata/media_art with this audio stream?
        # TODO: port _stream_audio

        # TODO: Stop any prior stream

        # TODO: dynamic session info
        session_info = models.SessionInfo(
            session_id=f"mass-session-{int(self._server.loop.time())}",
            codec="pcm",
            sample_rate=audio_format.sample_rate,
            channels=audio_format.channels,
            bit_depth=audio_format.bit_depth,
            now=int(self._server.loop.time() * 1_000_000),
            codec_header=None,
        )

        for player in self._players:
            player.send_message(models.SessionStartMessage(session_info))

        self._stream_task = self._server.loop.create_task(
            self._stream_audio(
                session_info.now + INITIAL_PLAYBACK_DELAY_US, audio_source, audio_format
            )
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

    def stop(self) -> None:
        """Stop playback of the group.

        Compared to pause, this also:
        - clears the audio source stream
        - clears metadata
        - and clears all buffers
        """
        if self._stream_task is None:
            return
        _ = self._stream_task.cancel()
        for player in self._players:
            player.send_message(
                models.SessionEndMessage(models.SessionEndPayload(player.player_id))
            )
        self._stream_task = None

    async def _stream_audio(
        self,
        start_time_us: int,
        audio_source: AsyncGenerator[bytes, None],
        audio_format: AudioFormat,  # noqa: ARG002
    ) -> None:
        samples_sent = 0

        chunk_timestamp_us = start_time_us

        async for chunk in audio_source:
            chunk_pos = 0
            while True:
                if chunk_pos >= len(chunk):
                    break
                c = chunk[
                    chunk_pos : chunk_pos + TARGET_CHUNK_BYTES
                ]  # TODO: maybe that's not so efficient?

                chunk_pos += len(c)
                samples_in_chunk = len(c) // STREAM_SAMPLE_SIZE
                samples_sent += samples_in_chunk

                for player in self._players:
                    player.send_audio_chunk(
                        timestamp_us=chunk_timestamp_us,
                        sample_count=samples_in_chunk,
                        audio_data=c,
                    )

                duration_of_samples_in_chunk = int(
                    samples_in_chunk / STREAM_SAMPLE_RATE * 1_000_000
                )
                chunk_timestamp_us += duration_of_samples_in_chunk

                time_until_next_chunk = chunk_timestamp_us - int(
                    self._server.loop.time() * 1_000_000
                )

                if time_until_next_chunk > BUFFER_DURATION_US:
                    await asyncio.sleep((time_until_next_chunk - BUFFER_DURATION_US) / 1_000_000)
