"""Push-based audio streaming API."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import UUID

from aiosendspin.server.channels import MAIN_CHANNEL

if TYPE_CHECKING:
    from aiosendspin.server.channels import ChannelRouter
    from aiosendspin.server.player_state import PlayerRegistry
    from aiosendspin.server.stream import AudioFormat


class PushStream:
    """
    Push-based audio streaming API.

    This class provides a push-based interface for streaming audio to players.
    Audio is prepared via prepare_audio(), then committed and sent via commit_audio().
    Backpressure is handled via wait_for_buffer_space() and timeline shifting.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        player_registry: PlayerRegistry,
        channel_router: ChannelRouter,
    ) -> None:
        """
        Create a new PushStream.

        Args:
            loop: Event loop for timing and async operations.
            player_registry: Registry for player state management.
            channel_router: Router for channel assignments.
        """
        self._loop = loop
        self._player_registry = player_registry
        self._channel_router = channel_router
        self._is_stopped = False

    @property
    def is_stopped(self) -> bool:
        """Whether this stream has been stopped."""
        return self._is_stopped

    def prepare_audio(
        self,
        pcm: bytes,
        audio_format: AudioFormat,
        *,
        channel_id: UUID = MAIN_CHANNEL,
    ) -> None:
        """
        Prepare PCM audio for the next commit.

        This is a synchronous method that stores the PCM data for encoding
        during commit_audio(). Calling twice for the same channel replaces
        the previous data (does not append).

        Args:
            pcm: Raw PCM audio data.
            audio_format: Format of the PCM data.
            channel_id: Channel to prepare audio for (default: MAIN_CHANNEL).
        """
        # Stub implementation - will be filled in Task 6

    async def commit_audio(self) -> int:
        """
        Encode and send all prepared audio to players.

        This is an asynchronous method that:
        1. Encodes prepared PCM for each required format
        2. Applies backpressure (timeline shift if needed)
        3. Assigns timestamps to encoded chunks
        4. Sends chunks to connected players

        Returns:
            The play_start_us timestamp for this commit.
        """
        # Stub implementation - will be filled in Task 8
        return 0

    async def wait_for_buffer_space(self) -> None:
        """
        Wait until there is buffer space available on players.

        This is useful for throttling audio production to match
        player consumption rates.
        """
        # Stub implementation - will be filled in Task 13

    def stop(self) -> None:
        """
        Stop the stream.

        After calling stop(), commit_audio() will raise StreamStoppedError.
        """
        self._is_stopped = True

    def clear(self) -> None:
        """
        Clear all pending audio and reset timing.

        This is used for seek operations where buffered audio is discarded.
        """
        # Stub implementation - will be filled in Task 12
