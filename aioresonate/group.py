"""Manages and synchronizes playback for a group of one or more players."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass


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

    async def play_media(
        self, audio_source: AsyncGenerator[bytes, None], audio_format: AudioFormat
    ) -> None:
        """Start playback of a new media stream.

        The library expects uncompressed PCM audio and will handle encoding.
        """
        # TODO: open questions:
        # - how to communicate to the caller what audio_format is preferred,
        #   especially on topology changes
        # - how to sync metadata/media_art with this audio stream?
        raise NotImplementedError

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
        raise NotImplementedError
