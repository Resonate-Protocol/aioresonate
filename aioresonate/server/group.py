"""Manages and synchronizes playback for a group of one or more clients."""

import asyncio
import base64
import logging
import uuid
from asyncio import QueueFull, Task
from collections.abc import AsyncGenerator, Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING, cast

import av
from av import logging as av_logging
from PIL import Image

from aioresonate.models import (
    BinaryMessageType,
    RepeatMode,
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
    SessionUpdateMetadata,
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

# The cyclic import is not an issue during runtime, so hide it
# pyright: reportImportCycles=none
if TYPE_CHECKING:
    from .client import Client
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


@dataclass
class Metadata:
    """Metadata for media playback."""

    title: str | None = None
    """Title of the current media."""
    artist: str | None = None
    """Artist of the current media."""
    album_artist: str | None = None
    """Album artist of the current media."""
    album: str | None = None
    """Album of the current media."""
    artwork_url: str | None = None
    """Artwork URL of the current media."""
    year: int | None = None
    """Release year of the current media."""
    track: int | None = None
    """Track number of the current media."""
    track_duration: int | None = None
    """Track duration in seconds."""
    playback_speed: int | None = 1
    """Speed factor, 1.0 is normal speed."""
    repeat: RepeatMode | None = None
    """Current repeat mode."""
    shuffle: bool | None = None
    """Whether shuffle is enabled."""

    # TODO: inject track_progress and timestamp when sending updates?

    def diff_update(self, last: "Metadata | None", timestamp: int) -> SessionUpdateMetadata:
        """Build a SessionUpdateMetadata containing only changed fields compared to last."""
        metadata_update = SessionUpdateMetadata(timestamp=timestamp)

        # Only include fields that have changed since the last metadata update
        if last is None or last.title != self.title:
            metadata_update.title = self.title
        if last is None or last.artist != self.artist:
            metadata_update.artist = self.artist
        if last is None or last.album_artist != self.album_artist:
            metadata_update.album_artist = self.album_artist
        if last is None or last.album != self.album:
            metadata_update.album = self.album
        if last is None or last.artwork_url != self.artwork_url:
            metadata_update.artwork_url = self.artwork_url
        if last is None or last.year != self.year:
            metadata_update.year = self.year
        if last is None or last.track != self.track:
            metadata_update.track = self.track
        if last is None or last.track_duration != self.track_duration:
            metadata_update.track_duration = self.track_duration
        if last is None or last.playback_speed != self.playback_speed:
            metadata_update.playback_speed = self.playback_speed
        if last is None or last.repeat != self.repeat:
            metadata_update.repeat = self.repeat
        if last is None or last.shuffle != self.shuffle:
            metadata_update.shuffle = self.shuffle

        return metadata_update

    @staticmethod
    def cleared_update(timestamp: int) -> SessionUpdateMetadata:
        """Build a SessionUpdateMetadata that clears all metadata fields."""
        metadata_update = SessionUpdateMetadata(timestamp=timestamp)
        metadata_update.title = None
        metadata_update.artist = None
        metadata_update.album_artist = None
        metadata_update.album = None
        metadata_update.artwork_url = None
        metadata_update.year = None
        metadata_update.track = None
        metadata_update.track_duration = None
        metadata_update.playback_speed = None
        metadata_update.repeat = None
        metadata_update.shuffle = None
        return metadata_update

    def snapshot_update(self, timestamp: int) -> SessionUpdateMetadata:
        """Build a SessionUpdateMetadata snapshot with all current values."""
        metadata_update = SessionUpdateMetadata(timestamp=timestamp)
        metadata_update.title = self.title
        metadata_update.artist = self.artist
        metadata_update.album_artist = self.album_artist
        metadata_update.album = self.album
        metadata_update.artwork_url = self.artwork_url
        metadata_update.year = self.year
        metadata_update.track = self.track
        metadata_update.track_duration = self.track_duration
        metadata_update.playback_speed = self.playback_speed
        metadata_update.repeat = self.repeat
        metadata_update.shuffle = self.shuffle
        return metadata_update


class ClientGroup:
    """
    A group of one or more clients for synchronized playback.

    Handles synchronized audio streaming across multiple clients with automatic
    format conversion and buffer management. Every client is always assigned to
    a group to simplify grouping requests.
    """

    _clients: list["Client"]
    """List of all clients in this group."""
    _player_formats: dict[str, AudioFormat]
    """Mapping of client IDs (with the player role) to their selected audio formats."""
    _client_art_formats: dict[str, PictureFormat]
    """Mapping of client IDs (with the metadata role) to their selected artwork formats."""
    _server: "ResonateServer"
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

    def __init__(self, server: "ResonateServer", *args: "Client") -> None:
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
        logger.debug(
            "ClientGroup initialized with %d client(s): %s",
            len(self._clients),
            [type(c).__name__ for c in self._clients],
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
            # Wait a bit to allow clients to process the stream end
            await asyncio.sleep(0.5)
        # TODO: open questions:
        # - how to communicate to the caller what audio_format is preferred,
        #   especially on topology changes
        # - how to sync metadata/media_art with this audio stream?

        self._stream_audio_format = audio_stream_format
        self._preferred_stream_codec = preferred_stream_codec

        for client in self._clients:
            if client.check_role(Roles.PLAYER):
                logger.debug("Selecting format for player %s", client.client_id)
                player_format = self.determine_player_format(
                    client, audio_stream_format, preferred_stream_codec
                )
                self._player_formats[client.client_id] = player_format
                logger.debug(
                    "Sending stream start to player %s with format %s",
                    client.client_id,
                    player_format,
                )
            else:
                logger.debug("Sending stream start to client %s", client.client_id)
                player_format = None
            self._send_stream_start_msg(client, player_format)

        self._stream_task = self._server.loop.create_task(
            self._stream_audio(
                int(self._server.loop.time() * 1_000_000) + INITIAL_PLAYBACK_DELAY_US,
                audio_stream,
                audio_stream_format,
            )
        )

        self._current_state = PlaybackStateType.PLAYING
        self._signal_event(GroupStateChangedEvent(PlaybackStateType.PLAYING))

    def determine_player_format(
        self,
        player: "Client",
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
        # TODO: move this to client.py instead
        player_info = player.info

        # Determine optimal sample rate
        sample_rate = source_format.sample_rate
        if (
            player_info.player_support
            and sample_rate not in player_info.player_support.support_sample_rates
        ):
            # Prefer lower rates that are closest to source, fallback to minimum
            lower_rates = [
                r for r in player_info.player_support.support_sample_rates if r < sample_rate
            ]
            sample_rate = (
                max(lower_rates)
                if lower_rates
                else min(player_info.player_support.support_sample_rates)
            )
            logger.debug("Adjusted sample_rate for player %s: %s", player.client_id, sample_rate)

        # Determine optimal bit depth
        bit_depth = source_format.bit_depth
        if (
            player_info.player_support
            and bit_depth not in player_info.player_support.support_bit_depth
        ):
            if 16 in player_info.player_support.support_bit_depth:
                bit_depth = 16
            else:
                raise NotImplementedError("Only 16bit is supported for now")
            logger.debug("Adjusted bit_depth for player %s: %s", player.client_id, bit_depth)

        # Determine optimal channel count
        channels = source_format.channels
        if (
            player_info.player_support
            and channels not in player_info.player_support.support_channels
        ):
            # Prefer stereo, then mono
            if 2 in player_info.player_support.support_channels:
                channels = 2
            elif 1 in player_info.player_support.support_channels:
                channels = 1
            else:
                raise NotImplementedError("Only mono and stereo are supported")
            logger.debug("Adjusted channels for player %s: %s", player.client_id, channels)

        # Determine optimal codec with fallback chain
        codec_fallbacks = [preferred_codec, AudioCodec.FLAC, AudioCodec.OPUS, AudioCodec.PCM]
        codec = None
        for candidate_codec in codec_fallbacks:
            if (
                player_info.player_support
                and candidate_codec.value in player_info.player_support.support_codecs
            ):
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
                        if (
                            condition
                            and player_info.player_support
                            and candidate_rate in player_info.player_support.support_sample_rates
                        ):
                            opus_sample_rate = candidate_rate
                            break

                    if opus_sample_rate is None:
                        logger.error(
                            "Player %s does not support any Opus sample rates, trying next codec",
                            player.client_id,
                        )
                        continue  # Try next codec in fallback chain

                    # Opus is viable, adjust sample rate and use it
                    if sample_rate != opus_sample_rate:
                        logger.debug(
                            "Adjusted sample_rate for Opus on player %s: %s -> %s",
                            player.client_id,
                            sample_rate,
                            opus_sample_rate,
                        )
                    sample_rate = opus_sample_rate

                codec = candidate_codec
                break

        if codec is None:
            raise ValueError(f"Player {player.client_id} does not support any known codec")

        if codec != preferred_codec:
            logger.info(
                "Falling back from preferred codec %s to %s for player %s",
                preferred_codec,
                codec,
                player.client_id,
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
        self, client: "Client", audio_format: AudioFormat | None = None
    ) -> None:
        """Send a stream start message to a client with the specified audio format for players."""
        logger.debug(
            "_send_stream_start_msg: client=%s, format=%s",
            client.client_id,
            audio_format,
        )
        if client.check_role(Roles.PLAYER):
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
        logger.debug("Sending stream start message to client %s: %s", client.client_id, stream_info)
        client.send_message(StreamStartMessage(stream_info))

    def _send_stream_end_msg(self, client: "Client") -> None:
        """Send a stream end message to a client to stop playback."""
        logger.debug("ending stream for %s (%s)", client.name, client.client_id)
        # Lifetime of album artwork is bound to the stream
        _ = self._client_art_formats.pop(client.client_id, None)
        client.send_message(StreamEndMessage())

    def stop(self) -> bool:
        """
        Stop playback for the group and clean up resources.

        Compared to pause(), this also:
        - Cancels the audio streaming task
        - Sends stream end messages to all clients
        - Clears all buffers and format mappings
        - Cleans up all audio encoders

        Returns:
            bool: True if an active stream was stopped, False if no stream was active.
        """
        if self._stream_task is None:
            logger.debug("stop called but no active stream task")
            return False
        logger.debug(
            "Stopping playback for group with clients: %s",
            [c.client_id for c in self._clients],
        )
        _ = self._stream_task.cancel()  # Don't care about cancellation result
        for client in self._clients:
            self._send_stream_end_msg(client)
            if client.check_role(Roles.PLAYER):
                del self._player_formats[client.client_id]

        self._audio_encoders.clear()
        self._audio_headers.clear()
        self._stream_task = None
        self._current_media_art = None

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

    def _send_media_art_to_client(self, client: "Client", image: Image.Image) -> None:
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
    def clients(self) -> list["Client"]:
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

    def remove_client(self, client: "Client") -> None:
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
                    del self._player_formats[client.client_id]
        if not self._clients:
            # Emit event for group deletion, no clients left
            self._signal_event(GroupDeletedEvent())
        else:
            # Emit event for client removal
            self._signal_event(GroupMemberRemovedEvent(client.client_id))
        # Each client needs to be in a group, add it to a new one
        client._set_group(ClientGroup(self._server, client))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    def add_client(self, client: "Client") -> None:
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
            # Join it to the current stream
            if client.check_role(Roles.PLAYER):
                player_format = self.determine_player_format(
                    client, self._stream_audio_format, self._preferred_stream_codec
                )
                self._player_formats[client.client_id] = player_format
            else:
                player_format = None
            self._send_stream_start_msg(client, player_format)

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
        player: "Client",
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
        player: "Client",
        request: "StreamRequestFormatPayload",
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

    def _update_player_format(self, player: "Client") -> None:
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

                    for player in self._clients:
                        if not player.check_role(Roles.PLAYER):
                            continue

                        self._update_player_format(player)
                        player_format = self._player_formats[player.client_id]

                        try:
                            sample_count, duration_us = self._resample_and_encode_to_player(
                                player, player_format, in_frame, resamplers, chunk_timestamp_us
                            )
                            duration_of_samples_in_chunk.append(duration_us)
                        except QueueFull:
                            logger.warning(
                                "Error sending audio chunk to %s, disconnecting player",
                                player.client_id,
                            )
                            await player.disconnect()

                        assert player.info.player_support is not None  # for type checking
                        # Calculate buffer duration for this player
                        player_buffer_capacity_samples = (
                            player.info.player_support.buffer_capacity
                        ) // ((player_format.bit_depth // 8) * player_format.channels)
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
            # TODO: Wait until all audio should be played, otherwise we cut off the audio
            self.stop()
