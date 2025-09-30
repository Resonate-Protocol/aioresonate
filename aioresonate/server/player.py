"""Player implementation and streaming helpers."""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, cast

import av
from av.logging import Capture

from aioresonate.models import BinaryMessageType, pack_binary_header_raw
from aioresonate.models.core import StreamStartMessage, StreamStartPayload
from aioresonate.models.player import (
    ClientHelloPlayerSupport,
    PlayerUpdatePayload,
    StreamStartPlayer,
)
from aioresonate.models.types import Roles

from .client import VolumeChangedEvent
from .group import AudioCodec, AudioFormat
from .streaming import (
    _BufferTracker,
    _DirectStreamContext,
    _samples_to_microseconds,
    build_flac_stream_header,
)

if TYPE_CHECKING:
    from .client import ResonateClient


class PlayerClient:
    """Player."""

    client: ResonateClient
    _volume: int = 100
    _muted: bool = False

    def __init__(self, client: ResonateClient) -> None:
        """Initialize player wrapper for a client."""
        self.client = client
        self._logger = client._logger.getChild("player")  # noqa: SLF001

    @property
    def support(self) -> ClientHelloPlayerSupport | None:
        """Return player capabilities advertised in the hello payload."""
        return self.client.info.player_support

    @property
    def muted(self) -> bool:
        """Mute state of this player."""
        return self._muted

    @property
    def volume(self) -> int:
        """Volume of this player."""
        return self._volume

    def set_volume(self, volume: int) -> None:
        """Set the volume of this player."""
        self._logger.debug("Setting volume from %d to %d", self._volume, volume)
        self._logger.error("NOT SUPPORTED BY SPEC YET")

    def mute(self) -> None:
        """Mute this player."""
        self._logger.debug("Muting player")
        self._logger.error("NOT SUPPORTED BY SPEC YET")

    def unmute(self) -> None:
        """Unmute this player."""
        self._logger.debug("Unmuting player")
        self._logger.error("NOT SUPPORTED BY SPEC YET")

    def handle_player_update(self, state: PlayerUpdatePayload) -> None:
        """Update internal mute/volume state from client report and emit event."""
        self._logger.debug("Received player state: volume=%d, muted=%s", state.volume, state.muted)
        if self._muted != state.muted or self._volume != state.volume:
            self._volume = state.volume
            self._muted = state.muted
            self.client._signal_event(  # noqa: SLF001
                VolumeChangedEvent(volume=self._volume, muted=self._muted)
            )

    def determine_optimal_format(
        self,
        source_format: AudioFormat,
    ) -> AudioFormat:
        """
        Determine the optimal audio format for this client given a source format.

        Prefers higher quality within the client's capabilities and falls back gracefully.

        Args:
            source_format: The source audio format to match against.
            preferred_codec: Preferred audio codec (e.g., Opus). Falls back when unsupported.

        Returns:
            AudioFormat: The optimal format for this client.
        """
        support = self.support

        # Determine optimal sample rate
        sample_rate = source_format.sample_rate
        if support and sample_rate not in support.support_sample_rates:
            # Prefer lower rates that are closest to source, fallback to minimum
            lower_rates = [r for r in support.support_sample_rates if r < sample_rate]
            sample_rate = max(lower_rates) if lower_rates else min(support.support_sample_rates)
            self._logger.debug(
                "Adjusted sample_rate for client %s: %s", self.client.client_id, sample_rate
            )

        # Determine optimal bit depth
        bit_depth = source_format.bit_depth
        if support and bit_depth not in support.support_bit_depth:
            if 16 in support.support_bit_depth:
                bit_depth = 16
            else:
                raise NotImplementedError("Only 16bit is supported for now")
            self._logger.debug(
                "Adjusted bit_depth for client %s: %s", self.client.client_id, bit_depth
            )

        # Determine optimal channel count
        channels = source_format.channels
        if support and channels not in support.support_channels:
            # Prefer stereo, then mono
            if 2 in support.support_channels:
                channels = 2
            elif 1 in support.support_channels:
                channels = 1
            else:
                raise NotImplementedError("Only mono and stereo are supported")
            self._logger.debug(
                "Adjusted channels for client %s: %s", self.client.client_id, channels
            )

        # Determine optimal codec with fallback chain
        codec_fallbacks = [AudioCodec.FLAC, AudioCodec.OPUS, AudioCodec.PCM]
        codec = None
        for candidate_codec in codec_fallbacks:
            if support and candidate_codec.value in support.support_codecs:
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
                        if condition and support and candidate_rate in support.support_sample_rates:
                            opus_sample_rate = candidate_rate
                            break

                    if opus_sample_rate is None:
                        self._logger.error(
                            "Client %s does not support any Opus sample rates, trying next codec",
                            self.client.client_id,
                        )
                        continue  # Try next codec in fallback chain

                    # Opus is viable, adjust sample rate and use it
                    if sample_rate != opus_sample_rate:
                        self._logger.debug(
                            "Adjusted sample_rate for Opus on client %s: %s -> %s",
                            self.client.client_id,
                            sample_rate,
                            opus_sample_rate,
                        )
                    sample_rate = opus_sample_rate

                codec = candidate_codec
                break

        if codec is None:
            raise ValueError(f"Client {self.client.client_id} does not support any known codec")

        # FLAC and PCM support any sample rate, no adjustment needed
        return AudioFormat(sample_rate, bit_depth, channels, codec)

    def _build_encoder(
        self,
        audio_format: AudioFormat,
        input_audio_layout: str,
        input_audio_format: str,
    ) -> tuple[av.AudioCodecContext | None, str | None, int]:
        """
        Create and open an encoder if needed.

        Returns:
            tuple of (encoder, header_b64, samples_per_chunk).
            For PCM, returns (None, None, default_samples_per_chunk).
        """
        if audio_format.codec == AudioCodec.PCM:
            # Default to ~25ms chunks for PCM
            samples_per_chunk = int(audio_format.sample_rate * 0.025)
            return None, None, samples_per_chunk

        encoder = cast(
            "av.AudioCodecContext", av.AudioCodecContext.create(audio_format.codec.value, "w")
        )
        encoder.sample_rate = audio_format.sample_rate
        encoder.layout = input_audio_layout
        encoder.format = input_audio_format
        if audio_format.codec == AudioCodec.FLAC:
            # Only default compression level for now
            encoder.options = {"compression_level": "5"}
        with Capture() as logs:
            encoder.open()
        for log in logs:
            self._logger.debug("Opening AudioCodecContext log from av: %s", log)
        header = bytes(encoder.extradata) if encoder.extradata else b""
        # For FLAC, we need to construct a proper FLAC stream header ourselves
        # since ffmpeg only provides the StreamInfo metadata block in extradata:
        # See https://datatracker.ietf.org/doc/rfc9639/ Section 8.1
        if audio_format.codec == AudioCodec.FLAC and header:
            header = build_flac_stream_header(header)
        codec_header_b64 = base64.b64encode(header).decode()
        samples_per_chunk = (
            int(encoder.frame_size) if encoder.frame_size else int(audio_format.sample_rate * 0.025)
        )
        return encoder, codec_header_b64, samples_per_chunk

    def _send_pcm(self, chunk: bytes, timestamp_us: int) -> None:
        header = pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, timestamp_us)
        self.client.send_message(header + chunk)

    async def _play_media_direct(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        audio_format: AudioFormat,
        *,
        play_start_time_us: int,
        stream_start_time_us: int = 0,
    ) -> int:
        """Stream pre-formatted PCM for internal coordination by groups and sessions."""
        # Internal check; suppress style warning for accessing private API
        self.client._ensure_role(Roles.PLAYER)  # noqa: SLF001
        support = self.support
        assert support is not None, "Player support info required"

        # Validate input format
        if audio_format.bit_depth != 16:
            raise ValueError("Only 16 bit PCM is supported")
        if audio_format.channels not in (1, 2):
            raise ValueError("Only mono or stereo are supported")

        bytes_per_sample = 2 if audio_format.bit_depth == 16 else 3
        input_audio_format = "s16" if audio_format.bit_depth == 16 else "s24"
        input_audio_layout = "stereo" if audio_format.channels == 2 else "mono"

        # Setup encoder if needed and prepare codec header
        encoder, codec_header_b64, samples_per_chunk = self._build_encoder(
            audio_format, input_audio_layout, input_audio_format
        )

        # Send stream start to this client
        player_stream_info = StreamStartPlayer(
            codec=audio_format.codec.value,
            sample_rate=audio_format.sample_rate,
            channels=audio_format.channels,
            bit_depth=audio_format.bit_depth,
            codec_header=codec_header_b64,
        )
        self.client.send_message(StreamStartMessage(StreamStartPayload(player=player_stream_info)))

        frame_stride_bytes = audio_format.channels * bytes_per_sample
        buffer_capacity_bytes = support.buffer_capacity
        buffer_tracker = _BufferTracker(
            loop=self.client._server.loop,  # noqa: SLF001
            logger=self._logger,
            client_id=self.client.client_id,
            capacity_bytes=buffer_capacity_bytes,
        )
        context = _DirectStreamContext(
            client=self.client,
            audio_format=audio_format,
            input_audio_format=input_audio_format,
            input_audio_layout=input_audio_layout,
            samples_per_chunk=samples_per_chunk,
            buffer_tracker=buffer_tracker,
            play_start_time_us=play_start_time_us,
            encoder=encoder,
            frame_stride_bytes=frame_stride_bytes,
            send_pcm=self._send_pcm,
        )

        # Skip initial offset within the stream
        bytes_to_skip_total = (
            int((stream_start_time_us * audio_format.sample_rate) / 1_000_000) * frame_stride_bytes
        )
        pending = await context._skip_initial_bytes(  # noqa: SLF001
            audio_stream, bytes_to_skip_total
        )
        if bytes_to_skip_total > 0 and not pending:
            return play_start_time_us

        input_buffer = pending

        async for chunk in audio_stream:
            if not chunk:
                continue
            input_buffer.extend(chunk)
            await context.drain_ready_chunks(input_buffer, force_flush=False)

        await context.drain_ready_chunks(input_buffer, force_flush=True)
        await context.flush_encoder()

        buffer_tracker.prune_consumed()
        end_timestamp_us = play_start_time_us + _samples_to_microseconds(
            context.samples_sent_total,
            audio_format.sample_rate,
        )

        self._logger.debug(
            "Completed direct stream for client %s: pcm=%sB compressed=%sB max_buffer=%s/%sB",
            self.client.client_id,
            context.pcm_bytes_consumed,
            context.compressed_bytes_sent,
            buffer_tracker.max_usage_bytes,
            buffer_capacity_bytes,
        )

        return end_timestamp_us
