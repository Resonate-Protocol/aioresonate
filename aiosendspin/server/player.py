"""Player implementation and streaming helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.models.core import ServerCommandMessage, ServerCommandPayload
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    PlayerCommandPayload,
    PlayerStatePayload,
)
from aiosendspin.models.types import PlayerCommand

from .events import VolumeChangedEvent
from .stream import AudioCodec, AudioFormat

if TYPE_CHECKING:
    from .client import SendspinClient


class PlayerClient:
    """Player."""

    client: SendspinClient
    _volume: int = 100
    _muted: bool = False

    def __init__(self, client: SendspinClient) -> None:
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
        if not self.support or PlayerCommand.VOLUME not in self.support.supported_commands:
            self._logger.warning("Player does not support the 'volume' command")
            return

        self._logger.debug("Setting volume from %d to %d", self._volume, volume)
        self.client.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(
                    player=PlayerCommandPayload(
                        command=PlayerCommand.VOLUME,
                        volume=volume,
                    )
                )
            )
        )

    def mute(self) -> None:
        """Mute this player."""
        if not self.support or PlayerCommand.MUTE not in self.support.supported_commands:
            self._logger.warning("Player does not support the 'mute' command")
            return

        self._logger.debug("Muting player")
        self.client.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(
                    player=PlayerCommandPayload(
                        command=PlayerCommand.MUTE,
                        mute=True,
                    )
                )
            )
        )

    def unmute(self) -> None:
        """Unmute this player."""
        if not self.support or PlayerCommand.MUTE not in self.support.supported_commands:
            self._logger.warning("Player does not support the 'mute' command")
            return

        self._logger.debug("Unmuting player")
        self.client.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(
                    player=PlayerCommandPayload(
                        command=PlayerCommand.MUTE,
                        mute=False,
                    )
                )
            )
        )

    def handle_player_update(self, state: PlayerStatePayload) -> None:
        """Update internal mute/volume state from client report and emit event."""
        changed = False

        if state.volume is not None:
            if not self.support or PlayerCommand.VOLUME not in self.support.supported_commands:
                self._logger.warning(
                    "Client sent volume field without declaring 'volume' in supported_commands"
                )
            elif self._volume != state.volume:
                self._volume = state.volume
                changed = True

        if state.muted is not None:
            if not self.support or PlayerCommand.MUTE not in self.support.supported_commands:
                self._logger.warning(
                    "Client sent muted field without declaring 'mute' in supported_commands"
                )
            elif self._muted != state.muted:
                self._muted = state.muted
                changed = True

        if changed:
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
        Properly handles per-codec capabilities by filtering formats by codec first.

        Args:
            source_format: The source audio format to match against.

        Returns:
            AudioFormat: The optimal format for this client.
        """
        support = self.support
        if not support or not support.supported_formats:
            raise ValueError(f"Client {self.client.client_id} has no supported formats")

        # Get available codecs
        support_codecs = {fmt.codec for fmt in support.supported_formats}

        # Determine optimal codec with fallback chain
        codec_fallbacks = [AudioCodec.FLAC, AudioCodec.OPUS, AudioCodec.PCM]
        selected_codec = None

        for candidate_codec in codec_fallbacks:
            if candidate_codec in support_codecs:
                selected_codec = candidate_codec
                break

        if selected_codec is None:
            raise ValueError(f"Client {self.client.client_id} does not support any known codec")

        # Filter supported_formats by the selected codec to get per-codec capabilities
        codec_formats = [fmt for fmt in support.supported_formats if fmt.codec == selected_codec]

        # Get supported values for this specific codec
        codec_sample_rates = {fmt.sample_rate for fmt in codec_formats}
        codec_bit_depths = {fmt.bit_depth for fmt in codec_formats}
        codec_channels = {fmt.channels for fmt in codec_formats}

        # Determine optimal sample rate for this codec
        sample_rate = source_format.sample_rate

        # Special handling for Opus - use specific sample rates
        if selected_codec == AudioCodec.OPUS:
            opus_rate_candidates = [
                (8000, sample_rate <= 8000),
                (12000, sample_rate <= 12000),
                (16000, sample_rate <= 16000),
                (24000, sample_rate <= 24000),
                (48000, True),  # Default fallback
            ]

            opus_sample_rate = None
            for candidate_rate, condition in opus_rate_candidates:
                if condition and candidate_rate in codec_sample_rates:
                    opus_sample_rate = candidate_rate
                    break

            if opus_sample_rate is None:
                raise ValueError(
                    f"Client {self.client.client_id} does not support any Opus sample rates"
                )

            if sample_rate != opus_sample_rate:
                self._logger.debug(
                    "Adjusted sample_rate for Opus on client %s: %s -> %s",
                    self.client.client_id,
                    sample_rate,
                    opus_sample_rate,
                )
            sample_rate = opus_sample_rate
        elif sample_rate not in codec_sample_rates:
            # For other codecs, prefer lower rates closest to source
            lower_rates = [r for r in codec_sample_rates if r < sample_rate]
            sample_rate = max(lower_rates) if lower_rates else min(codec_sample_rates)
            self._logger.debug(
                "Adjusted sample_rate for client %s: %s", self.client.client_id, sample_rate
            )

        # Determine optimal bit depth for this codec
        bit_depth = source_format.bit_depth
        if bit_depth not in codec_bit_depths:
            if 16 in codec_bit_depths:
                bit_depth = 16
            else:
                raise NotImplementedError("Only 16bit is supported for now")
            self._logger.debug(
                "Adjusted bit_depth for client %s: %s", self.client.client_id, bit_depth
            )

        # Determine optimal channel count for this codec
        channels = source_format.channels
        if channels not in codec_channels:
            # Prefer stereo, then mono
            if 2 in codec_channels:
                channels = 2
            elif 1 in codec_channels:
                channels = 1
            else:
                raise NotImplementedError("Only mono and stereo are supported")
            self._logger.debug(
                "Adjusted channels for client %s: %s", self.client.client_id, channels
            )

        return AudioFormat(sample_rate, bit_depth, channels, selected_codec)
