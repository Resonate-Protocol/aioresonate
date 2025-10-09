"""Player implementation and streaming helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aioresonate.models.player import ClientHelloPlayerSupport, PlayerUpdatePayload

from .events import VolumeChangedEvent
from .stream import AudioCodec, AudioFormat

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
