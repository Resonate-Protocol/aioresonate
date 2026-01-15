"""Player implementation and streaming helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.models.core import ServerCommandMessage, ServerCommandPayload
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    PlayerCommandPayload,
    PlayerStatePayload,
    SupportedAudioFormat,
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

    # Opus encoder constraints
    _OPUS_SAMPLE_RATES = frozenset({8000, 12000, 16000, 24000, 48000})
    _OPUS_BIT_DEPTHS = frozenset({16})  # Opus only accepts s16 input
    _FLAC_BIT_DEPTHS = frozenset({16, 24, 32})
    _PCM_BIT_DEPTHS = frozenset({16, 24, 32})

    def _is_format_supported_by_server(self, fmt: SupportedAudioFormat) -> bool:
        """Check if the server can encode to this format."""
        if fmt.sample_rate <= 0:
            return False
        if fmt.codec == AudioCodec.OPUS:
            if fmt.sample_rate not in self._OPUS_SAMPLE_RATES:
                return False
            if fmt.bit_depth not in self._OPUS_BIT_DEPTHS:
                return False
        elif fmt.codec == AudioCodec.FLAC:
            if fmt.bit_depth not in self._FLAC_BIT_DEPTHS:
                return False
        elif fmt.codec == AudioCodec.PCM:
            if fmt.bit_depth not in self._PCM_BIT_DEPTHS:
                return False
        return fmt.channels in (1, 2)

    def determine_optimal_format(self, source_format: AudioFormat) -> AudioFormat:  # noqa: ARG002
        """
        Determine the optimal audio format for this client.

        Uses the client's preferred format (first in supported_formats per spec),
        falling back to subsequent formats if the server can't encode the preferred one.
        """
        support = self.support
        if not support or not support.supported_formats:
            raise ValueError(f"Client {self.client.client_id} has no supported formats")

        # Use client's preferred format, falling back if server can't encode it
        for fmt in support.supported_formats:
            if self._is_format_supported_by_server(fmt):
                return AudioFormat(fmt.sample_rate, fmt.bit_depth, fmt.channels, fmt.codec)

        raise ValueError(f"Client {self.client.client_id} has no formats the server can encode")
