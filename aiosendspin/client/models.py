"""Shared data structures for the Sendspin client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiosendspin.models.types import AudioCodec

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass(frozen=True, slots=True)
class PairingSupport:
    """Operator wiring for PIN pairing: an operator can perform the pairing gesture here.

    The gesture itself is reported by calling ``SendspinClient.open_pairing_window``.
    Its presence enables offering ``static_pin``; ``pin_display`` additionally
    enables ``dynamic_pin``.
    """

    gesture_prompt: Callable[[bool], Awaitable[None]] | None = None
    """Optional operator prompt: awaited with ``True`` when a gated attempt starts
    waiting for a pairing window, and with ``False`` when the wait ends."""
    pin_display: Callable[[str | None], Awaitable[None]] | None = None
    """Out-channel that surfaces a derived dynamic PIN; called with ``None`` when the
    pairing exchange ends (success or failure) so the channel can clear."""


@dataclass(slots=True)
class PCMFormat:
    """PCM audio format description."""

    sample_rate: int
    """Sample rate in Hz (e.g., 48000, 44100)."""
    channels: int
    """Number of audio channels (1=mono, 2=stereo)."""
    bit_depth: int
    """Bits per sample (e.g., 16, 24, 32)."""

    def __post_init__(self) -> None:
        """Validate the provided PCM audio format."""
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if self.bit_depth not in (16, 24, 32):
            raise ValueError("bit_depth must be 16, 24, or 32")

    @property
    def frame_size(self) -> int:
        """Return bytes per PCM frame."""
        return self.channels * (self.bit_depth // 8)


@dataclass(slots=True)
class AudioFormat:
    """Audio format description including codec type."""

    codec: AudioCodec
    """Audio codec used for encoding."""
    pcm_format: PCMFormat
    """Format of decoded PCM audio."""
    codec_header: bytes | None = None
    """Optional codec-specific header bytes (e.g., FLAC streaminfo)."""


@dataclass(slots=True)
class ServerInfo:
    """Information about the connected server."""

    server_id: str
    name: str
