"""Shared data structures for the Sendspin client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from aiosendspin.models.types import AudioCodec

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


# Places a static pairing secret may be found, per the spec's pair-method descriptor.
SECRET_LOCATIONS: frozenset[str] = frozenset({"device", "leaflet", "operator"})

# Visual out-channel for a derived dynamic PIN, cleared by a ``None`` call.
type PinDisplay = Callable[[str | None], Awaitable[None]]


class PinSpeaker(Protocol):
    """Speaks a derived dynamic PIN through the device's audio out-channel."""

    def __call__(self, pin: str | None, *, languages: tuple[str, ...]) -> Awaitable[None]:
        """Speak ``pin``, or stop speaking when it is ``None``.

        Return once emission has started rather than awaiting its completion, since the
        pairing exchange is blocked meanwhile. ``languages`` holds the operator's BCP 47
        preferences in descending order, and is empty when the server declared none.
        """


@dataclass(frozen=True, slots=True)
class PairingSupport:
    """Operator wiring for PIN pairing: an operator can perform the pairing gesture here.

    The gesture itself is reported by calling ``SendspinClient.open_pairing_window``.
    Its presence enables offering ``static_pin``, unless ``offer_static_pin`` declines
    it. Either PIN out-channel additionally enables ``dynamic_pin``.
    """

    gesture_prompt: Callable[[bool], Awaitable[None]] | None = None
    """Optional operator prompt: awaited with ``True`` when a gated attempt starts
    waiting for a pairing window, and with ``False`` when the wait ends."""
    pin_display: PinDisplay | None = None
    """Visual out-channel that surfaces a derived dynamic PIN; called with ``None`` when the
    pairing exchange ends (success or failure) so the channel can clear."""
    pin_speaker: PinSpeaker | None = None
    """Spoken out-channel for the derived dynamic PIN, which also receives the operator's
    language preferences."""
    offer_static_pin: bool = True
    """Whether to offer ``static_pin`` at all, for a device with no per-device PIN to hand out."""
    secret_locations: tuple[str, ...] = ()
    """Where the operator finds a configured static secret, from ``SECRET_LOCATIONS``.

    Applies to every static-secret method the client offers.
    """

    def __post_init__(self) -> None:
        """Reject a secret location the descriptor cannot carry."""
        unknown = sorted(set(self.secret_locations) - SECRET_LOCATIONS)
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"unknown secret_locations: {names}")


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
