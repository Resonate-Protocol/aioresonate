"""Tests for player-role format capability filtering."""

from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.models.types import AudioCodec
from aiosendspin.server.roles.player.capabilities import can_encode_format


def test_can_encode_format_accepts_pcm_32_bit() -> None:
    """PCM 32-bit should be considered encodable."""
    fmt = SupportedAudioFormat(codec=AudioCodec.PCM, sample_rate=48_000, bit_depth=32, channels=2)
    assert can_encode_format(fmt)


def test_can_encode_format_accepts_flac_32_bit() -> None:
    """FLAC 32-bit should be considered encodable."""
    fmt = SupportedAudioFormat(codec=AudioCodec.FLAC, sample_rate=48_000, bit_depth=32, channels=2)
    assert can_encode_format(fmt)


def test_can_encode_format_rejects_opus_32_bit() -> None:
    """Opus with 32-bit input should be rejected."""
    fmt = SupportedAudioFormat(codec=AudioCodec.OPUS, sample_rate=48_000, bit_depth=32, channels=2)
    assert not can_encode_format(fmt)


def test_can_encode_format_rejects_invalid_opus_sample_rate() -> None:
    """Opus with unsupported sample rate should be rejected."""
    fmt = SupportedAudioFormat(codec=AudioCodec.OPUS, sample_rate=44_100, bit_depth=16, channels=2)
    assert not can_encode_format(fmt)
