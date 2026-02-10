"""Tests for audio format helper utilities."""

import sys

import pytest

from aiosendspin.server.audio import AudioFormat, _convert_s32_to_s24, _resolve_audio_format


def test_resolve_audio_format_24_bit_uses_s32_in_pyav() -> None:
    """24-bit wire format should map to s32 for PyAV processing."""
    wire_bytes, av_format, layout, av_bytes = _resolve_audio_format(
        AudioFormat(sample_rate=48_000, bit_depth=24, channels=2)
    )
    assert wire_bytes == 3
    assert av_format == "s32"
    assert layout == "stereo"
    assert av_bytes == 4


def test_resolve_audio_format_32_bit_is_supported() -> None:
    """32-bit PCM should be supported by resolver."""
    wire_bytes, av_format, layout, av_bytes = _resolve_audio_format(
        AudioFormat(sample_rate=44_100, bit_depth=32, channels=1)
    )
    assert wire_bytes == 4
    assert av_format == "s32"
    assert layout == "mono"
    assert av_bytes == 4


def test_convert_s32_to_s24_drops_least_significant_byte() -> None:
    """s32->s24 conversion should drop the LSB byte per sample."""
    # Two s32 samples with distinct bytes to verify byte order behavior.
    samples = bytes([0x01, 0x11, 0x21, 0x31, 0x02, 0x12, 0x22, 0x32])
    converted = _convert_s32_to_s24(samples)
    if sys.byteorder == "little":
        assert converted == bytes([0x11, 0x21, 0x31, 0x12, 0x22, 0x32])
    else:
        assert converted == bytes([0x01, 0x11, 0x21, 0x02, 0x12, 0x22])


def test_convert_s32_to_s24_rejects_invalid_length() -> None:
    """Invalid byte lengths must be rejected."""
    with pytest.raises(ValueError, match="multiple of 4"):
        _convert_s32_to_s24(b"\x00\x01\x02")
