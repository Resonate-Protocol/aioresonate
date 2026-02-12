"""Tests for audio format helper utilities."""

import sys

import pytest

from aiosendspin.server import audio as audio_module
from aiosendspin.server.audio import AudioFormat, _convert_s32_to_s24

S32_SAMPLES = bytes([0x01, 0x11, 0x21, 0x31, 0x02, 0x12, 0x22, 0x32])


def _expected_s24_samples() -> bytes:
    if sys.byteorder == "little":
        return bytes([0x11, 0x21, 0x31, 0x12, 0x22, 0x32])
    return bytes([0x01, 0x11, 0x21, 0x02, 0x12, 0x22])


def test_resolve_audio_format_24_bit_uses_s32_in_pyav() -> None:
    """24-bit wire format should map to s32 for PyAV processing."""
    wire_bytes, av_format, layout, av_bytes = AudioFormat(
        sample_rate=48_000, bit_depth=24, channels=2
    ).resolve_av_format()
    assert wire_bytes == 3
    assert av_format == "s32"
    assert layout == "stereo"
    assert av_bytes == 4


def test_resolve_audio_format_32_bit_is_supported() -> None:
    """32-bit PCM should be supported by resolver."""
    wire_bytes, av_format, layout, av_bytes = AudioFormat(
        sample_rate=44_100, bit_depth=32, channels=1
    ).resolve_av_format()
    assert wire_bytes == 4
    assert av_format == "s32"
    assert layout == "mono"
    assert av_bytes == 4


def test_convert_s32_to_s24_drops_least_significant_byte_python_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """s32->s24 conversion should drop the LSB byte per sample in fallback path."""
    monkeypatch.setattr(audio_module, "_get_numpy", lambda: None)
    converted = _convert_s32_to_s24(S32_SAMPLES)
    assert converted == _expected_s24_samples()


def test_convert_s32_to_s24_drops_least_significant_byte_numpy_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """s32->s24 conversion should drop the LSB byte per sample in NumPy path."""
    np = pytest.importorskip("numpy")
    monkeypatch.setattr(audio_module, "_get_numpy", lambda: np)
    converted = _convert_s32_to_s24(S32_SAMPLES)
    assert converted == _expected_s24_samples()


def test_convert_s32_to_s24_rejects_invalid_length() -> None:
    """Invalid byte lengths must be rejected."""
    with pytest.raises(ValueError, match="multiple of 4"):
        _convert_s32_to_s24(b"\x00\x01\x02")
