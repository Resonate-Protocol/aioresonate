"""Encoder/decoder round-trip and header tests for the shared codecs."""

from __future__ import annotations

import pytest

from aiosendspin.audio.codecs import create_decoder, create_encoder
from tests.conftest import sine_pcm_16bit


def _roundtrip(codec: str, pcm: bytes) -> bytes:
    enc = create_encoder(codec, sample_rate=48000, bit_depth=16, channels=2)
    frames = enc.process(pcm, 0, 0) + enc.flush()
    dec = create_decoder(
        codec, sample_rate=48000, bit_depth=16, channels=2, codec_header=enc.get_codec_header()
    )
    out = bytearray()
    for frame, _dur in frames:
        out += dec.decode(frame)
    out += dec.flush()
    return bytes(out)


def test_pcm_roundtrip_is_bit_exact() -> None:
    """Raw PCM survives encode+decode unchanged (the loopback confidence anchor)."""
    pcm = sine_pcm_16bit(48000)
    assert _roundtrip("pcm", pcm) == pcm


@pytest.mark.parametrize("codec", ["flac", "opus"])
def test_lossy_codec_roundtrip_recovers_audio(codec: str) -> None:
    """FLAC/Opus decode back to roughly the same amount of audio (structural check)."""
    pcm = sine_pcm_16bit(48000)
    out = _roundtrip(codec, pcm)
    # Within one decoder block of the input; never empty.
    assert len(out) >= len(pcm) - 4608 * 4
    assert len(out) <= len(pcm) + 4608 * 4


def test_codec_headers_match_their_wire_format() -> None:
    """Only flac carries a header, the `fLaC` marker plus STREAMINFO."""
    assert (
        create_encoder("pcm", sample_rate=48000, bit_depth=16, channels=2).get_codec_header()
        is None
    )
    flac = create_encoder("flac", sample_rate=48000, bit_depth=16, channels=2).get_codec_header()
    assert flac is not None
    assert flac[:4] == b"fLaC"
    # Opus is configured from the declared format, so it sends no header.
    assert (
        create_encoder("opus", sample_rate=48000, bit_depth=16, channels=2).get_codec_header()
        is None
    )


def test_unknown_codec_rejected() -> None:
    """Unsupported codec identifiers raise rather than silently no-op."""
    with pytest.raises(ValueError, match="Unsupported source codec"):
        create_encoder("mp3", sample_rate=48000, bit_depth=16, channels=2)
    with pytest.raises(ValueError, match="Unsupported source codec"):
        create_decoder("mp3", sample_rate=48000, bit_depth=16, channels=2, codec_header=None)


def test_flac_decoder_preserves_multichannel_frame_width() -> None:
    """FLAC decoding preserves every declared channel."""
    sample_rate = 48000
    channels = 6
    pcm = bytes(sample_rate * channels * 2)
    encoder = create_encoder("flac", sample_rate=sample_rate, bit_depth=16, channels=channels)
    frames = encoder.process(pcm, 0, 0) + encoder.flush()
    decoder = create_decoder(
        "flac",
        sample_rate=sample_rate,
        bit_depth=16,
        channels=channels,
        codec_header=encoder.get_codec_header(),
    )

    decoded = b"".join(decoder.decode(frame) for frame, _ in frames) + decoder.flush()

    assert len(pcm) <= len(decoded) <= len(pcm) + 4608 * channels * 2
