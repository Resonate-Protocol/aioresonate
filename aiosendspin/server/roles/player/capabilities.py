"""Server-side format validation for player roles.

Provides utilities to check if the server can encode a client's requested format
based on codec-specific constraints (sample rates, bit depths, channels).
"""

from __future__ import annotations

from aiosendspin.models import AudioCodec
from aiosendspin.models.player import SupportedAudioFormat
from aiosendspin.server.roles.player.audio_transformers import OpusEncoder

VALID_BIT_DEPTHS: frozenset[int] = frozenset({16, 24})
VALID_CHANNELS: frozenset[int] = frozenset({1, 2})


def can_encode_format(fmt: SupportedAudioFormat) -> bool:
    """Check if the server can encode this format.

    Validates against server encoding constraints:
    - Bit depth: 16 or 24
    - Channels: 1 or 2
    - Opus: sample rate must be one of 8k, 12k, 16k, 24k, 48k
    - FLAC/PCM: any sample rate

    Args:
        fmt: The format to validate.

    Returns:
        True if the server can encode this format.
    """
    if fmt.bit_depth not in VALID_BIT_DEPTHS:
        return False
    if fmt.channels not in VALID_CHANNELS:
        return False
    if fmt.codec == AudioCodec.OPUS:
        return fmt.sample_rate in OpusEncoder.VALID_SAMPLE_RATES
    return True


def filter_encodable_formats(
    formats: list[SupportedAudioFormat],
) -> list[SupportedAudioFormat]:
    """Filter to server-encodable formats, preserving client priority order.

    Args:
        formats: Client's supported formats in priority order.

    Returns:
        Formats the server can encode, maintaining the client's priority order.
    """
    return [fmt for fmt in formats if can_encode_format(fmt)]
