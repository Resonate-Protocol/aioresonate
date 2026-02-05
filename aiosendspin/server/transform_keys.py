"""Transform key helpers for deterministic transformer reuse."""

# TODO: could we move this to push_stream? or audio_transformers?

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


# TODO: wait, duplicate in audio_transformers.py?
@dataclass(frozen=True, slots=True)
class TransformKey:
    """Stable identity for transformed output.

    channel_id is stored as int (UUID.int) for faster hashing - int hash is O(1)
    vs UUID hash which requires attribute access and method calls.
    """

    channel_id: int  # UUID.int value for fast hashing
    transformer_type: type
    sample_rate: int
    bit_depth: int
    channels: int
    frame_duration_us: int
    options: tuple[tuple[str, str], ...]


def normalize_options(options: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    """Normalize options mapping into a deterministic, hashable tuple."""
    if not options:
        return ()
    return tuple(sorted(((key, value) for key, value in options.items()), key=lambda kv: kv[0]))
