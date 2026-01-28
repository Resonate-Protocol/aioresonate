"""Transform key helpers for deterministic transformer reuse."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class TransformKey:
    """Stable identity for transformed output."""

    channel_id: UUID
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
