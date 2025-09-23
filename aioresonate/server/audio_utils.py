"""Shared audio helper functions for aioresonate."""

from __future__ import annotations


def build_flac_stream_header(extradata: bytes) -> bytes:
    """Return a complete FLAC stream header for encoder ``extradata``."""
    if not extradata:
        return extradata
    return b"fLaC\x80" + len(extradata).to_bytes(3, "big") + extradata
