"""Tests for :mod:`aiosendspin.noise.constants`."""

from aiosendspin.noise.constants import (
    PROTOCOL_VERSION,
    PSK_ID_LABEL,
    SENTINEL_PSK,
)
from aiosendspin.noise.keys import psk_id_for


def test_sentinel_psk_matches_spec_hex() -> None:
    """SENTINEL_PSK equals the spec's published hex value."""
    expected = "1b5e24dbc1aed95fc2a5a338a90c05df44bd10f5ec1f4cd66cbf86272767b9d3"
    assert SENTINEL_PSK.hex() == expected


def test_sentinel_psk_id_matches_spec_constant() -> None:
    """psk_id derivation produces the spec's published Sentinel psk_id string."""
    assert psk_id_for(SENTINEL_PSK) == "GFsV9tLaSQm9HcFWpKsgYQOr7wFTvNUtkmFwuVz3zoo"


def test_psk_id_label_is_literal_utf8_no_nul() -> None:
    """PSK_ID_LABEL is literal UTF-8 with no NUL terminator or quotes."""
    assert PSK_ID_LABEL == b"sendspin-psk-id-v1"
    assert b"\x00" not in PSK_ID_LABEL


def test_protocol_version_is_one() -> None:
    """PROTOCOL_VERSION is currently 1 and only 1."""
    assert PROTOCOL_VERSION == 1
