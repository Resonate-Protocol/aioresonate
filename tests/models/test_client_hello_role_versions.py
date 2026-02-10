"""Tests for client/hello role version handling and support object validation."""

from __future__ import annotations

import pytest

from aiosendspin.models.core import ClientHelloPayload
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec


def test_player_support_required_only_for_v1_role_id() -> None:
    """player@v2 without player@v1_support must not raise."""
    payload = ClientHelloPayload(
        client_id="c1",
        name="Client",
        version=1,
        supported_roles=["player@v2"],
    )
    assert payload.player_support is None


def test_player_support_still_required_for_player_v1() -> None:
    """player@v1 requires player@v1_support (player_support alias)."""
    with pytest.raises(ValueError, match="player@v1_support"):
        ClientHelloPayload(
            client_id="c1",
            name="Client",
            version=1,
            supported_roles=["player@v1"],
        )


def test_player_support_accepted_for_player_v1() -> None:
    """player@v1 accepts player@v1_support via player_support alias."""
    payload = ClientHelloPayload(
        client_id="c1",
        name="Client",
        version=1,
        supported_roles=["player@v1"],
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM, sample_rate=48000, bit_depth=16, channels=2
                )
            ],
            buffer_capacity=100_000,
            supported_commands=[],
        ),
    )
    assert payload.player_support is not None


def test_player_support_serializes_with_spec_alias_name() -> None:
    """Serialized payload should use player@v1_support instead of legacy player_support."""
    payload = ClientHelloPayload(
        client_id="c1",
        name="Client",
        version=1,
        supported_roles=["player@v1"],
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM, sample_rate=48000, bit_depth=16, channels=2
                )
            ],
            buffer_capacity=100_000,
            supported_commands=[],
        ),
    )
    serialized = payload.to_dict()
    assert "player@v1_support" in serialized
    assert "player_support" not in serialized
