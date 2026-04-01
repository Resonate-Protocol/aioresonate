"""Tests for player model payloads."""

from __future__ import annotations

import pytest

from aiosendspin.models.player import PlayerCommandPayload, PlayerStatePayload
from aiosendspin.models.types import PlayerCommand


def test_player_state_static_delay_serializes_at_zero() -> None:
    """static_delay_ms=0 is always serialized (not omitted by omit_none)."""
    payload = PlayerStatePayload()
    data = payload.to_dict()
    assert "static_delay_ms" in data
    assert data["static_delay_ms"] == 0


def test_player_state_static_delay_range_valid() -> None:
    """Maximum value 5000 is accepted."""
    payload = PlayerStatePayload(static_delay_ms=5000)
    assert payload.static_delay_ms == 5000


def test_player_state_static_delay_range_invalid() -> None:
    """Values above 5000 are rejected."""
    with pytest.raises(ValueError, match="static_delay_ms"):
        PlayerStatePayload(static_delay_ms=5001)


def test_player_state_static_delay_negative_invalid() -> None:
    """Negative values are rejected."""
    with pytest.raises(ValueError, match="static_delay_ms"):
        PlayerStatePayload(static_delay_ms=-1)


def test_player_state_supported_commands_serializes() -> None:
    """supported_commands serializes enum values as strings."""
    payload = PlayerStatePayload(supported_commands=[PlayerCommand.SET_STATIC_DELAY])
    data = payload.to_dict()
    assert data["supported_commands"] == ["set_static_delay"]


def test_player_state_backward_compat_no_delay() -> None:
    """Old clients that don't send static_delay_ms get default 0."""
    data = '{"volume": 50}'
    payload = PlayerStatePayload.from_json(data)
    assert payload.static_delay_ms == 0


def test_player_command_set_static_delay_valid() -> None:
    """SET_STATIC_DELAY command accepts valid delay value."""
    cmd = PlayerCommandPayload(command=PlayerCommand.SET_STATIC_DELAY, static_delay_ms=300)
    assert cmd.static_delay_ms == 300


def test_player_command_set_static_delay_missing() -> None:
    """SET_STATIC_DELAY command requires static_delay_ms."""
    with pytest.raises(ValueError, match="static_delay_ms must be provided"):
        PlayerCommandPayload(command=PlayerCommand.SET_STATIC_DELAY)


def test_player_command_set_static_delay_out_of_range() -> None:
    """SET_STATIC_DELAY command rejects out-of-range values."""
    with pytest.raises(ValueError, match="static_delay_ms"):
        PlayerCommandPayload(command=PlayerCommand.SET_STATIC_DELAY, static_delay_ms=6000)


def test_player_command_volume_rejects_static_delay() -> None:
    """VOLUME command rejects static_delay_ms parameter."""
    with pytest.raises(ValueError, match="static_delay_ms should not"):
        PlayerCommandPayload(command=PlayerCommand.VOLUME, volume=50, static_delay_ms=100)


def test_player_state_rejects_invalid_supported_commands() -> None:
    """State-level supported_commands only allows set_static_delay."""
    with pytest.raises(ValueError, match="Invalid state-level"):
        PlayerStatePayload(supported_commands=[PlayerCommand.VOLUME])
