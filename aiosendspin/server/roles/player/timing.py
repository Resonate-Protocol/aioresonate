"""Player-specific transport timing helpers."""

from __future__ import annotations

from aiosendspin.server.roles.player.types import PlayerRoleProtocol


def get_player_static_delay_us(role: object | None) -> int:
    """Return static delay in microseconds for player roles, else 0."""
    if not isinstance(role, PlayerRoleProtocol):
        return 0
    return max(role.get_static_delay_ms(), 0) * 1_000


def effective_player_timestamp_us(role: object | None, timestamp_us: int) -> int:
    """Return the player's effective playback timestamp."""
    return timestamp_us - get_player_static_delay_us(role)


def min_safe_raw_player_timestamp_us(
    role: object | None,
    now_us: int,
    *,
    lead_us: int = 0,
) -> int:
    """Return the minimum raw timestamp that is still safe to send."""
    return now_us + max(0, lead_us) + get_player_static_delay_us(role)
