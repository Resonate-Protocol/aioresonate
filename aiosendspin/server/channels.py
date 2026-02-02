"""Channel routing infrastructure for multi-channel audio streaming."""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

# Main channel - default channel for all players
MAIN_CHANNEL: UUID = UUID("00000000-0000-0000-0000-000000000000")

# Type alias for channel resolver callbacks
ChannelResolver = Callable[[str], UUID]


def default_channel_resolver(player_id: str) -> UUID:  # noqa: ARG001
    """
    Return MAIN_CHANNEL for all players.

    Args:
        player_id: The player's client_id (unused in default implementation).

    Returns:
        MAIN_CHANNEL for all players.
    """
    return MAIN_CHANNEL
