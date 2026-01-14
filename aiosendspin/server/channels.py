"""Channel routing infrastructure for multi-channel audio streaming."""

from __future__ import annotations

from uuid import UUID

# Main channel - default channel for all players
MAIN_CHANNEL: UUID = UUID("00000000-0000-0000-0000-000000000000")


class ChannelRouter:
    """
    Manages player-to-channel assignments for multi-channel streaming.

    Players not explicitly assigned default to MAIN_CHANNEL implicitly.
    Only explicitly assigned players are tracked in the internal mapping.
    """

    def __init__(self) -> None:
        """Create a new ChannelRouter."""
        self._assignments: dict[str, UUID] = {}

    def get_channel(self, player_id: str) -> UUID:
        """
        Get the channel a player is assigned to.

        Args:
            player_id: The player's client_id.

        Returns:
            The assigned channel UUID, or MAIN_CHANNEL if not explicitly assigned.
        """
        return self._assignments.get(player_id, MAIN_CHANNEL)

    def set_channel(self, player_id: str, channel_id: UUID) -> None:
        """
        Assign a player to a channel.

        Args:
            player_id: The player's client_id.
            channel_id: The channel UUID to assign to.
        """
        self._assignments[player_id] = channel_id

    def get_players_on_channel(self, channel_id: UUID) -> list[str]:
        """
        Get all players explicitly assigned to a channel.

        Args:
            channel_id: The channel UUID to query.

        Returns:
            List of player_ids assigned to the channel.
        """
        return [
            player_id
            for player_id, assigned_channel in self._assignments.items()
            if assigned_channel == channel_id
        ]

    def remove_player(self, player_id: str) -> None:
        """
        Remove a player's channel assignment.

        The player will return to implicit MAIN_CHANNEL.

        Args:
            player_id: The player's client_id.
        """
        self._assignments.pop(player_id, None)
