"""Represents a single player device connected to the server."""

from .group import PlayerGroup


class PlayerInstance:
    """A Player that is connected to a ResonateServer.

    Playback is handled through groups, use PlayerInstance.group to get the
    assigned group.
    """

    def disconnect(self) -> None:
        """Disconnect the player from the server."""
        raise NotImplementedError

    @property
    def group(self) -> PlayerGroup:
        """Get the group assigned to this player."""
        raise NotImplementedError

    @property
    def player_id(self) -> str:
        """The unique identifier of this Player."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """The human-readable name of this Player."""
        raise NotImplementedError

    @property
    def capabilities(self) -> dict[str, str]:
        """List of capabilities supported by this player."""
        raise NotImplementedError

    def set_volume(self, volume: int) -> None:
        """Set the volume of this player."""
        raise NotImplementedError

    def mute(self) -> None:
        """Mute this player."""
        raise NotImplementedError

    def unmute(self) -> None:
        """Unmute this player."""
        raise NotImplementedError

    @property
    def muted(self) -> bool:
        """Mute state of this player."""
        raise NotImplementedError

    @property
    def volume(self) -> int:
        """Volume of this player."""
        raise NotImplementedError
