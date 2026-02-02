"""Shared player role protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient


@runtime_checkable
class PlayerRoleProtocol(Protocol):
    """Protocol for player role implementations."""

    @property
    def role_id(self) -> str:
        """Return the versioned role identifier."""
        ...

    _client: SendspinClient

    def get_player_volume(self) -> int | None:
        """Return the player volume if supported."""
        ...

    def get_player_muted(self) -> bool | None:
        """Return the player mute state if supported."""
        ...

    def set_player_volume(self, volume: int) -> None:
        """Set the player volume if supported."""
        ...

    def set_player_mute(self, muted: bool) -> None:  # noqa: FBT001
        """Set the player mute state if supported."""
        ...

    def get_player_supported_sample_rates(self) -> set[int] | None:
        """Return supported sample rates if available."""
        ...
