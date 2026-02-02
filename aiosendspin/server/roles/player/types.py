"""Shared player role protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aiosendspin.models import AudioCodec
    from aiosendspin.models.player import SupportedAudioFormat
    from aiosendspin.server.audio import AudioFormat
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

    def get_supported_formats(self) -> list[SupportedAudioFormat] | None:
        """Return formats both client and server support, in client priority order."""
        ...

    def set_preferred_format(self, audio_format: AudioFormat, codec: AudioCodec) -> bool:
        """Set preferred format if compatible. Returns True on success."""
        ...
