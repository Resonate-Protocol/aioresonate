"""Resonate Server implementation to connect to and manage many Resonate Players.

ResonateServer is the core of the music listening experience, responsible for:
- Managing connected players
- Orchestrating synchronized grouped playback
"""

from collections.abc import Awaitable, Callable
from enum import Enum, auto

from aiohttp import ClientWebSocketResponse, web

from .group import PlayerGroup
from .instance import PlayerInstance


class ResonateEvent(Enum):
    """Event type used by ResonateServer.add_event_callback()."""

    PLAYER_CONNECTED = auto()
    PLAYER_DISCONNECTED = auto()
    GROUP_UPDATED = auto()


class ResonateServer:
    """Resonate Server implementation to connect to and manage many Resonate Players."""

    def __init__(self) -> None:
        """Initialize a new Resonate Server."""
        raise NotImplementedError

    async def on_player_connect(self, request: web.Request) -> web.WebSocketResponse:
        """Handle an incoming WebSocket connection from a Resonate client."""
        raise NotImplementedError

    async def connect_to_player(self, url: str) -> ClientWebSocketResponse:
        """Connect to the Resonate player at the given URL."""
        raise NotImplementedError

    def add_event_callback(
        self, callback: Callable[[ResonateEvent], Awaitable[None]]
    ) -> Callable[[], None]:
        """Register a callback to listen for state changes of the server.

        State changes include:
        - A new player was connected
        - A player disconnected
        - Changes in Groups
        """
        # TODO: callback should also give info about what player connected
        raise NotImplementedError

    @property
    def players(self) -> list[PlayerInstance]:
        """Get the list of all players connected to this server."""
        raise NotImplementedError

    def get_player(self, player_id: str) -> PlayerInstance:
        """Get the player with the given id."""
        raise NotImplementedError

    @property
    def groups(self) -> list[PlayerGroup]:
        """Get the list of all groups."""
        raise NotImplementedError
