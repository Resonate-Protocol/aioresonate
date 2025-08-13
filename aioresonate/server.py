"""Resonate Server implementation to connect to and manage many Resonate Players.

ResonateServer is the core of the music listening experience, responsible for:
- Managing connected players
- Orchestrating synchronized grouped playback
"""

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass

from aiohttp import ClientWebSocketResponse, web
from aiohttp.client import ClientSession

from .group import PlayerGroup
from .instance import PlayerInstance


class ResonateEvent:
    """Event type used by ResonateServer.add_event_callback()."""


@dataclass
class PlayerAdded(ResonateEvent):
    """A new player was added."""

    player_id: str


@dataclass
class PlayerRemoved(ResonateEvent):
    """A player disconnected from the server."""

    player_id: str


# TODO: add grouped events


class ResonateServer:
    """Resonate Server implementation to connect to and manage many Resonate Players."""

    _players: set[PlayerInstance]
    _groups: set[PlayerGroup]
    loop: asyncio.AbstractEventLoop
    _event_cbs: list[Callable[[ResonateEvent], Coroutine[None, None, None]]]

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        """Initialize a new Resonate Server."""
        self._players = set()
        self._groups = set()
        self.loop = loop
        self._event_cbs = []

    async def on_player_connect(
        self, request: web.Request
    ) -> web.WebSocketResponse | ClientWebSocketResponse:
        """Handle an incoming WebSocket connection from a Resonate client."""
        instance = PlayerInstance(self, request=request, url=None, wsock_client=None)
        # TODO: only add once we know its id, see connect_to_player
        try:
            self._players.add(instance)
            return await instance.handle_client()
        finally:
            self._players.remove(instance)

    async def connect_to_player(self, url: str) -> web.WebSocketResponse | ClientWebSocketResponse:
        """Connect to the Resonate player at the given URL."""
        # TODO catch any exceptions from ws_connect
        async with ClientSession() as session:
            wsock = await session.ws_connect(url)
            instance = PlayerInstance(self, request=None, url=url, wsock_client=wsock)
            try:
                return await instance.handle_client()
            finally:
                self._on_player_remove(instance)

    def add_event_listener(
        self, callback: Callable[[ResonateEvent], Coroutine[None, None, None]]
    ) -> Callable[[], None]:
        """Register a callback to listen for state changes of the server.

        State changes include:
        - A new player was connected
        - A player disconnected
        - Changes in Groups

        Returns function to remove the listener.
        """
        self._event_cbs.append(callback)
        return lambda: self._event_cbs.remove(callback)

    def _signal_event(self, event: ResonateEvent) -> None:
        for cb in self._event_cbs:
            _ = self.loop.create_task(cb(event))

    def _on_player_add(self, instance: PlayerInstance) -> None:
        """
        Register the player to the server and notify that the player connected.

        Should only be called once all data like the player id was received.
        """
        if instance in self._players:
            return

        self._players.add(instance)
        self._signal_event(PlayerAdded(instance.player_id))

    def _on_player_remove(self, instance: PlayerInstance) -> None:
        if instance not in self._players:
            return

        self._players.remove(instance)
        self._signal_event(PlayerRemoved(instance.player_id))

    @property
    def players(self) -> list[PlayerInstance]:
        """Get the list of all players connected to this server."""
        raise NotImplementedError

    def get_player(self, player_id: str) -> PlayerInstance | None:
        """Get the player with the given id."""
        for player in self.players:
            if player.player_id == player_id:
                return player
        return None

    @property
    def groups(self) -> list[PlayerGroup]:
        """Get the list of all groups."""
        raise NotImplementedError
