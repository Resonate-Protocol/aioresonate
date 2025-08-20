"""Resonate Server implementation to connect to and manage many Resonate Players."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass

from aiohttp import ClientWebSocketResponse, web
from aiohttp.client import ClientSession

from .group import PlayerGroup
from .player import Player

logger = logging.getLogger(__name__)


class ResonateEvent:
    """Base event type used by ResonateServer.add_event_listener()."""


@dataclass
class PlayerAddedEvent(ResonateEvent):
    """A new player was added."""

    player_id: str


@dataclass
class PlayerRemovedEvent(ResonateEvent):
    """A player disconnected from the server."""

    player_id: str


class ResonateServer:
    """Resonate Server implementation to connect to and manage many Resonate Players."""

    _players: set[Player]
    _groups: set[PlayerGroup]
    loop: asyncio.AbstractEventLoop
    _event_cbs: list[Callable[[ResonateEvent], Coroutine[None, None, None]]]
    _id: str
    _name: str

    def __init__(self, loop: asyncio.AbstractEventLoop, server_id: str, server_name: str) -> None:
        """Initialize a new Resonate Server."""
        self._players = set()
        self._groups = set()
        self.loop = loop
        self._event_cbs = []
        self._id = server_id
        self._name = server_name
        logger.debug("ResonateServer initialized: id=%s, name=%s", server_id, server_name)

    async def on_player_connect(
        self, request: web.Request
    ) -> web.WebSocketResponse | ClientWebSocketResponse:
        """Handle an incoming WebSocket connection from a Resonate client."""
        logger.debug("Incoming player connection from %s", request.remote)
        player = Player(self, request=request, url=None, wsock_client=None)
        # TODO: only add once we know its id, see connect_to_player
        try:
            self._players.add(player)
            return await player.handle_client()
        finally:
            self._players.remove(player)

    async def connect_to_player(self, url: str) -> web.WebSocketResponse | ClientWebSocketResponse:
        """Connect to the Resonate player at the given URL."""
        logger.debug("Connecting to player at URL: %s", url)
        # TODO catch any exceptions from ws_connect
        async with ClientSession() as session:
            wsock = await session.ws_connect(url)
            player = Player(self, request=None, url=url, wsock_client=wsock)
            try:
                return await player.handle_client()
            finally:
                self._on_player_remove(player)

    def add_event_listener(
        self, callback: Callable[[ResonateEvent], Coroutine[None, None, None]]
    ) -> Callable[[], None]:
        """Register a callback to listen for state changes of the server.

        State changes include:
        - A new player was connected
        - A player disconnected

        Returns a function to remove the listener.
        """
        self._event_cbs.append(callback)
        return lambda: self._event_cbs.remove(callback)

    def _signal_event(self, event: ResonateEvent) -> None:
        for cb in self._event_cbs:
            _ = self.loop.create_task(cb(event))

    def _on_player_add(self, player: Player) -> None:
        """
        Register the player to the server and notify that the player connected.

        Should only be called once all data like the player id was received.
        """
        if player in self._players:
            return

        logger.debug("Adding player %s (%s) to server", player.player_id, player.name)
        self._players.add(player)
        self._signal_event(PlayerAddedEvent(player.player_id))

    def _on_player_remove(self, player: Player) -> None:
        if player not in self._players:
            return

        logger.debug("Removing player %s from server", player.player_id)
        self._players.remove(player)
        self._signal_event(PlayerRemovedEvent(player.player_id))

    @property
    def players(self) -> set[Player]:
        """Get the set of all players connected to this server."""
        return self._players

    def get_player(self, player_id: str) -> Player | None:
        """Get the player with the given id."""
        logger.debug("Looking for player with id: %s", player_id)
        for player in self.players:
            if player.player_id == player_id:
                logger.debug("Found player %s", player_id)
                return player
        logger.debug("Player %s not found", player_id)
        return None

    @property
    def groups(self) -> set[PlayerGroup]:
        """Get the set of all groups."""
        return self._groups

    @property
    def id(self) -> str:
        """Get the unique identifier of this server."""
        return self._id

    @property
    def name(self) -> str:
        """Get the name of this server."""
        return self._name
