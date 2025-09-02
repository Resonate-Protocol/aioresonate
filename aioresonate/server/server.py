"""Resonate Server implementation to connect to and manage many Resonate Players."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass

from aiohttp import ClientConnectionError, ClientWSTimeout, web
from aiohttp.client import ClientSession

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
    """All groups managed by this server."""
    _loop: asyncio.AbstractEventLoop
    _event_cbs: list[Callable[[ResonateEvent], Coroutine[None, None, None]]]
    _connection_tasks: dict[str, asyncio.Task[None]]
    """
    All tasks managing player connections.

    This only includes connections initiated via connect_to_player (Server -> Player).
    """
    _retry_events: dict[str, asyncio.Event]
    """
    For each connection task in _connection_tasks, this holds an asyncio.Event.

    This event is used to signal an immediate retry of the connection, in case the connection is
    sleeping during a backoff period.
    """
    _id: str
    _name: str
    _client_session: ClientSession
    """The client session used to connect to players."""
    _owns_session: bool
    """Whether this server instance owns the client session."""
    _app: web.Application | None
    """
    Web application instance for the server.

    This is used to handle incoming WebSocket connections from players.
    """
    _app_runner: web.AppRunner | None
    """App runner for the web application."""
    _tcp_site: web.TCPSite | None
    """TCP site for the web application."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        server_id: str,
        server_name: str,
        client_session: ClientSession | None = None,
    ) -> None:
        """
        Initialize a new Resonate Server.

        Args:
            loop: The asyncio event loop to use for asynchronous operations.
            server_id: Unique identifier for this server instance.
            server_name: Human-readable name for this server.
            client_session: Optional ClientSession for outgoing connections.
                If None, a new session will be created.
        """
        self._players = set()
        self._loop = loop
        self._event_cbs = []
        self._id = server_id
        self._name = server_name
        if client_session is None:
            self._client_session = ClientSession(loop=self._loop)
            self._owns_session = True
        else:
            self._client_session = client_session
            self._owns_session = False
        self._connection_tasks = {}
        self._retry_events = {}
        self._app = None
        self._app_runner = None
        self._tcp_site = None
        logger.debug("ResonateServer initialized: id=%s, name=%s", server_id, server_name)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Read-only access to the event loop used by this server."""
        return self._loop

    async def on_player_connect(self, request: web.Request) -> web.StreamResponse:
        """Handle an incoming WebSocket connection from a Resonate client."""
        logger.debug("Incoming player connection from %s", request.remote)

        player = Player(
            self,
            handle_player_connect=self._handle_player_connect,
            handle_player_disconnect=self._handle_player_disconnect,
            request=request,
        )
        await player._handle_client()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

        websocket = player.websocket_connection
        # This is a WebSocketResponse since we just created player
        # as client-initiated.
        assert isinstance(websocket, web.WebSocketResponse)
        return websocket

    def connect_to_player(self, url: str) -> None:
        """
        Connect to the Resonate player at the given URL.

        If an active connection already exists for this URL, nothing will happen.
        In case a connection attempt fails, a new connection will be attempted automatically.
        """
        logger.debug("Connecting to player at URL: %s", url)
        prev_task = self._connection_tasks.get(url)
        if prev_task is not None:
            logger.debug("Connection is already active for URL: %s", url)
            # Signal immediate retry if we have a retry event (connection is in backoff)
            if retry_event := self._retry_events.get(url):
                logger.debug("Signaling immediate retry for URL: %s", url)
                retry_event.set()
        else:
            # Create retry event for this connection
            self._retry_events[url] = asyncio.Event()
            self._connection_tasks[url] = self._loop.create_task(
                self._handle_player_connection(url)
            )

    def disconnect_from_player(self, url: str) -> None:
        """
        Disconnect from the Resonate player that was previously connected at the given URL.

        If no connection was established at this URL, or the connection is already closed,
        this will do nothing.

        NOTE: this will only disconnect connections that were established via connect_to_player.
        """
        connection_task = self._connection_tasks.pop(url, None)
        if connection_task is not None:
            logger.debug("Disconnecting from player at URL: %s", url)
            _ = connection_task.cancel()  # Don't care about cancellation result

    async def _handle_player_connection(self, url: str) -> None:
        """Handle the actual connection to a player."""
        # Exponential backoff settings
        backoff = 1.0
        max_backoff = 300.0  # 5 minutes

        try:
            while True:
                player: Player | None = None
                retry_event = self._retry_events.get(url)

                try:
                    async with self._client_session.ws_connect(
                        url,
                        heartbeat=30,
                        # Pyright doesn't recognise the signature
                        timeout=ClientWSTimeout(ws_close=10, ws_receive=60),  # pyright: ignore[reportCallIssue]
                    ) as wsock:
                        # Reset backoff on successful connect
                        backoff = 1.0
                        player = Player(
                            self,
                            handle_player_connect=self._handle_player_connect,
                            handle_player_disconnect=self._handle_player_disconnect,
                            wsock_client=wsock,
                        )
                        await player._handle_client()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                    if self._client_session.closed:
                        logger.debug("Client session closed, stopping connection task for %s", url)
                        break
                    if player.closing:
                        break
                except asyncio.CancelledError:
                    break
                except TimeoutError:
                    logger.debug("Connection task for %s timed out", url)
                except ClientConnectionError:
                    logger.debug("Connection task for %s failed", url)

                if backoff >= max_backoff:
                    break

                logger.debug("Trying to reconnect to player at %s in %.1fs", url, backoff)

                # Use asyncio.wait_for with the retry event to allow immediate retry
                if retry_event is not None:
                    try:
                        # Always returns True when event is set
                        _ = await asyncio.wait_for(retry_event.wait(), timeout=backoff)
                        logger.debug("Immediate retry requested for %s", url)
                        # Clear the event for next time
                        retry_event.clear()
                    except TimeoutError:
                        # Normal timeout, continue with exponential backoff
                        pass
                else:
                    await asyncio.sleep(backoff)

                # Increase backoff for next retry (exponential)
                backoff *= 2
        except asyncio.CancelledError:
            pass
        except Exception:
            # NOTE: Intentional catch-all to log unexpected exceptions so they are visible.
            logger.exception("Unexpected error occurred")
        finally:
            _ = self._connection_tasks.pop(url, None)  # Cleanup connection tasks dict
            _ = self._retry_events.pop(url, None)  # Cleanup retry events dict

    def add_event_listener(
        self, callback: Callable[[ResonateEvent], Coroutine[None, None, None]]
    ) -> Callable[[], None]:
        """
        Register a callback to listen for state changes of the server.

        State changes include:
        - A new player was connected
        - A player disconnected

        Returns a function to remove the listener.
        """
        self._event_cbs.append(callback)
        return lambda: self._event_cbs.remove(callback)

    def _signal_event(self, event: ResonateEvent) -> None:
        """Signal an event to all registered listeners."""
        for cb in self._event_cbs:
            _ = self._loop.create_task(cb(event))  # Fire and forget event callback

    def _handle_player_connect(self, player: Player) -> None:
        """
        Register the player to the server.

        Should only be called once all data like the player id was received.
        """
        if player in self._players:
            return

        logger.debug("Adding player %s (%s) to server", player.player_id, player.name)
        self._players.add(player)
        self._signal_event(PlayerAddedEvent(player.player_id))

    def _handle_player_disconnect(self, player: Player) -> None:
        """Unregister the player from the server."""
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
    def id(self) -> str:
        """Get the unique identifier of this server."""
        return self._id

    @property
    def name(self) -> str:
        """Get the name of this server."""
        return self._name

    async def start_server(self, port: int = 8927, host: str = "0.0.0.0") -> None:
        """Start an HTTP server to handle incoming Resonate connections on /resonate."""
        if self._app is not None:
            logger.warning("Server is already running")
            return

        logger.info("Starting Resonate server on port %d", port)
        self._app = web.Application()
        # Create perpetual WebSocket route for player connections
        _ = self._app.router.add_get("/resonate", self.on_player_connect)
        self._app_runner = web.AppRunner(self._app)
        await self._app_runner.setup()

        try:
            self._tcp_site = web.TCPSite(
                self._app_runner,
                host=host if host != "0.0.0.0" else None,
                port=port,
            )
            await self._tcp_site.start()
            logger.info("Resonate server started successfully on %s:%d", host, port)
        except OSError as e:
            logger.error("Failed to start server on %s:%d: %s", host, port, e)
            if self._app_runner:
                await self._app_runner.cleanup()
                self._app_runner = None
            if self._app:
                await self._app.shutdown()
                self._app = None
            raise

    async def stop_server(self) -> None:
        """Stop the HTTP server."""
        if self._tcp_site:
            await self._tcp_site.stop()
            self._tcp_site = None
            logger.debug("TCP site stopped")

        if self._app_runner:
            await self._app_runner.cleanup()
            self._app_runner = None
            logger.debug("App runner cleaned up")

        if self._app:
            await self._app.shutdown()
            self._app = None

    async def close(self) -> None:
        """Close the server and cleanup resources."""
        await self.stop_server()
        if self._owns_session and not self._client_session.closed:
            await self._client_session.close()
            logger.debug("Closed internal client session for server %s", self._name)
