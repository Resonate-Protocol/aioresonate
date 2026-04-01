"""Sendspin Server implementation to connect to and manage many Sendspin clients."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from ipaddress import ip_address

from aiohttp import (
    ClientConnectionError,
    ClientResponseError,
    ClientTimeout,
    ClientWSTimeout,
    web,
)
from aiohttp.client import ClientSession
from zeroconf import (
    InterfaceChoice,
    IPVersion,
    NonUniqueNameException,
    ServiceStateChange,
    Zeroconf,
)
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

from aiosendspin.models.core import ClientHelloPayload
from aiosendspin.models.types import ConnectionReason, GoodbyeReason
from aiosendspin.util import create_task, get_local_ip

from .client import SendspinClient
from .clock import Clock, LoopClock
from .connection import SendspinConnection
from .group import SendspinGroup

logger = logging.getLogger(__name__)


# Abort reconnection attempts after exponential backoff reaches this ceiling
MAX_RECONNECT_BACKOFF_S = 300.0
# Only consider a connection stable if it lasts at least this long, otherwise
# a successful but broken session may cause a reconnection every second.
STABLE_SERVER_INITIATED_SESSION_S = 10.0


class SendspinEvent:
    """Base event type used by SendspinServer.add_event_listener()."""


@dataclass
class ClientAddedEvent(SendspinEvent):
    """A new persistent client/device was added."""

    client_id: str


@dataclass
class ClientUpdatedEvent(SendspinEvent):
    """A client's hello payload changed on reconnect."""

    client_id: str


@dataclass
class ClientRemovedEvent(SendspinEvent):
    """A persistent client/device was removed from the server."""

    client_id: str


@dataclass(frozen=True, slots=True)
class ExternalStreamStartRequest:
    """Request payload for externally managed player connection on stream start."""

    client_id: str
    server: SendspinServer
    connection_reason: ConnectionReason = ConnectionReason.PLAYBACK


ExternalStreamStartCallback = Callable[[ExternalStreamStartRequest], None]


def _get_first_valid_ip(addresses: list[str]) -> str | None:
    """Get the first valid IP address, filtering out link-local and unspecified addresses."""
    for addr_str in addresses:
        try:
            addr = ip_address(addr_str)
        except ValueError:
            continue
        if not addr.is_link_local and not addr.is_unspecified:
            return addr_str
    return None


class SendspinServer:
    """Sendspin Server implementation to connect to and manage many Sendspin clients."""

    API_PATH = "/sendspin"  # Fixed by protocol
    _pending_connections: set[SendspinConnection]
    """Incoming connections that have not finished their handshake/message loop yet."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        server_id: str,
        server_name: str,
        client_session: ClientSession | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        """Initialize a Sendspin server instance."""
        self._loop = loop
        self._id = server_id
        self._name = server_name
        self._clock: Clock = clock or LoopClock(loop)

        self._clients: dict[str, SendspinClient] = {}
        self._event_cbs: list[Callable[[SendspinServer, SendspinEvent], None]] = []

        if client_session is None:
            self._client_session = ClientSession(loop=self._loop, timeout=ClientTimeout(total=30))
            self._owns_session = True
        else:
            self._client_session = client_session
            self._owns_session = False

        self._connection_tasks: dict[str, asyncio.Task[None]] = {}
        self._retry_events: dict[str, asyncio.Event] = {}
        self._initial_connect_waiters: dict[str, list[asyncio.Future[None]]] = {}
        self._initial_connect_succeeded: set[str] = set()
        self._connection_reasons: dict[str, ConnectionReason] = {}  # url → reason
        self._client_urls: dict[str, str] = {}  # client_id → url
        self._external_stream_start_cbs: dict[str, ExternalStreamStartCallback] = {}
        self._external_registration_timeouts: dict[str, asyncio.Handle] = {}
        self._reclaim_timeouts: dict[str, asyncio.Handle] = {}
        self._pending_connections = set()

        self._mdns_client_urls: dict[str, str] = {}
        self._app: web.Application | None = None
        self._app_runner: web.AppRunner | None = None
        self._tcp_site: web.TCPSite | None = None
        self._zc: AsyncZeroconf | None = None
        self._mdns_service: AsyncServiceInfo | None = None
        self._mdns_browser: AsyncServiceBrowser | None = None

        logger.debug("SendspinServer initialized: id=%s, name=%s", server_id, server_name)

    def _create_web_application(self) -> web.Application:
        app = web.Application()
        app.router.add_get(self.API_PATH, self.on_client_connect)
        return app

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Return the asyncio loop used by the server."""
        return self._loop

    @property
    def clock(self) -> Clock:
        """Time source used for timestamping."""
        return self._clock

    @property
    def id(self) -> str:
        """Return the server identifier advertised to clients."""
        return self._id

    @property
    def name(self) -> str:
        """Return the human-readable server name."""
        return self._name

    @property
    def clients(self) -> list[SendspinClient]:
        """All known clients (including disconnected)."""
        return list(self._clients.values())

    @property
    def connected_clients(self) -> list[SendspinClient]:
        """All currently connected clients."""
        return [c for c in self._clients.values() if c.is_connected]

    def get_client(self, client_id: str) -> SendspinClient | None:
        """Get a persistent client device by id, if known."""
        return self._clients.get(client_id)

    def get_or_create_client(self, client_id: str) -> SendspinClient:
        """Get or create a persistent client device by id."""
        client = self._clients.get(client_id)
        if client is None:
            client = SendspinClient(self, client_id=client_id)
            self._clients[client_id] = client
            SendspinGroup(self, client)
        return client

    def on_client_first_connect(self, client_id: str) -> None:
        """Fire ClientAddedEvent when a client completes its first handshake."""
        client = self._clients.get(client_id)
        if client is not None:
            self._fire_client_added_event_once(client)

    async def remove_client(self, client_id: str) -> None:
        """Remove a client from the persistent registry."""
        self._cancel_external_registration_timeout(client_id)
        self._cancel_reclaim_timeout(client_id)
        self._external_stream_start_cbs.pop(client_id, None)
        self._client_urls.pop(client_id, None)

        client = self._clients.pop(client_id, None)
        if client is None:
            return
        await client.group.remove_client(client)
        self._signal_event(ClientRemovedEvent(client_id))

    def register_external_player(
        self,
        hello: ClientHelloPayload,
        *,
        on_stream_start: ExternalStreamStartCallback,
        timeout_s: float = 0.0,
    ) -> SendspinClient:
        """Register an externally orchestrated client identity.

        Use this for clients that do not connect through Sendspin's standard
        discovery/reconnect mechanisms, or that require an external call to
        start their Sendspin client connection.

        Args:
            hello: Client identity/capability payload to preload while disconnected.
                This must match the hello payload sent when the client later
                connects to this server.
            on_stream_start: Callback invoked when playback needs this client to
                connect to the server through its usual external mechanism.
            timeout_s: Optional registration deadline in seconds. If > 0 and the
                client transport does not attach before the deadline, the client
                is fully unregistered from this server.
        """
        if timeout_s < 0:
            raise ValueError("timeout_s must be >= 0")

        client = self.get_or_create_client(hello.client_id)
        if client.is_connected:
            raise RuntimeError(
                f"Cannot register external player {hello.client_id!r} while client is connected"
            )
        client.preinitialize_client_from_hello(hello)
        self._fire_client_added_event_once(client)
        self._external_stream_start_cbs[hello.client_id] = on_stream_start
        self._cancel_reclaim_timeout(hello.client_id)
        if timeout_s > 0:
            self._schedule_external_registration_timeout(hello.client_id, timeout_s)
        else:
            self._cancel_external_registration_timeout(hello.client_id)
        return client

    def _fire_client_added_event_once(self, client: SendspinClient) -> None:
        """Fire ClientAddedEvent once per persistent client lifetime."""
        if client._added_event_fired:  # noqa: SLF001
            return
        client._added_event_fired = True  # noqa: SLF001
        self._signal_event(ClientAddedEvent(client.client_id))

    def unregister_external_player(self, client_id: str) -> None:
        """Remove external stream-start callback for a registered client."""
        self._cancel_external_registration_timeout(client_id)
        self._external_stream_start_cbs.pop(client_id, None)

    def is_external_player(self, client_id: str) -> bool:
        """Whether this client is externally managed for playback connects."""
        return client_id in self._external_stream_start_cbs

    def on_client_transport_attached(self, client_id: str) -> None:
        """Cancel pending connection-deadline timeouts once transport is attached."""
        self._cancel_external_registration_timeout(client_id)
        self._cancel_reclaim_timeout(client_id)

    def add_event_listener(
        self, callback: Callable[[SendspinServer, SendspinEvent], None]
    ) -> Callable[[], None]:
        """Register a callback for server events and return an unsubscribe callable."""
        self._event_cbs.append(callback)

        def _remove() -> None:
            with suppress(ValueError):
                self._event_cbs.remove(callback)

        return _remove

    def _signal_event(self, event: SendspinEvent) -> None:
        for cb in self._event_cbs:
            try:
                cb(self, event)
            except Exception:
                logger.exception("Error in event listener")

    def _signal_client_updated(self, client_id: str) -> None:
        """Emit a ClientUpdatedEvent (called from SendspinClient)."""
        self._signal_event(ClientUpdatedEvent(client_id))

    async def on_client_connect(self, request: web.Request) -> web.StreamResponse:
        """Handle an incoming WebSocket connection from a Sendspin client."""
        logger.debug("Incoming client connection from %s", request.remote)

        conn = SendspinConnection(self, request=request)
        self._pending_connections.add(conn)
        try:
            await conn._handle_client()  # noqa: SLF001
        finally:
            self._pending_connections.discard(conn)

        websocket = conn.websocket_connection
        assert isinstance(websocket, web.WebSocketResponse)
        return websocket

    def connect_to_client(
        self, url: str, *, connection_reason: ConnectionReason = ConnectionReason.DISCOVERY
    ) -> None:
        """Start a background connection attempt to a client URL.

        Initial connection failures are logged and stop the background task.
        Automatic retries only happen after at least one successful connection.
        """
        self._connection_reasons[url] = connection_reason
        prev_task = self._connection_tasks.get(url)
        if prev_task is not None:
            if retry_event := self._retry_events.get(url):
                retry_event.set()
            return

        self._initial_connect_succeeded.discard(url)
        self._retry_events[url] = asyncio.Event()
        self._connection_tasks[url] = create_task(
            self._handle_client_connection(url),
            eager_start=False,
        )

    async def connect_to_client_and_wait(
        self, url: str, *, connection_reason: ConnectionReason = ConnectionReason.DISCOVERY
    ) -> None:
        """Connect to a client and wait for the initial connection attempt.

        Raises:
            ClientConnectionError: If the initial connection to the client fails.
            ClientResponseError: If the client responds with an error HTTP status.
            TimeoutError: If the initial connection attempt times out.
            Exception: Other unexpected errors during the initial connection attempt.
        """
        self._connection_reasons[url] = connection_reason
        if url in self._initial_connect_succeeded:
            return

        waiter: asyncio.Future[None] = self._loop.create_future()
        self._initial_connect_waiters.setdefault(url, []).append(waiter)

        prev_task = self._connection_tasks.get(url)
        if prev_task is not None:
            if retry_event := self._retry_events.get(url):
                retry_event.set()
        else:
            self._initial_connect_succeeded.discard(url)
            self._retry_events[url] = asyncio.Event()
            self._connection_tasks[url] = create_task(
                self._handle_client_connection(url),
                eager_start=False,
            )

        await waiter

    def get_connection_reason(self, url: str) -> ConnectionReason:
        """Get the connection reason for a URL (for use by SendspinConnection)."""
        return self._connection_reasons.get(url, ConnectionReason.DISCOVERY)

    def register_client_url(self, client_id: str, url: str) -> None:
        """Record the URL used to connect to a client.

        If the client was previously registered with a different URL, the
        connection to that stale URL is cancelled.
        """
        previous_url = self._client_urls.get(client_id)
        self._client_urls[client_id] = url
        if previous_url is not None and previous_url != url:
            self.disconnect_from_client(previous_url)

    def get_client_url(self, client_id: str) -> str | None:
        """Get the URL for a client (for reconnection)."""
        return self._client_urls.get(client_id)

    def reclaim_client_for_playback(self, client_id: str, timeout_s: float = 30.0) -> bool:
        """Attempt to reconnect to a client for playback.

        Returns True if reconnection was initiated, False if no URL available.
        Used when starting playback to reclaim clients that disconnected with 'another_server'.

        The client must reconnect within timeout_s seconds; otherwise it is
        fully unregistered from this server.
        """
        if timeout_s < 0:
            raise ValueError("timeout_s must be >= 0")
        url = self._client_urls.get(client_id)
        if url is None:
            return False

        self.connect_to_client(url, connection_reason=ConnectionReason.PLAYBACK)
        if timeout_s > 0:
            self._schedule_reclaim_timeout(client_id, timeout_s)
        else:
            self._cancel_reclaim_timeout(client_id)
        return True

    def request_client_playback_connection(self, client_id: str) -> bool:
        """Request that a disconnected client connect for playback."""
        external_cb = self._external_stream_start_cbs.get(client_id)
        if external_cb is None:
            return self.reclaim_client_for_playback(client_id)

        request = ExternalStreamStartRequest(
            client_id=client_id,
            server=self,
        )
        try:
            external_cb(request)
        except Exception:
            logger.exception(
                "External stream-start callback failed for client_id=%s",
                client_id,
            )
            return False
        return True

    def disconnect_from_client(self, url: str) -> None:
        """Disconnect a server-initiated connection previously established via connect_to_client."""
        connection_task = self._connection_tasks.pop(url, None)
        if connection_task is not None:
            connection_task.cancel()

    def _cancel_external_registration_timeout(self, client_id: str) -> None:
        """Cancel a pending external-registration timeout for a client."""
        handle = self._external_registration_timeouts.pop(client_id, None)
        if handle is not None:
            handle.cancel()

    def _cancel_reclaim_timeout(self, client_id: str) -> None:
        """Cancel a pending reclaim timeout for a client."""
        handle = self._reclaim_timeouts.pop(client_id, None)
        if handle is not None:
            handle.cancel()

    def _schedule_external_registration_timeout(self, client_id: str, timeout_s: float) -> None:
        """Schedule full unregister if an externally registered client never connects."""
        self._cancel_external_registration_timeout(client_id)
        self._external_registration_timeouts[client_id] = self._loop.call_later(
            timeout_s,
            lambda: create_task(self._expire_external_registration(client_id)),
        )

    def _schedule_reclaim_timeout(self, client_id: str, timeout_s: float) -> None:
        """Schedule full unregister if reclaim never reconnects the client."""
        self._cancel_reclaim_timeout(client_id)
        self._reclaim_timeouts[client_id] = self._loop.call_later(
            timeout_s,
            lambda: create_task(self._expire_reclaim(client_id)),
        )

    async def _expire_external_registration(self, client_id: str) -> None:
        """Handle external-registration timeout."""
        self._external_registration_timeouts.pop(client_id, None)
        await self._full_unregister_disconnected_client(client_id)

    async def _expire_reclaim(self, client_id: str) -> None:
        """Handle reclaim timeout."""
        self._reclaim_timeouts.pop(client_id, None)
        await self._full_unregister_disconnected_client(client_id)

    async def _full_unregister_disconnected_client(self, client_id: str) -> None:
        """Fully unregister a client if it still has no active connection."""
        client = self._clients.get(client_id)
        if client is not None and client.connection is not None:
            return
        self.unregister_external_player(client_id)
        await self.remove_client(client_id)

    def _resolve_initial_connect_waiters(self, url: str, err: BaseException | None = None) -> None:
        """Resolve or fail waiters for an initial connection attempt."""
        waiters = self._initial_connect_waiters.pop(url, [])
        for waiter in waiters:
            if waiter.done():
                continue
            if err is None:
                waiter.set_result(None)
            else:
                waiter.set_exception(err)

    async def _handle_client_connection(self, url: str) -> None:  # noqa: PLR0915
        """Handle a server-initiated WebSocket connection task."""
        backoff = 1.0
        first_connection_succeeded = False

        try:
            while True:
                retry_event = self._retry_events.get(url)
                try:
                    async with self._client_session.ws_connect(
                        url,
                        heartbeat=30,
                        timeout=ClientWSTimeout(ws_close=10, ws_receive=60),  # pyright: ignore[reportCallIssue]
                    ) as wsock:
                        if not first_connection_succeeded:
                            first_connection_succeeded = True
                            self._initial_connect_succeeded.add(url)
                            self._resolve_initial_connect_waiters(url)
                        connection_started_s = time.monotonic()
                        conn = SendspinConnection(self, wsock_client=wsock, url=url)
                        await conn._handle_client()  # noqa: SLF001
                        session_duration_s = time.monotonic() - connection_started_s

                    if session_duration_s >= STABLE_SERVER_INITIATED_SESSION_S:
                        backoff = 1.0

                    if not conn.should_retry_server_initiated_connection:
                        if conn.goodbye_reason == GoodbyeReason.ANOTHER_SERVER:
                            logger.debug(
                                "Not reconnecting to %s after goodbye reason another_server",
                                url,
                            )
                        break

                    if self._client_session.closed:
                        break

                except asyncio.CancelledError:
                    if not first_connection_succeeded:
                        self._resolve_initial_connect_waiters(url, asyncio.CancelledError())
                    break
                except TimeoutError:
                    if not first_connection_succeeded:
                        logger.debug("Initial connection to %s timed out", url)
                        self._resolve_initial_connect_waiters(url, TimeoutError())
                        return
                    logger.debug("Connection task for %s timed out", url)
                except (ClientConnectionError, ClientResponseError) as err:
                    if not first_connection_succeeded:
                        logger.debug("Initial connection to %s failed: %s", url, err)
                        self._resolve_initial_connect_waiters(url, err)
                        return
                    logger.debug("Connection task for %s failed: %s", url, err)

                if backoff >= MAX_RECONNECT_BACKOFF_S:
                    break

                logger.debug("Trying to reconnect to client at %s in %.1fs", url, backoff)
                if retry_event is not None:
                    try:
                        await asyncio.wait_for(retry_event.wait(), timeout=backoff)
                        retry_event.clear()
                    except TimeoutError:
                        pass
                else:
                    await asyncio.sleep(backoff)
                backoff *= 2
        except asyncio.CancelledError:
            if not first_connection_succeeded:
                self._resolve_initial_connect_waiters(url, asyncio.CancelledError())
        except Exception as err:
            if not first_connection_succeeded:
                self._resolve_initial_connect_waiters(url, err)
            logger.exception("Unexpected error occurred")
        finally:
            self._connection_tasks.pop(url, None)
            self._retry_events.pop(url, None)
            self._initial_connect_succeeded.discard(url)
            self._connection_reasons.pop(url, None)

    async def start_server(
        self,
        port: int = 8927,
        host: str = "0.0.0.0",
        advertise_addresses: list[str] | None = None,
        *,
        discover_clients: bool = True,
    ) -> None:
        """Start the server HTTP listener and (optionally) mDNS discovery/advertising."""
        if self._app is not None:
            logger.warning("Server is already running")
            return

        logger.info("Starting Sendspin server on port %d", port)
        self._app = self._create_web_application()
        self._app_runner = web.AppRunner(self._app)
        await self._app_runner.setup()

        try:
            self._tcp_site = web.TCPSite(
                self._app_runner,
                host=host if host != "0.0.0.0" else None,
                port=port,
            )
            await self._tcp_site.start()
            logger.info("Sendspin server started successfully on %s:%d", host, port)

            self._zc = AsyncZeroconf(
                ip_version=IPVersion.V4Only,
                interfaces=[host] if host != "0.0.0.0" else InterfaceChoice.Default,
            )

            if advertise_addresses is not None:
                addresses = advertise_addresses
            elif local_ip := get_local_ip():
                addresses = [local_ip]
            else:
                addresses = []

            if addresses:
                await self._start_mdns_advertising(
                    addresses=addresses, port=port, path=self.API_PATH
                )
            else:
                logger.warning(
                    "No IP addresses available for mDNS advertising. "
                    "Clients may not be able to discover this server. "
                    "Consider specifying addresses manually via advertise_addresses."
                )

            if discover_clients:
                await self._start_mdns_discovery()
        except OSError as e:
            logger.error("Failed to start server on %s:%d: %s", host, port, e)
            await self._stop_mdns()
            if self._app_runner:
                await self._app_runner.cleanup()
                self._app_runner = None
            if self._app:
                await self._app.shutdown()
                self._app = None
            raise

    async def stop_server(self) -> None:
        """Stop the server HTTP listener and mDNS services."""
        await self._stop_mdns()

        if self._tcp_site:
            await self._tcp_site.stop()
            self._tcp_site = None

        if self._app_runner:
            await self._app_runner.cleanup()
            self._app_runner = None

        if self._app:
            await self._app.shutdown()
            self._app = None

    async def close(self) -> None:
        """Close the server and disconnect all active connections."""
        for task in self._connection_tasks.values():
            task.cancel()

        # Close pending incoming connections so their handlers can exit promptly.
        for conn in list(self._pending_connections):
            wsock = conn.websocket_connection
            if wsock.closed:
                continue
            logger.debug("Closing pending client connection")
            try:
                async with asyncio.timeout(1.0):
                    await wsock.close()
            except TimeoutError:
                logger.debug("Timeout while closing pending client websocket")

        disconnect_tasks: list[asyncio.Task[None]] = []
        for client in self._clients.values():
            if client.connection is None:
                continue
            disconnect_tasks.append(
                create_task(client.connection.disconnect(retry_connection=False))
            )
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        await self.stop_server()
        if self._owns_session and not self._client_session.closed:
            await self._client_session.close()

    async def _start_mdns_advertising(self, addresses: list[str], port: int, path: str) -> None:
        assert self._zc is not None
        if self._mdns_service is not None:
            await self._zc.async_unregister_service(self._mdns_service)

        service_type = "_sendspin-server._tcp.local."
        properties: dict[str, str] = {}
        if self._name:
            properties["name"] = self._name
        properties["path"] = path

        info = AsyncServiceInfo(
            type_=service_type,
            name=f"{self._id}.{service_type}",
            server=f"{self._id}.local.",
            parsed_addresses=addresses,
            port=port,
            properties=properties,
        )
        try:
            await self._zc.async_register_service(info)
            self._mdns_service = info
            logger.debug("mDNS advertising server on port %d with path %s", port, path)
        except NonUniqueNameException:
            logger.error("Sendspin server with identical name present in the local network!")

    async def _start_mdns_discovery(self) -> None:
        assert self._zc is not None

        service_type = "_sendspin._tcp.local."
        self._mdns_browser = AsyncServiceBrowser(
            self._zc.zeroconf,
            service_type,
            handlers=[self._on_mdns_service_state_change],
        )

    def _on_mdns_service_state_change(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        if state_change in (ServiceStateChange.Added, ServiceStateChange.Updated):

            def _schedule_add() -> None:
                create_task(self._handle_service_added(zeroconf, service_type, name))

            self._loop.call_soon_threadsafe(_schedule_add)
        elif state_change is ServiceStateChange.Removed:
            self._loop.call_soon_threadsafe(lambda: self._handle_service_removed(name))

    async def _handle_service_added(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        info = AsyncServiceInfo(service_type, name)
        if not info.load_from_cache(zeroconf):
            await info.async_request(zeroconf, 3000)

        if not info.parsed_addresses():
            return

        address = _get_first_valid_ip(info.parsed_addresses())
        if address is None:
            return

        port = info.port
        path = None
        if info.properties:
            for k, v in info.properties.items():
                key = k.decode() if isinstance(k, bytes) else k
                if key == "path" and v is not None:
                    path = v.decode() if isinstance(v, bytes) else v
                    break

        if port is None:
            return
        if path is None or not str(path).startswith("/"):
            return

        url = f"ws://{address}:{port}{path}"
        self._mdns_client_urls[name] = url
        self.connect_to_client(url)

    def _handle_service_removed(self, name: str) -> None:
        url = self._mdns_client_urls.pop(name, None)
        if url is not None:
            self.disconnect_from_client(url)
            create_task(self._cleanup_retained_clients_for_removed_mdns_url(url))

    async def _cleanup_retained_clients_for_removed_mdns_url(self, url: str) -> None:
        """Remove retained ANOTHER_SERVER clients when their mDNS URL disappears."""
        client_ids = [
            client_id for client_id, known_url in self._client_urls.items() if known_url == url
        ]
        for client_id in client_ids:
            self._client_urls.pop(client_id, None)

        for client_id in client_ids:
            client = self._clients.get(client_id)
            if client is None:
                continue

            if client.cleanup_on_mdns_removal and not client.is_connected:
                await self.remove_client(client_id)

    async def _stop_mdns(self) -> None:
        if self._zc is None:
            return
        try:
            if self._mdns_browser is not None:
                await self._mdns_browser.async_cancel()
            if self._mdns_service is not None:
                await self._zc.async_unregister_service(self._mdns_service)
        finally:
            await self._zc.async_close()
            self._zc = None
            self._mdns_service = None
            self._mdns_browser = None
