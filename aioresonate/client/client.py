"""Resonate Client implementation to connect to a Resonate Server."""

import asyncio
import logging

from aiohttp import ClientSession, web

logger = logging.getLogger(__name__)


async def _get_ip_pton(ip_string: str) -> bytes:
    """Return socket pton for a local ip."""
    # TODO: this is also in server, move to utils?
    try:
        return await asyncio.to_thread(socket.inet_pton, socket.AF_INET, ip_string)
    except OSError:
        return await asyncio.to_thread(socket.inet_pton, socket.AF_INET6, ip_string)


class ResonateClient:
    """Resonate Client implementation to connect to a Resonate Server."""

    _loop: asyncio.AbstractEventLoop
    _id: str
    _name: str
    _client_session: ClientSession
    """The client session used to connect to clients."""
    _owns_session: bool
    """Whether this server instance owns the client session."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        client_id: str,
        client_name: str,
        client_session: ClientSession | None = None,
    ) -> None:
        """Initialize Resonate Client."""
        self.loop = loop
        self._id = client_id
        self._name = client_name
        self._session = client_session or ClientSession(loop=loop)
        self.connected = False

    async def on_client_connect(self, request: web.Request) -> web.StreamResponse:
        """Handle an incoming WebSocket connection from a Resonate client."""
        logger.debug("Incoming client connection from %s", request.remote)

        client = Client(
            self,
            handle_client_connect=self._handle_client_connect,
            handle_client_disconnect=self._handle_client_disconnect,
            request=request,
        )
        await client._handle_client()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

        websocket = client.websocket_connection
        # This is a WebSocketResponse since we just created client
        # as client-initiated.
        assert isinstance(websocket, web.WebSocketResponse)
        return websocket

    async def _start_mdns_advertising(self, host: str, port: int, path: str) -> None:
        """Start advertising this server via mDNS."""
        assert self._zc is not None
        if self._mdns_service is not None:
            await self._zc.async_unregister_service(self._mdns_service)

        properties = {"path": path}
        service_type = "_resonate-server._tcp.local."
        info = AsyncServiceInfo(
            type_=service_type,
            name=f"{self._name}.{service_type}",
            addresses=[await _get_ip_pton(host)] if host != "0.0.0.0" else None,
            port=port,
            properties=properties,
        )
        await self._zc.async_register_service(info)
        self._mdns_service = info

        logger.debug("mDNS advertising server on port %d with path %s", port, path)

    def start_server_initiated(self, port: int | None = None, host: str = "0.0.0.0") -> None:
        """
        Start the client.

        Resonate support two modes of operation for the initial establishment of the
        WebSocket connection between server and client.

        This will use the server-initiated mode, which will:
        - Start a resonate compatible WebSocket endpoint to accept connections from
          the server.
        - Start advertising the server on the network via mDNS.

        Args:
            port: The port to listen on. If None, a random port will be used.
        """
        if self._app is not None:
            logger.warning("Server is already running")
            return

        api_path = "/resonate"
        logger.info("Starting Resonate server on port %d", port)
        self._app = web.Application()
        # Create perpetual WebSocket route for client connections
        _ = self._app.router.add_get(api_path, self.on_client_connect)
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
            # Start mDNS advertise and discovery
            self._zc = AsyncZeroconf(
                ip_version=IPVersion.V4Only, interfaces=InterfaceChoice.Default
            )
            await self._start_mdns_advertising(host=host, port=port, path=api_path)
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
