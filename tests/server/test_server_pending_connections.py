"""Tests for pending incoming connection shutdown handling."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

from aiosendspin.server.server import SendspinServer


class _FakeWebSocket:
    """Minimal websocket test double used by pending-connection tests."""

    def __init__(self, *, raise_timeout: bool = False) -> None:
        self.raise_timeout = raise_timeout
        self.closed = False
        self.close_calls = 0

    async def close(self) -> None:
        """Close websocket or simulate timeout."""
        self.close_calls += 1
        if self.raise_timeout:
            raise TimeoutError
        self.closed = True


class _FakePendingConnection:
    """Minimal pending-connection test double."""

    def __init__(self, websocket: _FakeWebSocket) -> None:
        self._websocket = websocket

    @property
    def websocket_connection(self) -> _FakeWebSocket:
        """Expose websocket in the same shape as SendspinConnection."""
        return self._websocket


def _make_server() -> SendspinServer:
    """Create a server instance with a mocked client session."""
    loop = asyncio.get_running_loop()
    client_session = MagicMock()
    client_session.closed = True
    client_session.close = AsyncMock()
    return SendspinServer(
        loop=loop, server_id="srv", server_name="server", client_session=client_session
    )


@pytest.mark.asyncio
async def test_close_closes_pending_client_connections() -> None:
    """close() should close websockets for pending incoming connections."""
    server = _make_server()
    websocket = _FakeWebSocket()
    server._pending_connections.add(_FakePendingConnection(websocket))  # noqa: SLF001
    server.stop_server = AsyncMock()  # type: ignore[method-assign]

    await server.close()

    assert websocket.close_calls == 1
    assert websocket.closed is True
    server.stop_server.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_ignores_pending_connection_close_timeout() -> None:
    """close() should continue shutdown when pending websocket close times out."""
    server = _make_server()
    websocket = _FakeWebSocket(raise_timeout=True)
    server._pending_connections.add(_FakePendingConnection(websocket))  # noqa: SLF001
    server.stop_server = AsyncMock()  # type: ignore[method-assign]

    await server.close()

    assert websocket.close_calls == 1
    server.stop_server.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_client_connect_cleans_pending_set_on_handler_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """on_client_connect() should always remove connection from pending set."""
    server = _make_server()
    request = MagicMock(spec=web.Request)
    request.remote = "127.0.0.1"

    class _FailingConnection:
        """Connection double whose handler fails immediately."""

        def __init__(
            self,
            _server: SendspinServer,
            *,
            request: web.Request,  # noqa: ARG002
        ) -> None:
            self.websocket_connection = web.WebSocketResponse()

        async def _handle_client(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr("aiosendspin.server.server.SendspinConnection", _FailingConnection)

    with pytest.raises(RuntimeError, match="boom"):
        await server.on_client_connect(request)

    assert server._pending_connections == set()  # noqa: SLF001
