"""Tests for initial server-initiated connection behavior."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientConnectionError

from aiosendspin.server.server import SendspinServer


class _FailingInitialConnectSession:
    """Client session whose ws_connect fails immediately."""

    def __init__(self) -> None:
        self.closed = False
        self.calls = 0

    def ws_connect(self, *_args: object, **_kwargs: object) -> object:
        """Raise an initial connection error."""
        self.calls += 1
        raise ClientConnectionError("boom")

    async def close(self) -> None:
        """Close session."""
        self.closed = True


class _SuccessfulConnectContext:
    """Async context manager returning a websocket stub."""

    async def __aenter__(self) -> object:
        return MagicMock()

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        return None


class _SuccessfulInitialConnectSession:
    """Client session whose first connection succeeds."""

    def __init__(self) -> None:
        self.closed = True
        self.calls = 0

    def ws_connect(self, *_args: object, **_kwargs: object) -> _SuccessfulConnectContext:
        """Return a successful websocket context manager."""
        self.calls += 1
        return _SuccessfulConnectContext()

    async def close(self) -> None:
        """Close session."""
        self.closed = True


def _make_server(client_session: object) -> SendspinServer:
    """Create server with injected client session test double."""
    loop = asyncio.get_running_loop()
    return SendspinServer(
        loop=loop,
        server_id="srv",
        server_name="server",
        client_session=client_session,
    )


async def _wait_for_connection_task_cleanup(server: SendspinServer, url: str) -> None:
    """Wait until a connection task is removed from bookkeeping."""
    for _ in range(50):
        task = server._connection_tasks.get(url)  # noqa: SLF001
        if task is None:
            return
        if task.done():
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_connect_to_client_and_wait_raises_on_initial_connection_failure() -> None:
    """Initial connection failure should propagate to waiting caller."""
    session = _FailingInitialConnectSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    with pytest.raises(ClientConnectionError):
        await server.connect_to_client_and_wait(url)

    await _wait_for_connection_task_cleanup(server, url)
    assert session.calls == 1
    assert url not in server._connection_tasks  # noqa: SLF001
    assert url not in server._retry_events  # noqa: SLF001


@pytest.mark.asyncio
async def test_connect_to_client_stops_after_initial_failure_without_retry() -> None:
    """Background connection should stop when first attempt fails."""
    session = _FailingInitialConnectSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    server.connect_to_client(url)
    await _wait_for_connection_task_cleanup(server, url)

    assert session.calls == 1
    assert url not in server._connection_tasks  # noqa: SLF001


@pytest.mark.asyncio
async def test_connect_to_client_and_wait_returns_on_initial_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waiting connect call should return once first connection succeeds."""
    session = _SuccessfulInitialConnectSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    class _FakeConnection:
        """Connection double used to bypass full websocket lifecycle."""

        closing = False

        def __init__(
            self,
            _server: SendspinServer,
            *,
            wsock_client: object,
            url: str | None = None,  # noqa: ARG002
        ) -> None:
            self._wsock_client = wsock_client

        async def _handle_client(self) -> None:
            return

    monkeypatch.setattr("aiosendspin.server.server.SendspinConnection", _FakeConnection)

    await server.connect_to_client_and_wait(url)

    assert session.calls == 1
