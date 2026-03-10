"""Tests for initial server-initiated connection behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientConnectionError

from aiosendspin.models.types import GoodbyeReason
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


class _PersistentSuccessfulSession:
    """Client session with successful connects and open lifecycle."""

    def __init__(self) -> None:
        self.closed = False
        self.calls = 0

    def ws_connect(self, *_args: object, **_kwargs: object) -> _SuccessfulConnectContext:
        self.calls += 1
        return _SuccessfulConnectContext()

    async def close(self) -> None:
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


@pytest.mark.asyncio
async def test_server_initiated_stops_retrying_on_another_server_goodbye(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server-initiated loop must stop when client disconnects with ANOTHER_SERVER."""
    session = _PersistentSuccessfulSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    class _FakeConnection:
        closing = False

        def __init__(
            self,
            _server: SendspinServer,
            *,
            wsock_client: object,  # noqa: ARG002
            url: str | None = None,  # noqa: ARG002
        ) -> None:
            self.goodbye_reason = GoodbyeReason.ANOTHER_SERVER
            self.should_retry_server_initiated_connection = False

        async def _handle_client(self) -> None:
            return

    monkeypatch.setattr("aiosendspin.server.server.SendspinConnection", _FakeConnection)

    server.connect_to_client(url)

    for _ in range(40):
        if url not in server._connection_tasks:  # noqa: SLF001
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("Connection task was not cleaned up after ANOTHER_SERVER goodbye")

    assert session.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("monotonic_values", "expected_sleeps"),
    [
        # Repeated unstable sessions should grow backoff: 1s -> 2s -> 4s.
        ([0.0, 5.0, 6.0, 9.0, 10.0, 13.0, 14.0, 16.0], [1.0, 2.0, 4.0]),
        # Unstable then stable should reset, then unstable grows again.
        ([0.0, 5.0, 6.0, 18.0, 19.0, 23.0, 24.0, 27.0], [1.0, 1.0, 2.0]),
        # Repeated stable sessions should keep reconnect delay at 1s.
        ([0.0, 12.0, 13.0, 24.0, 25.0, 35.0, 36.0, 49.0], [1.0, 1.0, 1.0]),
    ],
)
async def test_server_initiated_backoff_resets_only_after_stable_session(
    monkeypatch: pytest.MonkeyPatch,
    monotonic_values: list[float],
    expected_sleeps: list[float],
) -> None:
    """Reconnect backoff should reset only after sufficiently long sessions."""
    session = _PersistentSuccessfulSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"
    max_attempts = len(monotonic_values) // 2
    attempts = 0
    sleep_calls: list[float] = []
    monotonic_iter = iter(monotonic_values)

    class _FakeConnection:
        closing = False

        def __init__(
            self,
            _server: SendspinServer,
            *,
            wsock_client: object,  # noqa: ARG002
            url: str | None = None,  # noqa: ARG002
        ) -> None:
            nonlocal attempts
            attempts += 1
            self.should_retry_server_initiated_connection = attempts < max_attempts
            self.goodbye_reason = None

        async def _handle_client(self) -> None:
            return

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    def _fake_monotonic() -> float:
        return next(monotonic_iter)

    monkeypatch.setattr("aiosendspin.server.server.SendspinConnection", _FakeConnection)
    monkeypatch.setattr(
        "aiosendspin.server.server.time",
        SimpleNamespace(monotonic=_fake_monotonic),
    )
    monkeypatch.setattr("aiosendspin.server.server.asyncio.sleep", _fake_sleep)

    await server._handle_client_connection(url)  # noqa: SLF001

    assert sleep_calls == expected_sleeps


@pytest.mark.asyncio
async def test_mdns_removal_cleans_retained_another_server_client() -> None:
    """Removing mDNS entry should clean retained ANOTHER_SERVER clients."""
    session = _PersistentSuccessfulSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    retained_client = MagicMock()
    retained_client.is_connected = False
    retained_client.cleanup_on_mdns_removal = True

    server._clients = {"client-1": retained_client}  # noqa: SLF001
    server._client_urls = {"client-1": url}  # noqa: SLF001
    server._mdns_client_urls = {"service._sendspin._tcp.local.": url}  # noqa: SLF001
    server.remove_client = AsyncMock()  # type: ignore[method-assign]

    server._handle_service_removed("service._sendspin._tcp.local.")  # noqa: SLF001
    await asyncio.sleep(0)

    server.remove_client.assert_awaited_once_with("client-1")
    assert "client-1" not in server._client_urls  # noqa: SLF001


@pytest.mark.asyncio
async def test_mdns_removal_keeps_non_retained_client() -> None:
    """MDNS removal should not remove clients that are not marked for mDNS cleanup."""
    session = _PersistentSuccessfulSession()
    server = _make_server(session)
    url = "ws://127.0.0.1:9999/sendspin"

    retained_client = MagicMock()
    retained_client.is_connected = False
    retained_client.cleanup_on_mdns_removal = False

    server._clients = {"client-1": retained_client}  # noqa: SLF001
    server._client_urls = {"client-1": url}  # noqa: SLF001
    server._mdns_client_urls = {"service._sendspin._tcp.local.": url}  # noqa: SLF001
    server.remove_client = AsyncMock()  # type: ignore[method-assign]

    server._handle_service_removed("service._sendspin._tcp.local.")  # noqa: SLF001
    await asyncio.sleep(0)

    server.remove_client.assert_not_awaited()
