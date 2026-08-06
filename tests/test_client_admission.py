"""Tests for the client-side connection arbiter (multi-server admission)."""
# ruff: noqa: SLF001 - exercises the client's internal admission methods directly.

from __future__ import annotations

import asyncio

import pytest

from aiosendspin.client.client import SendspinClient
from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.types import Activity, GoodbyeReason, PairAbortReason, Roles
from aiosendspin.noise.trust_store import InMemoryClientPairingStore

from .conftest import make_sdk_client


class _FakeConnection:
    """Stand-in exposing only the surface the arbiter touches on a connection."""

    def __init__(
        self,
        *,
        server_id: str | None = "server",
        activities: list[Activity] | None = None,
        connected: bool = True,
        pairing_attempt_in_progress: bool = False,
        client: SendspinClient | None = None,
    ) -> None:
        """Record the connection's arbitration-relevant state and call sinks."""
        self.server_id = server_id
        self.activities = activities or []
        self.connected = connected
        self.pairing_attempt_in_progress = pairing_attempt_in_progress
        self._client = client
        self.goodbye_reason: GoodbyeReason | None = None
        self.pair_abort_reason: PairAbortReason | None = None
        self.disconnected = False

    @property
    def is_pairing(self) -> bool:
        """Whether this connection is currently a pairing connection."""
        return Activity.PAIRING in self.activities

    async def send_goodbye(self, reason: GoodbyeReason) -> None:
        """Record the goodbye reason the arbiter sent."""
        self.goodbye_reason = reason

    async def send_pair_abort(self, reason: PairAbortReason) -> None:
        """Record the pair/abort reason the arbiter sent."""
        self.pair_abort_reason = reason

    async def disconnect(self) -> None:
        """Mark disconnected and notify the owning client."""
        self.disconnected = True
        if self._client is not None:
            self._client.on_connection_closed(self)  # type: ignore[arg-type]


def _client() -> SendspinClient:
    return make_sdk_client(client_name="Test Client", roles=[Roles.CONTROLLER])


# --- _activity_rank ---


def test_activity_rank_orders_management_over_playback_over_pairing() -> None:
    """Management > playback > pairing > none."""
    rank = SendspinClient._activity_rank
    assert rank([Activity.MANAGEMENT, Activity.PLAYBACK]) == 3
    assert rank([Activity.PLAYBACK]) == 2
    assert rank([Activity.PAIRING]) == 1
    assert rank([]) == 0


# --- _should_admit_connection ---


async def test_admits_when_no_current_connection() -> None:
    """The first connection is always admitted."""
    client = _client()
    incoming = _FakeConnection(activities=[Activity.PAIRING])
    assert client._should_admit_connection(incoming) is True


async def test_admits_when_current_connection_disconnected() -> None:
    """A dead holder never blocks an incoming connection."""
    client = _client()
    client._admitted_connection = _FakeConnection(connected=False)  # type: ignore[assignment]
    incoming = _FakeConnection(activities=[Activity.PAIRING])
    assert client._should_admit_connection(incoming) is True


async def test_higher_rank_displaces_lower() -> None:
    """A management connection displaces an admitted playback connection."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.PLAYBACK]
    )
    incoming = _FakeConnection(activities=[Activity.MANAGEMENT])
    assert client._should_admit_connection(incoming) is True


async def test_lower_rank_rejected() -> None:
    """A playback connection does not displace an admitted management connection."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.MANAGEMENT]
    )
    incoming = _FakeConnection(activities=[Activity.PLAYBACK])
    assert client._should_admit_connection(incoming) is False


async def test_equal_nonempty_rank_admitted() -> None:
    """An incoming playback connection displaces an admitted playback connection."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.PLAYBACK]
    )
    incoming = _FakeConnection(activities=[Activity.PLAYBACK])
    assert client._should_admit_connection(incoming) is True


async def test_inflight_pairing_not_displaced_by_pairing() -> None:
    """An in-flight pairing attempt is protected from a competing incoming pairing."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.PAIRING], pairing_attempt_in_progress=True
    )
    incoming = _FakeConnection(activities=[Activity.PAIRING])
    assert client._should_admit_connection(incoming) is False


async def test_inflight_pairing_not_displaced_by_playback() -> None:
    """An in-flight pairing attempt is protected from an incoming playback connection."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.PAIRING], pairing_attempt_in_progress=True
    )
    incoming = _FakeConnection(activities=[Activity.PLAYBACK])
    assert client._should_admit_connection(incoming) is False


async def test_inflight_pairing_displaced_by_management() -> None:
    """Management still outranks an in-flight pairing attempt."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.PAIRING], pairing_attempt_in_progress=True
    )
    incoming = _FakeConnection(activities=[Activity.MANAGEMENT])
    assert client._should_admit_connection(incoming) is True


async def test_idle_pairing_displaced_by_playback() -> None:
    """A pairing connection with no attempt in progress is not protected."""
    client = _client()
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        activities=[Activity.PAIRING], pairing_attempt_in_progress=False
    )
    incoming = _FakeConnection(activities=[Activity.PLAYBACK])
    assert client._should_admit_connection(incoming) is True


async def test_pairing_displaces_idle() -> None:
    """A pairing connection displaces an admitted idle (no-activity) connection."""
    client = _client()
    client._admitted_connection = _FakeConnection(activities=[])  # type: ignore[assignment]
    incoming = _FakeConnection(activities=[Activity.PAIRING])
    assert client._should_admit_connection(incoming) is True


async def test_both_idle_incoming_wins_when_last_playback() -> None:
    """Two idle connections resolve in favour of the last-playback server."""
    client = _client()
    client.last_playback_server_id = "server-A"
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        server_id="server-B", activities=[]
    )
    incoming = _FakeConnection(server_id="server-A", activities=[])
    assert client._should_admit_connection(incoming) is True


async def test_both_idle_incoming_loses_when_not_last_playback() -> None:
    """An idle incoming connection does not displace an idle holder it can't beat."""
    client = _client()
    client.last_playback_server_id = "server-A"
    client._admitted_connection = _FakeConnection(  # type: ignore[assignment]
        server_id="server-A", activities=[]
    )
    incoming = _FakeConnection(server_id="server-B", activities=[])
    assert client._should_admit_connection(incoming) is False


# --- _admit_connection / _reject_connection ---


async def test_admit_displaces_previous_with_another_server() -> None:
    """Admitting a new connection sends the prior holder another_server and drops it."""
    client = _client()
    previous = _FakeConnection(activities=[Activity.PLAYBACK], client=client)
    client._admitted_connection = previous  # type: ignore[assignment]
    incoming = _FakeConnection(activities=[Activity.MANAGEMENT], client=client)

    await client._admit_connection(incoming)  # type: ignore[arg-type]

    assert client._admitted_connection is incoming  # type: ignore[comparison-overlap]
    assert previous.goodbye_reason is GoodbyeReason.ANOTHER_SERVER
    assert previous.disconnected is True


async def test_admit_same_connection_is_noop() -> None:
    """Re-admitting the current connection neither dismisses nor disconnects it."""
    client = _client()
    current = _FakeConnection(activities=[Activity.PLAYBACK], client=client)
    client._admitted_connection = current  # type: ignore[assignment]

    await client._admit_connection(current)  # type: ignore[arg-type]

    assert client._admitted_connection is current  # type: ignore[comparison-overlap]
    assert current.disconnected is False
    assert current.goodbye_reason is None


async def test_admit_displaces_previous_pairing_with_pair_abort() -> None:
    """Displacing an in-flight pairing aborts it rather than sending a goodbye."""
    client = _client()
    previous = _FakeConnection(activities=[Activity.PAIRING], client=client)
    client._admitted_connection = previous  # type: ignore[assignment]
    incoming = _FakeConnection(activities=[Activity.MANAGEMENT], client=client)

    await client._admit_connection(incoming)  # type: ignore[arg-type]

    assert previous.pair_abort_reason is PairAbortReason.CONCURRENT_ATTEMPT
    assert previous.goodbye_reason is None
    assert previous.disconnected is True


async def test_reject_sends_concurrent_attempt_and_disconnects() -> None:
    """A rejected incoming connection gets concurrent_attempt and is dropped."""
    client = _client()
    incoming = _FakeConnection(activities=[Activity.PLAYBACK], client=client)

    await client._reject_connection(incoming)  # type: ignore[arg-type]

    assert incoming.goodbye_reason is GoodbyeReason.CONCURRENT_ATTEMPT
    assert incoming.disconnected is True
    assert client._admitted_connection is None


async def test_reject_pairing_sends_pair_abort() -> None:
    """A rejected incoming pairing connection gets a pair/abort, not a goodbye."""
    client = _client()
    incoming = _FakeConnection(activities=[Activity.PAIRING], client=client)

    await client._reject_connection(incoming)  # type: ignore[arg-type]

    assert incoming.pair_abort_reason is PairAbortReason.CONCURRENT_ATTEMPT
    assert incoming.goodbye_reason is None
    assert incoming.disconnected is True


# --- note_playback_activity ---


async def test_admit_playback_connection_records_last_playback() -> None:
    """Admitting a connection that carries the playback activity records its server."""
    client = _client()
    incoming = _FakeConnection(server_id="server-A", activities=[Activity.PLAYBACK], client=client)

    await client._admit_connection(incoming)  # type: ignore[arg-type]

    assert client.last_playback_server_id == "server-A"


async def test_admit_idle_connection_does_not_record() -> None:
    """Admitting an idle connection leaves the last-playback pointer untouched."""
    client = _client()
    client.last_playback_server_id = "server-A"
    incoming = _FakeConnection(server_id="server-B", activities=[], client=client)

    await client._admit_connection(incoming)  # type: ignore[arg-type]

    assert client.last_playback_server_id == "server-A"


async def test_admit_pairing_connection_does_not_record() -> None:
    """A pairing connection is not playback and never becomes the last-playback server."""
    client = _client()
    incoming = _FakeConnection(server_id="server-B", activities=[Activity.PAIRING], client=client)

    await client._admit_connection(incoming)  # type: ignore[arg-type]

    assert client.last_playback_server_id is None


async def test_note_playback_activity_ignores_non_admitted() -> None:
    """A connection that is not the admitted holder cannot move the pointer."""
    client = _client()
    admitted = _FakeConnection(server_id="server-A", activities=[], client=client)
    client._admitted_connection = admitted  # type: ignore[assignment]
    other = _FakeConnection(server_id="server-B", activities=[Activity.PLAYBACK], client=client)

    await client.note_playback_activity(other)  # type: ignore[arg-type]

    assert client.last_playback_server_id is None


async def test_note_playback_activity_persists_later_activation() -> None:
    """A server/activate that adds playback after admission is persisted, not just cached."""
    store = InMemoryClientPairingStore()
    client = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER], pairing_store=store)
    admitted = _FakeConnection(server_id="server-A", activities=[], client=client)
    client._admitted_connection = admitted  # type: ignore[assignment]
    admitted.activities = [Activity.PLAYBACK]  # a later activate declares playback

    await client.note_playback_activity(admitted)  # type: ignore[arg-type]

    assert await store.get_last_playback_server_id() == "server-A"


async def test_failed_last_playback_write_retries_on_next_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed store write leaves the cached pointer stale so a later activation retries."""
    store = InMemoryClientPairingStore()
    client = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER], pairing_store=store)
    admitted = _FakeConnection(server_id="server-A", activities=[Activity.PLAYBACK], client=client)
    client._admitted_connection = admitted  # type: ignore[assignment]
    written: list[str | None] = []

    async def flaky(server_id: str | None) -> None:
        written.append(server_id)
        if len(written) == 1:
            raise OSError("store unavailable")

    monkeypatch.setattr(store, "set_last_playback_server_id", flaky)

    with pytest.raises(OSError, match="store unavailable"):
        await client.note_playback_activity(admitted)  # type: ignore[arg-type]
    assert client.last_playback_server_id is None

    await client.note_playback_activity(admitted)  # type: ignore[arg-type]

    assert written == ["server-A", "server-A"]


async def test_failed_last_playback_write_leaves_admission_unpublished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed store write during admission keeps the previous connection admitted."""
    store = InMemoryClientPairingStore()
    client = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER], pairing_store=store)
    previous = _FakeConnection(server_id="server-A", activities=[], client=client)
    client._admitted_connection = previous  # type: ignore[assignment]
    incoming = _FakeConnection(server_id="server-B", activities=[Activity.PLAYBACK], client=client)

    async def boom(_server_id: str | None) -> None:
        raise OSError("store unavailable")

    monkeypatch.setattr(store, "set_last_playback_server_id", boom)

    with pytest.raises(OSError, match="store unavailable"):
        await client._admit_connection(incoming)  # type: ignore[arg-type]

    assert client._admitted_connection is previous
    assert not previous.disconnected


async def test_failed_last_playback_read_retries_on_next_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed store read leaves the loader armed so a later admission retries it."""
    store = InMemoryClientPairingStore()
    await store.set_last_playback_server_id("server-Z")
    client = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER], pairing_store=store)
    original = store.get_last_playback_server_id
    reads: list[None] = []

    async def flaky() -> str | None:
        reads.append(None)
        if len(reads) == 1:
            raise OSError("store unavailable")
        return await original()

    monkeypatch.setattr(store, "get_last_playback_server_id", flaky)

    with pytest.raises(OSError, match="store unavailable"):
        await client._ensure_last_playback_loaded()
    assert client.last_playback_server_id is None

    await client._ensure_last_playback_loaded()

    assert client.last_playback_server_id == "server-Z"


# --- on_connection_closed ---


async def test_admitted_close_clears_slot_and_notifies() -> None:
    """Losing the admitted connection clears the slot and fires the disconnect callback."""
    client = _client()
    fired: list[bool] = []
    client.add_disconnect_listener(lambda: fired.append(True))
    admitted = _FakeConnection(client=client)
    client._admitted_connection = admitted  # type: ignore[assignment]

    client.on_connection_closed(admitted)  # type: ignore[arg-type]

    assert client._admitted_connection is None
    assert fired == [True]


async def test_provisional_close_is_silent() -> None:
    """A provisional connection closing is not reported as a client disconnect."""
    client = _client()
    fired: list[bool] = []
    client.add_disconnect_listener(lambda: fired.append(True))
    provisional = _FakeConnection(client=client)
    client._provisional_connections.add(provisional)  # type: ignore[arg-type]

    client.on_connection_closed(provisional)  # type: ignore[arg-type]

    assert provisional not in client._provisional_connections
    assert fired == []


# --- provisional bring-up timeout ---


class _BlockingWebSocket:
    """A websocket whose ``receive()`` never returns, to probe bring-up deadlines."""

    closed = False

    def __init__(self) -> None:
        """Set up the never-set event backing ``receive``."""
        self._never = asyncio.Event()

    async def receive(self) -> object:
        """Block forever (until the awaiting task is cancelled)."""
        await self._never.wait()
        raise AssertionError  # pragma: no cover - the event is never set

    async def close(self) -> None:
        """Mark the socket closed."""
        self.closed = True


async def test_provisional_connection_times_out_during_bringup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incoming connection drops itself if bring-up never reaches server/activate."""
    monkeypatch.setattr("aiosendspin.client.connection.PROVISIONAL_CONNECTION_TIMEOUT_S", 0.02)
    client = _client()
    connection = SendspinConnection(client)

    async def fake_handshake(ws: object, **_: object) -> None:
        connection._connected = True
        connection._ws = ws  # type: ignore[assignment]

    async def fake_inner() -> None:
        await asyncio.Event().wait()  # never reaches server/activate

    monkeypatch.setattr(connection, "_run_noise_handshake", fake_handshake)
    monkeypatch.setattr(connection, "_run_inner_handshake", fake_inner)

    with pytest.raises(TimeoutError):
        await connection.attach_websocket(_BlockingWebSocket(), expected_server_id=None)  # type: ignore[arg-type]
    assert connection._closed.is_set()


async def test_client_initiated_connection_times_out_during_bringup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client-initiated dial drops itself if bring-up never reaches server/activate."""
    monkeypatch.setattr("aiosendspin.client.connection.PROVISIONAL_CONNECTION_TIMEOUT_S", 0.02)
    client = _client()
    connection = SendspinConnection(client)

    async def fake_handshake(raw_ws: object, **_: object) -> None:
        connection._connected = True
        connection._ws = raw_ws  # type: ignore[assignment]

    async def fake_inner() -> None:
        await asyncio.Event().wait()  # never reaches server/activate

    monkeypatch.setattr(connection, "_run_noise_handshake", fake_handshake)
    monkeypatch.setattr(connection, "_run_inner_handshake", fake_inner)

    with pytest.raises(TimeoutError):
        await connection.connect(_BlockingWebSocket(), expected_server_id=None)  # type: ignore[arg-type]
    assert connection._closed.is_set()


async def test_client_seeds_last_playback_server_from_store() -> None:
    """The client seeds its last-playback tiebreak from the persisted store value."""
    store = InMemoryClientPairingStore()
    await store.set_last_playback_server_id("server-Z")
    sdk = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER], pairing_store=store)

    await sdk._ensure_last_playback_loaded()

    assert sdk.last_playback_server_id == "server-Z"
