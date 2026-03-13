"""Tests for multi-server support (connection reasons and client reclaim)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from aiohttp import web

from aiosendspin.models.core import (
    ClientHelloMessage,
    ClientHelloPayload,
    ServerHelloMessage,
    ServerStateMessage,
    ServerStatePayload,
)
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, ConnectionReason, PlayerCommand, Roles
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.roles.registry import ROLE_FACTORIES

if TYPE_CHECKING:
    from aiosendspin.models.types import ServerMessage


@dataclass
class _MockServer:
    """Mock server for testing connection reason lookup."""

    loop: asyncio.AbstractEventLoop
    clock: LoopClock
    id: str = "srv"
    name: str = "server"
    remove_client: AsyncMock = field(default_factory=AsyncMock)

    _connection_reasons: dict[str, ConnectionReason] = field(default_factory=dict)
    _client_urls: dict[str, str] = field(default_factory=dict)
    _clients: dict[str, SendspinClient] = field(default_factory=dict)
    _connection_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False

    def get_connection_reason(self, url: str) -> ConnectionReason:
        return self._connection_reasons.get(url, ConnectionReason.DISCOVERY)

    def register_client_url(self, client_id: str, url: str) -> None:
        self._client_urls[client_id] = url

    def get_client_url(self, client_id: str) -> str | None:
        return self._client_urls.get(client_id)

    def get_or_create_client(self, client_id: str) -> SendspinClient:
        client = self._clients.get(client_id)
        if client is None:
            client = SendspinClient(self, client_id=client_id)
            self._clients[client_id] = client
            SendspinGroup(self, client)
        return client


class _DummyConnection:
    def __init__(self) -> None:
        self.sent_messages: list[ServerMessage] = []

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: ServerMessage) -> None:
        self.sent_messages.append(message)

    def send_role_message(self, role: str, message: ServerMessage) -> None:  # noqa: ARG002
        self.sent_messages.append(message)

    def send_binary(
        self,
        data: bytes,  # noqa: ARG002
        *,
        role: str,  # noqa: ARG002
        timestamp_us: int,  # noqa: ARG002
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,  # noqa: ARG002
        buffer_byte_count: int | None = None,  # noqa: ARG002
        duration_us: int | None = None,  # noqa: ARG002
    ) -> bool:
        return True


def _player_hello(client_id: str) -> ClientHelloPayload:
    return ClientHelloPayload(
        client_id=client_id,
        name=client_id,
        version=1,
        supported_roles=[Roles.PLAYER.value],
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM,
                    channels=2,
                    sample_rate=48000,
                    bit_depth=16,
                )
            ],
            buffer_capacity=100_000,
            supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
        ),
    )


@pytest.fixture
async def mock_server() -> _MockServer:
    """Create a mock server with connection reason tracking."""
    loop = asyncio.get_running_loop()
    return _MockServer(loop=loop, clock=LoopClock(loop))


class TestConnectionReasonLookup:
    """Tests for connection reason lookup from server."""

    def test_get_connection_reason_defaults_to_discovery(self, mock_server: _MockServer) -> None:
        """Connection reason defaults to DISCOVERY for unknown URLs."""
        assert mock_server.get_connection_reason("ws://unknown:1234") == ConnectionReason.DISCOVERY

    def test_get_connection_reason_returns_stored_reason(self, mock_server: _MockServer) -> None:
        """Connection reason returns the stored value for known URLs."""
        url = "ws://192.168.1.100:8927/sendspin"
        mock_server._connection_reasons[url] = ConnectionReason.PLAYBACK  # noqa: SLF001

        assert mock_server.get_connection_reason(url) == ConnectionReason.PLAYBACK


class TestClientUrlTracking:
    """Tests for client URL registration and lookup."""

    def test_register_and_get_client_url(self, mock_server: _MockServer) -> None:
        """Client URL can be registered and retrieved."""
        mock_server.register_client_url("client-1", "ws://192.168.1.50:8927/sendspin")

        assert mock_server.get_client_url("client-1") == "ws://192.168.1.50:8927/sendspin"

    def test_get_client_url_returns_none_for_unknown(self, mock_server: _MockServer) -> None:
        """get_client_url returns None for unknown client IDs."""
        assert mock_server.get_client_url("unknown-client") is None


class TestConnectionSendsCorrectReason:
    """Tests that SendspinConnection uses the correct connection_reason in server/hello."""

    @pytest.mark.asyncio
    async def test_server_initiated_uses_discovery_by_default(
        self, mock_server: _MockServer
    ) -> None:
        """Server-initiated connection with no stored reason uses DISCOVERY."""
        url = "ws://192.168.1.100:8927/sendspin"
        # No reason stored for this URL

        # Create a connection with URL but no stored reason
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock(), url=url)

        # Capture priority messages — server/hello must use the priority queue
        sent_messages: list[ServerMessage] = []
        original_priority_send = conn.send_priority_message

        def capture_priority_send(msg: ServerMessage) -> None:
            sent_messages.append(msg)
            original_priority_send(msg)

        conn.send_priority_message = capture_priority_send  # type: ignore[method-assign]

        # Simulate receiving client/hello
        await conn._handle_message(  # noqa: SLF001
            ClientHelloMessage(payload=_player_hello("client-1")),
            timestamp_us=0,
        )

        # Find the server/hello message
        server_hello = next(m for m in sent_messages if isinstance(m, ServerHelloMessage))
        assert server_hello.payload.connection_reason == ConnectionReason.DISCOVERY

    @pytest.mark.asyncio
    async def test_server_initiated_uses_playback_when_stored(
        self, mock_server: _MockServer
    ) -> None:
        """Server-initiated connection uses stored PLAYBACK reason."""
        url = "ws://192.168.1.100:8927/sendspin"
        mock_server._connection_reasons[url] = ConnectionReason.PLAYBACK  # noqa: SLF001

        conn = SendspinConnection(mock_server, wsock_client=AsyncMock(), url=url)

        # Capture priority messages — server/hello must use the priority queue
        sent_messages: list[ServerMessage] = []
        original_priority_send = conn.send_priority_message

        def capture_priority_send(msg: ServerMessage) -> None:
            sent_messages.append(msg)
            original_priority_send(msg)

        conn.send_priority_message = capture_priority_send  # type: ignore[method-assign]

        await conn._handle_message(  # noqa: SLF001
            ClientHelloMessage(payload=_player_hello("client-1")),
            timestamp_us=0,
        )

        server_hello = next(m for m in sent_messages if isinstance(m, ServerHelloMessage))
        assert server_hello.payload.connection_reason == ConnectionReason.PLAYBACK

    @pytest.mark.asyncio
    async def test_client_initiated_uses_discovery(self, mock_server: _MockServer) -> None:
        """Client-initiated connection (no URL) uses DISCOVERY."""
        request = MagicMock(spec=web.Request)
        request.remote = "192.168.1.50"

        conn = SendspinConnection(mock_server, request=request)
        # Prepare the websocket response mock
        conn._wsock_server = AsyncMock()  # noqa: SLF001
        conn._wsock_server.closed = False  # noqa: SLF001

        # Capture priority messages — server/hello must use the priority queue
        sent_messages: list[ServerMessage] = []
        original_priority_send = conn.send_priority_message

        def capture_priority_send(msg: ServerMessage) -> None:
            sent_messages.append(msg)
            original_priority_send(msg)

        conn.send_priority_message = capture_priority_send  # type: ignore[method-assign]

        await conn._handle_message(  # noqa: SLF001
            ClientHelloMessage(payload=_player_hello("client-1")),
            timestamp_us=0,
        )

        server_hello = next(m for m in sent_messages if isinstance(m, ServerHelloMessage))
        assert server_hello.payload.connection_reason == ConnectionReason.DISCOVERY


class TestHandshakeOrdering:
    """Tests for server/hello ordering relative to role messages."""

    @pytest.mark.asyncio
    async def test_server_hello_queued_before_role_messages(
        self, mock_server: _MockServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """server/hello must be enqueued before any role-scoped server messages."""
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock())
        call_order: list[str] = []

        original_priority_send = conn.send_priority_message
        original_role_send = conn.send_role_message

        def capture_priority_send(msg: ServerMessage) -> None:
            call_order.append(msg.type)
            original_priority_send(msg)

        def capture_role_send(role: str, msg: ServerMessage) -> None:
            call_order.append(msg.type)
            original_role_send(role, msg)

        conn.send_priority_message = capture_priority_send  # type: ignore[method-assign]
        conn.send_role_message = capture_role_send  # type: ignore[method-assign]

        original_attach_connection = SendspinClient.attach_connection

        def attach_with_early_role_message(
            self: SendspinClient,
            connection: SendspinConnection,
            *,
            client_info: ClientHelloPayload,
            active_roles: list[str],
        ) -> None:
            connection.send_role_message(
                "metadata",
                ServerStateMessage(payload=ServerStatePayload()),
            )
            original_attach_connection(
                self, connection, client_info=client_info, active_roles=active_roles
            )

        monkeypatch.setattr(SendspinClient, "attach_connection", attach_with_early_role_message)

        await conn._handle_message(  # noqa: SLF001
            ClientHelloMessage(payload=_player_hello("client-1")),
            timestamp_us=0,
        )

        assert call_order
        assert call_order[0] == "server/hello"
        assert call_order.index("server/state") > 0


class TestCustomRoleSupportParsing:
    """Tests for custom role support-key handling in inbound client/hello messages."""

    @pytest.mark.parametrize(
        ("role_id", "support_key", "expected_attr", "support_payload"),
        [
            (
                "player@_custom_version",
                "player@_custom_version_support",
                "player_support",
                {
                    "supported_formats": [
                        {"codec": "pcm", "sample_rate": 48000, "bit_depth": 16, "channels": 2}
                    ],
                    "buffer_capacity": 100_000,
                    "supported_commands": [],
                },
            ),
            (
                "artwork@_custom_version",
                "artwork@_custom_version_support",
                "artwork_support",
                {
                    "channels": [
                        {
                            "source": "album",
                            "format": "jpeg",
                            "media_width": 300,
                            "media_height": 300,
                        }
                    ]
                },
            ),
            (
                "visualizer@_custom_version",
                "visualizer@_custom_version_support",
                "visualizer_support",
                {"buffer_capacity": 100_000},
            ),
        ],
    )
    def test_deserialize_client_hello_maps_custom_support(
        self,
        role_id: str,
        support_key: str,
        expected_attr: str,
        support_payload: dict[str, object],
    ) -> None:
        """Custom role support keys are mapped into existing support fields."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": [role_id],
                    support_key: support_payload,
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert getattr(msg.payload, expected_attr) is not None

    @pytest.mark.parametrize(
        ("role_id", "missing_support_key"),
        [
            ("player@_custom_version", "player@_custom_version_support"),
            ("artwork@_custom_version", "artwork@_custom_version_support"),
            ("visualizer@_custom_version", "visualizer@_custom_version_support"),
        ],
    )
    def test_deserialize_client_hello_requires_custom_support_key(
        self, role_id: str, missing_support_key: str
    ) -> None:
        """Custom role IDs require their matching custom support keys."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": [role_id],
                },
            }
        ).decode()

        with pytest.raises(ValueError, match=missing_support_key):
            SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001

    def test_deserialize_client_hello_logs_legacy_support_conversion_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Legacy support alias conversion warning is preserved for v1 roles."""
        caplog.set_level("WARNING")
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["player@v1"],
                    "player_support": {
                        "supported_formats": [
                            {
                                "codec": "pcm",
                                "sample_rate": 48000,
                                "bit_depth": 16,
                                "channels": 2,
                            }
                        ],
                        "buffer_capacity": 100_000,
                        "supported_commands": [],
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert "client/hello message used deprecated field names" in caplog.text

    def test_legacy_visualizer_support_key_is_not_normalized_to_v1(self) -> None:
        """Legacy visualizer_support must not be rewritten to visualizer@v1_support."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["visualizer@_draft_r1"],
                    "visualizer_support": {
                        "types": ["loudness", "f_peak"],
                        "buffer_capacity": 65536,
                        "batch_max": 8,
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.visualizer_support is not None

    def test_family_order_prefers_first_role_and_does_not_require_second_version_support(
        self,
    ) -> None:
        """When v1 is listed before v2, parser must not require v2 support key."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["player@v1", "player@v2"],
                    "player@v1_support": {
                        "supported_formats": [
                            {
                                "codec": "pcm",
                                "sample_rate": 48000,
                                "bit_depth": 16,
                                "channels": 2,
                            }
                        ],
                        "buffer_capacity": 100_000,
                        "supported_commands": [],
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.player_support is not None

    def test_family_order_prefers_first_custom_role_and_requires_matching_support_key(self) -> None:
        """When v2 is unregistered, parser falls back to first registered role in family."""
        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["player@v2", "player@v1"],
                    "player@v1_support": {
                        "supported_formats": [
                            {
                                "codec": "pcm",
                                "sample_rate": 48000,
                                "bit_depth": 16,
                                "channels": 2,
                            }
                        ],
                        "buffer_capacity": 100_000,
                        "supported_commands": [],
                    },
                },
            }
        ).decode()

        msg = SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001
        assert isinstance(msg, ClientHelloMessage)
        assert msg.payload.player_support is not None

    def test_family_order_prefers_registered_v2_and_requires_matching_support_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When v2 is registered and listed first, parser requires v2 support key."""
        monkeypatch.setitem(ROLE_FACTORIES, "player@v2", lambda _client: None)  # type: ignore[arg-type]

        raw = orjson.dumps(
            {
                "type": "client/hello",
                "payload": {
                    "client_id": "c1",
                    "name": "Client",
                    "version": 1,
                    "supported_roles": ["player@v2", "player@v1"],
                    "player@v1_support": {
                        "supported_formats": [
                            {
                                "codec": "pcm",
                                "sample_rate": 48000,
                                "bit_depth": 16,
                                "channels": 2,
                            }
                        ],
                        "buffer_capacity": 100_000,
                        "supported_commands": [],
                    },
                },
            }
        ).decode()

        with pytest.raises(ValueError, match="player@v2_support"):
            SendspinConnection._deserialize_client_message(raw)  # noqa: SLF001


class TestClientUrlRegistration:
    """Tests that client URLs are registered after successful handshake."""

    @pytest.mark.asyncio
    async def test_url_registered_after_handshake(self, mock_server: _MockServer) -> None:
        """Client URL is registered after receiving client/hello for server-initiated connection."""
        url = "ws://192.168.1.100:8927/sendspin"
        conn = SendspinConnection(mock_server, wsock_client=AsyncMock(), url=url)

        await conn._handle_message(  # noqa: SLF001
            ClientHelloMessage(payload=_player_hello("my-speaker")),
            timestamp_us=0,
        )

        assert mock_server.get_client_url("my-speaker") == url

    @pytest.mark.asyncio
    async def test_url_not_registered_for_client_initiated(self, mock_server: _MockServer) -> None:
        """Client URL is NOT registered for client-initiated connections (no URL to store)."""
        request = MagicMock(spec=web.Request)
        request.remote = "192.168.1.50"

        conn = SendspinConnection(mock_server, request=request)
        conn._wsock_server = AsyncMock()  # noqa: SLF001
        conn._wsock_server.closed = False  # noqa: SLF001

        await conn._handle_message(  # noqa: SLF001
            ClientHelloMessage(payload=_player_hello("my-speaker")),
            timestamp_us=0,
        )

        # No URL should be registered since we don't know the client's WebSocket URL
        assert mock_server.get_client_url("my-speaker") is None


@dataclass
class _MockServerWithReclaim:
    """Mock server that tracks reclaim calls."""

    loop: asyncio.AbstractEventLoop
    clock: LoopClock
    id: str = "srv"
    name: str = "server"

    _reclaim_calls: list[str] = field(default_factory=list)

    def request_client_playback_connection(self, client_id: str) -> bool:
        self._reclaim_calls.append(client_id)
        return True

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False


class TestAutomaticReclaim:
    """Tests for automatic client reclaim on playback start and group join."""

    @pytest.mark.asyncio
    async def test_start_stream_reclaims_disconnected_clients(self) -> None:
        """start_stream() reclaims disconnected clients in the group."""
        loop = asyncio.get_running_loop()
        server = _MockServerWithReclaim(loop=loop, clock=LoopClock(loop))

        # Create a client and group
        client = SendspinClient(server, client_id="speaker-1")
        group = SendspinGroup(server, client)

        # Connect and then disconnect the client (simulating another_server goodbye)
        conn = _DummyConnection()
        client.attach_connection(
            conn,
            client_info=_player_hello("speaker-1"),
            active_roles=[Roles.PLAYER.value],
        )
        client.mark_connected()

        # Store URL for reclaim
        server._client_urls = {"speaker-1": "ws://192.168.1.50:8927/sendspin"}  # type: ignore[attr-defined]  # noqa: SLF001

        # Disconnect the client
        client.detach_connection(None)
        assert not client.is_connected

        # Start a stream - should trigger reclaim
        group.start_stream()

        assert "speaker-1" in server._reclaim_calls  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_add_client_reclaims_if_group_has_active_playback(self) -> None:
        """add_client() reclaims disconnected client when group has active playback."""
        loop = asyncio.get_running_loop()
        server = _MockServerWithReclaim(loop=loop, clock=LoopClock(loop))

        # Create two clients - one connected, one disconnected
        client1 = SendspinClient(server, client_id="speaker-1")
        client2 = SendspinClient(server, client_id="speaker-2")
        group1 = SendspinGroup(server, client1)
        SendspinGroup(server, client2)

        # Connect client1 and start playback
        conn1 = _DummyConnection()
        client1.attach_connection(
            conn1,
            client_info=_player_hello("speaker-1"),
            active_roles=[Roles.PLAYER.value],
        )
        client1.mark_connected()
        group1.start_stream()

        # Clear reclaim calls from start_stream (client1 was connected)
        server._reclaim_calls.clear()  # noqa: SLF001

        # Connect and disconnect client2
        conn2 = _DummyConnection()
        client2.attach_connection(
            conn2,
            client_info=_player_hello("speaker-2"),
            active_roles=[Roles.PLAYER.value],
        )
        client2.mark_connected()
        client2.detach_connection(None)
        assert not client2.is_connected

        # Store URL for reclaim
        server._client_urls = {"speaker-2": "ws://192.168.1.51:8927/sendspin"}  # type: ignore[attr-defined]  # noqa: SLF001

        # Add disconnected client2 to group1 (which has active playback)
        await group1.add_client(client2)

        assert "speaker-2" in server._reclaim_calls  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_add_client_does_not_reclaim_if_connected(self) -> None:
        """add_client() does not reclaim if client is already connected."""
        loop = asyncio.get_running_loop()
        server = _MockServerWithReclaim(loop=loop, clock=LoopClock(loop))

        # Create two clients
        client1 = SendspinClient(server, client_id="speaker-1")
        client2 = SendspinClient(server, client_id="speaker-2")
        group1 = SendspinGroup(server, client1)
        SendspinGroup(server, client2)

        # Connect both clients
        conn1 = _DummyConnection()
        client1.attach_connection(
            conn1,
            client_info=_player_hello("speaker-1"),
            active_roles=[Roles.PLAYER.value],
        )
        client1.mark_connected()

        conn2 = _DummyConnection()
        client2.attach_connection(
            conn2,
            client_info=_player_hello("speaker-2"),
            active_roles=[Roles.PLAYER.value],
        )
        client2.mark_connected()

        # Start playback on group1
        group1.start_stream()
        server._reclaim_calls.clear()  # noqa: SLF001

        # Add connected client2 to group1
        await group1.add_client(client2)

        # Should not try to reclaim since client2 is connected
        assert "speaker-2" not in server._reclaim_calls  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_add_client_does_not_reclaim_if_no_active_playback(self) -> None:
        """add_client() does not reclaim if group has no active playback."""
        loop = asyncio.get_running_loop()
        server = _MockServerWithReclaim(loop=loop, clock=LoopClock(loop))

        # Create two clients
        client1 = SendspinClient(server, client_id="speaker-1")
        client2 = SendspinClient(server, client_id="speaker-2")
        group1 = SendspinGroup(server, client1)
        SendspinGroup(server, client2)

        # Connect client1 but don't start playback
        conn1 = _DummyConnection()
        client1.attach_connection(
            conn1,
            client_info=_player_hello("speaker-1"),
            active_roles=[Roles.PLAYER.value],
        )
        client1.mark_connected()

        # Connect and disconnect client2
        conn2 = _DummyConnection()
        client2.attach_connection(
            conn2,
            client_info=_player_hello("speaker-2"),
            active_roles=[Roles.PLAYER.value],
        )
        client2.mark_connected()
        client2.detach_connection(None)

        # Add disconnected client2 to group1 (no active playback)
        await group1.add_client(client2)

        # Should not try to reclaim since no active playback
        assert "speaker-2" not in server._reclaim_calls  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_add_client_replaces_stale_client_with_same_client_id(self) -> None:
        """add_client() replaces stale client objects that share the same client_id."""
        loop = asyncio.get_running_loop()
        server = _MockServerWithReclaim(loop=loop, clock=LoopClock(loop))

        owner = SendspinClient(server, client_id="speaker-1")
        stale = SendspinClient(server, client_id="speaker-2")
        replacement = SendspinClient(server, client_id="speaker-2")

        group1 = SendspinGroup(server, owner, stale)
        SendspinGroup(server, replacement)

        await group1.add_client(replacement)

        # Group membership should contain only the replacement object for speaker-2.
        speaker2_members = [c for c in group1.clients if c.client_id == "speaker-2"]
        assert speaker2_members == [replacement]
        assert stale not in group1.clients
