"""Tests for SendspinClient multi-role management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.roles import PlayerRole


@dataclass(slots=True)
class _DummyServer:
    loop: Any
    clock: Any
    id: str = "srv"
    name: str = "server"


@dataclass(slots=True)
class _DummyGroup:
    clients: list[SendspinClient]

    def on_client_connected(self, client: SendspinClient) -> None:  # noqa: ARG002
        return

    def _register_client_events(self, client: SendspinClient) -> None:  # noqa: ARG002
        return

    def _unregister_client_events(self, client: SendspinClient) -> None:  # noqa: ARG002
        return


class _FakeConnection:
    def __init__(self) -> None:
        self.sent_json: list[object] = []
        self.sent_binary: list[bytes] = []
        self.high_water = False
        self.buffer_tracker = None

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:
        self.sent_json.append(message)

    def try_send_binary(
        self,
        data: bytes,
        *,
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,  # noqa: ARG002
    ) -> bool:
        self.sent_binary.append(data)
        if (
            self.buffer_tracker is not None
            and buffer_end_time_us is not None
            and buffer_byte_count is not None
        ):
            self.buffer_tracker.register(buffer_end_time_us, buffer_byte_count)
        return True

    def queue_high_water(self, threshold: float = 0.8) -> bool:  # noqa: ARG002
        return self.high_water


def _make_client_hello() -> MagicMock:
    """Create a mock ClientHelloPayload for player role."""
    hello = MagicMock()
    hello.client_id = "test-client"
    hello.name = "Test Client"
    hello.player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.PCM,
                channels=2,
                sample_rate=48000,
                bit_depth=16,
            ),
        ],
        supported_commands=[PlayerCommand.VOLUME],
        buffer_capacity=100_000,
    )
    return hello


class TestClientRoles:
    """Tests for client role management."""

    def test_player_role_accessor_returns_player_role(self, mock_loop: Any) -> None:
        """player_role property returns the player role instance."""
        server = _DummyServer(loop=mock_loop, clock=LoopClock(mock_loop))
        group = _DummyGroup(clients=[])
        client = SendspinClient(server, client_id="test")
        client._group = group  # noqa: SLF001
        group.clients.append(client)

        conn = _FakeConnection()
        hello = _make_client_hello()

        client.attach_connection(conn, client_info=hello, active_roles=["player@v1"])
        client.mark_connected()

        assert client.player_role is not None
        assert isinstance(client.player_role, PlayerRole)

    def test_player_role_has_role_family(self, mock_loop: Any) -> None:
        """PlayerRole has role_family='player'."""
        server = _DummyServer(loop=mock_loop, clock=LoopClock(mock_loop))
        group = _DummyGroup(clients=[])
        client = SendspinClient(server, client_id="test")
        client._group = group  # noqa: SLF001
        group.clients.append(client)

        conn = _FakeConnection()
        hello = _make_client_hello()

        client.attach_connection(conn, client_info=hello, active_roles=["player@v1"])

        assert client.player_role is not None
        assert client.player_role.role_family == "player"

    def test_active_roles_includes_player_role(self, mock_loop: Any) -> None:
        """active_roles includes PlayerRole when player role is active."""
        server = _DummyServer(loop=mock_loop, clock=LoopClock(mock_loop))
        group = _DummyGroup(clients=[])
        client = SendspinClient(server, client_id="test")
        client._group = group  # noqa: SLF001
        group.clients.append(client)

        conn = _FakeConnection()
        hello = _make_client_hello()

        client.attach_connection(conn, client_info=hello, active_roles=["player@v1"])

        roles = client.active_roles
        assert len(roles) == 1
        assert roles[0] is client.player_role

    def test_active_roles_empty_when_no_roles(self, mock_loop: Any) -> None:
        """active_roles returns empty list when no roles active."""
        server = _DummyServer(loop=mock_loop, clock=LoopClock(mock_loop))
        client = SendspinClient(server, client_id="test")

        assert client.active_roles == []
