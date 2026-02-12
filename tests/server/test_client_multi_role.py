"""Tests for SendspinClient multi-role management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand
from aiosendspin.server.audio_transformers import TransformerPool
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.clock import LoopClock
from aiosendspin.server.roles import PlayerV1Role


@dataclass(slots=True)
class _DummyServer:
    loop: Any
    clock: Any
    id: str = "srv"
    name: str = "server"

    def is_external_player(self, client_id: str) -> bool:  # noqa: ARG002
        return False


class _DummyGroup:
    def __init__(self, clients: list[SendspinClient]) -> None:
        self.clients = clients
        self.transformer_pool = TransformerPool()

    def on_client_connected(self, client: SendspinClient) -> None:  # noqa: ARG002
        return

    def _register_client_events(self, client: SendspinClient) -> None:  # noqa: ARG002
        return

    def _unregister_client_events(self, client: SendspinClient) -> None:  # noqa: ARG002
        return

    def group_role(self, family: str) -> None:  # noqa: ARG002
        return None

    def get_channel_for_player(self, player_id: str) -> UUID:  # noqa: ARG002
        return MAIN_CHANNEL


class _FakeConnection:
    def __init__(self) -> None:
        self.sent_json: list[object] = []
        self.sent_binary: list[bytes] = []
        self.buffer_tracker = None

    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:
        self.sent_json.append(message)

    def send_role_message(self, role: str, message: object) -> None:  # noqa: ARG002
        self.sent_json.append(message)

    def send_binary(
        self,
        data: bytes,
        *,
        role: str,  # noqa: ARG002
        timestamp_us: int,  # noqa: ARG002
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,
    ) -> bool:
        self.sent_binary.append(data)
        if (
            self.buffer_tracker is not None
            and buffer_end_time_us is not None
            and buffer_byte_count is not None
        ):
            self.buffer_tracker.register(buffer_end_time_us, buffer_byte_count, duration_us or 0)
        return True


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
        """client.role('player@v1') returns the player role instance."""
        server = _DummyServer(loop=mock_loop, clock=LoopClock(mock_loop))
        group = _DummyGroup(clients=[])
        client = SendspinClient(server, client_id="test")
        client._group = group  # noqa: SLF001
        group.clients.append(client)

        conn = _FakeConnection()
        hello = _make_client_hello()

        client.attach_connection(conn, client_info=hello, active_roles=["player@v1"])
        client.mark_connected()

        role = client.role("player@v1")
        assert role is not None
        assert isinstance(role, PlayerV1Role)

    def test_player_role_has_role_family(self, mock_loop: Any) -> None:
        """PlayerV1Role has role_family='player'."""
        server = _DummyServer(loop=mock_loop, clock=LoopClock(mock_loop))
        group = _DummyGroup(clients=[])
        client = SendspinClient(server, client_id="test")
        client._group = group  # noqa: SLF001
        group.clients.append(client)

        conn = _FakeConnection()
        hello = _make_client_hello()

        client.attach_connection(conn, client_info=hello, active_roles=["player@v1"])

        role = client.role("player@v1")
        assert role is not None
        assert role.role_family == "player"

    def test_active_roles_includes_player_role(self, mock_loop: Any) -> None:
        """active_roles includes PlayerV1Role when player role is active."""
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
        assert roles[0] is client.role("player@v1")

    def test_active_roles_empty_when_no_roles(self, mock_loop: Any) -> None:
        """active_roles returns empty list when no roles active."""
        server = _DummyServer(loop=mock_loop, clock=LoopClock(mock_loop))
        client = SendspinClient(server, client_id="test")

        assert client.active_roles == []
