"""Tests for SendspinGroup integration with PushStream."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiosendspin.models.types import Roles
from aiosendspin.server.channels import ChannelRouter
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.player_state import PlayerRegistry
from aiosendspin.server.push_stream import PushStream


class TestGroupStartStream:
    """Tests for SendspinGroup.start_stream() integration."""

    @pytest.fixture
    def mock_loop(self) -> MagicMock:
        """Create a mock event loop."""
        loop = MagicMock()
        loop.time.return_value = 1000.0
        return loop

    @pytest.fixture
    def mock_server(self, mock_loop: MagicMock) -> MagicMock:
        """Create a mock server with player registry."""
        server = MagicMock()
        server.loop = mock_loop
        server.player_registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        return server

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock client for the group."""
        client = MagicMock()
        client.client_id = "test-client"
        client.check_role.return_value = True
        return client

    def test_start_stream_returns_push_stream(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """start_stream() should return a PushStream instance."""
        group = SendspinGroup(mock_server, mock_client)
        stream = group.start_stream()

        assert isinstance(stream, PushStream)

    def test_start_stream_uses_server_registry(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """start_stream() should use the server's player registry."""
        group = SendspinGroup(mock_server, mock_client)
        stream = group.start_stream()

        # The stream should use the server's player registry
        assert stream._player_registry is mock_server.player_registry  # noqa: SLF001

    def test_start_stream_uses_server_loop(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """start_stream() should use the server's event loop."""
        group = SendspinGroup(mock_server, mock_client)
        stream = group.start_stream()

        # The stream should use the server's loop
        assert stream._loop is mock_server.loop  # noqa: SLF001

    def test_group_stop_stops_stream(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """Group stop should call stream stop."""
        group = SendspinGroup(mock_server, mock_client)
        stream = group.start_stream()

        assert not stream.is_stopped

        group.stop_stream()

        assert stream.is_stopped

    def test_multiple_start_stream_returns_new_instances(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """Each call to start_stream() should return a new PushStream."""
        group = SendspinGroup(mock_server, mock_client)
        stream1 = group.start_stream()
        stream2 = group.start_stream()

        assert stream1 is not stream2

    def test_start_stream_with_channel_router(
        self,
        mock_server: MagicMock,
        mock_client: MagicMock,
    ) -> None:
        """start_stream() can accept a custom channel router."""
        group = SendspinGroup(mock_server, mock_client)
        custom_router = ChannelRouter()
        stream = group.start_stream(channel_router=custom_router)

        assert stream._channel_router is custom_router  # noqa: SLF001


class TestPlayerJoinWithActiveStream:
    """Tests for player joining a group with an active PushStream."""

    @pytest.fixture
    def mock_loop(self) -> MagicMock:
        """Create a mock event loop."""
        loop = MagicMock()
        loop.time.return_value = 1000.0
        return loop

    @pytest.fixture
    def mock_server(self, mock_loop: MagicMock) -> MagicMock:
        """Create a mock server with player registry."""
        server = MagicMock()
        server.loop = mock_loop
        server.player_registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        return server

    @pytest.fixture
    def mock_owner_client(self) -> MagicMock:
        """Create a mock owner client for the group."""
        client = MagicMock()
        client.client_id = "owner-client"
        client.check_role.return_value = True
        client.group = MagicMock()
        client.group.stop = AsyncMock()
        return client

    @pytest.fixture
    def mock_player_client(self) -> MagicMock:
        """Create a mock player client to join the group."""
        client = MagicMock()
        client.client_id = "player-client"
        # Make check_role return True only for PLAYER role
        client.check_role.side_effect = lambda role: role == Roles.PLAYER
        # Mock the group property for ungroup() call
        client.group = MagicMock()
        client.group.stop = AsyncMock()
        client.group._clients = []  # noqa: SLF001
        client.ungroup = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_player_join_triggers_on_player_join(
        self,
        mock_server: MagicMock,
        mock_owner_client: MagicMock,
        mock_player_client: MagicMock,
    ) -> None:
        """Adding a player to a group with active stream calls on_player_join."""
        group = SendspinGroup(mock_server, mock_owner_client)
        stream = group.start_stream()

        # Mock on_player_join to track calls
        with patch.object(stream, "on_player_join") as mock_on_player_join:
            await group.add_client(mock_player_client)

            mock_on_player_join.assert_called_once_with(mock_player_client.client_id)

    @pytest.mark.asyncio
    async def test_non_player_join_does_not_trigger_on_player_join(
        self,
        mock_server: MagicMock,
        mock_owner_client: MagicMock,
    ) -> None:
        """Adding a non-player client does not call on_player_join."""
        group = SendspinGroup(mock_server, mock_owner_client)
        stream = group.start_stream()

        # Create a non-player client (visualizer only, no special handling)
        visualizer_client = MagicMock()
        visualizer_client.client_id = "visualizer-client"
        # Returns False for all roles to avoid special handling code paths
        visualizer_client.check_role.return_value = False
        visualizer_client.group = MagicMock()
        visualizer_client.group.stop = AsyncMock()
        visualizer_client.group._clients = []  # noqa: SLF001
        visualizer_client.ungroup = AsyncMock()

        with patch.object(stream, "on_player_join") as mock_on_player_join:
            await group.add_client(visualizer_client)

            mock_on_player_join.assert_not_called()

    @pytest.mark.asyncio
    async def test_player_join_without_active_stream(
        self,
        mock_server: MagicMock,
        mock_owner_client: MagicMock,
        mock_player_client: MagicMock,
    ) -> None:
        """Adding a player without an active stream does not crash."""
        group = SendspinGroup(mock_server, mock_owner_client)
        # No stream started

        # Should not raise
        await group.add_client(mock_player_client)

    @pytest.mark.asyncio
    async def test_player_join_with_stopped_stream(
        self,
        mock_server: MagicMock,
        mock_owner_client: MagicMock,
        mock_player_client: MagicMock,
    ) -> None:
        """Adding a player to a stopped stream does not call on_player_join."""
        group = SendspinGroup(mock_server, mock_owner_client)
        stream = group.start_stream()
        stream.stop()  # Stop the stream

        with patch.object(stream, "on_player_join") as mock_on_player_join:
            await group.add_client(mock_player_client)

            mock_on_player_join.assert_not_called()
