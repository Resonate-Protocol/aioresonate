"""Tests for SendspinGroup integration with PushStream."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

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
