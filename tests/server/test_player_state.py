"""Tests for PlayerRecord and PlayerRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiosendspin.models import AudioCodec
from aiosendspin.models.types import Roles
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.player_state import PlayerRecord, PlayerRegistry
from aiosendspin.server.stream import AudioFormat


class TestPlayerRecordDefaults:
    """Tests for PlayerRecord default values."""

    def test_creates_with_client_id(self, mock_loop: MagicMock) -> None:
        """PlayerRecord should store the client_id."""
        record = PlayerRecord(
            client_id="test-player-1",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        assert record.client_id == "test-player-1"

    def test_default_volume_is_100(self, mock_loop: MagicMock) -> None:
        """Default volume should be 100."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        assert record.volume == 100

    def test_default_muted_is_false(self, mock_loop: MagicMock) -> None:
        """Default muted state should be False."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        assert record.muted is False

    def test_default_group_id_is_none(self, mock_loop: MagicMock) -> None:
        """Default group_id should be None."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        assert record.group_id is None

    def test_default_connection_is_none(self, mock_loop: MagicMock) -> None:
        """Default connection should be None."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        assert record.connection is None


class TestPlayerRecordProperties:
    """Tests for PlayerRecord readable/writable properties."""

    def test_volume_is_writable(self, mock_loop: MagicMock) -> None:
        """Volume should be settable."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        record.volume = 50
        assert record.volume == 50

    def test_muted_is_writable(self, mock_loop: MagicMock) -> None:
        """Muted should be settable."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        record.muted = True
        assert record.muted is True

    def test_group_id_is_writable(self, mock_loop: MagicMock) -> None:
        """Group ID should be settable."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        record.group_id = "group-123"
        assert record.group_id == "group-123"

    def test_preferred_format_stores_format(self, mock_loop: MagicMock) -> None:
        """Preferred format should be storable."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.OPUS)
        record.preferred_format = fmt
        assert record.preferred_format == fmt


class TestPlayerRecordConnection:
    """Tests for PlayerRecord connection management."""

    def test_connection_can_be_set(self, mock_loop: MagicMock) -> None:
        """Connection should be settable."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        mock_client: SendspinClient = MagicMock()
        record.connection = mock_client
        assert record.connection is mock_client

    def test_is_connected_true_when_connection_present(self, mock_loop: MagicMock) -> None:
        """is_connected should be True when connection is set."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        record.connection = MagicMock()
        assert record.is_connected is True

    def test_is_connected_false_when_no_connection(self, mock_loop: MagicMock) -> None:
        """is_connected should be False when connection is None."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        assert record.is_connected is False


class TestPlayerRecordBufferTracker:
    """Tests for PlayerRecord buffer tracker ownership."""

    def test_has_buffer_tracker(self, mock_loop: MagicMock) -> None:
        """PlayerRecord should own a BufferTracker."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        assert record.buffer_tracker is not None

    def test_buffer_tracker_has_correct_client_id(self, mock_loop: MagicMock) -> None:
        """BufferTracker should have the same client_id."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        assert record.buffer_tracker.client_id == "test-player"

    def test_buffer_tracker_has_correct_capacity(self, mock_loop: MagicMock) -> None:
        """BufferTracker should have the provided capacity."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=50_000,
        )
        assert record.buffer_tracker.capacity_bytes == 50_000


class TestPlayerRecordDisconnect:
    """Tests for disconnect tracking."""

    def test_mark_disconnected_records_time(self, mock_loop: MagicMock) -> None:
        """mark_disconnected should record the disconnect time."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        record.mark_disconnected(time_us=1_000_000)
        assert record.disconnect_time_us == 1_000_000

    def test_disconnect_time_initially_none(self, mock_loop: MagicMock) -> None:
        """Disconnect time should be None initially."""
        record = PlayerRecord(
            client_id="test-player",
            loop=mock_loop,
            buffer_capacity_bytes=100_000,
        )
        assert record.disconnect_time_us is None


# =============================================================================
# PlayerRegistry Tests
# =============================================================================


class TestPlayerRegistryGetOrCreate:
    """Tests for PlayerRegistry get_or_create behavior."""

    def test_creates_new_player_record(self, mock_loop: MagicMock) -> None:
        """get_or_create should create a new PlayerRecord if not found."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        record = registry.get_or_create("player-1")
        assert record is not None
        assert record.client_id == "player-1"

    def test_returns_same_instance_for_same_id(self, mock_loop: MagicMock) -> None:
        """get_or_create should return the same instance for the same client_id."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        record1 = registry.get_or_create("player-1")
        record2 = registry.get_or_create("player-1")
        assert record1 is record2

    def test_creates_different_instances_for_different_ids(self, mock_loop: MagicMock) -> None:
        """get_or_create should create different instances for different client_ids."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        record1 = registry.get_or_create("player-1")
        record2 = registry.get_or_create("player-2")
        assert record1 is not record2
        assert record1.client_id == "player-1"
        assert record2.client_id == "player-2"


class TestPlayerRegistryGet:
    """Tests for PlayerRegistry get behavior."""

    def test_returns_none_if_not_found(self, mock_loop: MagicMock) -> None:
        """Get should return None if client_id is not found."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        assert registry.get("nonexistent") is None

    def test_returns_record_if_found(self, mock_loop: MagicMock) -> None:
        """Get should return the record if client_id exists."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        created = registry.get_or_create("player-1")
        found = registry.get("player-1")
        assert found is created


class TestPlayerRegistryGetConnected:
    """Tests for PlayerRegistry get_connected behavior."""

    def test_returns_empty_when_no_players(self, mock_loop: MagicMock) -> None:
        """get_connected should return empty list when no players exist."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        assert registry.get_connected() == []

    def test_returns_empty_when_all_disconnected(self, mock_loop: MagicMock) -> None:
        """get_connected should return empty list when all players are disconnected."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        registry.get_or_create("player-1")
        registry.get_or_create("player-2")
        assert registry.get_connected() == []

    def test_returns_only_connected_players(self, mock_loop: MagicMock) -> None:
        """get_connected should return only players with connections."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        record1 = registry.get_or_create("player-1")
        record2 = registry.get_or_create("player-2")
        record3 = registry.get_or_create("player-3")

        record1.connection = MagicMock()
        record3.connection = MagicMock()

        connected = registry.get_connected()
        assert len(connected) == 2
        assert record1 in connected
        assert record2 not in connected
        assert record3 in connected


class TestPlayerRegistryGetInGroup:
    """Tests for PlayerRegistry get_in_group behavior."""

    def test_returns_empty_when_no_players_in_group(self, mock_loop: MagicMock) -> None:
        """get_in_group should return empty list when no players in group."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        registry.get_or_create("player-1")
        assert registry.get_in_group("group-123") == []

    def test_returns_players_in_group(self, mock_loop: MagicMock) -> None:
        """get_in_group should return players belonging to the group."""
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        record1 = registry.get_or_create("player-1")
        record2 = registry.get_or_create("player-2")
        record3 = registry.get_or_create("player-3")

        record1.group_id = "group-A"
        record2.group_id = "group-B"
        record3.group_id = "group-A"

        group_a = registry.get_in_group("group-A")
        assert len(group_a) == 2
        assert record1 in group_a
        assert record3 in group_a

        group_b = registry.get_in_group("group-B")
        assert len(group_b) == 1
        assert record2 in group_b


class TestPlayerRegistryCleanup:
    """Tests for PlayerRegistry cleanup_expired behavior."""

    def test_does_not_remove_connected_players(self, mock_loop: MagicMock) -> None:
        """cleanup_expired should not remove connected players."""
        mock_loop.time.return_value = 10.0  # 10 seconds
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        record = registry.get_or_create("player-1")
        record.connection = MagicMock()

        registry.cleanup_expired(timeout_us=1_000_000)  # 1 second timeout
        assert registry.get("player-1") is record

    def test_does_not_remove_recently_disconnected(self, mock_loop: MagicMock) -> None:
        """cleanup_expired should not remove recently disconnected players."""
        mock_loop.time.return_value = 10.0  # 10 seconds = 10_000_000 us
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        record = registry.get_or_create("player-1")
        record.mark_disconnected(time_us=9_500_000)  # Disconnected 0.5s ago

        registry.cleanup_expired(timeout_us=1_000_000)  # 1 second timeout
        assert registry.get("player-1") is record

    def test_removes_long_disconnected_players(self, mock_loop: MagicMock) -> None:
        """cleanup_expired should remove players disconnected longer than timeout."""
        mock_loop.time.return_value = 10.0  # 10 seconds = 10_000_000 us
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        record = registry.get_or_create("player-1")
        record.mark_disconnected(time_us=5_000_000)  # Disconnected 5s ago

        registry.cleanup_expired(timeout_us=1_000_000)  # 1 second timeout
        assert registry.get("player-1") is None

    def test_keeps_players_without_disconnect_time(self, mock_loop: MagicMock) -> None:
        """cleanup_expired should not remove players that were never marked disconnected."""
        mock_loop.time.return_value = 10.0
        registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)
        record = registry.get_or_create("player-1")
        # Never disconnected, disconnect_time_us is None

        registry.cleanup_expired(timeout_us=1_000_000)
        assert registry.get("player-1") is record


# =============================================================================
# Server Integration Tests (Task 3.1)
# =============================================================================


class TestServerPlayerRegistryIntegration:
    """Tests for PlayerRegistry integration with SendspinServer."""

    @pytest.fixture
    def mock_server(self, mock_loop: MagicMock) -> MagicMock:
        """Create a mock server with PlayerRegistry."""
        server = MagicMock()
        server.loop = mock_loop
        server._player_registry = PlayerRegistry(loop=mock_loop, default_buffer_capacity=100_000)  # noqa: SLF001
        return server

    @pytest.fixture
    def mock_player_client(self) -> MagicMock:
        """Create a mock client with player role."""
        client = MagicMock(spec=SendspinClient)
        client.client_id = "player-1"
        client.check_role.side_effect = lambda role: role == Roles.PLAYER
        return client

    def test_connect_attaches_client_to_player_record(
        self, mock_server: MagicMock, mock_player_client: MagicMock
    ) -> None:
        """On connect, server should attach client to PlayerRecord."""
        registry = mock_server._player_registry  # noqa: SLF001

        # Simulate connect
        record = registry.get_or_create(mock_player_client.client_id)
        record.connection = mock_player_client

        assert record.connection is mock_player_client
        assert record.is_connected is True

    def test_disconnect_detaches_but_keeps_record(
        self, mock_server: MagicMock, mock_player_client: MagicMock, mock_loop: MagicMock
    ) -> None:
        """On disconnect, PlayerRecord should remain but connection becomes None."""
        mock_loop.time.return_value = 1.0
        registry = mock_server._player_registry  # noqa: SLF001

        # Simulate connect then disconnect
        record = registry.get_or_create(mock_player_client.client_id)
        record.connection = mock_player_client

        # Disconnect
        record.connection = None
        record.mark_disconnected(time_us=1_000_000)

        assert record.connection is None
        assert record.is_connected is False
        assert registry.get(mock_player_client.client_id) is record

    def test_reconnect_reuses_same_record(
        self, mock_server: MagicMock, mock_player_client: MagicMock, mock_loop: MagicMock
    ) -> None:
        """On reconnect, same PlayerRecord instance should be reused."""
        mock_loop.time.return_value = 1.0
        registry = mock_server._player_registry  # noqa: SLF001

        # First connect
        record1 = registry.get_or_create(mock_player_client.client_id)
        record1.connection = mock_player_client
        record1.volume = 75  # Set some state

        # Disconnect
        record1.connection = None
        record1.mark_disconnected(time_us=1_000_000)

        # Reconnect (same client_id)
        new_mock_client = MagicMock(spec=SendspinClient)
        new_mock_client.client_id = "player-1"

        record2 = registry.get_or_create(new_mock_client.client_id)
        record2.connection = new_mock_client

        assert record1 is record2
        assert record2.volume == 75  # State preserved
        assert record2.is_connected is True

    def test_reconnect_preserves_group_membership(
        self, mock_server: MagicMock, mock_player_client: MagicMock, mock_loop: MagicMock
    ) -> None:
        """On reconnect, group_id should be preserved for restoration."""
        mock_loop.time.return_value = 1.0
        registry = mock_server._player_registry  # noqa: SLF001

        # First connect and join group
        record = registry.get_or_create(mock_player_client.client_id)
        record.connection = mock_player_client
        record.group_id = "group-abc"

        # Disconnect
        record.connection = None
        record.mark_disconnected(time_us=1_000_000)

        # Reconnect
        new_mock_client = MagicMock(spec=SendspinClient)
        new_mock_client.client_id = "player-1"

        record_after = registry.get_or_create(new_mock_client.client_id)
        record_after.connection = new_mock_client

        assert record_after.group_id == "group-abc"
