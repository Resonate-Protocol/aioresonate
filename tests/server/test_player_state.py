"""Tests for PlayerRecord and PlayerRegistry."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from aiosendspin.models import AudioCodec
from aiosendspin.server.player_state import PlayerRecord
from aiosendspin.server.stream import AudioFormat

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient


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
