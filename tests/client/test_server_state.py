"""Tests for client handling of server/state role clearing."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.client.time_sync import SendspinTimeFilter
from aiosendspin.clock import ManualClock
from aiosendspin.models.core import ServerStatePayload
from aiosendspin.models.metadata import SessionUpdateMetadata
from aiosendspin.models.types import UndefinedField


def _make_connection() -> tuple[SendspinConnection, MagicMock]:
    conn = SendspinConnection.__new__(SendspinConnection)
    client = MagicMock()
    client.clock = ManualClock(now_us_value=1_000_000)
    conn._client = client  # noqa: SLF001
    conn._time_filter = SendspinTimeFilter()  # noqa: SLF001
    conn._init_state_trackers()  # noqa: SLF001
    return conn, client


def test_absent_role_does_not_fire_callback() -> None:
    """A role omitted from server/state (UndefinedField) fires no callback."""
    conn, client = _make_connection()
    payload = ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=1))
    conn._handle_server_state(payload)  # noqa: SLF001
    client.notify_metadata_callback.assert_called_once()
    assert client.notify_metadata_callback.call_args.args[0] is payload
    client.notify_scheduled_metadata.assert_not_called()
    client.notify_color_callback.assert_not_called()
    client.notify_controller_callback.assert_not_called()


def test_whole_role_null_fires_callback() -> None:
    """A whole-role null clears the role and fires its callback."""
    conn, client = _make_connection()
    conn._handle_server_state(ServerStatePayload(color=None))  # noqa: SLF001
    client.notify_color_callback.assert_called_once()
    client.notify_scheduled_color.assert_not_called()
    client.notify_metadata_callback.assert_not_called()
    client.notify_controller_callback.assert_not_called()


def test_listener_keeps_raw_diff_while_internal_state_is_merged() -> None:
    """Listeners retain wire diffs while the scheduler keeps merged state."""
    conn, client = _make_connection()
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(
            metadata=SessionUpdateMetadata(timestamp=1, title="First", artist="Artist")
        )
    )
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=2, title="Second"))
    )

    delivered = client.notify_metadata_callback.call_args_list[1].args[0].metadata
    assert isinstance(delivered.artist, UndefinedField)
    assert conn._metadata_state.confirmed is not None  # noqa: SLF001
    assert conn._metadata_state.confirmed.artist == "Artist"  # noqa: SLF001
