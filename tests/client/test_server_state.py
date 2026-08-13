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
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=1))
    )
    client.notify_metadata_callback.assert_called_once()
    client.notify_effective_metadata.assert_called_once()
    client.notify_color_callback.assert_not_called()
    client.notify_controller_callback.assert_not_called()


def test_whole_role_null_fires_callback() -> None:
    """A whole-role null clears the role and fires its callback."""
    conn, client = _make_connection()
    conn._handle_server_state(ServerStatePayload(color=None))  # noqa: SLF001
    client.notify_color_callback.assert_called_once()
    client.notify_effective_color.assert_called_once()
    client.notify_metadata_callback.assert_not_called()
    client.notify_controller_callback.assert_not_called()


def test_raw_listener_keeps_diff_while_effective_listener_gets_snapshot() -> None:
    """Raw listeners retain wire diffs while effective listeners receive merged state."""
    conn, client = _make_connection()
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(
            metadata=SessionUpdateMetadata(timestamp=1, title="First", artist="Artist")
        )
    )
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=2, title="Second"))
    )

    raw = client.notify_metadata_callback.call_args_list[1].args[0].metadata
    effective = client.notify_effective_metadata.call_args_list[1].args[0].metadata
    assert isinstance(raw.artist, UndefinedField)
    assert effective.artist == "Artist"
