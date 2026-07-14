"""Tests for client handling of server/state role clearing."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.core import ServerStatePayload
from aiosendspin.models.metadata import SessionUpdateMetadata


def _make_connection() -> tuple[SendspinConnection, MagicMock]:
    conn = SendspinConnection.__new__(SendspinConnection)
    client = MagicMock()
    conn._client = client  # noqa: SLF001
    return conn, client


def test_absent_role_does_not_fire_callback() -> None:
    """A role omitted from server/state (UndefinedField) fires no callback."""
    conn, client = _make_connection()
    conn._handle_server_state(  # noqa: SLF001
        ServerStatePayload(metadata=SessionUpdateMetadata(timestamp=1))
    )
    client.notify_metadata_callback.assert_called_once()
    client.notify_color_callback.assert_not_called()
    client.notify_controller_callback.assert_not_called()


def test_whole_role_null_fires_callback() -> None:
    """A whole-role null clears the role and fires its callback."""
    conn, client = _make_connection()
    conn._handle_server_state(ServerStatePayload(color=None))  # noqa: SLF001
    client.notify_color_callback.assert_called_once()
    client.notify_metadata_callback.assert_not_called()
    client.notify_controller_callback.assert_not_called()
