"""Bring-up failure handling for incoming (server-initiated) connections."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.types import Roles
from tests.conftest import make_sdk_client

if TYPE_CHECKING:
    from aiohttp import web


async def test_attach_websocket_unexpected_failure_disconnects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bring-up failure outside the expected set tears the connection down and propagates."""
    sdk = make_sdk_client(client_name="c", roles=[Roles.CONTROLLER])
    disconnects: list[SendspinConnection] = []

    async def boom(
        self: SendspinConnection,  # noqa: ARG001
        ws: web.WebSocketResponse,  # noqa: ARG001
        *,
        expected_server_id: str | None = None,  # noqa: ARG001
    ) -> None:
        raise UnicodeDecodeError("utf-8", b"x", 0, 1, "bad")

    async def record_disconnect(self: SendspinConnection) -> None:
        disconnects.append(self)

    monkeypatch.setattr(SendspinConnection, "attach_websocket", boom)
    monkeypatch.setattr(SendspinConnection, "disconnect", record_disconnect)

    with pytest.raises(UnicodeDecodeError):
        await sdk.attach_websocket(MagicMock())

    assert len(disconnects) == 1
    assert not sdk._provisional_connections  # noqa: SLF001
