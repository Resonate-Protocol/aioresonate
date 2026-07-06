"""Tests for SendspinClient.handle_availability_change role iteration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.noise.keys import Identity
from aiosendspin.noise.trust_store import InMemoryServerPairingStore
from aiosendspin.server import SendspinServer


def _make_server() -> SendspinServer:
    loop = asyncio.get_running_loop()
    client_session = MagicMock()
    client_session.closed = True
    client_session.close = AsyncMock()
    return SendspinServer(
        loop=loop,
        identity=Identity.generate(),
        server_name="server",
        client_session=client_session,
        pairing_store=InMemoryServerPairingStore(),
    )


@pytest.mark.asyncio
async def test_availability_change_survives_role_rebuild_during_hook() -> None:
    """A hook that rebuilds _roles mid-iteration must not raise dictionary-changed-size."""
    server = _make_server()
    client = server.get_or_create_client("dev")

    async def _rebuild() -> None:
        client._roles["late@v1"] = MagicMock()  # noqa: SLF001

    rebuilder = MagicMock()
    rebuilder.on_availability_changed.return_value = _rebuild()
    second = MagicMock()
    second.on_availability_changed.return_value = None
    client._roles = {"a@v1": rebuilder, "b@v1": second}  # noqa: SLF001

    await client.handle_availability_change(available=True)

    assert "late@v1" in client._roles  # noqa: SLF001
