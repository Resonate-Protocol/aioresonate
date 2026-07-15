"""Tests for the client spec-compliance signalling helpers."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.noise.keys import Identity
from aiosendspin.noise.trust_store import InMemoryServerPairingStore
from aiosendspin.server import SendspinServer
from aiosendspin.server.compliance import ClientComplianceError


def _make_server(*, strict_clients: bool = False) -> SendspinServer:
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
        strict_clients=strict_clients,
    )


@pytest.mark.asyncio
async def test_strict_clients_defaults_to_false() -> None:
    """strict_clients is opt-in."""
    server = _make_server()
    assert server.strict_clients is False


@pytest.mark.asyncio
async def test_flag_noncompliance_lenient_logs_every_time(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Lenient mode logs each occurrence (no dedup)."""
    server = _make_server()
    client = server.get_or_create_client("dev")
    with caplog.at_level(logging.INFO):
        client.flag_noncompliance("legacy thing")
        client.flag_noncompliance("legacy thing")
    hits = [r for r in caplog.records if "non-compliant client: legacy thing" in r.message]
    assert len(hits) == 2


@pytest.mark.asyncio
async def test_flag_noncompliance_strict_raises() -> None:
    """Strict mode raises ClientComplianceError."""
    server = _make_server(strict_clients=True)
    client = server.get_or_create_client("dev")
    with pytest.raises(ClientComplianceError, match="legacy thing"):
        client.flag_noncompliance("legacy thing")
