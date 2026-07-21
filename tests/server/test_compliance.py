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


def _make_server(*, allow_noncompliant_clients: bool = True) -> SendspinServer:
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
        allow_noncompliant_clients=allow_noncompliant_clients,
    )


@pytest.mark.asyncio
async def test_allow_noncompliant_clients_defaults_to_true() -> None:
    """Rejecting non-compliant clients is opt-in."""
    server = _make_server()
    assert server.allow_noncompliant_clients is True


@pytest.mark.asyncio
async def test_flag_noncompliance_lenient_dedups_per_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Lenient mode logs each distinct reason once, not on every occurrence."""
    server = _make_server()
    client = server.get_or_create_client("dev")
    with caplog.at_level(logging.INFO):
        client.flag_noncompliance("legacy thing")
        client.flag_noncompliance("legacy thing")
        client.flag_noncompliance("other thing")
    messages = [r.message for r in caplog.records if "non-compliant client" in r.message]
    assert messages == [
        "non-compliant client: legacy thing",
        "non-compliant client: other thing",
    ]


@pytest.mark.asyncio
async def test_flag_noncompliance_relogs_after_reconnect(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A new connection re-logs a reason already seen on the previous one."""
    server = _make_server()
    client = server.get_or_create_client("dev")
    with caplog.at_level(logging.INFO):
        client.flag_noncompliance("legacy thing")
        client._noncompliance_logged.clear()  # attach_connection resets this  # noqa: SLF001
        client.flag_noncompliance("legacy thing")
    hits = [r for r in caplog.records if "non-compliant client: legacy thing" in r.message]
    assert len(hits) == 2


@pytest.mark.asyncio
async def test_flag_noncompliance_strict_raises() -> None:
    """Strict mode raises ClientComplianceError."""
    server = _make_server(allow_noncompliant_clients=False)
    client = server.get_or_create_client("dev")
    with pytest.raises(ClientComplianceError, match="legacy thing"):
        client.flag_noncompliance("legacy thing")
