"""The client's configured Noise cipher suite drives its handshake."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from aiosendspin.client.connection import SendspinConnection
from aiosendspin.models.types import Roles
from aiosendspin.noise.session import NoiseCipherSuite
from aiosendspin.noise.trust_store import PskCategory
from tests.conftest import make_sdk_client

if TYPE_CHECKING:
    import pytest


async def test_handshake_uses_configured_cipher_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-default cipher suite is passed through to the Noise handshake."""
    sdk = make_sdk_client(
        client_name="c", roles=[Roles.CONTROLLER], cipher_suite=NoiseCipherSuite.AESGCM
    )
    connection = SendspinConnection(sdk)
    captured: dict[str, NoiseCipherSuite] = {}

    async def fake_handshake(_raw_ws: object, *, suite: NoiseCipherSuite, **_: object) -> MagicMock:
        captured["suite"] = suite
        result = MagicMock()
        result.psk.category = PskCategory.SENTINEL
        return result

    monkeypatch.setattr("aiosendspin.client.connection.run_handshake_client", fake_handshake)
    await connection._run_noise_handshake(MagicMock(), expected_server_id=None)  # noqa: SLF001

    assert captured["suite"] is NoiseCipherSuite.AESGCM
