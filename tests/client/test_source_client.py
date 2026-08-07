"""Tests for source-specific client API wiring."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aiosendspin.client.client import SendspinClient
from aiosendspin.models.types import Roles
from tests.conftest import make_sdk_client


async def test_source_role_requires_source_support() -> None:
    """A source client must provide its versioned support object."""
    with pytest.raises(ValueError, match="source_support"):
        make_sdk_client(client_name="source", roles=[Roles.SOURCE])


async def test_send_available_uses_admitted_connection() -> None:
    """The public availability API delegates to the active connection."""
    client = SendspinClient.__new__(SendspinClient)
    connection = AsyncMock()
    client._admitted_connection = connection  # type: ignore[assignment]  # noqa: SLF001

    await client.send_available(available=False)

    connection.send_available.assert_awaited_once_with(available=False)
