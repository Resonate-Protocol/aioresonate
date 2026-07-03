"""Stream lifecycle messages carry a server_transmitted send-time timestamp."""

from __future__ import annotations

import pytest

from aiosendspin.models.core import (
    StreamClearPayload,
    StreamEndPayload,
    StreamStartPayload,
)


@pytest.mark.parametrize(
    "payload",
    [StreamStartPayload(), StreamClearPayload(), StreamEndPayload()],
)
def test_stream_payload_always_serializes_server_transmitted(
    payload: StreamStartPayload | StreamClearPayload | StreamEndPayload,
) -> None:
    """server_transmitted is a required field, present on the wire even when unset."""
    assert "server_transmitted" in payload.to_dict()
