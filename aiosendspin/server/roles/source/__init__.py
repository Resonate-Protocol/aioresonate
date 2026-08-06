"""Server-side source role: receive and decode audio captured by source clients."""

from __future__ import annotations

from aiosendspin.server.roles.registry import register_role

from .events import (
    SourceSignalChangedEvent,
    SourceStreamEndedEvent,
    SourceStreamStartedEvent,
)
from .stream import SourceStream
from .v1 import SourceV1Role

# Source captures local audio (potentially a microphone), so it may only be
# activated on a long-term paired connection.
register_role(
    "source@v1",
    lambda client: SourceV1Role(client=client),
    requires_pairing=True,
)

__all__ = [
    "SourceSignalChangedEvent",
    "SourceStream",
    "SourceStreamEndedEvent",
    "SourceStreamStartedEvent",
    "SourceV1Role",
]
