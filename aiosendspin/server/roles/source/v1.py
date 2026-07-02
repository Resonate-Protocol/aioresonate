"""Source role (v1) - per-connection behavior.

A source client captures audio from a local input (line-in, turntable, Bluetooth,
microphone) and streams it to the server. This role tracks the announced input
stream format, forwards captured audio frames to the group for decode and
distribution, and issues ``server/command`` start/stop requests (the server is the
sole initiator of source streaming).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiosendspin.models.core import ServerCommandMessage, ServerCommandPayload
from aiosendspin.models.source import SourceCommandPayload
from aiosendspin.models.types import BinaryMessageType, SourceCommand, SourceSignal
from aiosendspin.server.roles.base import Role
from aiosendspin.server.roles.source.group import SourceGroupRole

if TYPE_CHECKING:
    from aiosendspin.models.core import (
        ClientStatePayload,
        ClientStreamEndPayload,
        ClientStreamStartPayload,
    )
    from aiosendspin.models.source import ClientStreamStartSource
    from aiosendspin.server.client import SendspinClient

logger = logging.getLogger(__name__)

_SOURCE_AUDIO_CHUNK = BinaryMessageType.SOURCE_AUDIO_CHUNK.value


@dataclass
class SourceRoleState:
    """Persistent per-client source state (survives reconnects)."""

    stream_source: ClientStreamStartSource | None = None
    """Format announced by the most recent client_stream/start, if streaming."""
    signal: SourceSignal | None = None
    """Last reported line-sensing signal, if the source supports line sensing."""
    commanded: SourceCommand | None = None
    """Last start/stop command sent to this source (to avoid redundant commands)."""


class SourceV1Role(Role):
    """Per-connection implementation of the ``source@v1`` role."""

    def __init__(self, *, client: SendspinClient) -> None:
        """Initialize the role for a client connection."""
        self._client = client

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "source@v1"

    @property
    def role_family(self) -> str:
        """Role family name for protocol messages."""
        return "source"

    @property
    def client_id(self) -> str:
        """Identifier of the owning client."""
        return self._client.client_id

    def _state(self) -> SourceRoleState:
        return self._client.get_or_create_role_state("source", SourceRoleState)

    @property
    def stream_source(self) -> ClientStreamStartSource | None:
        """Format of the active input stream, or None if not streaming."""
        return self._state().stream_source

    @property
    def signal(self) -> SourceSignal | None:
        """Last reported line-sensing signal."""
        return self._state().signal

    # --- Lifecycle ---

    def on_connect(self) -> None:
        """Subscribe to the group source coordinator."""
        self._subscribe_to_group_role()

    def on_disconnect(self) -> None:
        """Unsubscribe from the group source coordinator."""
        self._unsubscribe_from_group_role()

    # --- Message hooks ---

    def on_client_state(self, payload: ClientStatePayload) -> None:
        """Track line-sensing signal and gate streaming on it (reference policy)."""
        if payload.source is None:
            return
        self._state().signal = payload.source.signal
        # If the source reports line sensing, follow it: start on signal, stop on silence.
        if payload.source.signal == SourceSignal.ABSENT:
            self.send_stop_command()
        elif payload.source.signal == SourceSignal.PRESENT:
            self.send_start_command()

    def on_client_stream_start(self, payload: ClientStreamStartPayload) -> None:
        """Record the announced input stream format and reset the decoder."""
        self._state().stream_source = payload.source
        if (group := self._group_source()) is not None:
            group.clear_decoder(self)

    def on_client_stream_end(self, payload: ClientStreamEndPayload) -> None:  # noqa: ARG002
        """End the current input stream and stop distributing its audio."""
        self._state().stream_source = None
        if (group := self._group_source()) is not None:
            group.stop_source(self)

    def on_client_binary(self, message_type: int, timestamp_us: int, data: bytes) -> None:
        """Forward captured source audio frames to the group for distribution."""
        if message_type != _SOURCE_AUDIO_CHUNK:
            return
        if self.stream_source is None:
            return  # no client_stream/start yet; ignore stray frames
        if (group := self._group_source()) is not None:
            group.enqueue(self, timestamp_us, data)

    # --- Commands ---

    def send_start_command(self) -> None:
        """Request this source to begin streaming."""
        if self._state().commanded == SourceCommand.START:
            return
        self._state().commanded = SourceCommand.START
        self.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(
                    source=SourceCommandPayload(command=SourceCommand.START)
                )
            )
        )

    def send_stop_command(self) -> None:
        """Request this source to stop streaming."""
        if self._state().commanded == SourceCommand.STOP:
            return
        self._state().commanded = SourceCommand.STOP
        self.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(
                    source=SourceCommandPayload(command=SourceCommand.STOP)
                )
            )
        )

    # --- Helpers ---

    def _group_source(self) -> SourceGroupRole | None:
        return self._group_role if isinstance(self._group_role, SourceGroupRole) else None
