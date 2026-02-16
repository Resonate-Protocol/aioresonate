"""SourceV1Role implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiosendspin.models import BinaryMessageType
from aiosendspin.models.core import (
    ClientCommandPayload,
    ClientStatePayload,
    InputStreamStartPayload,
    ServerCommandMessage,
    ServerCommandPayload,
)
from aiosendspin.models.source import (
    InputStreamStartSource,
    SourceCommandPayload,
    SourceStatePayload,
)
from aiosendspin.models.types import (
    SourceClientCommand,
    SourceCommand,
    SourceSignalType,
    SourceStateType,
)
from aiosendspin.server.roles.base import Role
from aiosendspin.util import create_task

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.roles.source.group import SourceGroupRole


@dataclass
class SourceRoleState:
    """Persistent source state across reconnects."""

    state: SourceStateType = SourceStateType.IDLE
    signal: SourceSignalType | None = None
    level: float | None = None
    input_stream_format: InputStreamStartSource | None = None
    last_event: SourceClientCommand | None = None
    last_event_ts_us: int | None = None


class SourceV1Role(Role):
    """Role implementation for source@v1 clients."""

    def __init__(self, client: SendspinClient | None = None) -> None:
        """Initialize role state for a single source@v1 client."""
        if client is None:
            raise ValueError("SourceV1Role requires a client")
        self._client = client
        self._group_role: SourceGroupRole | None = None

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "source@v1"

    @property
    def role_family(self) -> str:
        """Role family name used for state storage."""
        return "source"

    @property
    def state(self) -> SourceStateType:
        """Current source state."""
        return self._get_state().state

    @property
    def signal(self) -> SourceSignalType | None:
        """Most recent source signal state, if available."""
        return self._get_state().signal

    @property
    def last_event(self) -> SourceClientCommand | None:
        """Last source lifecycle event reported by the client."""
        return self._get_state().last_event

    @property
    def last_event_ts_us(self) -> int | None:
        """Server timestamp of the last source lifecycle event."""
        return self._get_state().last_event_ts_us

    @property
    def input_stream_format(self) -> InputStreamStartSource | None:
        """Current input stream format announced by the source."""
        return self._get_state().input_stream_format

    def _get_state(self) -> SourceRoleState:
        """Get or create persistent per-client source state."""
        return self._client.get_or_create_role_state(self.role_family, SourceRoleState)

    def on_connect(self) -> None:
        """Subscribe to group-level source coordination on connect."""
        self._subscribe_to_group_role()

    def on_disconnect(self) -> None:
        """Unsubscribe from group-level source coordination on disconnect."""
        self._unsubscribe_from_group_role()

    def requires_initial_state(self) -> bool:
        """Require initial server/state after connection."""
        return True

    def on_client_state(self, payload: ClientStatePayload) -> None:
        """Handle source state updates from client/state."""
        if payload.source is None:
            return
        self._update_source_state(payload.source)

    def on_input_stream_start(self, payload: InputStreamStartPayload) -> None:
        """Store current input stream format from input_stream/start."""
        state = self._get_state()
        state.input_stream_format = payload.source
        if self._group_role is not None:
            self._group_role.clear_decoder(self)

    def on_input_stream_end(self) -> None:
        """Clear input stream format when input_stream/end is received."""
        state = self._get_state()
        state.input_stream_format = None
        if self._group_role is not None:
            self._group_role.clear_decoder(self)

    def on_command(self, payload: ClientCommandPayload) -> None:
        """Record source lifecycle events from client/command."""
        if payload.source is None:
            return
        state = self._get_state()
        state.last_event = payload.source.command
        state.last_event_ts_us = self._client._server.clock.now_us()  # noqa: SLF001
        if self._group_role is not None:
            self._group_role.push_state()

    def on_client_binary(self, *, message_type: int, timestamp_us: int, payload: bytes) -> None:
        """Forward source binary frames to group ingest when stream is active."""
        if message_type != BinaryMessageType.SOURCE_AUDIO_CHUNK.value:
            return
        state = self._get_state()
        if state.state != SourceStateType.STREAMING:
            return
        if state.input_stream_format is None:
            return
        if self._group_role is None:
            return
        self._group_role.enqueue(self, timestamp_us, payload)

    def send_start_command(self) -> None:
        """Send server/command start to this source."""
        self.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(
                    source=SourceCommandPayload(command=SourceCommand.START)
                )
            )
        )

    def send_stop_command(self) -> None:
        """Send server/command stop to this source."""
        self.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(
                    source=SourceCommandPayload(command=SourceCommand.STOP)
                )
            )
        )

    def _update_source_state(self, source_state: SourceStatePayload) -> None:
        """Update cached source state and synchronize group playback state."""
        state = self._get_state()
        old = state.state
        state.state = source_state.state
        state.signal = source_state.signal
        state.level = source_state.level

        if self._group_role is not None:
            if old != SourceStateType.STREAMING and source_state.state == SourceStateType.STREAMING:
                self._group_role.start_source(self)
            elif (
                old == SourceStateType.STREAMING and source_state.state != SourceStateType.STREAMING
            ):
                create_task(self._group_role.stop_source(self))
            self._group_role.push_state()
