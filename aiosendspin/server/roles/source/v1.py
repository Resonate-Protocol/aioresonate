"""Source role (v1) - per-connection behavior.

A source client captures audio from a local input (line-in, turntable, Bluetooth,
microphone) and streams it to the server. This role tracks the announced input
stream format, validates inbound capture against the spec, and forwards accepted
frames to the group role.

Whether a source should be streaming at all is server policy, not library
behavior: the host decides, by calling ``send_start_command()`` and
``send_stop_command()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiosendspin.models.core import ServerCommandMessage, ServerCommandPayload
from aiosendspin.models.source import SourceCommandPayload
from aiosendspin.models.types import (
    BinaryMessageType,
    ClientStateType,
    SourceCommand,
    SourceSignal,
)
from aiosendspin.server.roles.base import Role
from aiosendspin.server.roles.source.group import SourceGroupRole
from aiosendspin.util import create_task

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
    """Per-client source state.

    ``signal`` is a property of the device and survives reconnects. The streaming
    fields do not: per spec, streaming state is per-connection and a previously
    sent ``start`` does not survive reconnection.
    """

    stream_source: ClientStreamStartSource | None = None
    """Format announced by the most recent client_stream/start, if streaming."""
    signal: SourceSignal | None = None
    """Last reported line-sensing signal, if the source supports line sensing."""
    commanded: SourceCommand | None = None
    """Last start/stop command sent on this connection (to avoid redundant commands)."""


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

    @property
    def is_streaming_requested(self) -> bool:
        """Whether this connection has been told to stream."""
        return self._state().commanded == SourceCommand.START

    # --- Lifecycle ---

    def on_connect(self) -> None:
        """Reset per-connection streaming state and subscribe to the group role."""
        self._reset_stream_state()
        self._subscribe_to_group_role()

    def on_disconnect(self) -> None:
        """Close the input stream and unsubscribe from the group role."""
        if (group := self._group_source()) is not None:
            group.end_stream(self)
        self._unsubscribe_from_group_role()
        self._reset_stream_state()

    def on_group_changed(self, group: object) -> None:
        """Re-publish an open capture stream to the group the source moved to.

        Unsubscribing closes the stream the old group's host was consuming, so a
        source that is mid-capture would otherwise go silent without ever sending
        ``client_stream/end``.
        """
        super().on_group_changed(group)
        source = self.stream_source
        if source is not None and (group_role := self._group_source()) is not None:
            group_role.start_stream(self, source)

    def _reset_stream_state(self) -> None:
        """Clear streaming state that must not survive a reconnect."""
        state = self._state()
        state.stream_source = None
        state.commanded = None

    # --- Message hooks ---

    def on_client_state(self, payload: ClientStatePayload) -> None:
        """Record the line-sensing signal and surface it to the host."""
        if payload.source is None:
            return
        signal = payload.source.signal
        if signal == self._state().signal:
            return
        self._state().signal = signal
        if (group := self._group_source()) is not None:
            group.report_signal(self, signal)

    def on_client_stream_start(self, payload: ClientStreamStartPayload) -> None:
        """Open the announced input stream, unless it was never requested."""
        if not self.is_streaming_requested:
            # Spec: an unsolicited client_stream/start is a protocol error. The
            # stream must not be treated as open, and the connection is closed.
            logger.warning(
                "Source %s sent client_stream/start without a start command; closing connection",
                self.client_id,
            )
            self._close_connection()
            return
        self._state().stream_source = payload.source
        if (group := self._group_source()) is not None:
            group.start_stream(self, payload.source)

    def on_client_stream_end(self, payload: ClientStreamEndPayload) -> None:  # noqa: ARG002
        """End the current input stream."""
        self._state().stream_source = None
        if (group := self._group_source()) is not None:
            group.end_stream(self)

    def on_client_binary(self, message_type: int, timestamp_us: int, data: bytes) -> None:
        """Forward captured source audio frames to the group role."""
        if message_type != _SOURCE_AUDIO_CHUNK:
            return
        if self.stream_source is None:
            # No open input stream (before client_stream/start or after
            # client_stream/end). Chunks may keep arriving after a stop command
            # until the client processes it, so this is expected and not an error.
            return
        if not self._is_available():
            # An unavailable source has not converged its time filter, so its
            # capture timestamps cannot be trusted.
            logger.debug(
                "Source %s is not available (state=%s); dropping captured frame",
                self.client_id,
                self._client.client_state.value,
            )
            return
        if (group := self._group_source()) is not None:
            group.push_audio(self, timestamp_us, data)

    # --- Commands ---

    def send_start_command(self) -> None:
        """Request this source to begin streaming."""
        if self._state().commanded == SourceCommand.START:
            return
        self._state().commanded = SourceCommand.START
        self._send_command(SourceCommand.START)

    def send_stop_command(self) -> None:
        """Request this source to stop streaming."""
        if self._state().commanded == SourceCommand.STOP:
            return
        self._state().commanded = SourceCommand.STOP
        self._send_command(SourceCommand.STOP)

    def _send_command(self, command: SourceCommand) -> None:
        self.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(source=SourceCommandPayload(command=command))
            )
        )

    # --- Helpers ---

    def _is_available(self) -> bool:
        """Whether the client reports itself as available to participate."""
        return self._client.client_state == ClientStateType.SYNCHRONIZED

    def _close_connection(self) -> None:
        """Close the connection after a protocol violation."""
        connection = self._client.connection
        if connection is not None:
            create_task(connection.disconnect(retry_connection=False))

    def _group_source(self) -> SourceGroupRole | None:
        return self._group_role if isinstance(self._group_role, SourceGroupRole) else None
