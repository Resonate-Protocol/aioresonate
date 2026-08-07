"""Group-level coordination for the source role.

A source client captures audio from a local input and streams it to the server.
``SourceGroupRole`` decodes those frames (per the format announced in
``client_stream/start``) and hands the resulting PCM to the host application as
a ``SourceStreamStartedEvent`` carrying an async iterator of chunks.

The library does not route that audio anywhere by itself. Feeding it into the
group's own ``PushStream`` is one policy, and not the right one for every host:
a host with its own audio pipeline (mixing, effects, multi-room routing) needs
the capture handed to it rather than injected into a stream it does not own.
Sources are tracked per client, so a host is free to mix several at once.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiosendspin.server.roles.base import GroupRole
from aiosendspin.server.roles.source.decoder import SourceDecoder
from aiosendspin.server.roles.source.events import (
    SourceSignalChangedEvent,
    SourceStreamEndedEvent,
    SourceStreamStartedEvent,
)
from aiosendspin.server.roles.source.stream import SourceAudioStream

if TYPE_CHECKING:
    from aiosendspin.models.source import ClientStreamStartSource
    from aiosendspin.models.types import SourceSignal
    from aiosendspin.server.group import SendspinGroup
    from aiosendspin.server.roles.base import Role
    from aiosendspin.server.roles.source.v1 import SourceV1Role

logger = logging.getLogger(__name__)


class SourceGroupRole(GroupRole):
    """Tracks source clients in a group and publishes their capture to the host."""

    role_family = "source"

    def __init__(self, group: SendspinGroup) -> None:
        """Initialize source coordination state for the group."""
        super().__init__(group)
        self._streams: dict[str, SourceAudioStream] = {}

    @property
    def active_streams(self) -> dict[str, SourceAudioStream]:
        """Open capture streams by client ID."""
        return dict(self._streams)

    # --- Membership ---

    def on_member_leave(self, role: Role) -> None:
        """Close the capture stream of a departing source."""
        if isinstance(role, _source_role_cls()):
            self.end_stream(role)

    def close(self) -> None:
        """Close every open capture stream when the group is torn down."""
        for client_id in list(self._streams):
            self._close_stream(client_id)

    # --- Stream lifecycle (driven by SourceV1Role) ---

    def start_stream(self, role: SourceV1Role, source: ClientStreamStartSource) -> None:
        """Open a capture stream for a source and publish it to the host.

        A ``client_stream/start`` while a stream is already open replaces it, so
        the previous stream is closed first and the host gets a fresh one for the
        new format.
        """
        self._close_stream(role.client_id)
        try:
            decoder = SourceDecoder(source)
        except Exception:
            logger.exception(
                "Source %s: cannot decode announced format %s; ignoring stream",
                role.client_id,
                source.codec,
            )
            return

        stream = SourceAudioStream(client_id=role.client_id, decoder=decoder)
        self._streams[role.client_id] = stream
        self.emit_group_event(
            SourceStreamStartedEvent(
                client_id=role.client_id,
                audio_format=decoder.audio_format,
                stream=stream,
            )
        )

    def end_stream(self, role: SourceV1Role) -> None:
        """Close a source's capture stream, if it has one open."""
        self._close_stream(role.client_id)

    def push_audio(self, role: SourceV1Role, timestamp_us: int, data: bytes) -> None:
        """Hand a captured frame to the source's open stream."""
        stream = self._streams.get(role.client_id)
        if stream is None:
            # No open input stream: the role rejects these before reaching us,
            # but a stream can also be closed between the two.
            return
        stream.push(timestamp_us, data)

    def report_signal(self, role: SourceV1Role, signal: SourceSignal | None) -> None:
        """Publish a source's line-sensing signal to the host."""
        self.emit_group_event(SourceSignalChangedEvent(client_id=role.client_id, signal=signal))

    # --- Helpers ---

    def _close_stream(self, client_id: str) -> None:
        stream = self._streams.pop(client_id, None)
        if stream is None:
            return
        stream.end()
        self.emit_group_event(SourceStreamEndedEvent(client_id=client_id))


def _source_role_cls() -> type[SourceV1Role]:
    """Return the SourceV1Role class (imported lazily to avoid a cycle)."""
    from aiosendspin.server.roles.source.v1 import SourceV1Role  # noqa: PLC0415

    return SourceV1Role
