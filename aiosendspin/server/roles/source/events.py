"""Source group role events.

These events are the library's entire source-role output: the host learns when a
source starts and stops capturing, gets the decoded audio through
``SourceStreamStartedEvent.stream``, and decides what to do with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiosendspin.server.events import GroupRoleEvent

if TYPE_CHECKING:
    from aiosendspin.models.types import SourceSignal
    from aiosendspin.server.audio import AudioFormat
    from aiosendspin.server.roles.source.stream import SourceAudioStream


class SourceEvent(GroupRoleEvent):
    """Base event type for source group role changes."""


@dataclass
class SourceStreamStartedEvent(SourceEvent):
    """A source client opened an input stream and is about to send audio.

    Iterate ``stream`` to receive the decoded capture. Iteration ends by itself
    when the source stops, disconnects, or the group goes away, so a host can
    simply run ``async for chunk in event.stream:`` in a task.
    """

    client_id: str
    """Identifier of the source client."""

    audio_format: AudioFormat
    """PCM format the stream yields."""

    stream: SourceAudioStream
    """Decoded capture, as an async iterator of chunks."""


@dataclass
class SourceStreamEndedEvent(SourceEvent):
    """A source client's input stream ended; its stream has been closed."""

    client_id: str
    """Identifier of the source client."""


@dataclass
class SourceSignalChangedEvent(SourceEvent):
    """A source reported a new line-sensing signal in ``client/state``.

    Purely informational: whether a signal should start or stop a source is
    server policy, so act on it by calling ``send_start_command()`` /
    ``send_stop_command()`` on the source role.
    """

    client_id: str
    """Identifier of the source client."""

    signal: SourceSignal | None
    """The reported signal, or None if the source cleared it."""
