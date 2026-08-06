"""Source role client events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiosendspin.server.events import ClientRoleEvent

if TYPE_CHECKING:
    from aiosendspin.audio.format import AudioFormat
    from aiosendspin.models.types import SignalState

    from .stream import SourceStream


@dataclass
class SourceStreamStartedEvent(ClientRoleEvent):
    """A source client began an input stream.

    ``handle`` is a single-consumer async iterator yielding ``(pcm, timestamp_us)``
    pairs of decoded native PCM. It ends when the source sends ``client_stream/end``.
    """

    audio_format: AudioFormat
    """Native PCM format of the decoded audio yielded by ``handle``."""
    handle: SourceStream
    """Async iterator over decoded ``(pcm, timestamp_us)`` chunks."""


@dataclass
class SourceStreamEndedEvent(ClientRoleEvent):
    """A source client ended its input stream; the matching handle is exhausted."""


@dataclass
class SourceSignalChangedEvent(ClientRoleEvent):
    """A source client reported signal/line-sense presence via client/state."""

    signal: SignalState
    """Signal presence; only emitted when the source advertised the 'line_sense' feature."""
