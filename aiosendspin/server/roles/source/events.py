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
    """A source client began an input stream."""

    audio_format: AudioFormat
    """Native PCM format of the decoded audio yielded by ``handle``."""
    handle: SourceStream
    """Async iterator over decoded ``(pcm, timestamp_us)`` chunks."""


@dataclass
class SourceStreamEndedEvent(ClientRoleEvent):
    """A source client ended its input stream."""


@dataclass
class SourceSignalChangedEvent(ClientRoleEvent):
    """A source client reported signal presence."""

    signal: SignalState
    """Reported signal presence."""
