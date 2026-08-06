"""Source role implementation: decode audio captured by a source client."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, ClassVar

from aiosendspin.audio.codecs import create_decoder
from aiosendspin.audio.format import AudioFormat
from aiosendspin.models.core import ServerCommandMessage, ServerCommandPayload
from aiosendspin.models.source import SourceCommandServerPayload
from aiosendspin.models.types import BinaryMessageType
from aiosendspin.server.roles.base import Role

from .events import (
    SourceSignalChangedEvent,
    SourceStreamEndedEvent,
    SourceStreamStartedEvent,
)
from .stream import SourceStream

if TYPE_CHECKING:
    from aiosendspin.models.core import ClientStatePayload
    from aiosendspin.models.source import ClientStreamStartPayload
    from aiosendspin.server.client import SendspinClient

logger = logging.getLogger(__name__)


class SourceV1Role(Role):
    """Per-connection role that decodes audio streamed up by a source client."""

    handled_binary_types: ClassVar[frozenset[int]] = frozenset(
        {BinaryMessageType.SOURCE_AUDIO_CHUNK.value}
    )

    def __init__(self, client: SendspinClient | None = None) -> None:
        """Initialize the source role."""
        if client is None:
            raise ValueError("SourceV1Role requires a client")
        self._client = client
        self._group_role = None
        self._decoder: object | None = None
        self._stream: SourceStream | None = None
        self._stream_active = False
        self._initial_state_received = False
        # Whether this connection has been asked to stream. Per-connection state: a
        # start does not survive a reconnect, so the server must ask again.
        self._start_requested = False
        # Timestamp of the most recent chunk, used to stamp the decoder flush tail.
        self._last_timestamp_us = 0

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "source@v1"

    @property
    def stream_active(self) -> bool:
        """Whether the client currently has an open input stream."""
        return self._stream_active

    @property
    def role_family(self) -> str:
        """Role family name for protocol messages."""
        return "source"

    def on_connect(self) -> None:
        """Source has no group role; nothing to subscribe."""

    def on_disconnect(self) -> None:
        """End any active stream so a waiting consumer is released."""
        self._end_stream()
        self._initial_state_received = False
        self._start_requested = False

    def on_deactivate(self) -> None:
        """End any active stream when the role leaves active_roles."""
        self._end_stream()
        self._initial_state_received = False
        self._start_requested = False
        super().on_deactivate()

    def requires_initial_state(self) -> bool:
        """Require synchronized client state before accepting captured audio."""
        return True

    # --- Server-initiated streaming control ---

    def request_start(self) -> None:
        """Ask the source client to begin streaming (server/command: start)."""
        self._start_requested = True
        self.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(source=SourceCommandServerPayload(command="start"))
            )
        )

    def request_stop(self) -> None:
        """Ask the source client to stop streaming (server/command: stop)."""
        self._start_requested = False
        self.send_message(
            ServerCommandMessage(
                payload=ServerCommandPayload(source=SourceCommandServerPayload(command="stop"))
            )
        )

    # --- Inbound stream handling ---

    def on_client_stream_start(self, payload: ClientStreamStartPayload) -> None:
        """Build a decoder and a fresh stream handle, then announce it."""
        if not self._start_requested:
            # The server never asked this connection to stream, so no stream is opened.
            # Flagging (rather than disconnecting here) lets strict servers reject the
            # client while tolerant ones survive a start crossing an in-flight stop.
            self._client.flag_noncompliance(
                "client_stream/start sent without a preceding source start command"
            )
            return
        source = payload.source
        # Restart on a second start with no update path: end the prior stream first.
        if self._stream_active:
            self._end_stream()

        audio_format = AudioFormat(
            sample_rate=source.sample_rate,
            bit_depth=source.bit_depth,
            channels=source.channels,
        )
        header = base64.b64decode(source.codec_header) if source.codec_header else None
        try:
            # Validate the declared shape here: a PCM decoder accepts anything, so an
            # impossible format would otherwise only fail inside the consumer.
            if source.sample_rate <= 0:
                msg = f"Unsupported sample rate: {source.sample_rate}"
                raise ValueError(msg)  # noqa: TRY301
            audio_format.resolve_av_format()
            self._decoder = create_decoder(
                source.codec.value,
                sample_rate=source.sample_rate,
                bit_depth=source.bit_depth,
                channels=source.channels,
                codec_header=header,
            )
        except (ValueError, ImportError):
            logger.exception("Failed to build source decoder for codec %r", source.codec)
            self._decoder = None
            return

        self._stream = SourceStream(audio_format)
        self._stream_active = True
        self._client._signal_event(  # noqa: SLF001
            SourceStreamStartedEvent(audio_format=audio_format, handle=self._stream)
        )

    def on_binary_chunk(self, message_type: int, timestamp_us: int, data: bytes) -> None:  # noqa: ARG002
        """Decode an inbound source audio chunk into the active stream, if one is open.

        Chunks are dropped unless a stream is open and the client reports being
        available, since only then are its capture timestamps trustworthy.
        """
        if (
            not self._initial_state_received
            or not self._stream_active
            or self._stream is None
            or self._decoder is None
        ):
            return
        if not self._client.available:
            return
        try:
            pcm = self._decoder.decode(data)  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Failed to decode source audio chunk")
            return
        # Keep the flush-tail stamp monotonic even if a chunk arrives out of order.
        self._last_timestamp_us = max(self._last_timestamp_us, timestamp_us)
        self._stream._push(pcm, timestamp_us)  # noqa: SLF001

    def on_client_stream_end(self) -> None:
        """End the active stream and release its decoder."""
        self._end_stream()

    def _end_stream(self) -> None:
        """Drain decoder tail into the stream, close it, and announce the end.

        Every teardown path routes through here (client end, stream replacement,
        deactivation, disconnect) so a consumer always learns its handle died.
        """
        was_active = self._stream_active
        if self._stream is not None and self._decoder is not None:
            try:
                tail = self._decoder.flush()  # type: ignore[attr-defined]
            except Exception:
                logger.exception("Failed to flush source decoder")
                tail = b""
            # Tail continues just after the last chunk, so carry its timestamp.
            self._stream._push(tail, self._last_timestamp_us)  # noqa: SLF001
            self._stream._end()  # noqa: SLF001
        self._stream = None
        self._decoder = None
        self._stream_active = False
        self._last_timestamp_us = 0
        if was_active:
            self._client._signal_event(SourceStreamEndedEvent())  # noqa: SLF001

    # --- State hooks ---

    def on_client_state(self, payload: ClientStatePayload) -> None:
        """Surface a source's signal presence, only when it advertised line_sense."""
        if payload.available is not None:
            self._initial_state_received = True
        source = payload.source
        if source is None or source.signal is None or not self._line_sense_supported():
            return
        self._client._signal_event(  # noqa: SLF001
            SourceSignalChangedEvent(signal=source.signal)
        )

    def _line_sense_supported(self) -> bool:
        """Whether the source advertised the 'line_sense' feature in client/hello."""
        support = self._client.info.source_support
        return (
            support is not None
            and support.features is not None
            and bool(support.features.line_sense)
        )
