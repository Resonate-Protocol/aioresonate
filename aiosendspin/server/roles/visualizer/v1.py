"""Visualizer role implementation for draft visualizer streaming."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.models import pack_binary_header_raw
from aiosendspin.models.core import (
    StreamClearMessage,
    StreamClearPayload,
    StreamEndMessage,
    StreamEndPayload,
    StreamStartMessage,
    StreamStartPayload,
)
from aiosendspin.models.types import BinaryMessageType
from aiosendspin.models.visualizer import (
    ClientHelloVisualizerSupport,
    StreamStartVisualizer,
)
from aiosendspin.server.roles.base import (
    AudioChunk,
    AudioRequirements,
    BinaryHandling,
    Role,
    StreamRequirements,
)
from aiosendspin.server.roles.visualizer.features import VisualizerFeatureExtractor
from aiosendspin.server.roles.visualizer.packing import pack_visualizer_frames

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient


class VisualizerV1Role(Role):
    """Role implementation for draft visualizer streaming."""

    def __init__(self, client: SendspinClient | None = None) -> None:
        """Initialize VisualizerV1Role."""
        if client is None:
            msg = "VisualizerV1Role requires a client"
            raise ValueError(msg)
        self._client = client
        self._stream_started = False
        self._buffer_tracker = None
        self._group_role = None
        self._stream_config: StreamStartVisualizer | None = None
        self._extractor: VisualizerFeatureExtractor | None = None

    @property
    def role_id(self) -> str:
        """Versioned role identifier."""
        return "visualizer@_draft_r1"

    @property
    def role_family(self) -> str:
        """Role family name for protocol messages."""
        return "visualizer"

    def get_stream_requirements(self) -> StreamRequirements:
        """Visualizer role sends binary streams."""
        return StreamRequirements()

    def get_audio_requirements(self) -> AudioRequirements:
        """Return audio requirements for visualizer analysis."""
        return AudioRequirements(
            sample_rate=48_000,
            bit_depth=16,
            channels=2,
            frame_duration_us=25_000,
        )

    def get_binary_handling(self, message_type: int) -> BinaryHandling | None:
        """Return handling policy for visualizer binary frames."""
        if message_type == BinaryMessageType.VISUALIZATION_DATA.value:
            return BinaryHandling(drop_late=True, grace_period_us=2_000_000, buffer_track=False)
        return None

    def on_connect(self) -> None:
        """Initialize stream config and subscribe to group role."""
        self._init_stream_config()
        self._subscribe_to_group_role()

    def on_disconnect(self) -> None:
        """Unsubscribe from VisualizerGroupRole."""
        self._unsubscribe_from_group_role()
        self._stream_started = False
        self._extractor = None

    def on_stream_start(self) -> None:
        """Start extractor state for a new audio stream."""
        if self._stream_config is None:
            return
        # stream/end clears client-side visualizer config, so resend stream/start
        # at each new stream boundary.
        self._send_stream_start()
        req = self.get_audio_requirements()
        self._extractor = VisualizerFeatureExtractor(
            sample_rate=req.sample_rate,
            channels=req.channels,
            config=self._stream_config,
        )
        self._stream_started = True

    def on_audio_chunk(self, chunk: AudioChunk) -> None:
        """Process audio chunk and emit visualizer binary frame."""
        if not self.has_connection() or self._stream_config is None:
            return

        if self._extractor is None:
            self.on_stream_start()
            if self._extractor is None:
                return

        frame = self._extractor.process_chunk(chunk.data, chunk.timestamp_us)
        body = pack_visualizer_frames(frames=[frame], config=self._stream_config)
        header = pack_binary_header_raw(
            BinaryMessageType.VISUALIZATION_DATA.value, frame.timestamp_us
        )
        packed = header + body
        self._client.send_binary(
            packed,
            role_family=self.role_family,
            timestamp_us=frame.timestamp_us,
            message_type=BinaryMessageType.VISUALIZATION_DATA.value,
        )

    def on_stream_clear(self) -> None:
        """Reset visualizer state and notify client to clear buffered data."""
        if self._extractor is not None:
            self._extractor.reset()
        self.send_message(StreamClearMessage(payload=StreamClearPayload(roles=["visualizer"])))
        self.reset_binary_timing()

    def on_stream_end(self) -> None:
        """Reset visualizer state and notify client that stream has ended."""
        self._extractor = None
        self._stream_started = False
        self.send_message(StreamEndMessage(payload=StreamEndPayload(roles=["visualizer"])))
        self.reset_binary_timing()

    def on_stream_request_format(self, payload: object) -> None:  # noqa: ARG002
        """Ignore runtime visualizer renegotiation for now."""
        self._client._logger.debug(  # noqa: SLF001
            "Ignoring visualizer stream/request-format from client %s",
            self._client.client_id,
        )

    def _init_stream_config(self) -> None:
        """Parse visualizer support config from client/hello."""
        support_raw = self._client.info.visualizer_support
        if support_raw is None:
            raise ValueError("visualizer support object missing for draft visualizer role")
        support = ClientHelloVisualizerSupport.from_dict(
            self._normalize_support_payload(support_raw)
        )
        self._stream_config = StreamStartVisualizer.from_support(support)

    def _normalize_support_payload(self, support_raw: object) -> dict[str, object]:
        """Normalize legacy/minimal support payloads to draft visualizer schema."""
        if isinstance(support_raw, dict):
            payload: dict[str, object] = dict(support_raw)
        elif isinstance(support_raw, ClientHelloVisualizerSupport):
            # Legacy and draft payloads represented by model object.
            payload = support_raw.to_dict()
        else:
            raise TypeError("visualizer support object must be a JSON object")

        # Backward-compatible defaults when client sends only buffer capacity.
        if "types" not in payload:
            payload["types"] = ["loudness", "f_peak"]
        if "batch_max" not in payload:
            payload["batch_max"] = 8

        raw_types = payload.get("types")
        if isinstance(raw_types, list):
            normalized_types = [v for v in raw_types if isinstance(v, str)]
            if "spectrum" in normalized_types and payload.get("spectrum") is None:
                normalized_types = [v for v in normalized_types if v != "spectrum"]
            if not normalized_types:
                normalized_types = ["loudness", "f_peak"]
            payload["types"] = normalized_types

        return payload

    def _send_stream_start(self) -> None:
        """Send stream/start with negotiated visualizer configuration."""
        if self._stream_config is None:
            return
        message = StreamStartMessage(payload=StreamStartPayload(visualizer=self._stream_config))
        self.send_message(message)
        self._stream_started = True
