"""Tests for PlayerRole stream lifecycle message payloads."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.models import AudioCodec, unpack_binary_header
from aiosendspin.models.core import StreamClearMessage, StreamEndMessage, StreamStartMessage
from aiosendspin.models.types import BinaryMessageType
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.roles import AudioChunk, AudioRequirements, PlayerRole
from aiosendspin.server.transformers import PcmPassthrough


def test_player_role_stream_clear_uses_role_family() -> None:
    """PlayerRole.stream/clear uses unversioned role family."""
    client = MagicMock()
    client.send_message = MagicMock()
    client.buffer_tracker = None

    role = PlayerRole(_client=client)
    role._has_transport = True  # noqa: SLF001
    role.clear_stream()

    msg = client.send_message.call_args.args[0]
    assert isinstance(msg, StreamClearMessage)
    assert msg.payload.roles == ["player"]


def test_player_role_stream_end_uses_role_family() -> None:
    """PlayerRole.stream/end omits roles (end all streams)."""
    client = MagicMock()
    client.send_message = MagicMock()
    client.buffer_tracker = None

    role = PlayerRole(_client=client)
    role._has_transport = True  # noqa: SLF001
    role.end_stream()

    msg = client.send_message.call_args.args[0]
    assert isinstance(msg, StreamEndMessage)
    assert msg.payload.roles is None


def test_player_role_send_cached_chunk_packs_header_and_tracks_duration() -> None:
    """Catch-up uses role-controlled header packing and accurate duration tracking."""

    class _Tracker:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def register(self, timestamp_us: int, byte_count: int) -> None:
            self.calls.append((timestamp_us, byte_count))

        def reset(self) -> None:
            return

    tracker = _Tracker()

    sent: list[bytes] = []
    client = MagicMock()
    client.buffer_tracker = tracker
    client.queue_high_water = MagicMock(return_value=False)

    def _try_send_binary(
        data: bytes,
        *,
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,  # noqa: ARG001
    ) -> bool:
        sent.append(data)
        if buffer_end_time_us is not None and buffer_byte_count is not None:
            tracker.register(buffer_end_time_us, buffer_byte_count)
        return True

    client.try_send_binary = MagicMock(side_effect=_try_send_binary)
    client.send_message = MagicMock()

    role = PlayerRole(_client=client)
    payload = b"\x01\x02\x03"
    timestamp_us = 123_000
    duration_us = 40_000
    byte_count = len(payload)

    assert role.send_cached_chunk(payload, timestamp_us, duration_us, byte_count)
    assert sent, "Expected a binary send"

    header = unpack_binary_header(sent[0])
    assert header.message_type == BinaryMessageType.AUDIO_CHUNK.value
    assert header.timestamp_us == timestamp_us
    assert sent[0][9:] == payload

    assert tracker.calls == [(timestamp_us + duration_us, byte_count)]


def test_player_role_send_stream_start_drops_without_transport() -> None:
    """send_stream_start() is a no-op when no transport attached."""
    client = MagicMock()
    client.send_message = MagicMock()

    role = PlayerRole(_client=client)
    role._has_transport = False  # noqa: SLF001

    audio_format = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    role.send_stream_start(audio_format, AudioCodec.PCM)

    client.send_message.assert_not_called()


def test_player_role_clear_stream_drops_without_transport() -> None:
    """clear_stream() is a no-op for JSON message when no transport attached."""
    client = MagicMock()
    client.send_message = MagicMock()
    client.buffer_tracker = None

    role = PlayerRole(_client=client)
    role._has_transport = False  # noqa: SLF001

    role.clear_stream()

    client.send_message.assert_not_called()


def test_player_role_end_stream_drops_without_transport() -> None:
    """end_stream() is a no-op for JSON message when no transport attached."""
    client = MagicMock()
    client.send_message = MagicMock()
    client.buffer_tracker = None

    role = PlayerRole(_client=client)
    role._has_transport = False  # noqa: SLF001

    role.end_stream()

    client.send_message.assert_not_called()


# --- Tests for hook-based streaming methods ---


def test_player_role_on_stream_start_sends_message() -> None:
    """on_stream_start() sends stream/start with format info."""
    client = MagicMock()
    client.send_message = MagicMock()

    role = PlayerRole(_client=client)
    role._has_transport = True  # noqa: SLF001
    role._audio_requirements = AudioRequirements(  # noqa: SLF001
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(),
    )

    role.on_stream_start()

    client.send_message.assert_called_once()
    msg = client.send_message.call_args.args[0]
    assert isinstance(msg, StreamStartMessage)
    assert msg.payload.player.sample_rate == 48000
    assert msg.payload.player.codec == AudioCodec.PCM


def test_player_role_on_audio_chunk_returns_true() -> None:
    """on_audio_chunk() returns True when chunk sent successfully."""
    client = MagicMock()
    client.queue_high_water.return_value = False
    client.try_send_binary.return_value = True

    role = PlayerRole(_client=client)
    role._has_transport = True  # noqa: SLF001

    chunk = AudioChunk(data=b"audio", timestamp_us=1000, duration_us=25000, byte_count=5)
    result = role.on_audio_chunk(chunk)

    assert result is True
    client.try_send_binary.assert_called_once()


def test_player_role_on_audio_chunk_returns_false_on_backpressure() -> None:
    """on_audio_chunk() returns False when queue is full."""
    client = MagicMock()
    client.queue_high_water.return_value = True  # Queue full

    role = PlayerRole(_client=client)
    role._has_transport = True  # noqa: SLF001

    chunk = AudioChunk(data=b"audio", timestamp_us=1000, duration_us=25000, byte_count=5)
    result = role.on_audio_chunk(chunk)

    assert result is False
