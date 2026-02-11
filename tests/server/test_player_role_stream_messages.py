"""Tests for PlayerV1Role stream lifecycle message payloads."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.models import AudioCodec, unpack_binary_header
from aiosendspin.models.core import StreamClearMessage, StreamEndMessage, StreamStartMessage
from aiosendspin.models.types import BinaryMessageType
from aiosendspin.server.roles import AudioChunk, AudioRequirements, PlayerV1Role
from aiosendspin.server.roles.player.audio_transformers import PcmPassthrough


def test_player_role_on_stream_clear_uses_role_family() -> None:
    """PlayerV1Role.on_stream_clear() sends stream/clear with unversioned role family."""
    client = MagicMock()
    client.send_role_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._buffer_tracker = None  # noqa: SLF001
    role.on_stream_clear()

    _role, msg = client.send_role_message.call_args.args
    assert isinstance(msg, StreamClearMessage)
    assert msg.payload.roles == ["player"]


def test_player_role_on_stream_end_uses_role_family() -> None:
    """PlayerV1Role.on_stream_end() omits roles (end all streams)."""
    client = MagicMock()
    client.send_role_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._buffer_tracker = None  # noqa: SLF001
    role.on_stream_end()

    _role, msg = client.send_role_message.call_args.args
    assert isinstance(msg, StreamEndMessage)
    assert msg.payload.roles is None


def test_player_role_on_audio_chunk_packs_header_and_tracks_duration() -> None:
    """on_audio_chunk uses role-controlled header packing and accurate duration tracking."""

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

    def _send_binary(
        data: bytes,
        *,
        role_family: str,  # noqa: ARG001
        timestamp_us: int,  # noqa: ARG001
        message_type: int,  # noqa: ARG001
        buffer_end_time_us: int | None = None,
        buffer_byte_count: int | None = None,
        duration_us: int | None = None,  # noqa: ARG001
    ) -> bool:
        sent.append(data)
        if buffer_end_time_us is not None and buffer_byte_count is not None:
            tracker.register(buffer_end_time_us, buffer_byte_count)
        return True

    client.send_binary = MagicMock(side_effect=_send_binary)
    client.send_role_message = MagicMock()

    role = PlayerV1Role(client=client)
    payload = b"\x01\x02\x03"
    timestamp_us = 123_000
    duration_us = 40_000
    byte_count = len(payload)

    chunk = AudioChunk(
        data=payload, timestamp_us=timestamp_us, duration_us=duration_us, byte_count=byte_count
    )
    assert role.on_audio_chunk(chunk) is None
    assert sent, "Expected a binary send"

    header = unpack_binary_header(sent[0])
    assert header.message_type == BinaryMessageType.AUDIO_CHUNK.value
    assert header.timestamp_us == timestamp_us
    assert sent[0][9:] == payload

    assert tracker.calls == [(timestamp_us + duration_us, byte_count)]


def test_player_role_on_stream_start_drops_without_transport() -> None:
    """on_stream_start() is a no-op when no transport attached."""
    client = MagicMock()
    client.send_role_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = None  # noqa: SLF001
    role._audio_requirements = AudioRequirements(  # noqa: SLF001
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
    )

    role.on_stream_start()

    client.send_role_message.assert_not_called()


def test_player_role_on_stream_clear_drops_without_transport() -> None:
    """on_stream_clear() is a no-op for JSON message when no transport attached."""
    client = MagicMock()
    client.send_role_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = None  # noqa: SLF001

    role.on_stream_clear()

    client.send_role_message.assert_not_called()


def test_player_role_on_stream_end_drops_without_transport() -> None:
    """on_stream_end() is a no-op for JSON message when no transport attached."""
    client = MagicMock()
    client.send_role_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = None  # noqa: SLF001

    role.on_stream_end()

    client.send_role_message.assert_not_called()


# --- Tests for hook-based streaming methods ---


def test_player_role_on_stream_start_sets_pending_flag() -> None:
    """on_stream_start() sets pending flag, message sent on first chunk."""
    client = MagicMock()
    client.send_role_message = MagicMock()
    client.send_binary = MagicMock(return_value=True)

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._audio_requirements = AudioRequirements(  # noqa: SLF001
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
    )

    role.on_stream_start()

    # Message is deferred until first chunk
    client.send_role_message.assert_not_called()
    assert role._pending_stream_start is True  # noqa: SLF001

    # First chunk triggers the stream/start message
    chunk = AudioChunk(data=b"\x00" * 100, timestamp_us=0, duration_us=25000, byte_count=100)
    role.on_audio_chunk(chunk)

    client.send_role_message.assert_called_once()
    _role, msg = client.send_role_message.call_args.args
    assert isinstance(msg, StreamStartMessage)
    assert msg.payload.player.sample_rate == 48000
    assert msg.payload.player.codec == AudioCodec.PCM
    assert role._pending_stream_start is False  # noqa: SLF001


def test_player_role_on_audio_chunk_sends_binary() -> None:
    """on_audio_chunk() sends a chunk when connection is present."""
    client = MagicMock()
    client.send_binary.return_value = True

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001

    chunk = AudioChunk(data=b"audio", timestamp_us=1000, duration_us=25000, byte_count=5)
    result = role.on_audio_chunk(chunk)

    assert result is None
    client.send_binary.assert_called_once()
