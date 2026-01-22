"""Tests for PlayerRole stream lifecycle message payloads."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.models import unpack_binary_header
from aiosendspin.models.core import StreamClearMessage, StreamEndMessage
from aiosendspin.models.types import BinaryMessageType
from aiosendspin.server.roles import PlayerRole


def test_player_role_stream_clear_uses_role_family() -> None:
    """PlayerRole.stream/clear uses unversioned role family."""
    client = MagicMock()
    client.send_message = MagicMock()
    client.buffer_tracker = None

    role = PlayerRole(_client=client)
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
