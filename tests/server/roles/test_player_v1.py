"""Tests for the simplified PlayerRole (v1) implementation."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.models import AudioCodec, unpack_binary_header
from aiosendspin.models.core import StreamClearMessage, StreamEndMessage, StreamStartMessage
from aiosendspin.models.types import BinaryMessageType
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.roles.base import AudioChunk, AudioRequirements, StreamRequirements
from aiosendspin.server.roles.player_v1 import PlayerRole
from aiosendspin.server.transformers import FlacEncoder, PcmPassthrough

# --- Basic properties ---


def test_player_role_has_role_id() -> None:
    """PlayerRole has role_id of 'player@v1'."""
    client = MagicMock()
    role = PlayerRole(client=client)
    assert role.role_id == "player@v1"


def test_player_role_has_role_family() -> None:
    """PlayerRole has role_family of 'player'."""
    client = MagicMock()
    role = PlayerRole(client=client)
    assert role.role_family == "player"


def test_player_role_has_preferred_format_property() -> None:
    """PlayerRole exposes preferred_format property."""
    client = MagicMock()
    audio_format = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    role = PlayerRole(client=client, preferred_format=audio_format)
    assert role.preferred_format == audio_format


# --- StreamRequirements ---


def test_player_role_get_stream_requirements_returns_stream_requirements() -> None:
    """PlayerRole.get_stream_requirements() returns StreamRequirements."""
    client = MagicMock()
    role = PlayerRole(client=client)
    req = role.get_stream_requirements()
    assert isinstance(req, StreamRequirements)


# --- AudioRequirements ---


def test_player_role_get_audio_requirements_returns_stored_requirements() -> None:
    """PlayerRole.get_audio_requirements() returns stored requirements."""
    client = MagicMock()
    audio_req = AudioRequirements(sample_rate=48000, bit_depth=16, channels=2)
    role = PlayerRole(client=client, audio_requirements=audio_req)
    assert role.get_audio_requirements() is audio_req


def test_player_role_get_audio_requirements_returns_none_when_not_set() -> None:
    """PlayerRole.get_audio_requirements() returns None when not set."""
    client = MagicMock()
    role = PlayerRole(client=client)
    assert role.get_audio_requirements() is None


# --- BinaryHandling ---


def test_player_role_get_binary_handling_returns_handling_for_audio_chunk() -> None:
    """PlayerRole returns BinaryHandling for AUDIO_CHUNK message type."""
    client = MagicMock()
    role = PlayerRole(client=client)

    handling = role.get_binary_handling(BinaryMessageType.AUDIO_CHUNK.value)

    assert handling is not None
    assert handling.drop_late is True
    assert handling.grace_period_us == 2_000_000
    assert handling.rate_limit is True
    assert handling.rate_limit_factor == 2.0
    assert handling.buffer_track is True


def test_player_role_get_binary_handling_returns_none_for_unknown_type() -> None:
    """PlayerRole returns None for unknown message types."""
    client = MagicMock()
    role = PlayerRole(client=client)

    handling = role.get_binary_handling(999)  # Unknown type

    assert handling is None


# --- on_connect / on_disconnect ---


def test_player_role_on_connect_resets_stream_state() -> None:
    """on_connect() resets stream started flag."""
    client = MagicMock()
    role = PlayerRole(client=client)
    role._stream_started = True  # noqa: SLF001
    role.on_connect()
    assert role._stream_started is False  # noqa: SLF001


def test_player_role_on_disconnect_resets_stream_state() -> None:
    """on_disconnect() resets stream started flag."""
    client = MagicMock()
    role = PlayerRole(client=client)
    role._stream_started = True  # noqa: SLF001
    role.on_disconnect()
    assert role._stream_started is False  # noqa: SLF001


# --- on_stream_start ---


def test_player_role_on_stream_start_sends_message_with_pcm() -> None:
    """on_stream_start() sends stream/start with PCM codec when using PcmPassthrough."""
    client = MagicMock()
    client.send_message = MagicMock()

    audio_req = AudioRequirements(
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
    )
    role = PlayerRole(client=client, audio_requirements=audio_req)
    role._has_transport = True  # noqa: SLF001

    role.on_stream_start()

    client.send_message.assert_called_once()
    msg = client.send_message.call_args.args[0]
    assert isinstance(msg, StreamStartMessage)
    assert msg.payload.player.sample_rate == 48000
    assert msg.payload.player.bit_depth == 16
    assert msg.payload.player.channels == 2
    assert msg.payload.player.codec == AudioCodec.PCM
    assert msg.payload.player.codec_header is None


def test_player_role_on_stream_start_sends_message_with_flac() -> None:
    """on_stream_start() sends stream/start with FLAC codec when using FlacEncoder."""
    client = MagicMock()
    client.send_message = MagicMock()

    encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
    # Force encoder to initialize so we get a header
    encoder._ensure_initialized()  # noqa: SLF001

    audio_req = AudioRequirements(sample_rate=48000, bit_depth=16, channels=2, transformer=encoder)
    role = PlayerRole(client=client, audio_requirements=audio_req)
    role._has_transport = True  # noqa: SLF001

    role.on_stream_start()

    client.send_message.assert_called_once()
    msg = client.send_message.call_args.args[0]
    assert isinstance(msg, StreamStartMessage)
    assert msg.payload.player.codec == AudioCodec.FLAC
    assert msg.payload.player.codec_header is not None  # FLAC has header


def test_player_role_on_stream_start_sets_stream_started_flag() -> None:
    """on_stream_start() sets _stream_started to True."""
    client = MagicMock()
    client.send_message = MagicMock()

    audio_req = AudioRequirements(
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
    )
    role = PlayerRole(client=client, audio_requirements=audio_req)
    role._has_transport = True  # noqa: SLF001
    role._stream_started = False  # noqa: SLF001

    role.on_stream_start()

    assert role._stream_started is True  # noqa: SLF001


def test_player_role_on_stream_start_noop_without_audio_requirements() -> None:
    """on_stream_start() is no-op when no audio requirements."""
    client = MagicMock()
    client.send_message = MagicMock()

    role = PlayerRole(client=client)
    role._has_transport = True  # noqa: SLF001

    role.on_stream_start()

    client.send_message.assert_not_called()


def test_player_role_on_stream_start_noop_without_transport() -> None:
    """on_stream_start() is no-op when no transport."""
    client = MagicMock()
    client.send_message = MagicMock()

    audio_req = AudioRequirements(
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
    )
    role = PlayerRole(client=client, audio_requirements=audio_req)
    role._has_transport = False  # noqa: SLF001

    role.on_stream_start()

    client.send_message.assert_not_called()


# --- on_audio_chunk ---


def test_player_role_on_audio_chunk_returns_true_on_success() -> None:
    """on_audio_chunk() returns True when chunk sent successfully."""
    client = MagicMock()
    client.try_send_binary.return_value = True

    role = PlayerRole(client=client)
    role._has_transport = True  # noqa: SLF001

    chunk = AudioChunk(data=b"audio", timestamp_us=1000, duration_us=25000, byte_count=5)
    result = role.on_audio_chunk(chunk)

    assert result is True
    client.try_send_binary.assert_called_once()


def test_player_role_on_audio_chunk_packs_binary_header() -> None:
    """on_audio_chunk() packs binary header with timestamp."""
    sent_data: list[bytes] = []
    client = MagicMock()

    def capture_send(data: bytes, **kwargs: object) -> bool:  # noqa: ARG001
        sent_data.append(data)
        return True

    client.try_send_binary.side_effect = capture_send

    role = PlayerRole(client=client)
    role._has_transport = True  # noqa: SLF001

    chunk = AudioChunk(data=b"\x01\x02\x03", timestamp_us=123_456, duration_us=25000, byte_count=3)
    role.on_audio_chunk(chunk)

    assert len(sent_data) == 1
    header = unpack_binary_header(sent_data[0])
    assert header.message_type == BinaryMessageType.AUDIO_CHUNK.value
    assert header.timestamp_us == 123_456
    assert sent_data[0][9:] == b"\x01\x02\x03"


def test_player_role_on_audio_chunk_passes_buffer_metadata() -> None:
    """on_audio_chunk() passes buffer tracking metadata to try_send_binary."""
    client = MagicMock()
    client.try_send_binary.return_value = True

    role = PlayerRole(client=client)
    role._has_transport = True  # noqa: SLF001

    chunk = AudioChunk(data=b"audio", timestamp_us=1000, duration_us=25000, byte_count=100)
    role.on_audio_chunk(chunk)

    call_kwargs = client.try_send_binary.call_args.kwargs
    assert call_kwargs["buffer_end_time_us"] == 1000 + 25000
    assert call_kwargs["buffer_byte_count"] == 100
    assert call_kwargs["duration_us"] == 25000


def test_player_role_on_audio_chunk_returns_false_on_send_failure() -> None:
    """on_audio_chunk() returns False when try_send_binary fails."""
    client = MagicMock()
    client.try_send_binary.return_value = False  # Send failed

    role = PlayerRole(client=client)
    role._has_transport = True  # noqa: SLF001

    chunk = AudioChunk(data=b"audio", timestamp_us=1000, duration_us=25000, byte_count=5)
    result = role.on_audio_chunk(chunk)

    assert result is False


# --- on_stream_clear ---


def test_player_role_on_stream_clear_sends_message() -> None:
    """on_stream_clear() sends stream/clear message."""
    client = MagicMock()
    client.send_message = MagicMock()

    role = PlayerRole(client=client)
    role._has_transport = True  # noqa: SLF001
    role._buffer_tracker = None  # noqa: SLF001

    role.on_stream_clear()

    client.send_message.assert_called_once()
    msg = client.send_message.call_args.args[0]
    assert isinstance(msg, StreamClearMessage)
    assert msg.payload.roles == ["player"]


def test_player_role_on_stream_clear_resets_stream_started() -> None:
    """on_stream_clear() resets _stream_started flag."""
    client = MagicMock()
    client.send_message = MagicMock()

    role = PlayerRole(client=client)
    role._has_transport = True  # noqa: SLF001
    role._stream_started = True  # noqa: SLF001
    role._buffer_tracker = None  # noqa: SLF001

    role.on_stream_clear()

    assert role._stream_started is False  # noqa: SLF001


def test_player_role_on_stream_clear_resets_buffer_tracker() -> None:
    """on_stream_clear() resets buffer tracker if present."""
    client = MagicMock()
    client.send_message = MagicMock()
    buffer_tracker = MagicMock()

    role = PlayerRole(client=client)
    role._has_transport = True  # noqa: SLF001
    role._buffer_tracker = buffer_tracker  # noqa: SLF001

    role.on_stream_clear()

    buffer_tracker.reset.assert_called_once()


def test_player_role_on_stream_clear_noop_without_transport() -> None:
    """on_stream_clear() is no-op when no transport."""
    client = MagicMock()
    client.send_message = MagicMock()

    role = PlayerRole(client=client)
    role._has_transport = False  # noqa: SLF001

    role.on_stream_clear()

    client.send_message.assert_not_called()


# --- on_stream_end ---


def test_player_role_on_stream_end_sends_message() -> None:
    """on_stream_end() sends stream/end message."""
    client = MagicMock()
    client.send_message = MagicMock()

    role = PlayerRole(client=client)
    role._has_transport = True  # noqa: SLF001
    role._buffer_tracker = None  # noqa: SLF001

    role.on_stream_end()

    client.send_message.assert_called_once()
    msg = client.send_message.call_args.args[0]
    assert isinstance(msg, StreamEndMessage)
    # stream/end omits roles (ends all streams)
    assert msg.payload.roles is None


def test_player_role_on_stream_end_resets_stream_started() -> None:
    """on_stream_end() resets _stream_started flag."""
    client = MagicMock()
    client.send_message = MagicMock()

    role = PlayerRole(client=client)
    role._has_transport = True  # noqa: SLF001
    role._stream_started = True  # noqa: SLF001
    role._buffer_tracker = None  # noqa: SLF001

    role.on_stream_end()

    assert role._stream_started is False  # noqa: SLF001


def test_player_role_on_stream_end_resets_buffer_tracker() -> None:
    """on_stream_end() resets buffer tracker if present."""
    client = MagicMock()
    client.send_message = MagicMock()
    buffer_tracker = MagicMock()

    role = PlayerRole(client=client)
    role._has_transport = True  # noqa: SLF001
    role._buffer_tracker = buffer_tracker  # noqa: SLF001

    role.on_stream_end()

    buffer_tracker.reset.assert_called_once()


def test_player_role_on_stream_end_noop_without_transport() -> None:
    """on_stream_end() is no-op when no transport."""
    client = MagicMock()
    client.send_message = MagicMock()

    role = PlayerRole(client=client)
    role._has_transport = False  # noqa: SLF001

    role.on_stream_end()

    client.send_message.assert_not_called()
