"""Tests for the simplified PlayerV1Role (v1) implementation."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from aiosendspin.models import AudioCodec, unpack_binary_header
from aiosendspin.models.core import StreamClearMessage, StreamEndMessage, StreamStartMessage
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import BinaryMessageType, PlayerCommand
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.roles import PlayerV1Role
from aiosendspin.server.roles.base import AudioChunk, AudioRequirements, StreamRequirements
from aiosendspin.server.roles.player.audio_transformers import FlacEncoder, PcmPassthrough

# --- Basic properties ---


def _make_client_stub() -> MagicMock:
    client = MagicMock()
    state_store: dict[str, object] = {}

    def get_or_create_role_state(family: str, cls: type[object]) -> object:
        state_store.setdefault(family, cls())
        return state_store[family]

    client.get_or_create_role_state.side_effect = get_or_create_role_state
    client.info = MagicMock()
    client.info.player_support = None
    client.group = MagicMock()
    client._server = MagicMock()  # noqa: SLF001
    client._logger = MagicMock()  # noqa: SLF001
    client.client_id = "test-client"
    client.connection = None
    client.send_role_message = MagicMock()
    return client


def test_player_role_has_role_id() -> None:
    """PlayerV1Role has role_id of 'player@v1'."""
    client = _make_client_stub()
    role = PlayerV1Role(client=client)
    assert role.role_id == "player@v1"


def test_player_role_has_role_family() -> None:
    """PlayerV1Role has role_family of 'player'."""
    client = _make_client_stub()
    role = PlayerV1Role(client=client)
    assert role.role_family == "player"


def test_player_role_has_preferred_format_property() -> None:
    """PlayerV1Role exposes preferred_format property."""
    client = _make_client_stub()
    audio_format = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
    role = PlayerV1Role(client=client, preferred_format=audio_format)
    assert role.preferred_format == audio_format


# --- StreamRequirements ---


def test_player_role_get_stream_requirements_returns_stream_requirements() -> None:
    """PlayerV1Role.get_stream_requirements() returns StreamRequirements."""
    client = _make_client_stub()
    role = PlayerV1Role(client=client)
    req = role.get_stream_requirements()
    assert isinstance(req, StreamRequirements)


# --- AudioRequirements ---


def test_player_role_get_audio_requirements_returns_stored_requirements() -> None:
    """PlayerV1Role.get_audio_requirements() returns stored requirements."""
    client = _make_client_stub()
    audio_req = AudioRequirements(sample_rate=48000, bit_depth=16, channels=2)
    role = PlayerV1Role(client=client, audio_requirements=audio_req)
    assert role.get_audio_requirements() is audio_req


def test_player_role_get_audio_requirements_returns_none_when_not_set() -> None:
    """PlayerV1Role.get_audio_requirements() returns None when not set."""
    client = _make_client_stub()
    role = PlayerV1Role(client=client)
    assert role.get_audio_requirements() is None


def test_player_role_get_audio_requirements_refreshes_when_channel_changes() -> None:
    """PlayerV1Role refreshes cached requirements if resolver channel changed."""
    client = _make_client_stub()
    cached_channel = uuid4()
    refreshed_channel = uuid4()
    client.group.get_channel_for_player.return_value = refreshed_channel

    audio_req = AudioRequirements(
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
        channel_id=cached_channel,
    )
    role = PlayerV1Role(client=client, audio_requirements=audio_req)
    refreshed_req = AudioRequirements(
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
        channel_id=refreshed_channel,
    )

    def _refresh(*, force: bool = False) -> None:
        assert force is True
        role._audio_requirements = refreshed_req  # noqa: SLF001

    role._ensure_audio_requirements = MagicMock(side_effect=_refresh)  # type: ignore[method-assign]  # noqa: SLF001

    req = role.get_audio_requirements()

    role._ensure_audio_requirements.assert_called_once()  # type: ignore[attr-defined]  # noqa: SLF001
    assert req is refreshed_req


# --- BinaryHandling ---


def test_player_role_get_binary_handling_returns_handling_for_audio_chunk() -> None:
    """PlayerV1Role returns BinaryHandling for AUDIO_CHUNK message type."""
    client = _make_client_stub()
    role = PlayerV1Role(client=client)

    handling = role.get_binary_handling(BinaryMessageType.AUDIO_CHUNK.value)

    assert handling is not None
    assert handling.drop_late is True
    assert handling.grace_period_us == 2_000_000
    assert handling.buffer_track is True


def test_player_role_get_binary_handling_returns_none_for_unknown_type() -> None:
    """PlayerV1Role returns None for unknown message types."""
    client = _make_client_stub()
    role = PlayerV1Role(client=client)

    handling = role.get_binary_handling(999)  # Unknown type

    assert handling is None


# --- on_connect / on_disconnect ---


def test_player_role_on_connect_resets_stream_state() -> None:
    """on_connect() resets stream started flag."""
    client = _make_client_stub()
    role = PlayerV1Role(client=client)
    role._stream_started = True  # noqa: SLF001
    role.on_connect()
    assert role._stream_started is False  # noqa: SLF001


def test_player_role_on_disconnect_resets_stream_state() -> None:
    """on_disconnect() resets stream started flag."""
    client = _make_client_stub()
    role = PlayerV1Role(client=client)
    role._stream_started = True  # noqa: SLF001
    role.on_disconnect()
    assert role._stream_started is False  # noqa: SLF001


# --- on_stream_start ---


def test_player_role_on_stream_start_sets_pending_flag() -> None:
    """on_stream_start() sets _pending_stream_start to True (deferred send)."""
    client = _make_client_stub()
    client.send_message = MagicMock()

    audio_req = AudioRequirements(
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
    )
    role = PlayerV1Role(client=client, audio_requirements=audio_req)
    role._client.connection = MagicMock()  # noqa: SLF001

    role.on_stream_start()

    # Message is deferred until first audio chunk
    client.send_role_message.assert_not_called()
    assert role._pending_stream_start is True  # noqa: SLF001


def test_player_role_on_stream_start_resets_binary_timing() -> None:
    """on_stream_start() should reset stale timing so late-drop grace reapplies."""
    client = _make_client_stub()

    audio_req = AudioRequirements(
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
    )
    role = PlayerV1Role(client=client, audio_requirements=audio_req)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._stream_start_time_us = 12345  # noqa: SLF001
    role._late_skips_since_log = 4  # noqa: SLF001
    role._last_late_log_s = 42.0  # noqa: SLF001

    role.on_stream_start()

    assert role._stream_start_time_us is None  # noqa: SLF001
    assert role._late_skips_since_log == 0  # noqa: SLF001
    assert role._last_late_log_s == 0.0  # noqa: SLF001
    assert role._pending_stream_start is True  # noqa: SLF001


def test_player_role_on_audio_chunk_sends_deferred_stream_start_with_pcm() -> None:
    """on_audio_chunk() sends deferred stream/start with PCM codec."""
    client = _make_client_stub()
    client.send_message = MagicMock()
    client.send_binary = MagicMock(return_value=True)

    audio_req = AudioRequirements(
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
    )
    role = PlayerV1Role(client=client, audio_requirements=audio_req)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._pending_stream_start = True  # noqa: SLF001

    chunk = AudioChunk(data=b"\x00" * 100, timestamp_us=0, duration_us=25000, byte_count=100)
    role.on_audio_chunk(chunk)

    client.send_role_message.assert_called_once()
    _role, msg = client.send_role_message.call_args.args
    assert isinstance(msg, StreamStartMessage)
    assert msg.payload.player.sample_rate == 48000
    assert msg.payload.player.bit_depth == 16
    assert msg.payload.player.channels == 2
    assert msg.payload.player.codec == AudioCodec.PCM
    assert msg.payload.player.codec_header is None
    assert role._pending_stream_start is False  # noqa: SLF001
    assert role._stream_started is True  # noqa: SLF001


def test_player_role_on_audio_chunk_sends_deferred_stream_start_with_flac() -> None:
    """on_audio_chunk() sends deferred stream/start with FLAC codec and header."""
    client = _make_client_stub()
    client.send_message = MagicMock()
    client.send_binary = MagicMock(return_value=True)

    encoder = FlacEncoder(sample_rate=48000, bit_depth=16, channels=2)
    # Force encoder to initialize so we get a header
    encoder._ensure_initialized()  # noqa: SLF001

    audio_req = AudioRequirements(sample_rate=48000, bit_depth=16, channels=2, transformer=encoder)
    role = PlayerV1Role(client=client, audio_requirements=audio_req)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._pending_stream_start = True  # noqa: SLF001

    chunk = AudioChunk(data=b"\x00" * 100, timestamp_us=0, duration_us=25000, byte_count=100)
    role.on_audio_chunk(chunk)

    client.send_role_message.assert_called_once()
    _role, msg = client.send_role_message.call_args.args
    assert isinstance(msg, StreamStartMessage)
    assert msg.payload.player.codec == AudioCodec.FLAC
    assert msg.payload.player.codec_header is not None  # FLAC has header
    assert role._pending_stream_start is False  # noqa: SLF001
    assert role._stream_started is True  # noqa: SLF001


def test_player_role_on_stream_start_sets_stream_started_flag_on_first_chunk() -> None:
    """_stream_started is set to True when stream/start is sent on first chunk."""
    client = _make_client_stub()
    client.send_message = MagicMock()
    client.send_binary = MagicMock(return_value=True)

    audio_req = AudioRequirements(
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
    )
    role = PlayerV1Role(client=client, audio_requirements=audio_req)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._stream_started = False  # noqa: SLF001

    role.on_stream_start()
    assert role._stream_started is False  # noqa: SLF001 - not yet

    chunk = AudioChunk(data=b"\x00" * 100, timestamp_us=0, duration_us=25000, byte_count=100)
    role.on_audio_chunk(chunk)

    assert role._stream_started is True  # noqa: SLF001


def test_player_role_on_stream_start_noop_without_audio_requirements() -> None:
    """on_stream_start() is no-op when no audio requirements."""
    client = _make_client_stub()
    client.send_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001

    role.on_stream_start()

    client.send_role_message.assert_not_called()


def test_player_role_on_stream_start_noop_without_transport() -> None:
    """on_stream_start() is no-op when no transport."""
    client = _make_client_stub()
    client.send_message = MagicMock()

    audio_req = AudioRequirements(
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=PcmPassthrough(sample_rate=48000, bit_depth=16, channels=2),
    )
    role = PlayerV1Role(client=client, audio_requirements=audio_req)
    role._client.connection = None  # noqa: SLF001

    role.on_stream_start()

    client.send_role_message.assert_not_called()


# --- on_audio_chunk ---


def test_player_role_on_audio_chunk_sends_on_success() -> None:
    """on_audio_chunk() sends chunk when connected."""
    client = MagicMock()
    client.send_binary.return_value = True

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._stream_started = True  # noqa: SLF001

    chunk = AudioChunk(data=b"audio", timestamp_us=1000, duration_us=25000, byte_count=5)
    result = role.on_audio_chunk(chunk)

    assert result is None
    client.send_binary.assert_called_once()


def test_player_role_on_audio_chunk_packs_binary_header() -> None:
    """on_audio_chunk() packs binary header with timestamp."""
    sent_data: list[bytes] = []
    client = MagicMock()

    def capture_send(data: bytes, **kwargs: object) -> bool:  # noqa: ARG001
        sent_data.append(data)
        return True

    client.send_binary.side_effect = capture_send

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._stream_started = True  # noqa: SLF001

    chunk = AudioChunk(data=b"\x01\x02\x03", timestamp_us=123_456, duration_us=25000, byte_count=3)
    role.on_audio_chunk(chunk)

    assert len(sent_data) == 1
    header = unpack_binary_header(sent_data[0])
    assert header.message_type == BinaryMessageType.AUDIO_CHUNK.value
    assert header.timestamp_us == 123_456
    assert sent_data[0][9:] == b"\x01\x02\x03"


def test_player_role_on_audio_chunk_passes_buffer_metadata() -> None:
    """on_audio_chunk() passes buffer tracking metadata to send_binary."""
    client = MagicMock()
    client.send_binary.return_value = True

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._stream_started = True  # noqa: SLF001

    chunk = AudioChunk(data=b"audio", timestamp_us=1000, duration_us=25000, byte_count=100)
    role.on_audio_chunk(chunk)

    call_kwargs = client.send_binary.call_args.kwargs
    assert call_kwargs["buffer_end_time_us"] == 1000 + 25000
    assert call_kwargs["buffer_byte_count"] == 100
    assert call_kwargs["duration_us"] == 25000


def test_player_role_on_audio_chunk_ignores_send_return_value() -> None:
    """on_audio_chunk() is fire-and-forget and ignores send return values."""
    client = MagicMock()
    client.send_binary.return_value = None

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._stream_started = True  # noqa: SLF001

    chunk = AudioChunk(data=b"audio", timestamp_us=1000, duration_us=25000, byte_count=5)
    result = role.on_audio_chunk(chunk)

    assert result is None


def test_player_role_on_audio_chunk_drops_when_stream_not_started() -> None:
    """on_audio_chunk() drops stale chunks after lifecycle reset."""
    client = MagicMock()
    client.send_binary.return_value = True
    client.send_role_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._stream_started = False  # noqa: SLF001
    role._pending_stream_start = False  # noqa: SLF001

    chunk = AudioChunk(data=b"audio", timestamp_us=1000, duration_us=25000, byte_count=5)
    role.on_audio_chunk(chunk)

    client.send_role_message.assert_not_called()
    client.send_binary.assert_not_called()


# --- on_stream_clear ---


def test_player_role_on_stream_clear_sends_message() -> None:
    """on_stream_clear() sends stream/clear message."""
    client = MagicMock()
    client.send_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._buffer_tracker = None  # noqa: SLF001

    role.on_stream_clear()

    client.send_role_message.assert_called_once()
    _role, msg = client.send_role_message.call_args.args
    assert isinstance(msg, StreamClearMessage)
    assert msg.payload.roles == ["player"]


def test_player_role_on_stream_clear_resets_stream_started() -> None:
    """on_stream_clear() resets _stream_started flag."""
    client = MagicMock()
    client.send_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._stream_started = True  # noqa: SLF001
    role._buffer_tracker = None  # noqa: SLF001

    role.on_stream_clear()

    assert role._stream_started is False  # noqa: SLF001


def test_player_role_on_stream_clear_resets_buffer_tracker() -> None:
    """on_stream_clear() resets buffer tracker if present."""
    client = MagicMock()
    client.send_message = MagicMock()
    buffer_tracker = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._buffer_tracker = buffer_tracker  # noqa: SLF001

    role.on_stream_clear()

    buffer_tracker.reset.assert_called_once()


def test_player_role_on_stream_clear_noop_without_transport() -> None:
    """on_stream_clear() is no-op when no transport."""
    client = MagicMock()
    client.send_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = None  # noqa: SLF001

    role.on_stream_clear()

    client.send_role_message.assert_not_called()


# --- on_stream_end ---


def test_player_role_on_stream_end_sends_message() -> None:
    """on_stream_end() sends stream/end message."""
    client = MagicMock()
    client.send_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._buffer_tracker = None  # noqa: SLF001

    role.on_stream_end()

    client.send_role_message.assert_called_once()
    _role, msg = client.send_role_message.call_args.args
    assert isinstance(msg, StreamEndMessage)
    # stream/end omits roles (ends all streams)
    assert msg.payload.roles is None


def test_player_role_on_stream_end_resets_stream_started() -> None:
    """on_stream_end() resets _stream_started flag."""
    client = MagicMock()
    client.send_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._stream_started = True  # noqa: SLF001
    role._buffer_tracker = None  # noqa: SLF001

    role.on_stream_end()

    assert role._stream_started is False  # noqa: SLF001


def test_player_role_on_stream_end_resets_buffer_tracker() -> None:
    """on_stream_end() resets buffer tracker if present."""
    client = MagicMock()
    client.send_message = MagicMock()
    buffer_tracker = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._buffer_tracker = buffer_tracker  # noqa: SLF001

    role.on_stream_end()

    buffer_tracker.reset.assert_called_once()


def test_player_role_on_stream_end_noop_without_transport() -> None:
    """on_stream_end() is no-op when no transport."""
    client = MagicMock()
    client.send_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = None  # noqa: SLF001

    role.on_stream_end()

    client.send_role_message.assert_not_called()


def test_player_role_drops_audio_chunk_after_stream_end() -> None:
    """Chunks arriving after stream/end must be suppressed."""
    client = MagicMock()
    client.send_binary.return_value = True
    client.send_message = MagicMock()

    role = PlayerV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role._stream_started = True  # noqa: SLF001

    role.on_stream_end()
    role.on_audio_chunk(
        AudioChunk(data=b"audio", timestamp_us=1000, duration_us=25000, byte_count=5)
    )

    # only stream/end should be sent
    client.send_role_message.assert_called_once()
    client.send_binary.assert_not_called()


def test_player_role_on_group_changed_resets_buffer_and_timing() -> None:
    """on_group_changed() should reset stale buffer/timing state from old group."""
    client = _make_client_stub()
    role = PlayerV1Role(client=client)

    state = role._state()  # noqa: SLF001
    buffer_tracker = MagicMock()
    state.buffer_tracker = buffer_tracker
    role._stream_started = True  # noqa: SLF001
    role._pending_stream_start = True  # noqa: SLF001
    role._stream_start_time_us = 12345  # noqa: SLF001

    role.on_group_changed(object())

    buffer_tracker.reset.assert_called_once()
    assert role._stream_started is False  # noqa: SLF001
    assert role._pending_stream_start is False  # noqa: SLF001
    assert role._stream_start_time_us is None  # noqa: SLF001


# --- _ensure_preferred_format ---


def _make_player_support(*formats: SupportedAudioFormat) -> ClientHelloPlayerSupport:
    return ClientHelloPlayerSupport(
        supported_formats=list(formats),
        buffer_capacity=65536,
        supported_commands=[PlayerCommand.VOLUME],
    )


def test_ensure_preferred_format_sets_format_from_compatible_list() -> None:
    """_ensure_preferred_format() picks compatible[0] as the preferred format."""
    client = _make_client_stub()
    client.info.player_support = _make_player_support(
        SupportedAudioFormat(codec=AudioCodec.FLAC, channels=2, sample_rate=48000, bit_depth=16),
    )
    role = PlayerV1Role(client=client)

    role._ensure_preferred_format()  # noqa: SLF001

    assert role._preferred_format == AudioFormat(sample_rate=48000, bit_depth=16, channels=2)  # noqa: SLF001
    assert role._preferred_codec == AudioCodec.FLAC  # noqa: SLF001


def test_ensure_preferred_format_resets_to_new_priority_on_reconnect() -> None:
    """On reconnect, _ensure_preferred_format() always resets to the new first priority.

    This holds even when the previously stored format is still in the compatible list.
    """
    client = _make_client_stub()

    # First connect: FLAC is first priority
    client.info.player_support = _make_player_support(
        SupportedAudioFormat(codec=AudioCodec.FLAC, channels=2, sample_rate=48000, bit_depth=16),
        SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=96000, bit_depth=24),
    )
    role = PlayerV1Role(client=client)
    role._ensure_preferred_format()  # noqa: SLF001

    assert role._preferred_codec == AudioCodec.FLAC  # noqa: SLF001

    # Reconnect: client reorders — PCM 96k/24 is now first (e.g. user passed --audio-format)
    # FLAC is still in the list (compatible), but must no longer be preferred
    client.info.player_support = _make_player_support(
        SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=96000, bit_depth=24),
        SupportedAudioFormat(codec=AudioCodec.FLAC, channels=2, sample_rate=48000, bit_depth=16),
    )
    role._ensure_preferred_format()  # noqa: SLF001

    assert role._preferred_format == AudioFormat(sample_rate=96000, bit_depth=24, channels=2)  # noqa: SLF001
    assert role._preferred_codec == AudioCodec.PCM  # noqa: SLF001


def test_set_preferred_format_persists_across_reconnect() -> None:
    """set_preferred_format() remains active after reconnect until cleared."""
    client = _make_client_stub()
    client.info.player_support = _make_player_support(
        SupportedAudioFormat(codec=AudioCodec.FLAC, channels=2, sample_rate=48000, bit_depth=16),
        SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=96000, bit_depth=24),
    )
    role = PlayerV1Role(client=client)
    role._ensure_preferred_format()  # noqa: SLF001

    assert role.set_preferred_format(
        AudioFormat(sample_rate=96000, bit_depth=24, channels=2),
        AudioCodec.PCM,
    )

    # Reconnect with FLAC still first priority - sticky override should still win.
    role._ensure_preferred_format()  # noqa: SLF001
    assert role._preferred_format == AudioFormat(sample_rate=96000, bit_depth=24, channels=2)  # noqa: SLF001
    assert role._preferred_codec == AudioCodec.PCM  # noqa: SLF001


def test_set_preferred_format_codec_only_uses_first_matching_codec_format() -> None:
    """Codec-only override picks first compatible format for that codec."""
    client = _make_client_stub()
    client.info.player_support = _make_player_support(
        SupportedAudioFormat(codec=AudioCodec.FLAC, channels=2, sample_rate=44100, bit_depth=16),
        SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=96000, bit_depth=24),
        SupportedAudioFormat(codec=AudioCodec.FLAC, channels=2, sample_rate=48000, bit_depth=16),
    )
    role = PlayerV1Role(client=client)
    role._ensure_preferred_format()  # noqa: SLF001

    assert role.set_preferred_format(None, AudioCodec.FLAC)
    assert role._preferred_format == AudioFormat(sample_rate=44100, bit_depth=16, channels=2)  # noqa: SLF001
    assert role._preferred_codec == AudioCodec.FLAC  # noqa: SLF001


def test_set_preferred_format_clear_mid_stream_notifies_group() -> None:
    """Clearing override mid-stream should defer stream/start and invalidate caches."""
    client = _make_client_stub()
    client.group.has_active_stream = False
    client.info.player_support = _make_player_support(
        SupportedAudioFormat(codec=AudioCodec.FLAC, channels=2, sample_rate=48000, bit_depth=16),
        SupportedAudioFormat(codec=AudioCodec.PCM, channels=2, sample_rate=96000, bit_depth=24),
    )
    role = PlayerV1Role(client=client)
    role._ensure_preferred_format()  # noqa: SLF001
    assert role.set_preferred_format(
        AudioFormat(sample_rate=96000, bit_depth=24, channels=2),
        AudioCodec.PCM,
    )

    role._pending_stream_start = False  # noqa: SLF001
    client.group.on_role_format_changed.reset_mock()
    client.group.has_active_stream = True

    assert role.set_preferred_format(None)
    assert role._pending_stream_start is True  # noqa: SLF001
    client.group.on_role_format_changed.assert_called_once_with(role)


def test_ensure_preferred_format_noop_when_no_player_support() -> None:
    """_ensure_preferred_format() does nothing when player_support is None."""
    client = _make_client_stub()
    client.info.player_support = None
    role = PlayerV1Role(client=client)
    role._preferred_format = AudioFormat(sample_rate=44100, bit_depth=16, channels=2)  # noqa: SLF001

    role._ensure_preferred_format()  # noqa: SLF001

    # Unchanged — no support info yet
    assert role._preferred_format == AudioFormat(sample_rate=44100, bit_depth=16, channels=2)  # noqa: SLF001


def test_ensure_preferred_format_noop_when_no_compatible_formats() -> None:
    """_ensure_preferred_format() does nothing when all formats are unsupported by server."""
    client = _make_client_stub()
    # 999-channel format is not encodable by the server
    client.info.player_support = _make_player_support(
        SupportedAudioFormat(codec=AudioCodec.FLAC, channels=2, sample_rate=48000, bit_depth=8),
    )
    role = PlayerV1Role(client=client)
    role._preferred_format = AudioFormat(sample_rate=44100, bit_depth=16, channels=2)  # noqa: SLF001

    role._ensure_preferred_format()  # noqa: SLF001

    # Unchanged — no compatible formats found; warning logged
    assert role._preferred_format == AudioFormat(sample_rate=44100, bit_depth=16, channels=2)  # noqa: SLF001


def test_preferred_format_override_used_as_fallback_when_no_client_support() -> None:
    """preferred_format returns _preferred_format_override when _preferred_format is None."""
    client = _make_client_stub()
    client.info.player_support = None
    override = AudioFormat(sample_rate=44100, bit_depth=16, channels=2)
    role = PlayerV1Role(client=client, preferred_format=override)

    # _ensure_preferred_format() is a no-op without player_support, so override is used
    role._ensure_preferred_format()  # noqa: SLF001

    assert role.preferred_format is override


def test_preferred_format_override_superseded_after_client_hello() -> None:
    """Once client sends supported_formats, _ensure_preferred_format() uses the client list.

    The constructor override becomes irrelevant after the first client/hello.
    """
    client = _make_client_stub()
    override = AudioFormat(sample_rate=44100, bit_depth=16, channels=2)
    client.info.player_support = _make_player_support(
        SupportedAudioFormat(codec=AudioCodec.FLAC, channels=2, sample_rate=48000, bit_depth=16),
    )
    role = PlayerV1Role(client=client, preferred_format=override)

    role._ensure_preferred_format()  # noqa: SLF001

    # _preferred_format now set from client hello, so override is shadowed
    assert role._preferred_format == AudioFormat(sample_rate=48000, bit_depth=16, channels=2)  # noqa: SLF001
    assert role.preferred_format == AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
