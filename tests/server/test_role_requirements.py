"""Tests for Role base class and requirement declarations."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID

import pytest

from aiosendspin.server.roles import (
    AudioChunk,
    AudioRequirements,
    PlayerV1Role,
    Role,
    StreamRequirements,
)


class TestStreamRequirements:
    """Tests for StreamRequirements dataclass."""

    def test_stream_requirements_creates_with_defaults(self) -> None:
        """StreamRequirements can be instantiated with defaults."""
        req = StreamRequirements()
        assert req is not None

    def test_stream_requirements_is_frozen(self) -> None:
        """StreamRequirements is immutable."""
        req = StreamRequirements()
        with pytest.raises(AttributeError):
            req.foo = "bar"  # type: ignore[attr-defined]


class TestAudioRequirements:
    """Tests for AudioRequirements dataclass."""

    def test_audio_requirements_with_pcm_format(self) -> None:
        """AudioRequirements captures PCM format without transformer."""
        req = AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
        )
        assert req.sample_rate == 48000
        assert req.bit_depth == 16
        assert req.channels == 2
        assert req.transformer is None
        assert req.channel_id is None
        assert req.frame_duration_us is None
        assert req.transform_options is None

    def test_audio_requirements_with_transformer(self) -> None:
        """AudioRequirements can include a transformer."""

        class DummyTransformer:
            def process(self, pcm: bytes, _timestamp_us: int, _duration_us: int) -> bytes:
                return pcm

            def get_header(self) -> bytes | None:
                return None

            def reset(self) -> None:
                pass

        transformer = DummyTransformer()
        req = AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            transformer=transformer,
        )
        assert req.transformer is transformer

    def test_audio_requirements_with_channel_id(self) -> None:
        """AudioRequirements can specify a channel."""
        channel = UUID("12345678-1234-5678-1234-567812345678")
        req = AudioRequirements(
            sample_rate=48000,
            bit_depth=16,
            channels=2,
            channel_id=channel,
        )
        assert req.channel_id == channel

    def test_audio_requirements_is_frozen(self) -> None:
        """AudioRequirements is immutable."""
        req = AudioRequirements(sample_rate=48000, bit_depth=16, channels=2)
        with pytest.raises(AttributeError):
            req.sample_rate = 44100  # type: ignore[misc]


class TestPlayerRoleRequirements:
    """Tests for PlayerV1Role requirement declarations."""

    def test_player_role_declares_stream_requirements(self) -> None:
        """PlayerV1Role returns StreamRequirements."""
        client = MagicMock()
        role = PlayerV1Role(client=client)
        req = role.get_stream_requirements()
        assert req is not None
        assert isinstance(req, StreamRequirements)

    def test_player_role_family_is_player(self) -> None:
        """PlayerV1Role.role_family is 'player'."""
        client = MagicMock()
        role = PlayerV1Role(client=client)
        assert role.role_family == "player"


class TestRoleBaseClass:
    """Tests for Role base class capabilities."""

    def test_role_family_is_abstract(self) -> None:
        """Role requires role_family property."""

        class IncompleteRole(Role):
            @property
            def role_id(self) -> str:
                return "test@v1"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        with pytest.raises(TypeError, match="role_family"):
            IncompleteRole()  # type: ignore[abstract]

    def test_get_stream_requirements_defaults_to_none(self) -> None:
        """Roles that don't stream return None from get_stream_requirements()."""

        class NonStreamingRole(Role):
            @property
            def role_id(self) -> str:
                return "test@v1"

            @property
            def role_family(self) -> str:
                return "test"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        role = NonStreamingRole()
        assert role.get_stream_requirements() is None

    def test_get_audio_requirements_defaults_to_none(self) -> None:
        """Roles that don't need audio return None from get_audio_requirements()."""

        class NonAudioRole(Role):
            @property
            def role_id(self) -> str:
                return "test@v1"

            @property
            def role_family(self) -> str:
                return "test"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        role = NonAudioRole()
        assert role.get_audio_requirements() is None

    def test_on_stream_start_is_noop_by_default(self) -> None:
        """Roles that don't override on_stream_start() don't crash."""

        class TestRole(Role):
            @property
            def role_id(self) -> str:
                return "test@v1"

            @property
            def role_family(self) -> str:
                return "test"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        role = TestRole()
        role.on_stream_start()  # Should not raise

    def test_on_stream_clear_is_noop_by_default(self) -> None:
        """Roles that don't override on_stream_clear() don't crash."""

        class TestRole(Role):
            @property
            def role_id(self) -> str:
                return "test@v1"

            @property
            def role_family(self) -> str:
                return "test"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        role = TestRole()
        role.on_stream_clear()  # Should not raise

    def test_on_stream_end_is_noop_by_default(self) -> None:
        """Roles that don't override on_stream_end() don't crash."""

        class TestRole(Role):
            @property
            def role_id(self) -> str:
                return "test@v1"

            @property
            def role_family(self) -> str:
                return "test"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        role = TestRole()
        role.on_stream_end()  # Should not raise

    def test_on_audio_chunk_is_noop_by_default(self) -> None:
        """Roles that don't override on_audio_chunk() are no-op."""

        class TestRole(Role):
            @property
            def role_id(self) -> str:
                return "test@v1"

            @property
            def role_family(self) -> str:
                return "test"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        role = TestRole()
        chunk = AudioChunk(data=b"test", timestamp_us=0, duration_us=1000, byte_count=4)
        result = role.on_audio_chunk(chunk)
        assert result is None

    def test_send_message_drops_silently_without_transport(self) -> None:
        """send_message() is a no-op when no transport attached."""

        class TestRole(Role):
            def __init__(self, client: MagicMock) -> None:
                self._client = client
                self._client.connection = None

            @property
            def role_id(self) -> str:
                return "test@v1"

            @property
            def role_family(self) -> str:
                return "test"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        mock_client = MagicMock()
        role = TestRole(mock_client)

        role.send_message({"type": "test"})
        mock_client.send_role_message.assert_not_called()

    def test_send_message_forwards_to_client_with_transport(self) -> None:
        """send_message() forwards to client when transport attached."""

        class TestRole(Role):
            def __init__(self, client: MagicMock) -> None:
                self._client = client
                self._client.connection = MagicMock()

            @property
            def role_id(self) -> str:
                return "test@v1"

            @property
            def role_family(self) -> str:
                return "test"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        mock_client = MagicMock()
        role = TestRole(mock_client)

        msg = {"type": "test"}
        role.send_message(msg)
        mock_client.send_role_message.assert_called_once_with("test", msg)


class TestAudioChunk:
    """Tests for AudioChunk dataclass."""

    def test_audio_chunk_has_required_fields(self) -> None:
        """AudioChunk captures all required metadata."""
        chunk = AudioChunk(
            data=b"\x00\x01\x02",
            timestamp_us=1_000_000,
            duration_us=25_000,
            byte_count=3,
        )
        assert chunk.data == b"\x00\x01\x02"
        assert chunk.timestamp_us == 1_000_000
        assert chunk.duration_us == 25_000
        assert chunk.byte_count == 3

    def test_audio_chunk_is_frozen(self) -> None:
        """AudioChunk is immutable."""
        chunk = AudioChunk(data=b"x", timestamp_us=0, duration_us=0, byte_count=1)
        with pytest.raises(AttributeError):
            chunk.data = b"y"  # type: ignore[misc]
