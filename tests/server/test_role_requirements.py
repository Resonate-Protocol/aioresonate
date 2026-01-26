"""Tests for Role base class and requirement declarations."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from aiosendspin.models import AudioCodec
from aiosendspin.server.audio import AudioFormat
from aiosendspin.server.roles import AudioRequirements, PlayerRole, Role, StreamRequirements


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

    def test_audio_requirements_with_target_format(self) -> None:
        """AudioRequirements captures target format."""
        fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2, codec=AudioCodec.OPUS)
        req = AudioRequirements(target_format=fmt)
        assert req.target_format == fmt

    def test_audio_requirements_with_channel_id(self) -> None:
        """AudioRequirements can specify a channel."""
        channel = UUID("12345678-1234-5678-1234-567812345678")
        fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
        req = AudioRequirements(target_format=fmt, channel_id=channel)
        assert req.channel_id == channel

    def test_audio_requirements_channel_defaults_to_none(self) -> None:
        """AudioRequirements channel_id defaults to None (main channel)."""
        fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
        req = AudioRequirements(target_format=fmt)
        assert req.channel_id is None

    def test_audio_requirements_is_frozen(self) -> None:
        """AudioRequirements is immutable."""
        fmt = AudioFormat(sample_rate=48000, bit_depth=16, channels=2)
        req = AudioRequirements(target_format=fmt)
        with pytest.raises(AttributeError):
            req.target_format = fmt  # type: ignore[misc]


class TestPlayerRoleRequirements:
    """Tests for PlayerRole requirement declarations."""

    def test_player_role_declares_stream_requirements(self) -> None:
        """PlayerRole returns StreamRequirements."""
        client = MagicMock()
        role = PlayerRole(_client=client)
        req = role.get_stream_requirements()
        assert req is not None
        assert isinstance(req, StreamRequirements)

    def test_player_role_family_is_player(self) -> None:
        """PlayerRole.role_family is 'player'."""
        client = MagicMock()
        role = PlayerRole(_client=client)
        assert role.role_family == "player"


class TestRoleBaseClass:
    """Tests for Role base class capabilities."""

    def test_role_family_is_abstract(self) -> None:
        """Role requires role_family property."""

        class IncompleteRole(Role):
            pass

        with pytest.raises(TypeError, match="role_family"):
            IncompleteRole()  # type: ignore[abstract]

    def test_get_stream_requirements_defaults_to_none(self) -> None:
        """Roles that don't stream return None from get_stream_requirements()."""

        @dataclass
        class NonStreamingRole(Role):
            role_family: str = "test"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        role = NonStreamingRole()
        assert role.get_stream_requirements() is None

    def test_get_audio_requirements_defaults_to_none(self) -> None:
        """Roles that don't need audio return None from get_audio_requirements()."""

        @dataclass
        class NonAudioRole(Role):
            role_family: str = "test"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        role = NonAudioRole()
        assert role.get_audio_requirements() is None

    def test_on_audio_chunk_is_noop_by_default(self) -> None:
        """Roles that don't override on_audio_chunk() don't crash."""

        @dataclass
        class NonAudioRole(Role):
            role_family: str = "test"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        role = NonAudioRole()
        # Should not raise
        role.on_audio_chunk(b"audio_data", 123_000)

    def test_on_transport_attach_sets_has_transport(self) -> None:
        """on_transport_attach() sets _has_transport to True."""

        @dataclass
        class TestRole(Role):
            role_family: str = "test"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        role = TestRole()
        assert not role._has_transport  # noqa: SLF001
        role.on_transport_attach()
        assert role._has_transport  # noqa: SLF001

    def test_on_transport_detach_clears_has_transport(self) -> None:
        """on_transport_detach() sets _has_transport to False."""

        @dataclass
        class TestRole(Role):
            role_family: str = "test"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        role = TestRole()
        role._has_transport = True  # noqa: SLF001
        role.on_transport_detach()
        assert not role._has_transport  # noqa: SLF001

    def test_send_message_drops_silently_without_transport(self) -> None:
        """send_message() is a no-op when no transport attached."""

        @dataclass
        class TestRole(Role):
            _client: MagicMock
            role_family: str = "test"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        mock_client = MagicMock()
        role = TestRole(_client=mock_client)
        role._has_transport = False  # noqa: SLF001

        role.send_message({"type": "test"})
        mock_client.send_message.assert_not_called()

    def test_send_message_forwards_to_client_with_transport(self) -> None:
        """send_message() forwards to client when transport attached."""

        @dataclass
        class TestRole(Role):
            _client: MagicMock
            role_family: str = "test"

            def on_connect(self) -> None:
                pass

            def on_disconnect(self) -> None:
                pass

        mock_client = MagicMock()
        role = TestRole(_client=mock_client)
        role._has_transport = True  # noqa: SLF001

        msg = {"type": "test"}
        role.send_message(msg)
        mock_client.send_message.assert_called_once_with(msg)
