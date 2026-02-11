"""Tests for base role classes and dataclasses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from aiosendspin.server.roles.base import (
    AudioChunk,
    AudioRequirements,
    BinaryHandling,
    Role,
    StreamRequirements,
)

if TYPE_CHECKING:
    from aiosendspin.models.types import ServerMessage


# --- BinaryHandling tests ---


def test_binary_handling_is_frozen_dataclass() -> None:
    """BinaryHandling should be an immutable dataclass."""
    handling = BinaryHandling()
    with pytest.raises(FrozenInstanceError):
        handling.drop_late = True  # type: ignore[misc]


def test_binary_handling_has_sensible_defaults() -> None:
    """BinaryHandling should have safe defaults (no dropping)."""
    handling = BinaryHandling()
    assert handling.drop_late is False
    assert handling.grace_period_us == 0
    assert handling.buffer_track is False


def test_binary_handling_stores_all_fields() -> None:
    """BinaryHandling should store all provided fields."""
    handling = BinaryHandling(
        drop_late=True,
        grace_period_us=2_000_000,
        buffer_track=True,
    )
    assert handling.drop_late is True
    assert handling.grace_period_us == 2_000_000
    assert handling.buffer_track is True


# --- StreamRequirements tests ---


def test_stream_requirements_is_frozen_dataclass() -> None:
    """StreamRequirements should be an immutable dataclass."""
    req = StreamRequirements()
    with pytest.raises(FrozenInstanceError):
        req.foo = "bar"  # type: ignore[attr-defined]


# --- AudioChunk tests ---


def test_audio_chunk_is_frozen_dataclass() -> None:
    """AudioChunk should be an immutable dataclass."""
    chunk = AudioChunk(data=b"audio", timestamp_us=1000, duration_us=25000, byte_count=5)
    with pytest.raises(FrozenInstanceError):
        chunk.data = b"new"  # type: ignore[misc]


def test_audio_chunk_stores_all_fields() -> None:
    """AudioChunk should store all provided fields."""
    chunk = AudioChunk(data=b"test", timestamp_us=5000, duration_us=10000, byte_count=4)
    assert chunk.data == b"test"
    assert chunk.timestamp_us == 5000
    assert chunk.duration_us == 10000
    assert chunk.byte_count == 4


# --- AudioRequirements tests ---


def test_audio_requirements_is_frozen_dataclass() -> None:
    """AudioRequirements should be an immutable dataclass."""
    req = AudioRequirements(sample_rate=48000, bit_depth=16, channels=2)
    with pytest.raises(FrozenInstanceError):
        req.sample_rate = 44100  # type: ignore[misc]


def test_audio_requirements_stores_all_fields() -> None:
    """AudioRequirements should store all provided fields."""
    transformer = MagicMock()
    channel_id = uuid4()
    req = AudioRequirements(
        sample_rate=48000,
        bit_depth=16,
        channels=2,
        transformer=transformer,
        channel_id=channel_id,
    )
    assert req.sample_rate == 48000
    assert req.bit_depth == 16
    assert req.channels == 2
    assert req.transformer is transformer
    assert req.channel_id == channel_id


def test_audio_requirements_defaults() -> None:
    """AudioRequirements should have sensible defaults."""
    req = AudioRequirements(sample_rate=48000, bit_depth=16, channels=2)
    assert req.transformer is None
    assert req.channel_id is None


# --- Role ABC tests ---


class ConcreteRole(Role):
    """Concrete implementation for testing the abstract Role class."""

    @property
    def role_id(self) -> str:
        """Return versioned role ID."""
        return "test@v1"

    @property
    def role_family(self) -> str:
        """Return role family name."""
        return "test"

    def on_connect(self) -> None:
        """Handle connection."""

    def on_disconnect(self) -> None:
        """Handle disconnection."""


def test_role_requires_role_id() -> None:
    """Role subclass must implement role_id property."""

    class MissingRoleId(Role):
        @property
        def role_family(self) -> str:
            return "test"

        def on_connect(self) -> None:
            pass

        def on_disconnect(self) -> None:
            pass

    with pytest.raises(TypeError, match="role_id"):
        MissingRoleId()  # type: ignore[abstract]


def test_role_requires_role_family() -> None:
    """Role subclass must implement role_family property."""

    class MissingRoleFamily(Role):
        @property
        def role_id(self) -> str:
            return "test@v1"

        def on_connect(self) -> None:
            pass

        def on_disconnect(self) -> None:
            pass

    with pytest.raises(TypeError, match="role_family"):
        MissingRoleFamily()  # type: ignore[abstract]


def test_role_requires_on_connect() -> None:
    """Role subclass must implement on_connect method."""

    class MissingOnConnect(Role):
        @property
        def role_id(self) -> str:
            return "test@v1"

        @property
        def role_family(self) -> str:
            return "test"

        def on_disconnect(self) -> None:
            pass

    with pytest.raises(TypeError, match="on_connect"):
        MissingOnConnect()  # type: ignore[abstract]


def test_role_requires_on_disconnect() -> None:
    """Role subclass must implement on_disconnect method."""

    class MissingOnDisconnect(Role):
        @property
        def role_id(self) -> str:
            return "test@v1"

        @property
        def role_family(self) -> str:
            return "test"

        def on_connect(self) -> None:
            pass

    with pytest.raises(TypeError, match="on_disconnect"):
        MissingOnDisconnect()  # type: ignore[abstract]


def test_role_get_stream_requirements_returns_none_by_default() -> None:
    """Role.get_stream_requirements() returns None by default."""
    client = MagicMock()
    role = ConcreteRole()
    role._client = client  # noqa: SLF001
    assert role.get_stream_requirements() is None


def test_role_get_audio_requirements_returns_none_by_default() -> None:
    """Role.get_audio_requirements() returns None by default."""
    client = MagicMock()
    role = ConcreteRole()
    role._client = client  # noqa: SLF001
    assert role.get_audio_requirements() is None


def test_role_get_binary_handling_returns_none_by_default() -> None:
    """Role.get_binary_handling() returns None by default."""
    client = MagicMock()
    role = ConcreteRole()
    role._client = client  # noqa: SLF001
    assert role.get_binary_handling(0) is None


def test_role_reset_binary_timing_clears_stream_start() -> None:
    """Role.reset_binary_timing() clears _stream_start_time_us."""
    client = MagicMock()
    role = ConcreteRole()
    role._client = client  # noqa: SLF001
    role._stream_start_time_us = 12345  # noqa: SLF001
    role.reset_binary_timing()
    assert role._stream_start_time_us is None  # noqa: SLF001


def test_role_send_message_drops_without_transport() -> None:
    """Role.send_message() silently drops messages when no transport attached."""
    client = MagicMock()
    role = ConcreteRole()
    role._client = client  # noqa: SLF001
    role._client.connection = None  # noqa: SLF001

    message: ServerMessage = MagicMock()
    role.send_message(message)

    client.send_role_message.assert_not_called()


def test_role_send_message_forwards_with_transport() -> None:
    """Role.send_message() forwards to client when transport attached."""
    client = MagicMock()
    role = ConcreteRole()
    role._client = client  # noqa: SLF001
    role._client.connection = MagicMock()  # noqa: SLF001

    message: ServerMessage = MagicMock()
    role.send_message(message)

    client.send_role_message.assert_called_once_with(role.role_family, message)


def test_role_on_stream_start_is_noop_by_default() -> None:
    """Role.on_stream_start() is a no-op by default."""
    role = ConcreteRole()
    role.on_stream_start()  # Should not raise


def test_role_on_audio_chunk_is_noop_by_default() -> None:
    """Role.on_audio_chunk() is a no-op by default."""
    role = ConcreteRole()
    chunk = AudioChunk(data=b"audio", timestamp_us=1000, duration_us=25000, byte_count=5)
    result = role.on_audio_chunk(chunk)
    assert result is None


def test_role_on_stream_clear_is_noop_by_default() -> None:
    """Role.on_stream_clear() is a no-op by default."""
    role = ConcreteRole()
    role.on_stream_clear()  # Should not raise


def test_role_on_stream_end_is_noop_by_default() -> None:
    """Role.on_stream_end() is a no-op by default."""
    role = ConcreteRole()
    role.on_stream_end()  # Should not raise


def test_role_has_buffer_tracker_none_by_default() -> None:
    """Role._buffer_tracker is None by default."""
    role = ConcreteRole()
    assert role._buffer_tracker is None  # noqa: SLF001


def test_role_has_stream_started_false_by_default() -> None:
    """Role._stream_started is False by default."""
    role = ConcreteRole()
    assert role._stream_started is False  # noqa: SLF001
