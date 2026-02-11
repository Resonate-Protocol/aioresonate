"""Tests for ArtworkV1Role (v1) implementation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiosendspin.models.artwork import ArtworkChannel
from aiosendspin.models.core import StreamStartMessage
from aiosendspin.models.types import ArtworkSource, PictureFormat
from aiosendspin.server.roles.artwork.v1 import ArtworkV1Role


def _make_client_stub() -> MagicMock:
    """Create a mock client for testing."""
    client = MagicMock()
    client.group = MagicMock()
    client.group.group_role.return_value = None
    client.info = MagicMock()
    client.info.artwork_support = None
    client.send_message = MagicMock()
    client.send_role_message = MagicMock()
    client.send_binary = MagicMock(return_value=True)
    client._logger = MagicMock()  # noqa: SLF001
    return client


def test_artwork_role_has_role_id() -> None:
    """ArtworkV1Role has role_id of 'artwork@v1'."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    assert role.role_id == "artwork@v1"


def test_artwork_role_has_role_family() -> None:
    """ArtworkV1Role has role_family of 'artwork'."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    assert role.role_family == "artwork"


def test_artwork_role_requires_client() -> None:
    """ArtworkV1Role raises ValueError if no client provided."""
    with pytest.raises(ValueError, match="requires a client"):
        ArtworkV1Role(client=None)


def test_artwork_role_on_connect_subscribes_to_group_role() -> None:
    """on_connect() subscribes to ArtworkGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = ArtworkV1Role(client=client)
    role.on_connect()

    client.group.group_role.assert_called_with("artwork")
    group_role.subscribe.assert_called_once_with(role)


def test_artwork_role_on_disconnect_unsubscribes_from_group_role() -> None:
    """on_disconnect() unsubscribes from ArtworkGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = ArtworkV1Role(client=client)
    role.on_connect()
    role.on_disconnect()

    group_role.unsubscribe.assert_called_once_with(role)


def test_artwork_role_init_channel_configs_from_support() -> None:
    """on_connect() initializes channel configs from client hello."""
    client = _make_client_stub()
    support = MagicMock()
    support.channels = [
        ArtworkChannel(
            source=ArtworkSource.ALBUM,
            format=PictureFormat.JPEG,
            media_width=300,
            media_height=300,
        ),
        ArtworkChannel(
            source=ArtworkSource.ARTIST,
            format=PictureFormat.PNG,
            media_width=400,
            media_height=400,
        ),
    ]
    client.info.artwork_support = support

    role = ArtworkV1Role(client=client)
    role.on_connect()

    configs = role.get_channel_configs()
    assert len(configs) == 2
    assert configs[0].source == ArtworkSource.ALBUM
    assert configs[1].source == ArtworkSource.ARTIST


def test_artwork_role_sends_stream_start_on_connect_with_transport() -> None:
    """on_connect() sends stream/start when transport is attached."""
    client = _make_client_stub()
    support = MagicMock()
    support.channels = [
        ArtworkChannel(
            source=ArtworkSource.ALBUM,
            format=PictureFormat.JPEG,
            media_width=300,
            media_height=300,
        ),
    ]
    client.info.artwork_support = support

    role = ArtworkV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001
    role.on_connect()

    client.send_role_message.assert_called()
    _role, msg = client.send_role_message.call_args.args
    assert isinstance(msg, StreamStartMessage)
    assert msg.payload.artwork is not None
    assert len(msg.payload.artwork.channels) == 1


def test_artwork_role_send_artwork() -> None:
    """send_artwork() sends binary message with header and image data."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001

    role.send_artwork(channel=0, image_data=b"image", timestamp_us=1000)

    client.send_binary.assert_called_once()
    kwargs = client.send_binary.call_args.kwargs
    assert kwargs["role_family"] == "artwork"
    assert kwargs["timestamp_us"] == 1000
    assert kwargs["message_type"] == 8  # ARTWORK_CHANNEL_0


def test_artwork_role_send_artwork_cleared() -> None:
    """send_artwork_cleared() sends empty binary message."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    role._client.connection = MagicMock()  # noqa: SLF001

    role.send_artwork_cleared(channel=1, timestamp_us=2000)

    client.send_binary.assert_called_once()
    kwargs = client.send_binary.call_args.kwargs
    assert kwargs["message_type"] == 9  # ARTWORK_CHANNEL_1


def test_artwork_role_send_artwork_noop_without_transport() -> None:
    """send_artwork() is a no-op when no transport."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    role._client.connection = None  # noqa: SLF001

    role.send_artwork(channel=0, image_data=b"image", timestamp_us=1000)

    client.send_binary.assert_not_called()


def test_artwork_role_has_no_audio_requirements() -> None:
    """ArtworkV1Role does not receive audio."""
    client = _make_client_stub()
    role = ArtworkV1Role(client=client)
    assert role.get_audio_requirements() is None
