"""Tests for ArtworkGroupRole events."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from aiosendspin.models.types import ArtworkSource
from aiosendspin.server.roles.artwork.events import ArtworkClearedEvent, ArtworkUpdatedEvent
from aiosendspin.server.roles.artwork.group import ArtworkGroupRole


def _make_group_stub() -> MagicMock:
    group = MagicMock()
    group._server = MagicMock()  # noqa: SLF001
    group._server.clock.now_us.return_value = 123_456  # noqa: SLF001
    return group


@pytest.mark.asyncio
async def test_set_album_artwork_emits_updated_event() -> None:
    """Setting album artwork emits ArtworkUpdatedEvent."""
    group = _make_group_stub()
    agr = ArtworkGroupRole(group)

    image = Image.new("RGB", (320, 240), (255, 0, 0))
    await agr.set_album_artwork(image)

    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ArtworkUpdatedEvent)
    assert event.source == ArtworkSource.ALBUM
    assert event.timestamp_us == 123_456
    assert event.width == 320
    assert event.height == 240


@pytest.mark.asyncio
async def test_clear_album_artwork_emits_cleared_event() -> None:
    """Clearing album artwork emits ArtworkClearedEvent."""
    group = _make_group_stub()
    agr = ArtworkGroupRole(group)
    image = Image.new("RGB", (100, 100), (0, 255, 0))
    await agr.set_album_artwork(image)
    group._signal_event.reset_mock()  # noqa: SLF001

    await agr.set_album_artwork(None)

    group._signal_event.assert_called_once()  # noqa: SLF001
    event = group._signal_event.call_args.args[0]  # noqa: SLF001
    assert isinstance(event, ArtworkClearedEvent)
    assert event.source == ArtworkSource.ALBUM
    assert event.timestamp_us == 123_456
    assert agr.get_album_artwork() is None
