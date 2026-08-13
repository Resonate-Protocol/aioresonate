"""Tests for ArtworkGroupRole events."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from PIL import Image

from aiosendspin.models.artwork import ArtworkChannel
from aiosendspin.models.types import ArtworkSource, PictureFormat
from aiosendspin.server.roles.artwork.events import ArtworkClearedEvent, ArtworkUpdatedEvent
from aiosendspin.server.roles.artwork.group import ArtworkGroupRole
from aiosendspin.server.roles.artwork.types import ArtworkRoleProtocol


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


@pytest.mark.asyncio
async def test_scheduled_artwork_clear_keeps_current_and_replays_in_order() -> None:
    """A scheduled clear replays after the current image."""
    group = _make_group_stub()
    agr = ArtworkGroupRole(group)
    image = Image.new("RGB", (10, 10), (255, 0, 0))
    await agr.set_album_artwork(image)
    await agr.set_album_artwork(None, timestamp_us=500_000)
    role = MagicMock(spec=ArtworkRoleProtocol)
    config = ArtworkChannel(
        source=ArtworkSource.ALBUM,
        format=PictureFormat.JPEG,
        media_width=10,
        media_height=10,
    )
    agr._process_and_encode_image = MagicMock(return_value=b"current")  # type: ignore[method-assign]  # noqa: SLF001

    await agr._send_artwork_replay(role, 0, config)  # noqa: SLF001

    assert agr.get_album_artwork() is image
    assert role.method_calls[0].args == (0, b"current", 123_456)
    assert role.method_calls[1].args == (0, 500_000)


@pytest.mark.asyncio
async def test_earlier_artwork_replaces_pending_without_committing_it() -> None:
    """An earlier artwork message discards the prior pending image."""
    group = _make_group_stub()
    agr = ArtworkGroupRole(group)
    current = Image.new("RGB", (10, 10), (255, 0, 0))
    later = Image.new("RGB", (10, 10), (0, 255, 0))
    earlier = Image.new("RGB", (10, 10), (0, 0, 255))
    await agr.set_album_artwork(current)
    await agr.set_album_artwork(later, timestamp_us=500_000)
    await agr.set_album_artwork(earlier, timestamp_us=400_000)

    assert agr.get_album_artwork() is current
    assert agr._pending_artwork[ArtworkSource.ALBUM].image is earlier  # noqa: SLF001


@pytest.mark.asyncio
async def test_later_artwork_commits_pending_before_storing_replacement() -> None:
    """A later artwork message commits pending before replacing it."""
    group = _make_group_stub()
    agr = ArtworkGroupRole(group)
    current = Image.new("RGB", (10, 10), (255, 0, 0))
    pending = Image.new("RGB", (10, 10), (0, 255, 0))
    replacement = Image.new("RGB", (10, 10), (0, 0, 255))
    await agr.set_album_artwork(current)
    await agr.set_album_artwork(pending, timestamp_us=500_000)
    await agr.set_album_artwork(replacement, timestamp_us=600_000)

    assert agr.get_album_artwork() is pending
    assert agr._pending_artwork[ArtworkSource.ALBUM].image is replacement  # noqa: SLF001


@pytest.mark.asyncio
async def test_live_artwork_waits_for_complete_replay() -> None:
    """A live update cannot interleave with current then pending replay."""
    group = _make_group_stub()
    agr = ArtworkGroupRole(group)
    current = Image.new("RGB", (10, 10), (255, 0, 0))
    pending = Image.new("RGB", (10, 10), (0, 255, 0))
    replacement = Image.new("RGB", (10, 10), (0, 0, 255))
    await agr.set_album_artwork(current)
    await agr.set_album_artwork(pending, timestamp_us=500_000)
    role = MagicMock(spec=ArtworkRoleProtocol)
    role.get_channel_configs.return_value = {
        0: ArtworkChannel(
            source=ArtworkSource.ALBUM,
            format=PictureFormat.JPEG,
            media_width=10,
            media_height=10,
        )
    }
    agr._members = [role]  # noqa: SLF001
    entered = asyncio.Event()
    release = asyncio.Event()
    sent_timestamps: list[int] = []

    async def send(
        _role: ArtworkRoleProtocol,
        _image: Image.Image | None,
        _channel: int,
        _config: ArtworkChannel,
        timestamp_us: int,
    ) -> None:
        sent_timestamps.append(timestamp_us)
        if len(sent_timestamps) == 1:
            entered.set()
            await release.wait()

    agr._encode_and_send_artwork = send  # type: ignore[method-assign]  # noqa: SLF001
    replay = asyncio.create_task(
        agr._send_artwork_replay(role, 0, role.get_channel_configs()[0])  # noqa: SLF001
    )
    await entered.wait()
    live = asyncio.create_task(agr.set_album_artwork(replacement, timestamp_us=600_000))
    await asyncio.sleep(0)

    assert sent_timestamps == [123_456]

    release.set()
    await asyncio.gather(replay, live)

    assert sent_timestamps == [123_456, 500_000, 600_000]
