"""Tests for public protocol callback hooks on the Sendspin client."""

from __future__ import annotations

import pytest

from aiosendspin.client.client import SendspinClient
from aiosendspin.models import pack_binary_header_raw
from aiosendspin.models.artwork import (
    ArtworkChannel,
    ClientHelloArtworkSupport,
    StreamArtworkChannelConfig,
    StreamStartArtwork,
)
from aiosendspin.models.core import ServerHelloPayload, StreamStartMessage, StreamStartPayload
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import (
    ArtworkSource,
    AudioCodec,
    BinaryMessageType,
    ConnectionReason,
    PictureFormat,
    Roles,
)


def _player_support() -> ClientHelloPlayerSupport:
    return ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.PCM,
                sample_rate=48_000,
                bit_depth=16,
                channels=2,
            )
        ],
        buffer_capacity=100_000,
        supported_commands=[],
    )


@pytest.mark.asyncio
async def test_server_hello_listener_receives_payload() -> None:
    """Client should expose server/hello through a public listener."""
    client = SendspinClient(
        client_id="client-1",
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
    )

    captured: list[ServerHelloPayload] = []
    client.add_server_hello_listener(captured.append)

    payload = ServerHelloPayload(
        server_id="server-1",
        name="Test Server",
        version=1,
        active_roles=[Roles.PLAYER.value],
        connection_reason=ConnectionReason.PLAYBACK,
    )

    client._handle_server_hello(payload)  # noqa: SLF001

    assert captured == [payload]
    assert client.server_info is not None
    assert client.server_info.server_id == "server-1"


@pytest.mark.asyncio
async def test_artwork_listener_receives_binary_frames_after_artwork_stream_start() -> None:
    """Client should expose artwork binary frames without private overrides."""
    client = SendspinClient(
        client_id="client-1",
        client_name="Test Client",
        roles=[Roles.ARTWORK],
        artwork_support=ClientHelloArtworkSupport(
            channels=[
                ArtworkChannel(
                    source=ArtworkSource.ALBUM,
                    format=PictureFormat.JPEG,
                    media_width=256,
                    media_height=256,
                )
            ]
        ),
    )
    captured: list[tuple[int, bytes]] = []
    client.add_artwork_listener(lambda channel, data: captured.append((channel, data)))

    await client._handle_stream_start(  # noqa: SLF001
        StreamStartMessage(
            payload=StreamStartPayload(
                artwork=StreamStartArtwork(
                    channels=[
                        StreamArtworkChannelConfig(
                            source=ArtworkSource.ALBUM,
                            format=PictureFormat.JPEG,
                            width=512,
                            height=512,
                        )
                    ]
                )
            )
        )
    )

    payload = b"artwork-bytes"
    client._handle_binary_message(  # noqa: SLF001
        pack_binary_header_raw(BinaryMessageType.ARTWORK_CHANNEL_0.value, 123_456) + payload
    )

    assert captured == [(0, payload)]
