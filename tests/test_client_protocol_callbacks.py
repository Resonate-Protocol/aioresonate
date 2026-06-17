"""Tests for public protocol callback hooks on the Sendspin client."""

from __future__ import annotations

import json
import struct
from unittest.mock import MagicMock

import pytest

from aiosendspin.client.client import AudioFormat, SendspinClient
from aiosendspin.models import pack_binary_header_raw
from aiosendspin.models.artwork import (
    ArtworkChannel,
    ClientHelloArtworkSupport,
    StreamArtworkChannelConfig,
    StreamStartArtwork,
)
from aiosendspin.models.core import ServerHelloPayload, StreamStartMessage, StreamStartPayload
from aiosendspin.models.player import (
    ClientHelloPlayerSupport,
    StreamStartPlayer,
    SupportedAudioFormat,
)
from aiosendspin.models.types import (
    ArtworkSource,
    AudioCodec,
    BinaryMessageType,
    ConnectionReason,
    MediaCommand,
    PictureFormat,
    Roles,
)
from aiosendspin.models.visualizer import (
    ClientHelloVisualizerSupport,
    VisualizerFrame,
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


def _artwork_support() -> ClientHelloArtworkSupport:
    return ClientHelloArtworkSupport(
        channels=[
            ArtworkChannel(
                source=ArtworkSource.ALBUM,
                format=PictureFormat.JPEG,
                media_width=256,
                media_height=256,
            )
        ]
    )


def _visualizer_support() -> ClientHelloVisualizerSupport:
    return ClientHelloVisualizerSupport(
        buffer_capacity=4096,
        rate_max=30,
        types=["loudness"],
    )


def _stream_start_player() -> StreamStartPlayer:
    return StreamStartPlayer(
        codec=AudioCodec.PCM,
        sample_rate=48_000,
        channels=2,
        bit_depth=16,
    )


@pytest.mark.asyncio
async def test_artwork_binary_dropped_when_only_player_stream_active() -> None:
    """Artwork binaries must be rejected when only the player stream is active."""
    client = SendspinClient(
        client_id="client-1",
        client_name="Test Client",
        roles=[Roles.PLAYER, Roles.ARTWORK],
        player_support=_player_support(),
        artwork_support=_artwork_support(),
    )
    captured: list[tuple[int, bytes]] = []
    client.add_artwork_listener(lambda channel, data: captured.append((channel, data)))

    await client._handle_stream_start(  # noqa: SLF001
        StreamStartMessage(payload=StreamStartPayload(player=_stream_start_player()))
    )

    client._handle_binary_message(  # noqa: SLF001
        pack_binary_header_raw(BinaryMessageType.ARTWORK_CHANNEL_0.value, 123_456) + b"art"
    )

    assert captured == []


@pytest.mark.asyncio
async def test_audio_binary_dropped_when_only_artwork_stream_active() -> None:
    """Audio binaries must be rejected when only the artwork stream is active."""
    client = SendspinClient(
        client_id="client-1",
        client_name="Test Client",
        roles=[Roles.PLAYER, Roles.ARTWORK],
        player_support=_player_support(),
        artwork_support=_artwork_support(),
    )
    captured: list[tuple[int, bytes, AudioFormat]] = []
    client.add_audio_chunk_listener(
        lambda ts, data, fmt: captured.append((ts, data, fmt)),
    )

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

    client._handle_binary_message(  # noqa: SLF001
        pack_binary_header_raw(BinaryMessageType.AUDIO_CHUNK.value, 123_456) + b"\x00\x00\x00\x00"
    )

    assert captured == []


@pytest.mark.asyncio
async def test_visualizer_binary_dropped_when_only_player_stream_active() -> None:
    """Visualizer binaries must be rejected when only the player stream is active."""
    client = SendspinClient(
        client_id="client-1",
        client_name="Test Client",
        roles=[Roles.PLAYER, Roles.VISUALIZER],
        player_support=_player_support(),
        visualizer_support=_visualizer_support(),
    )
    captured: list[list[VisualizerFrame]] = []
    client.add_visualizer_listener(captured.append)

    await client._handle_stream_start(  # noqa: SLF001
        StreamStartMessage(payload=StreamStartPayload(player=_stream_start_player()))
    )

    # Loudness frame: type byte + 8-byte timestamp + 2-byte value.
    loudness_payload = (
        bytes([BinaryMessageType.VISUALIZATION_LOUDNESS.value])
        + struct.pack(">q", 1_000)
        + struct.pack(">H", 42)
    )
    client._handle_binary_message(loudness_payload)  # noqa: SLF001

    assert captured == []


@pytest.mark.asyncio
async def test_artwork_binary_dispatched_when_artwork_stream_active() -> None:
    """Artwork binaries must reach listeners once artwork stream is active."""
    client = SendspinClient(
        client_id="client-1",
        client_name="Test Client",
        roles=[Roles.ARTWORK],
        artwork_support=_artwork_support(),
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

    payload = b"artwork-bytes-2"
    client._handle_binary_message(  # noqa: SLF001
        pack_binary_header_raw(BinaryMessageType.ARTWORK_CHANNEL_1.value, 234_567) + payload
    )

    assert captured == [(1, payload)]


@pytest.mark.asyncio
async def test_send_group_command_seek_forwards_position_ms() -> None:
    """send_group_command must include position_ms in the outgoing JSON for seek."""
    client = SendspinClient(
        client_id="client-1",
        client_name="Test Client",
        roles=[Roles.PLAYER],
        player_support=_player_support(),
    )

    mock_ws = MagicMock()
    mock_ws.closed = False
    client._ws = mock_ws  # noqa: SLF001
    client._connected = True  # noqa: SLF001

    sent: list[str] = []

    async def _capture(payload: str) -> None:
        sent.append(payload)

    client._send_message = _capture  # noqa: SLF001

    await client.send_group_command(MediaCommand.SEEK, position_ms=12_000)

    assert len(sent) == 1
    msg = json.loads(sent[0])
    assert msg["payload"]["controller"]["command"] == "seek"
    assert msg["payload"]["controller"]["position_ms"] == 12_000
