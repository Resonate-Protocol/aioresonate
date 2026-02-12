"""Tests for externally managed player registration and stream-start callbacks."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiosendspin.models.core import ClientHelloPayload
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles
from aiosendspin.server.client import SendspinClient
from aiosendspin.server.group import SendspinGroup
from aiosendspin.server.server import ExternalStreamStartRequest, SendspinServer


class _DummyConnection:
    async def disconnect(self, *, retry_connection: bool = True) -> None:  # noqa: ARG002
        return

    def send_message(self, message: object) -> None:  # noqa: ARG002
        return

    def send_role_message(self, role: str, message: object) -> None:  # noqa: ARG002
        return

    def send_binary(
        self,
        data: bytes,  # noqa: ARG002
        *,
        role: str,  # noqa: ARG002
        timestamp_us: int,  # noqa: ARG002
        message_type: int,  # noqa: ARG002
        buffer_end_time_us: int | None = None,  # noqa: ARG002
        buffer_byte_count: int | None = None,  # noqa: ARG002
        duration_us: int | None = None,  # noqa: ARG002
    ) -> bool:
        return True


def _player_hello(client_id: str) -> ClientHelloPayload:
    return ClientHelloPayload(
        client_id=client_id,
        name=client_id,
        version=1,
        supported_roles=[Roles.PLAYER.value],
        player_support=ClientHelloPlayerSupport(
            supported_formats=[
                SupportedAudioFormat(
                    codec=AudioCodec.PCM,
                    channels=2,
                    sample_rate=48000,
                    bit_depth=16,
                )
            ],
            buffer_capacity=100_000,
            supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
        ),
    )


def _make_server() -> SendspinServer:
    loop = asyncio.get_running_loop()
    client_session = MagicMock()
    client_session.closed = True
    client_session.close = AsyncMock()
    return SendspinServer(
        loop=loop,
        server_id="srv",
        server_name="server",
        client_session=client_session,
    )


@pytest.mark.asyncio
async def test_register_external_player_preloads_identity_and_fires_on_start_stream() -> None:
    """External player should be visible and trigger callback on stream start."""
    server = _make_server()
    callback_calls: list[ExternalStreamStartRequest] = []

    player = server.register_external_player(
        _player_hello("external-1"),
        on_stream_start=callback_calls.append,
    )

    assert player.client_id == "external-1"
    assert player.name == "external-1"
    assert player.info.client_id == "external-1"
    assert not player.is_connected
    assert server.is_external_player("external-1")

    player.group.start_stream()

    assert len(callback_calls) == 1
    assert callback_calls[0].client_id == "external-1"


@pytest.mark.asyncio
async def test_add_external_player_to_active_group_requests_external_connect() -> None:
    """Adding disconnected external player to active stream should call callback."""
    server = _make_server()
    callback_calls: list[ExternalStreamStartRequest] = []

    owner = SendspinClient(server, client_id="owner")
    SendspinGroup(server, owner)
    owner.attach_connection(
        _DummyConnection(),
        client_info=_player_hello("owner"),
        active_roles=[Roles.PLAYER.value],
    )
    owner.mark_connected()
    owner.group.start_stream()

    external = server.register_external_player(
        _player_hello("external-2"),
        on_stream_start=callback_calls.append,
    )

    await owner.group.add_client(external)

    assert len(callback_calls) == 1
    assert callback_calls[0].client_id == "external-2"
