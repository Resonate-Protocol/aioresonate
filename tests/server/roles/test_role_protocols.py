"""Protocol conformance tests for role implementations."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.server.roles.artwork.types import ArtworkRoleProtocol
from aiosendspin.server.roles.artwork.v1 import ArtworkV1Role
from aiosendspin.server.roles.controller.types import ControllerRoleProtocol
from aiosendspin.server.roles.controller.v1 import ControllerV1Role
from aiosendspin.server.roles.metadata.types import MetadataRoleProtocol
from aiosendspin.server.roles.metadata.v1 import MetadataV1Role
from aiosendspin.server.roles.player.types import PlayerRoleProtocol
from aiosendspin.server.roles.player.v1 import PlayerV1Role


def _make_basic_client_stub() -> MagicMock:
    client = MagicMock()
    client.group = MagicMock()
    client.group.group_role.return_value = None
    return client


def _make_artwork_client_stub() -> MagicMock:
    client = _make_basic_client_stub()
    client.info = MagicMock()
    client.info.artwork_support = None
    client.send_message = MagicMock()
    client.send_binary = MagicMock(return_value=True)
    client._logger = MagicMock()  # noqa: SLF001
    return client


def _make_player_client_stub() -> MagicMock:
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
    return client


def test_artwork_role_matches_protocol() -> None:
    """ArtworkV1Role satisfies ArtworkRoleProtocol."""
    client = _make_artwork_client_stub()
    role = ArtworkV1Role(client=client)
    assert isinstance(role, ArtworkRoleProtocol)


def test_controller_role_matches_protocol() -> None:
    """ControllerV1Role satisfies ControllerRoleProtocol."""
    client = _make_basic_client_stub()
    role = ControllerV1Role(client=client)
    assert isinstance(role, ControllerRoleProtocol)


def test_metadata_role_matches_protocol() -> None:
    """MetadataV1Role satisfies MetadataRoleProtocol."""
    client = _make_basic_client_stub()
    role = MetadataV1Role(client=client)
    assert isinstance(role, MetadataRoleProtocol)


def test_player_role_matches_protocol() -> None:
    """PlayerV1Role satisfies PlayerRoleProtocol."""
    client = _make_player_client_stub()
    role = PlayerV1Role(client=client)
    assert isinstance(role, PlayerRoleProtocol)
