"""Tests for ColorV1Role (v1) implementation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiosendspin.server.roles.color.v1 import ColorV1Role


def _make_client_stub() -> MagicMock:
    client = MagicMock()
    client.group = MagicMock()
    client.group.group_role.return_value = None
    return client


def test_color_v1_role_id() -> None:
    """ColorV1Role has role_id of 'color@v1'."""
    client = _make_client_stub()
    role = ColorV1Role(client=client)
    assert role.role_id == "color@v1"


def test_color_v1_role_family() -> None:
    """ColorV1Role has role_family of 'color'."""
    client = _make_client_stub()
    role = ColorV1Role(client=client)
    assert role.role_family == "color"


def test_color_v1_role_requires_client() -> None:
    """ColorV1Role raises ValueError if no client provided."""
    with pytest.raises(ValueError, match="requires a client"):
        ColorV1Role(client=None)


def test_color_v1_on_connect_subscribes_to_group_role() -> None:
    """on_connect() subscribes to ColorGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = ColorV1Role(client=client)
    role.on_connect()

    client.group.group_role.assert_called_with("color")
    group_role.subscribe.assert_called_once_with(role)


def test_color_v1_on_disconnect_unsubscribes_from_group_role() -> None:
    """on_disconnect() unsubscribes from ColorGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = ColorV1Role(client=client)
    role.on_connect()
    role.on_disconnect()

    group_role.unsubscribe.assert_called_once_with(role)


def test_color_v1_has_no_stream_requirements() -> None:
    """ColorV1Role does not send binary streams."""
    client = _make_client_stub()
    role = ColorV1Role(client=client)
    assert role.get_stream_requirements() is None


def test_color_v1_has_no_audio_requirements() -> None:
    """ColorV1Role does not receive audio."""
    client = _make_client_stub()
    role = ColorV1Role(client=client)
    assert role.get_audio_requirements() is None
