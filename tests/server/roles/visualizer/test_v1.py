"""Tests for VisualizerV1Role (v1) implementation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiosendspin.server.roles.visualizer.v1 import VisualizerV1Role


def _make_client_stub() -> MagicMock:
    """Create a mock client for testing."""
    client = MagicMock()
    client.group = MagicMock()
    client.group.group_role.return_value = None
    return client


def test_visualizer_role_has_role_id() -> None:
    """VisualizerV1Role has role_id of 'visualizer@v1'."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    assert role.role_id == "visualizer@v1"


def test_visualizer_role_has_role_family() -> None:
    """VisualizerV1Role has role_family of 'visualizer'."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    assert role.role_family == "visualizer"


def test_visualizer_role_requires_client() -> None:
    """VisualizerV1Role raises ValueError if no client provided."""
    with pytest.raises(ValueError, match="requires a client"):
        VisualizerV1Role(client=None)


def test_visualizer_role_on_connect_subscribes_to_group_role() -> None:
    """on_connect() subscribes to VisualizerGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = VisualizerV1Role(client=client)
    role.on_connect()

    client.group.group_role.assert_called_with("visualizer")
    group_role.subscribe.assert_called_once_with(role)


def test_visualizer_role_on_disconnect_unsubscribes_from_group_role() -> None:
    """on_disconnect() unsubscribes from VisualizerGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = VisualizerV1Role(client=client)
    role.on_connect()
    role.on_disconnect()

    group_role.unsubscribe.assert_called_once_with(role)


def test_visualizer_role_has_no_stream_requirements() -> None:
    """VisualizerV1Role (placeholder) does not send binary streams."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    assert role.get_stream_requirements() is None


def test_visualizer_role_has_no_audio_requirements() -> None:
    """VisualizerV1Role (placeholder) does not receive audio."""
    client = _make_client_stub()
    role = VisualizerV1Role(client=client)
    assert role.get_audio_requirements() is None
