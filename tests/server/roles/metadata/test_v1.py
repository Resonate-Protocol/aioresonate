"""Tests for MetadataRole (v1) implementation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiosendspin.server.roles.metadata.v1 import MetadataRole


def _make_client_stub() -> MagicMock:
    """Create a mock client for testing."""
    client = MagicMock()
    client.group = MagicMock()
    client.group.group_role.return_value = None
    return client


def test_metadata_role_has_role_id() -> None:
    """MetadataRole has role_id of 'metadata@v1'."""
    client = _make_client_stub()
    role = MetadataRole(client=client)
    assert role.role_id == "metadata@v1"


def test_metadata_role_has_role_family() -> None:
    """MetadataRole has role_family of 'metadata'."""
    client = _make_client_stub()
    role = MetadataRole(client=client)
    assert role.role_family == "metadata"


def test_metadata_role_requires_client() -> None:
    """MetadataRole raises ValueError if no client provided."""
    with pytest.raises(ValueError, match="requires a client"):
        MetadataRole(client=None)


def test_metadata_role_on_connect_subscribes_to_group_role() -> None:
    """on_connect() subscribes to MetadataGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = MetadataRole(client=client)
    role.on_connect()

    client.group.group_role.assert_called_with("metadata")
    group_role.subscribe.assert_called_once_with(role)


def test_metadata_role_on_disconnect_unsubscribes_from_group_role() -> None:
    """on_disconnect() unsubscribes from MetadataGroupRole."""
    client = _make_client_stub()
    group_role = MagicMock()
    client.group.group_role.return_value = group_role

    role = MetadataRole(client=client)
    role.on_connect()
    role.on_disconnect()

    group_role.unsubscribe.assert_called_once_with(role)


def test_metadata_role_has_no_stream_requirements() -> None:
    """MetadataRole does not send binary streams."""
    client = _make_client_stub()
    role = MetadataRole(client=client)
    assert role.get_stream_requirements() is None


def test_metadata_role_has_no_audio_requirements() -> None:
    """MetadataRole does not receive audio."""
    client = _make_client_stub()
    role = MetadataRole(client=client)
    assert role.get_audio_requirements() is None
