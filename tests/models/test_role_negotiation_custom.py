"""Tests for negotiating custom role IDs registered at runtime."""

from __future__ import annotations

from typing import Any

from aiosendspin.models.types import negotiate_active_roles
from aiosendspin.server.roles.registry import ROLE_FACTORIES


def _factory(_client: Any) -> object:
    return object()


def test_negotiate_accepts_registered_custom_role_in_known_family(
    monkeypatch: Any,
) -> None:
    """Registered custom role IDs should be negotiable without SUPPORTED_ROLE_VERSIONS edits."""
    monkeypatch.setitem(ROLE_FACTORIES, "player@_airplay_bridge", _factory)

    active_roles = negotiate_active_roles(["player@_airplay_bridge"])

    assert active_roles == ["player@_airplay_bridge"]


def test_negotiate_accepts_registered_custom_role_in_unknown_family(
    monkeypatch: Any,
) -> None:
    """Registered custom roles for unknown families should also be negotiable."""
    monkeypatch.setitem(ROLE_FACTORIES, "customaudio@v1", _factory)

    active_roles = negotiate_active_roles(["customaudio@v1"])

    assert active_roles == ["customaudio@v1"]


def test_negotiate_rejects_unregistered_custom_role_id() -> None:
    """Unregistered custom role IDs should remain non-negotiable."""
    active_roles = negotiate_active_roles(["player@_not_registered"])

    assert active_roles == []


def test_negotiate_prefers_first_client_role_when_custom_first(
    monkeypatch: Any,
) -> None:
    """Client order should determine same-family selection when custom role appears first."""
    monkeypatch.setitem(ROLE_FACTORIES, "player@_airplay_bridge", _factory)

    active_roles = negotiate_active_roles(["player@_airplay_bridge", "player@v1"])

    assert active_roles == ["player@_airplay_bridge"]


def test_negotiate_prefers_first_client_role_when_standard_first(
    monkeypatch: Any,
) -> None:
    """Client order should determine same-family selection when standard role appears first."""
    monkeypatch.setitem(ROLE_FACTORIES, "player@_airplay_bridge", _factory)

    active_roles = negotiate_active_roles(["player@v1", "player@_airplay_bridge"])

    assert active_roles == ["player@v1"]
