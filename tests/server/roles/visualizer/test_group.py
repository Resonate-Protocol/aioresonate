"""Tests for VisualizerGroupRole."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.server.roles.visualizer.group import VisualizerGroupRole


def _make_group_stub() -> MagicMock:
    """Create a mock group for testing."""
    return MagicMock()


def test_visualizer_group_role_family() -> None:
    """VisualizerGroupRole has role_family of 'visualizer'."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)
    assert vgr.role_family == "visualizer"


def test_visualizer_group_role_subscribe() -> None:
    """VisualizerGroupRole accepts subscriptions."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)

    member = MagicMock()
    vgr.subscribe(member)

    assert member in vgr._members  # noqa: SLF001


def test_visualizer_group_role_unsubscribe() -> None:
    """VisualizerGroupRole handles unsubscriptions."""
    group = _make_group_stub()
    vgr = VisualizerGroupRole(group)

    member = MagicMock()
    vgr.subscribe(member)
    vgr.unsubscribe(member)

    assert member not in vgr._members  # noqa: SLF001
