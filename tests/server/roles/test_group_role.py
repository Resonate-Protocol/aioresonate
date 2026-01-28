"""Tests for GroupRole subscription mechanism."""

from __future__ import annotations

from unittest.mock import MagicMock

from aiosendspin.server.roles.base import GroupRole, Role


class DummyGroupRole(GroupRole):
    """Test GroupRole implementation that tracks join/leave events."""

    role_family = "dummy"

    def __init__(self, group: object) -> None:
        """Initialize with event tracking lists."""
        super().__init__(group)  # type: ignore[arg-type]
        self.joined: list[Role] = []
        self.left: list[Role] = []

    def on_member_join(self, role: Role) -> None:
        """Track member join events."""
        self.joined.append(role)

    def on_member_leave(self, role: Role) -> None:
        """Track member leave events."""
        self.left.append(role)


def test_group_role_subscribe_adds_member() -> None:
    """Subscribe adds role to members list."""
    group = MagicMock()
    group_role = DummyGroupRole(group)
    role = MagicMock()

    group_role.subscribe(role)

    assert role in group_role._members  # noqa: SLF001


def test_group_role_subscribe_is_idempotent() -> None:
    """Subscribing twice does not add duplicate members."""
    group = MagicMock()
    group_role = DummyGroupRole(group)
    role = MagicMock()

    group_role.subscribe(role)
    group_role.subscribe(role)

    assert group_role._members.count(role) == 1  # noqa: SLF001
    assert len(group_role.joined) == 1  # on_member_join only called once


def test_group_role_unsubscribe_removes_member() -> None:
    """Unsubscribe removes role from members list."""
    group = MagicMock()
    group_role = DummyGroupRole(group)
    role = MagicMock()
    group_role._members.append(role)  # noqa: SLF001

    group_role.unsubscribe(role)

    assert role not in group_role._members  # noqa: SLF001


def test_group_role_unsubscribe_nonmember_is_noop() -> None:
    """Unsubscribing a non-member does nothing."""
    group = MagicMock()
    group_role = DummyGroupRole(group)
    role = MagicMock()

    group_role.unsubscribe(role)  # Should not raise

    assert len(group_role.left) == 0


def test_group_role_on_member_join_called() -> None:
    """Subscribe calls on_member_join hook."""
    group = MagicMock()
    group_role = DummyGroupRole(group)
    role = MagicMock()

    group_role.subscribe(role)

    assert role in group_role.joined


def test_group_role_on_member_leave_called() -> None:
    """Unsubscribe calls on_member_leave hook."""
    group = MagicMock()
    group_role = DummyGroupRole(group)
    role = MagicMock()
    group_role._members.append(role)  # noqa: SLF001

    group_role.unsubscribe(role)

    assert role in group_role.left


def test_group_role_stores_group_reference() -> None:
    """GroupRole stores reference to owning group."""
    group = MagicMock()
    group_role = DummyGroupRole(group)

    assert group_role._group is group  # noqa: SLF001
