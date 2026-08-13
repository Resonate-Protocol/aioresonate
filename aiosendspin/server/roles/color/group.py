"""ColorGroupRole - group-level color coordination."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from aiosendspin.models.color import SessionUpdateColor
from aiosendspin.models.core import ServerStateMessage, ServerStatePayload
from aiosendspin.server.roles.base import GroupRole, Role
from aiosendspin.server.roles.color.events import ColorClearedEvent, ColorUpdatedEvent
from aiosendspin.server.roles.color.state import Color
from aiosendspin.server.roles.color.types import ColorRoleProtocol

if TYPE_CHECKING:
    from aiosendspin.server.group import SendspinGroup


class ColorGroupRole(GroupRole):
    """Coordinate color palette across a group.

    Stores current color state and pushes updates to subscribed ColorV1Roles.
    """

    role_family = "color"

    def __init__(self, group: SendspinGroup) -> None:
        """Initialize ColorGroupRole."""
        super().__init__(group)
        self._current_color: Color | None = None
        self._pending_color: Color | None = None
        self._pending_update: SessionUpdateColor | None = None

    @property
    def color(self) -> Color | None:
        """Return current color palette."""
        return self._current_color

    def on_member_join(self, role: Role) -> None:
        """Send current color to newly joined member."""
        self._send_state_to_role(role)

    def _send_state_to_role(self, role: ColorRoleProtocol) -> None:
        """Send current color state to a single role."""
        timestamp = self._group._server.clock.now_us()  # noqa: SLF001
        if self._current_color is not None:
            color_update = self._current_color.snapshot_update(timestamp)
        else:
            color_update = SessionUpdateColor.cleared(timestamp)
        role.send_message(ServerStateMessage(ServerStatePayload(color=color_update)))
        if self._pending_update is not None:
            role.send_message(ServerStateMessage(ServerStatePayload(color=self._pending_update)))

    def set_color(self, color: Color | None, *, timestamp_us: int | None = None) -> None:
        """Set or schedule a color palette and push it to subscribed roles."""
        timestamp = (
            self._group._server.clock.now_us()  # noqa: SLF001
            if timestamp_us is None
            else timestamp_us
        )
        if color is not None:
            if timestamp_us is not None:
                color = replace(color, timestamp_us=timestamp_us)
            elif color.timestamp_us is None:
                color = replace(color, timestamp_us=timestamp)
            else:
                timestamp = color.timestamp_us

        had_pending = self._pending_update is not None
        if self._pending_update is not None:
            pending_timestamp = self._pending_update.timestamp
            if timestamp >= pending_timestamp:
                self._current_color = self._pending_color
            self._pending_color = None
            self._pending_update = None

        if not had_pending and color == self._current_color:
            return

        last_color = self._current_color
        if color is None:
            color_update = SessionUpdateColor.cleared(timestamp)
        else:
            color_update = color.diff_update(last_color, timestamp)

        if timestamp > self._group._server.clock.now_us():  # noqa: SLF001
            self._pending_color = color
            self._pending_update = color_update
        else:
            self._current_color = color

        for role in self._members:
            state_message = ServerStateMessage(ServerStatePayload(color=color_update))
            role.send_message(state_message)

        if color is None:
            self.emit_group_event(
                ColorClearedEvent(previous_color=last_color, timestamp_us=timestamp)
            )
            return
        self.emit_group_event(
            ColorUpdatedEvent(
                color=color,
                previous_color=last_color,
                timestamp_us=timestamp,
            )
        )

    def clear(self, *, timestamp_us: int | None = None) -> None:
        """Clear the color palette."""
        self.set_color(None, timestamp_us=timestamp_us)
