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
from aiosendspin.server.roles.scheduled_state import ScheduledRoleState

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
        self._state: ScheduledRoleState[Color, SessionUpdateColor] = ScheduledRoleState()

    @property
    def color(self) -> Color | None:
        """Return current color palette."""
        return self._state.current(self._now_us())

    def on_member_join(self, role: Role) -> None:
        """Send current color to newly joined member."""
        self._send_state_to_role(role)

    def _send_state_to_role(self, role: ColorRoleProtocol) -> None:
        """Send current color state to a single role."""
        timestamp = self._now_us()
        current = self._state.current(timestamp)
        if current is not None:
            color_update = current.snapshot_update(timestamp)
        else:
            color_update = SessionUpdateColor.cleared(timestamp)
        role.send_message(ServerStateMessage(ServerStatePayload(color=color_update)))
        if (pending_update := self._state.pending_update) is not None:
            role.send_message(ServerStateMessage(ServerStatePayload(color=pending_update)))

    def set_color(self, color: Color | None, *, timestamp_us: int | None = None) -> None:
        """Set or schedule a color palette and push it to subscribed roles."""
        now_us = self._now_us()
        current = self._state.current(now_us)
        timestamp = now_us if timestamp_us is None else timestamp_us
        if color is not None:
            if timestamp_us is not None:
                color = replace(color, timestamp_us=timestamp_us)
            elif color.timestamp_us is None:
                color = replace(color, timestamp_us=timestamp)
            else:
                timestamp = color.timestamp_us

        if not self._state.has_pending and not self._state.scheduled_fields and color == current:
            return

        last_color = current
        if color is None:
            color_update = SessionUpdateColor.cleared(timestamp)
        else:
            color_update = color.diff_update(
                last_color, timestamp, include=self._state.scheduled_fields
            )

        if timestamp > now_us:
            self._state.schedule(
                color, color_update, timestamp, set(color_update.to_dict()) - {"timestamp"}
            )
        else:
            self._state.apply(color, timestamp)

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
