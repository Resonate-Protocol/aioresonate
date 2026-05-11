"""Color state for the Sendspin protocol."""

from __future__ import annotations

from dataclasses import dataclass

from aiosendspin.models.color import SessionUpdateColor


@dataclass
class Color:
    """Color palette for the current audio."""

    background_dark: list[int] | None = None
    """Background color for dark mode as [R, G, B].

    Caller must ensure a WCAG contrast ratio of at least 4.5:1 with white text
    and with `on_dark` (if also set).
    """
    background_light: list[int] | None = None
    """Background color for light mode as [R, G, B].

    Caller must ensure a WCAG contrast ratio of at least 4.5:1 with black text
    and with `on_light` (if also set).
    """
    primary: list[int] | None = None
    """Dominant color as [R, G, B]. Not adjusted for contrast."""
    accent: list[int] | None = None
    """Secondary or complementary color as [R, G, B]. Not adjusted for contrast."""
    on_dark: list[int] | None = None
    """Light color for use on dark backgrounds as [R, G, B].

    Caller must ensure a WCAG contrast ratio of at least 4.5:1 with `background_dark`
    (if also set).
    """
    on_light: list[int] | None = None
    """Dark color for use on light backgrounds as [R, G, B].

    Caller must ensure a WCAG contrast ratio of at least 4.5:1 with `background_light`
    (if also set).
    """

    def equals(self, other: Color | None) -> bool:
        """Check if color palette is equal to another."""
        if other is None:
            return False
        return (
            self.background_dark == other.background_dark
            and self.background_light == other.background_light
            and self.primary == other.primary
            and self.accent == other.accent
            and self.on_dark == other.on_dark
            and self.on_light == other.on_light
        )

    def diff_update(self, last: Color | None, timestamp: int) -> SessionUpdateColor:
        """Build a SessionUpdateColor containing only changed fields compared to last."""
        update = SessionUpdateColor(timestamp=timestamp)
        if last is None or last.background_dark != self.background_dark:
            update.background_dark = self.background_dark
        if last is None or last.background_light != self.background_light:
            update.background_light = self.background_light
        if last is None or last.primary != self.primary:
            update.primary = self.primary
        if last is None or last.accent != self.accent:
            update.accent = self.accent
        if last is None or last.on_dark != self.on_dark:
            update.on_dark = self.on_dark
        if last is None or last.on_light != self.on_light:
            update.on_light = self.on_light
        return update

    def snapshot_update(self, timestamp: int) -> SessionUpdateColor:
        """Build a SessionUpdateColor snapshot with all current values."""
        update = SessionUpdateColor(timestamp=timestamp)
        update.background_dark = self.background_dark
        update.background_light = self.background_light
        update.primary = self.primary
        update.accent = self.accent
        update.on_dark = self.on_dark
        update.on_light = self.on_light
        return update

    @staticmethod
    def cleared_update(timestamp: int) -> SessionUpdateColor:
        """Build a SessionUpdateColor that clears all color fields."""
        update = SessionUpdateColor(timestamp=timestamp)
        update.background_dark = None
        update.background_light = None
        update.primary = None
        update.accent = None
        update.on_dark = None
        update.on_light = None
        return update
