"""
Color messages for the Sendspin protocol.

This module contains messages specific to clients with the color role, which
receive color palettes derived from the current audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin

from .types import UndefinedField, undefined_field

_RGB = tuple[int, int, int]
_RGB_LEN = 3


def _validate_rgb(name: str, value: _RGB) -> None:
    if len(value) != _RGB_LEN:
        raise ValueError(f"{name} must be (R, G, B) (length 3), got length {len(value)}")
    for component in value:
        if not (0 <= component <= 255):
            raise ValueError(f"{name} values must be 0-255, got {component}")


# Server -> Client: server/state color object
@dataclass
class SessionUpdateColor(DataClassORJSONMixin):
    """Color object in server/state message."""

    _RGB_FIELDS: ClassVar[tuple[str, ...]] = (
        "background_dark",
        "background_light",
        "primary",
        "accent",
        "on_dark",
        "on_light",
    )

    timestamp: int
    """Server clock time in microseconds for when these colors are valid."""
    background_dark: _RGB | None | UndefinedField = field(default_factory=undefined_field)
    """Background color for dark mode as `(R, G, B)`. Null clears the field."""
    background_light: _RGB | None | UndefinedField = field(default_factory=undefined_field)
    """Background color for light mode as `(R, G, B)`. Null clears the field."""
    primary: _RGB | None | UndefinedField = field(default_factory=undefined_field)
    """Dominant color as `(R, G, B)`. Null clears the field."""
    accent: _RGB | None | UndefinedField = field(default_factory=undefined_field)
    """Secondary or complementary color as `(R, G, B)`. Null clears the field."""
    on_dark: _RGB | None | UndefinedField = field(default_factory=undefined_field)
    """Light color for use on dark backgrounds as `(R, G, B)`. Null clears the field."""
    on_light: _RGB | None | UndefinedField = field(default_factory=undefined_field)
    """Dark color for use on light backgrounds as `(R, G, B)`. Null clears the field."""

    def __post_init__(self) -> None:
        """Validate RGB fields."""
        for name in self._RGB_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, UndefinedField) and value is not None:
                _validate_rgb(name, value)

    @classmethod
    def cleared(cls, timestamp: int) -> SessionUpdateColor:
        """Build a SessionUpdateColor that explicitly clears all color fields."""
        return cls(timestamp=timestamp, **dict.fromkeys(cls._RGB_FIELDS))

    class Config(BaseConfig):
        """Config for parsing json messages."""

        omit_default = True
