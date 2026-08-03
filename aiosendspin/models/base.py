"""
Shared serialization base for Sendspin protocol models.

The spec types most numeric wire fields as `integer`, and a strict client rejects a field
whose JSON type does not match: sendspin-cpp logs `Ignoring field 'track_duration':
expected integer in [0, 4294967295]` and drops the field entirely. Python does not enforce
`int` annotations at runtime and mashumaro passes primitives through unchanged, so a float
handed to a model by a caller would otherwise reach the wire verbatim as `217000.0`.

`SendspinConfig` closes that gap for every model that uses it, by coercing `int`-typed
fields as they are serialized. Deserialization already coerces through mashumaro's built-in
`int()` handling, so only the outbound direction needs this.
"""

from __future__ import annotations

import math
import operator
from typing import Any

from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin


def int_to_wire(value: Any) -> int:
    """
    Coerce a value annotated as `int` into a real `int` for the wire.

    Exact integers pass through by value, including `bool`, `IntEnum`, and anything else
    implementing `__index__` (NumPy integer scalars, for instance). Floats are rounded to
    the nearest integer: every integer field in the protocol carries a quantized unit
    (milliseconds, microseconds, hertz, counts), so a fractional part is finer than the
    wire can represent. Rounding rather than truncating keeps float-arithmetic artifacts
    such as `2.9 * 1000 == 2899.9999999999995` from losing a whole unit.

    Raises:
        TypeError: If the value is not a number at all.
        ValueError: If the value is a NaN or an infinity, which have no integer form.
    """
    try:
        return operator.index(value)
    except TypeError:
        pass

    if isinstance(value, float) or hasattr(value, "__float__"):
        as_float = float(value)
        if not math.isfinite(as_float):
            msg = f"cannot serialize non-finite value {value!r} as an integer"
            raise ValueError(msg)
        return round(as_float)

    msg = f"expected an integer, got {type(value).__name__}: {value!r}"
    raise TypeError(msg)


class SendspinConfig(BaseConfig):
    """
    Base mashumaro config for every Sendspin model.

    A model's `Config` must derive from this rather than from `BaseConfig` directly:
    mashumaro resolves `Config` by lookup, so a `Config(BaseConfig)` on a subclass
    replaces this one outright and silently drops the integer coercion. The
    `test_every_json_model_uses_sendspin_config` test in
    `tests/models/test_wire_int_coercion.py` checks every model for this.
    """

    serialization_strategy = {int: {"serialize": int_to_wire}}  # noqa: RUF012


class SendspinModel(DataClassORJSONMixin):
    """Base class for Sendspin protocol models. Applies `SendspinConfig` by default."""

    class Config(SendspinConfig):
        """Config for parsing json messages."""
