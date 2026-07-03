"""Client availability: the `available` boolean and legacy `state` wire compat.

New clients send the spec's `available` boolean. Legacy clients send a `state`
enum, which the server normalizes to `available` at deserialization time. The
payload itself has no `state` field.
"""

from __future__ import annotations

import pytest

from aiosendspin.models.core import ClientStatePayload


@pytest.mark.parametrize(
    ("state", "expected_available"),
    [
        ("synchronized", True),
        ("external_source", False),
        ("error", True),
    ],
)
def test_legacy_state_on_wire_normalizes_to_available(
    state: str,
    expected_available: bool,  # noqa: FBT001
) -> None:
    """A legacy client's `state` enum is translated to `available` on deserialization."""
    payload = ClientStatePayload.from_dict({"state": state})
    assert payload.available is expected_available


def test_new_available_wins_over_legacy_state_on_wire() -> None:
    """When both are present on the wire, `available` is authoritative."""
    payload = ClientStatePayload.from_dict({"available": True, "state": "external_source"})
    assert payload.available is True
