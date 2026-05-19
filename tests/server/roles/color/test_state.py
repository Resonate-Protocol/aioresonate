"""Tests for the Color domain dataclass and its WCAG invariants."""

from __future__ import annotations

import pytest

from aiosendspin.server.roles.color.state import Color


def test_empty_color_is_valid() -> None:
    """All-None Color passes (no constraints triggered)."""
    Color()


def test_primary_and_accent_alone_are_not_contrast_checked() -> None:
    """primary/accent have no spec contrast requirement."""
    Color(primary=(255, 0, 0), accent=(0, 255, 0))


def test_background_dark_fails_against_white_text() -> None:
    """A near-white background_dark cannot clear 4.5:1 vs white text."""
    with pytest.raises(ValueError, match="background_dark vs white text"):
        Color(background_dark=(240, 240, 240))


def test_background_light_fails_against_black_text() -> None:
    """A near-black background_light cannot clear 4.5:1 vs black text."""
    with pytest.raises(ValueError, match="background_light vs black text"):
        Color(background_light=(20, 20, 20))


def test_on_dark_fails_against_black_text() -> None:
    """on_dark must also serve as a light bg (4.5:1 vs black)."""
    with pytest.raises(ValueError, match="on_dark vs black text"):
        Color(on_dark=(50, 50, 50))


def test_on_light_fails_against_white_text() -> None:
    """on_light must also serve as a dark bg (4.5:1 vs white)."""
    with pytest.raises(ValueError, match="on_light vs white text"):
        Color(on_light=(220, 220, 220))


def test_background_dark_and_on_dark_pair_must_clear_min_contrast() -> None:
    """background_dark vs on_dark must clear 4.5:1 when both set."""
    with pytest.raises(ValueError, match="background_dark vs on_dark"):
        Color(background_dark=(0, 0, 0), on_dark=(40, 40, 40))


def test_valid_full_palette_passes() -> None:
    """A palette satisfying all spec contrast rules constructs cleanly."""
    Color(
        background_dark=(0, 0, 0),
        background_light=(255, 255, 255),
        primary=(180, 30, 30),
        accent=(30, 180, 30),
        on_dark=(255, 255, 255),
        on_light=(0, 0, 0),
    )


def test_rgb_wrong_length_raises() -> None:
    """Color rejects tuples that are not exactly 3 components."""
    with pytest.raises(ValueError, match=r"primary must be \(R, G, B\)"):
        Color(primary=(1, 2, 3, 4))  # type: ignore[arg-type]


def test_rgb_component_out_of_range_raises() -> None:
    """Color rejects component values outside 0-255."""
    with pytest.raises(ValueError, match="primary values must be 0-255"):
        Color(primary=(300, 0, 0))
