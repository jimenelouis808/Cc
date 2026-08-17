"""Element resolution and colour conversion."""

from __future__ import annotations

import pytest

from atomviz_studio.core import colors
from atomviz_studio.core.elements import (
    ELEMENTS,
    element_color,
    element_label,
    guess_element,
    is_stick_name,
    sort_symbols,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        # Atomic Blender object names.
        ("Carbon", "C"),
        ("Carbon_mesh", "C"),
        ("Carbon_ball.001", "C"),
        ("Nitrogen_mesh.014", "N"),
        ("Iron_F2+", "Fe"),
        # Hand-built / other tools.
        ("C", "C"),
        ("N.003", "N"),
        ("atom_S", "S"),
        ("Au_sphere", "Au"),
        ("boron", "B"),
        ("Aluminum", "Al"),
        ("Sulphur", "S"),
    ],
)
def test_guess_element_resolves_known_names(name, expected):
    assert guess_element(name) == expected


@pytest.mark.parametrize(
    "name",
    ["Sticks", "Sticks_Cylinder", "bond_01", "Camera", "Cube", "Vacancy", "AV_Haze"],
)
def test_guess_element_rejects_non_atoms(name):
    assert guess_element(name) is None


def test_full_names_beat_symbols():
    """'Carbon' must never resolve to calcium, 'Nitrogen' never to nickel."""
    assert guess_element("Carbon") == "C"
    assert guess_element("Nitrogen") == "N"
    assert guess_element("Cobalt") == "Co"


def test_stick_detection():
    assert is_stick_name("Sticks")
    assert is_stick_name("Carbon_sticks")
    assert not is_stick_name("Carbon_mesh")


def test_sort_symbols_orders_by_atomic_number():
    assert sort_symbols({"O", "H", "Fe", "C"}) == ["H", "C", "O", "Fe"]
    # Unknown symbols end up last instead of raising.
    assert sort_symbols({"X", "C"})[-1] == "X"


def test_element_helpers():
    assert element_color("C") == ELEMENTS["C"].cpk
    assert element_color("Zz") == "#b0b0b0"
    assert element_label("N").startswith("N - Nitrogen")


def test_hex_roundtrip():
    for value in ("#000000", "#ffffff", "#22d3ee", "#f472b6"):
        assert colors.linear_to_hex(colors.hex_to_linear(value)) == value


def test_hex_to_linear_is_darker_than_srgb():
    """Mid grey must be ~0.21 linear, not 0.5 — this is the classic bug."""
    r, _, _, alpha = colors.hex_to_linear("#808080")
    assert alpha == 1.0
    assert 0.2 < r < 0.23


def test_invalid_hex_raises():
    with pytest.raises(ValueError):
        colors.hex_to_linear("#12345")


def test_mix_lighten_darken():
    assert colors.mix_hex("#000000", "#ffffff", 0.0) == "#000000"
    assert colors.mix_hex("#000000", "#ffffff", 1.0) == "#ffffff"
    assert colors.lighten("#000000", 1.0) == "#ffffff"
    assert colors.darken("#ffffff", 1.0) == "#000000"
    assert colors.luminance("#ffffff") == pytest.approx(1.0, abs=1e-6)
    assert colors.luminance("#000000") == pytest.approx(0.0, abs=1e-6)
