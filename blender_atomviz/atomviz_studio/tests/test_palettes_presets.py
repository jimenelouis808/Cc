"""Palette resolution and preset catalogue integrity."""

from __future__ import annotations

import pytest

from atomviz_studio.core.colors import hex_to_srgb
from atomviz_studio.core.palettes import (
    DEFAULT_PALETTE,
    PALETTES,
    list_palettes,
    resolve_palette,
)
from atomviz_studio.core.presets import (
    DEFAULT_LOOK,
    DEFAULT_RENDER_PRESET,
    LOOKS,
    RENDER_PRESETS,
    get_look,
    list_looks,
    list_render_presets,
)

SYMBOLS = ["C", "H", "N", "B", "S", "Au"]


@pytest.mark.parametrize("key", sorted(PALETTES))
def test_every_palette_resolves_every_symbol(key):
    resolved = resolve_palette(key, SYMBOLS)
    assert set(resolved) == set(SYMBOLS)
    for value in resolved.values():
        hex_to_srgb(value)  # raises when malformed


def test_unknown_palette_falls_back_to_default():
    assert resolve_palette("does_not_exist", ["C"]) == resolve_palette(DEFAULT_PALETTE, ["C"])


def test_accent_palette_keeps_framework_neutral():
    resolved = resolve_palette("mono_accent", ["C", "H", "N", "B"], accent="#ff0000")
    # Heteroatoms are variants of the accent, the lightest element first.
    assert resolved["B"] == "#ff0000"
    assert resolved["C"] != resolved["N"]
    # Two different heteroatoms must not collide.
    assert resolved["N"] != resolved["B"]


def test_spectral_palette_is_monotonic_in_z():
    resolved = resolve_palette("spectral", ["H", "C", "Au"])
    assert len({resolved["H"], resolved["C"], resolved["Au"]}) == 3


def test_overrides_win():
    resolved = resolve_palette("cpk", ["C", "N"], overrides={"N": "#123456"})
    assert resolved["N"] == "#123456"
    # An override for an absent element is ignored rather than injected.
    assert "O" not in resolve_palette("cpk", ["C"], overrides={"O": "#123456"})


def test_palette_enum_items_are_unique():
    keys = [key for key, _, _ in list_palettes()]
    assert len(keys) == len(set(keys))


def test_render_presets_are_sane():
    assert DEFAULT_RENDER_PRESET in RENDER_PRESETS
    for key, preset in RENDER_PRESETS.items():
        assert preset.key == key
        assert preset.width > 0 and preset.height > 0
        assert preset.engine in {"EEVEE", "CYCLES"}
        assert preset.samples > 0
        assert preset.aspect == pytest.approx(preset.width / preset.height)
    assert len({k for k, _, _ in list_render_presets()}) == len(RENDER_PRESETS)


def test_looks_reference_existing_registries():
    from atomviz_studio.core.palettes import PALETTES as palettes

    assert DEFAULT_LOOK in LOOKS
    for key, look in LOOKS.items():
        assert look.key == key
        assert look.palette in palettes
        assert look.render_preset in RENDER_PRESETS
        hex_to_srgb(look.accent)
        hex_to_srgb(look.stick_color)
        for name in ("top", "bottom", "accent"):
            if name in look.background_params:
                hex_to_srgb(str(look.background_params[name]))
    assert len({k for k, _, _ in list_looks()}) == len(LOOKS)


def test_get_look_falls_back():
    assert get_look("nope").key == DEFAULT_LOOK


def test_look_effect_specs_have_expected_types():
    for look in LOOKS.values():
        if look.electricity:
            assert int(look.electricity["count"]) > 0
            hex_to_srgb(str(look.electricity["color"]))
        if look.lasers:
            assert int(look.lasers["count"]) > 0
            hex_to_srgb(str(look.lasers["color"]))
