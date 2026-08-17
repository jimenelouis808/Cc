"""Import every Blender-facing module against a fake ``bpy``.

This does not prove the node graphs are correct — only Blender can do that —
but it does catch the errors that are otherwise only discovered by installing
the add-on: typos at class-body level, wrong imports, property declarations
that would not survive registration, and stale references between modules.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from atomviz_studio.tests import bpy_stub

BLENDER_MODULES = [
    "atomviz_studio.core.compat",
    "atomviz_studio.core.nodes",
    "atomviz_studio.core.detect",
    "atomviz_studio.materials.styles",
    "atomviz_studio.materials.apply",
    "atomviz_studio.effects.backgrounds",
    "atomviz_studio.effects.lighting",
    "atomviz_studio.effects.electricity",
    "atomviz_studio.effects.lasers",
    "atomviz_studio.effects.postfx",
    "atomviz_studio.scene.camera",
    "atomviz_studio.scene.render",
    "atomviz_studio.looks.apply",
    "atomviz_studio.ui.props",
    "atomviz_studio.ui.operators",
    "atomviz_studio.ui.panels",
    "atomviz_studio.cli.render_cover",
]


@pytest.fixture()
def stubbed_bpy():
    """Install the fake bpy for the duration of one test, then clean up."""
    keys = bpy_stub.install()
    imported_before = {name for name in sys.modules if name.startswith("atomviz_studio")}
    yield
    for name in [n for n in sys.modules if n.startswith("atomviz_studio")]:
        if name not in imported_before:
            del sys.modules[name]
    bpy_stub.uninstall(keys)


@pytest.mark.parametrize("module_name", BLENDER_MODULES)
def test_module_imports(module_name, stubbed_bpy):
    module = importlib.import_module(module_name)
    assert module is not None


def test_registries_are_consistent(stubbed_bpy):
    """Styles, backgrounds and rigs referenced by the looks must exist."""
    from atomviz_studio.core.presets import LOOKS
    from atomviz_studio.effects.backgrounds import BACKGROUNDS
    from atomviz_studio.effects.lighting import LIGHT_RIGS
    from atomviz_studio.materials.styles import STYLES

    for look in LOOKS.values():
        assert look.style in STYLES, f"{look.key}: unknown style {look.style}"
        assert look.stick_style in STYLES, f"{look.key}: unknown bond style {look.stick_style}"
        assert look.background in BACKGROUNDS, f"{look.key}: unknown background {look.background}"
        assert look.lighting in LIGHT_RIGS, f"{look.key}: unknown rig {look.lighting}"


def test_style_params_merge(stubbed_bpy):
    from atomviz_studio.materials.styles import StyleParams

    params = StyleParams().merged({"roughness": 0.9, "emissive_dopants": 4.0, "unknown": 2.0})
    assert params.roughness == 0.9
    assert params.emissive_dopants == 4.0
    assert params.extras["unknown"] == 2.0
    # The original stays untouched.
    assert StyleParams().roughness != 0.9


def test_operator_and_panel_ids_are_unique(stubbed_bpy):
    from atomviz_studio.ui.operators import CLASSES as operator_classes
    from atomviz_studio.ui.panels import CLASSES as panel_classes

    ids = [cls.bl_idname for cls in operator_classes]
    assert len(ids) == len(set(ids))
    assert all(name.startswith("atomviz.") for name in ids)
    panel_ids = [cls.bl_idname for cls in panel_classes]
    assert len(panel_ids) == len(set(panel_ids))


def test_cli_parses_arguments(stubbed_bpy):
    from atomviz_studio.cli.render_cover import parse_args

    args = parse_args(["--xyz", "a.xyz", "--look", "laser_lab", "--out", "b.png", "--seed", "3"])
    assert args.look == "laser_lab"
    assert args.seed == 3
    assert str(args.xyz) == "a.xyz"
    assert parse_args(["--list"]).list is True
