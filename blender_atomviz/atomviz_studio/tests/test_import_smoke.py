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
    saved_bpy = bpy_stub.install()
    # Reload every add-on module against the stub, then restore what was there.
    saved_addon = {
        name: module for name, module in sys.modules.items() if name.startswith("atomviz_studio")
    }
    for name in list(saved_addon):
        if name != "atomviz_studio.tests.bpy_stub" and not name.startswith("atomviz_studio.tests"):
            del sys.modules[name]
    yield
    for name in [n for n in sys.modules if n.startswith("atomviz_studio")]:
        if name not in saved_addon:
            del sys.modules[name]
    sys.modules.update(saved_addon)
    bpy_stub.uninstall(saved_bpy)


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


def test_engine_candidates_cover_every_eevee_rename(stubbed_bpy):
    """The EEVEE identifier moved in 4.2 and moved back in 5.0."""
    from atomviz_studio.core.compat import engine_candidates

    eevee = engine_candidates("EEVEE")
    assert "BLENDER_EEVEE" in eevee and "BLENDER_EEVEE_NEXT" in eevee
    assert engine_candidates("CYCLES")[0] == "CYCLES"
    # Whatever the build offers, some engine is always reachable.
    assert set(engine_candidates("EEVEE")) == set(engine_candidates("CYCLES"))


def test_postfx_picks_a_compositor_api(stubbed_bpy):
    """Blender 5.0 replaced scene.node_tree with a compositing node group."""
    from atomviz_studio.effects import postfx

    assert isinstance(postfx.USES_NODE_GROUP, bool)
    assert set(postfx.GLARE_TYPES) >= {"NONE", "FOG_GLOW", "STREAKS"}
    for name in ("FOG_GLOW", "STREAKS", "GHOSTS", "SIMPLE_STAR"):
        assert name in postfx._GLARE_SOCKET_LABELS
