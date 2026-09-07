"""Static checks on the Tk layer that cannot be exercised without a display.

The widget code in :mod:`ramancarbon.gui.app` is the one part of this
package with no runtime test coverage: opening a window needs a display,
and the environment this was built in has no Tkinter at all. That is a real
gap, and these checks are a partial stand-in for it rather than a
replacement.

They caught two genuine bugs that nothing else would have: a helper
referenced through ``self`` that was never defined (an attribute lookup, so
the linter cannot see it either), and a tab-index-to-canvas map that had
drifted out of step with the order the tabs are actually created in.

What they check is internal consistency of the wiring, by parsing the
source. What they cannot check is whether the window looks right, whether
the layout fits, or whether any callback does something sensible.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "gui" / "app.py"
SOURCE = APP.read_text(encoding="utf-8")


def test_app_module_parses():
    ast.parse(SOURCE)


def test_every_canvas_is_mapped_to_a_tab_and_has_a_drawer():
    """Three lists have to agree, and nothing else keeps them in step."""
    created = set(re.findall(r'_make_canvas\(\w+, "(\w+)"', SOURCE))
    mapped = set(re.findall(r'\("(\w+)",\)', SOURCE))
    drawers = set(re.findall(r'"(\w+)": self\._draw_\w+', SOURCE))
    assert created, "no canvases found; the regex needs updating"
    assert created == mapped, f"canvas/tab mismatch: {created ^ mapped}"
    assert created == drawers, f"canvas/drawer mismatch: {created ^ drawers}"


def test_tab_index_map_matches_the_order_tabs_are_built():
    """The lazy redraw keys off the notebook's tab INDEX, so the map has to
    follow the order the tabs are added in, not the order the methods happen
    to be defined in the file."""
    build_order = re.findall(r"self\._build_tab_(\w+)\(\)", SOURCE)
    mapping = re.search(r"_tab_canvases = \{(.*?)\n        \}", SOURCE, re.S)
    assert mapping, "the tab map is missing"

    entries = re.findall(r"^\s*(\d+):\s*\(([^)]*)\)", mapping.group(1), re.M)
    indices = [int(n) for n, _ in entries]
    assert indices == list(range(len(build_order))), (
        f"the map covers {indices} but {len(build_order)} tabs are built"
    )

    # And the mapping has to name the canvas that tab actually creates. The
    # count alone passed while a rename left index 3 pointing at the wrong
    # panel, so check each builder's body for its _make_canvas call.
    for index, (_, canvases) in enumerate(entries):
        names = re.findall(r'"(\w+)"', canvases)
        builder = build_order[index]
        body = re.search(
            rf"def _build_tab_{builder}\(self\).*?(?=\n    def )", SOURCE, re.S
        )
        assert body, f"cannot find _build_tab_{builder}"
        created = re.findall(r'_make_canvas\(\w+, "(\w+)"', body.group(0))
        assert sorted(names) == sorted(created), (
            f"tab {index} ({builder}) creates {created} but the map says {names}"
        )


def test_every_self_attribute_used_is_assigned_somewhere():
    """Catches the class of bug a linter misses: ``self.something`` that is
    read but never set. Attribute access is invisible to static name
    resolution, so nothing else would flag it."""
    tree = ast.parse(SOURCE)
    app = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "RamanCarbonApp"
    )
    assigned: set[str] = set()
    read: set[str] = set()
    for node in ast.walk(app):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id != "self":
                continue
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                assigned.add(node.attr)
            else:
                read.add(node.attr)
    defined = {n.name for n in app.body if isinstance(n, (ast.FunctionDef,))}
    missing = read - assigned - defined
    assert not missing, f"read but never assigned: {sorted(missing)}"


def test_callbacks_named_in_commands_exist():
    """Every ``command=self._x`` has to resolve to a real method."""
    tree = ast.parse(SOURCE)
    app = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "RamanCarbonApp"
    )
    methods = {n.name for n in app.body if isinstance(n, ast.FunctionDef)}
    referenced = set(re.findall(r"command=self\.(\w+)", SOURCE))
    missing = referenced - methods
    assert not missing, f"commands with no method: {sorted(missing)}"


@pytest.mark.parametrize(
    "feature",
    [
        "_auto_preprocess",   # automatic preprocessing
        "_export_fit",        # deconvolution export
        "_export_all",        # full export
        "_combine_lasers",    # multi-wavelength
        "profile_var",        # lineshape selector
        "basis_var",          # area/height selector
        "indices_text",       # structural indices panel
        "interference_var",   # non-carbon scan, opt-in
        "_calibrate",         # silicon-line calibration
        "_bootstrap",         # resampled uncertainties
        "batch_text",         # batch statistics panel
        "_analyse_tmd",       # dichalcogenide analysis
        "tmd_text",           # TMD panel
        "n_d_var",            # configurable peaks per region
        "n_g_var",
        "_load_region_model",
    ],
)
def test_requested_features_are_wired_into_the_window(feature):
    """Each of these corresponds to something the interface must expose;
    a refactor that drops the widget should fail here."""
    assert feature in SOURCE


def test_interference_scan_defaults_to_off_in_the_interface():
    """It is opt-in in the analysis, so the checkbox must start unticked or
    the default would differ between the GUI and everything else."""
    assert 'self.interference_var = self.tk.BooleanVar(value=False)' in SOURCE


def test_normalisation_offers_0_100():
    from ramancarbon.gui.state import NORMALISATIONS

    assert any(key == "0-100" for key, _ in NORMALISATIONS)


def test_profile_choices_cover_pseudo_voigt_and_gaussian():
    from ramancarbon.gui.state import DECONVOLUTION_PROFILES

    keys = {key for key, _ in DECONVOLUTION_PROFILES}
    assert {"pseudo_voigt", "gaussian"} <= keys
    assert "" in keys  # "use the database defaults"
