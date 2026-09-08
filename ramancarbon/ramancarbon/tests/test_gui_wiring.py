"""Static checks on the Tk layer, and a real smoke test where Tk exists.

The widget code is the one part of this package that cannot be exercised
by ordinary unit tests: opening a window needs a display. Two things stand
in for that.

:class:`SECTIONS` below is parsed rather than imported, and the checks are
of internal consistency: every canvas is drawn by something, every tab
index maps to the canvases that tab really creates, every ``command=`` names
a method that exists, every ``self.x`` that is read is assigned somewhere.
Those caught two genuine bugs nothing else would have — a helper referenced
through ``self`` that was never defined, which is an attribute lookup and
so invisible to a linter, and a tab-index map that had drifted out of step
with the order the tabs are built in.

:func:`test_the_whole_suite_opens_and_every_section_works` is the real
thing, and it is skipped when Tkinter is missing. It builds the window,
visits all four sections, loads their demo data, runs their analyses and
walks every tab.

What none of this checks is whether the window *looks* right.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

GUI = Path(__file__).resolve().parents[1] / "gui"

#: ``module stem -> class name`` for every section of the suite.
SECTION_MODULES = {
    "app": "RamanCarbonApp",
    "tmd_app": "TMDApp",
    "xrd_app": "XRDApp",
    "echem_app": "EchemApp",
}


def source(stem: str) -> str:
    return (GUI / f"{stem}.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("stem", sorted(SECTION_MODULES) + ["suite", "base",
                                                            "xrd_state",
                                                            "echem_state",
                                                            "plots_xrd",
                                                            "plots_echem"])
def test_every_gui_module_parses(stem):
    ast.parse(source(stem))


# -- internal consistency ---------------------------------------------


def test_carbon_section_canvases_are_mapped_and_drawn():
    text = source("app")
    created = set(re.findall(r'_make_canvas\(\w+, "(\w+)"', text))
    mapped = set(re.findall(r'\("(\w+)",\)', text))
    drawers = set(re.findall(r'"(\w+)": self\._draw_\w+', text))
    assert created, "no canvases found; the regex needs updating"
    assert created == mapped, f"canvas/tab mismatch: {created ^ mapped}"
    assert created == drawers, f"canvas/drawer mismatch: {created ^ drawers}"


@pytest.mark.parametrize("stem", ["xrd_app", "echem_app"])
def test_new_sections_map_every_canvas_to_a_drawer(stem):
    text = source(stem)
    created = set(re.findall(r'make_canvas\(\w+, "(\w+)"', text))
    drawers = set(re.findall(r'"(\w+)": self\._draw_\w+', text))
    mapped = set(re.findall(r'"(\w+)"', re.search(
        r"_tab_canvases = \{(.*?)\n        \}", text, re.S).group(1)))
    assert created == drawers, f"{stem}: canvas/drawer mismatch {created ^ drawers}"
    assert created == mapped, f"{stem}: canvas/tab mismatch {created ^ mapped}"


def test_carbon_tab_index_map_matches_the_build_order():
    """The lazy redraw keys off the notebook's tab INDEX, so the map has to
    follow the order the tabs are added in, not the order the methods are
    defined in the file."""
    text = source("app")
    build_order = re.findall(r"self\._build_tab_(\w+)\(\)", text)
    mapping = re.search(r"_tab_canvases = \{(.*?)\n        \}", text, re.S)
    assert mapping, "the tab map is missing"
    entries = re.findall(r"^\s*(\d+):\s*\(([^)]*)\)", mapping.group(1), re.M)
    assert [int(n) for n, _ in entries] == list(range(len(build_order)))
    for index, (_, canvases) in enumerate(entries):
        names = re.findall(r'"(\w+)"', canvases)
        body = re.search(
            rf"def _build_tab_{build_order[index]}\(self\).*?(?=\n    def )",
            text, re.S,
        )
        assert body, f"cannot find _build_tab_{build_order[index]}"
        created = re.findall(r'_make_canvas\(\w+, "(\w+)"', body.group(0))
        assert sorted(names) == sorted(created), (
            f"tab {index} creates {created} but the map says {names}"
        )


@pytest.mark.parametrize("stem", ["xrd_app", "echem_app"])
def test_new_section_tab_maps_cover_every_tab(stem):
    text = source(stem)
    added = len(re.findall(r"self\.notebook\.add\(", text))
    mapping = re.search(r"_tab_canvases = \{(.*?)\n        \}", text, re.S)
    indices = [int(n) for n in re.findall(r"^\s*(\d+):", mapping.group(1), re.M)]
    assert indices == list(range(added)), (
        f"{stem}: {added} tabs are added but the map covers {indices}"
    )


@pytest.mark.parametrize("stem, name", sorted(SECTION_MODULES.items()))
def test_every_self_attribute_used_is_assigned_somewhere(stem, name):
    """Catches the bug a linter misses: ``self.something`` read but never
    set. Attribute access is invisible to static name resolution."""
    tree = ast.parse(source(stem))
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    assigned: set[str] = set()
    read: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            if child.value.id != "self":
                continue
            if isinstance(child.ctx, (ast.Store, ast.Del)):
                assigned.add(child.attr)
            else:
                read.add(child.attr)
    defined = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
    inherited = {
        "tk", "ttk", "root", "container", "palette", "fonts", "queue", "busy",
        "status_var", "progress", "make_canvas", "with_style", "mark_dirty",
        "flush_dirty", "run_async", "drain_queue", "build_status", "set_status",
        "flush_messages", "warn", "show_error",
    }
    missing = read - assigned - defined - inherited
    assert not missing, f"{stem}: read but never assigned: {sorted(missing)}"


@pytest.mark.parametrize("stem, name", sorted(SECTION_MODULES.items()))
def test_callbacks_named_in_commands_exist(stem, name):
    tree = ast.parse(source(stem))
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
    referenced = set(re.findall(r"command=self\.(\w+)", source(stem)))
    missing = referenced - methods
    assert not missing, f"{stem}: commands with no method: {sorted(missing)}"


# -- the four sections exist and are wired ----------------------------


def test_the_suite_declares_four_sections():
    from ramancarbon.gui.suite import SECTIONS

    keys = [key for key, _, _ in SECTIONS]
    assert keys == ["carbono", "tmd", "drx", "echem"]


def test_the_suite_builds_each_section_lazily():
    """Constructing all four at start imports matplotlib, builds a dozen
    figures and reads the reference library before the window appears."""
    text = source("suite")
    assert "_ensure" in text
    for key in ("carbono", "tmd", "drx", "echem"):
        assert f'key == "{key}"' in text or f'"{key}"' in text


def test_the_two_raman_sections_share_one_session():
    text = source("suite")
    assert "session=self.session" in text          # carbon section
    tmd = re.search(r"TMDApp\(([^)]*)\)", text, re.S)
    assert tmd and "self.session" in tmd.group(1)  # dichalcogenide section


@pytest.mark.parametrize(
    "stem, feature",
    [
        ("app", "_auto_preprocess"),
        ("app", "_export_fit"),
        ("app", "_export_all"),
        ("app", "_combine_lasers"),
        ("app", "profile_var"),
        ("app", "basis_var"),
        ("app", "indices_text"),
        ("app", "interference_var"),
        ("app", "_calibrate"),
        ("app", "_bootstrap"),
        ("app", "batch_text"),
        ("app", "n_d_var"),
        ("app", "n_g_var"),
        ("app", "_load_region_model"),
        ("tmd_app", "_analyse"),
        ("tmd_app", "tmd_text"),
        ("tmd_app", "oxide_text"),
        ("tmd_app", "material_var"),
        ("xrd_app", "_identify"),
        ("xrd_app", "_auto_refine"),
        ("xrd_app", "_prepare_manual"),
        ("xrd_app", "_refine_once"),
        ("xrd_app", "parameter_table"),
        ("xrd_app", "_add_cif_directory"),
        ("xrd_app", "texture_var"),
        ("xrd_app", "kalpha2_var"),
        ("echem_app", "_analyse"),
        ("echem_app", "mass_var"),
        ("echem_app", "reference_var"),
        ("echem_app", "ph_var"),
        ("echem_app", "resistance_var"),
        ("echem_app", "circuit_var"),
        ("echem_app", "reaction_var"),
        ("echem_app", "summary_table"),
        ("echem_app", "normalise_var"),
    ],
)
def test_requested_features_are_wired_into_the_window(stem, feature):
    assert feature in source(stem)


def test_interference_scan_defaults_to_off_in_the_interface():
    assert 'self.interference_var = self.tk.BooleanVar(value=False)' in source("app")


def test_the_non_faradaic_claim_defaults_to_off():
    """Most windows are not free of faradaic current, and claiming
    otherwise turns a C_dl that includes reaction into an ECSA."""
    assert 'self.non_faradaic_var = tk.BooleanVar(value=False)' in source("echem_app")


def test_normalisation_offers_0_100():
    from ramancarbon.gui.state import NORMALISATIONS

    assert any(key == "0-100" for key, _ in NORMALISATIONS)


def test_profile_choices_cover_pseudo_voigt_and_gaussian():
    from ramancarbon.gui.state import DECONVOLUTION_PROFILES

    keys = {key for key, _ in DECONVOLUTION_PROFILES}
    assert {"pseudo_voigt", "gaussian"} <= keys
    assert "" in keys  # "use the database defaults"


def test_axis_parsing_accepts_the_forms_people_type():
    from ramancarbon.gui.xrd_app import _parse_axis

    assert _parse_axis("001") == (0, 0, 1)
    assert _parse_axis("0 0 1") == (0, 0, 1)
    assert _parse_axis("1,0,-1") == (1, 0, -1)
    assert _parse_axis("") is None
    assert _parse_axis("basura") is None


# -- the real thing, where a display exists ---------------------------


def test_the_whole_suite_opens_and_every_section_works():
    """Build the window, visit all four sections, run their analyses.

    Skipped without Tkinter or a display. This is the only test that
    exercises the widget code as code rather than as text.
    """
    tkinter = pytest.importorskip("tkinter")
    import matplotlib

    matplotlib.use("Agg", force=False)
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:  # pragma: no cover - no display
        pytest.skip(f"sin pantalla: {exc}")

    from ramancarbon.gui.suite import SECTIONS, Suite

    try:
        suite = Suite(root)
        for index, (key, _, _) in enumerate(SECTIONS):
            suite.notebook.select(index)
            # update(), not update_idletasks(): selecting a tab fires a
            # VIRTUAL event, and virtual events are not idle tasks. With
            # update_idletasks() the section is never built and the test
            # fails on a window that works perfectly well in practice.
            root.update()
            section = suite.sections.get(key)
            assert section is not None, f"la sección {key} no se construyó"
            section._load_demo()
            root.update()
            if hasattr(section, "notebook"):
                for tab in range(len(section.notebook.tabs())):
                    section.notebook.select(tab)
                    root.update()
    finally:
        root.destroy()
