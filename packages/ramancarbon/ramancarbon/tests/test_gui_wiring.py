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
import inspect
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
    "xps_app": "XPSApp",
}


def source(stem: str) -> str:
    return (GUI / f"{stem}.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("stem", sorted(SECTION_MODULES) + ["suite", "base",
                                                            "xrd_state",
                                                            "echem_state",
                                                            "xps_state",
                                                            "plots_xps",
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


@pytest.mark.parametrize("stem", ["tmd_app", "xrd_app", "echem_app", "xps_app"])
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


@pytest.mark.parametrize("stem", ["tmd_app", "xrd_app", "echem_app", "xps_app"])
def test_new_section_tab_maps_cover_every_tab(stem):
    text = source(stem)
    added = len(re.findall(r"self\.notebook\.add\(", text))
    mapping = re.search(r"_tab_canvases = \{(.*?)\n        \}", text, re.S)
    indices = [int(n) for n in re.findall(r"^\s*(\d+):", mapping.group(1), re.M)]
    assert indices == list(range(added)), (
        f"{stem}: {added} tabs are added but the map covers {indices}"
    )


@pytest.mark.parametrize("stem", ["tmd_app", "xrd_app", "echem_app", "xps_app"])
def test_each_new_section_tab_lists_the_canvases_it_actually_builds(stem):
    """The map is keyed by the notebook's own tab index, so it follows the
    order the tabs are ADDED in, not the order the methods appear in the
    file. Inserting a tab in the middle and forgetting to shift the rest
    does not raise: it draws the wrong figure into the wrong tab, or
    leaves one blank until something else marks it dirty."""
    text = source(stem)
    build_order = re.findall(r"self\._build_tab_(\w+)\(\)", text)
    mapping = re.search(r"_tab_canvases = \{(.*?)\n        \}", text, re.S)
    entries = re.findall(r"^\s*(\d+):\s*\(([^)]*)\)", mapping.group(1), re.M)
    assert len(entries) == len(build_order), (
        f"{stem}: {len(build_order)} tabs built, {len(entries)} mapped")
    for index, (_, canvases) in enumerate(entries):
        listed = re.findall(r'"(\w+)"', canvases)
        body = re.search(
            rf"def _build_tab_{build_order[index]}\(self\).*?(?=\n    def )",
            text, re.S,
        )
        assert body, f"{stem}: cannot find _build_tab_{build_order[index]}"
        created = re.findall(r'make_canvas\(\w+, "(\w+)"', body.group(0))
        assert sorted(listed) == sorted(created), (
            f"{stem} tab {index} ({build_order[index]}) builds {created} "
            f"but the map says {listed}"
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
        "tk", "ttk", "root", "container", "palette", "figure_palette",
        "fonts", "queue", "busy",
        "status_var", "progress", "make_canvas", "with_style", "mark_dirty",
        "flush_dirty", "run_async", "drain_queue", "build_status", "set_status",
        "flush_messages", "warn", "show_error", "report_progress",
        "show_text", "ask_yes_no", "canvas_limits",
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


def test_the_suite_declares_five_sections():
    from ramancarbon.gui.suite import SECTIONS

    keys = [key for key, _, _ in SECTIONS]
    assert keys == ["carbono", "tmd", "drx", "echem", "xps"]


def test_the_suite_builds_each_section_lazily():
    """Constructing all five at start imports matplotlib, builds a dozen
    figures and reads the reference library before the window appears."""
    text = source("suite")
    assert "_ensure" in text
    for key in ("carbono", "tmd", "drx", "echem", "xps"):
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
        # What the sample is made of, and the peak marking that depends
        # on it: the two things that answer "is that 212 line a breathing
        # mode or my iron carbide".
        ("app", "elements_var"),
        ("app", "_explained_positions"),
        # The user's own Raman references: add, remove, and see which
        # entries came with the program and which did not.
        ("app", "phase_table"),
        ("app", "_add_user_phase"),
        ("app", "_remove_user_phase"),
        ("app", "_open_phase_folder"),
        ("app", "polymers_var"),
        ("app", "_reset_zoom"),
        ("tmd_app", "_analyse"),
        ("tmd_app", "tmd_text"),
        ("tmd_app", "oxide_text"),
        ("tmd_app", "material_var"),
        # The dichalcogenide section had one column and no tabs: no
        # preprocessing controls of its own, no way to see the fitted modes
        # as numbers, no export, and no way to read the catalogue it was
        # deciding with. Everything below is one of those.
        ("tmd_app", "_auto_preprocess"),
        ("tmd_app", "baseline_var"),
        ("tmd_app", "smooth_var"),
        ("tmd_app", "modes_table"),
        ("tmd_app", "phases_text"),
        ("tmd_app", "phases_var"),
        ("tmd_app", "_export_table"),
        ("tmd_app", "library_list"),
        ("tmd_app", "_on_library_select"),
        ("xrd_app", "_identify"),
        ("xrd_app", "_auto_refine"),
        ("xrd_app", "_prepare_manual"),
        ("xrd_app", "_refine_once"),
        ("xrd_app", "parameter_table"),
        ("xrd_app", "_add_cif_directory"),
        ("xrd_app", "texture_var"),
        ("xrd_app", "kalpha2_var"),
        # Peak finding on a nanocrystalline pattern. The defaults are
        # wrong for a CVD sample and there was no way to change them.
        ("xrd_app", "smooth_var"),
        ("xrd_app", "background_lambda_var"),
        ("xrd_app", "significance_var"),
        # Rietveld: phases in and out of the MODEL, the numbers it is
        # judged by, and a counter so a fit that returns instantly can be
        # told from one that never started.
        ("xrd_app", "model_phase_combo"),
        ("xrd_app", "_add_model_phase"),
        ("xrd_app", "_remove_model_phase"),
        ("xrd_app", "metrics_table"),
        ("xrd_app", "_queue_progress"),
        ("xrd_app", "_show_refinement_report"),
        ("echem_app", "_analyse"),
        ("echem_app", "mass_var"),
        ("echem_app", "reference_var"),
        ("echem_app", "ph_var"),
        ("echem_app", "resistance_var"),
        ("echem_app", "circuit_var"),
        ("echem_app", "reaction_var"),
        ("echem_app", "summary_table"),
        ("echem_app", "normalise_var"),
        # The cell form: everything a specific capacitance depends on has to
        # be enterable, and R_u can come straight off the impedance.
        ("echem_app", "electrolyte_var"),
        ("echem_app", "volume_var"),
        ("echem_app", "compensated_var"),
        ("echem_app", "_resistance_from_eis"),
        # The XPS section: the choices a high-resolution fit is made of have
        # to be visible and adjustable, not defaults nobody sees.
        ("xps_app", "region_var"),
        ("xps_app", "count_var"),
        ("xps_app", "state_list"),
        ("xps_app", "background_var"),
        ("xps_app", "link_var"),
        ("xps_app", "satellite_var"),
        ("xps_app", "reference_var"),
        ("xps_app", "reference_state_var"),
        ("xps_app", "transmission_var"),
        ("xps_app", "_compare_counts"),
        ("xps_app", "_export_tables"),
        ("xps_app", "_import_model"),
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
    """Build the window, visit all five sections, run their analyses.

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


def test_every_plot_preset_the_header_offers_exists():
    """The chooser is a list of strings; a typo in it would only show up
    when somebody saved a figure with that preset selected."""
    from ramancarbon.gui.suite import PLOT_PRESETS
    from ramancarbon.plotting.style import PRESETS, preset

    assert set(PLOT_PRESETS) <= set(PRESETS)
    for name in PLOT_PRESETS:
        assert preset(name).width_in > 0


def test_the_chosen_preset_is_remembered_and_used():
    """It is stored on the session, saved with the preferences, and read
    back by the figure-saving code."""
    import inspect

    from ramancarbon.gui import app as gui_app
    from ramancarbon.gui import suite as gui_suite
    from ramancarbon.gui.state import Session

    assert "plot_preset" in inspect.getsource(Session.__init__)
    assert "plot_preset" in inspect.getsource(Session.remember)
    assert "plot_preset" in inspect.getsource(gui_suite.Suite._on_preset_changed)
    assert "plot_preset" in inspect.getsource(gui_app.RamanCarbonApp._save_figure)


class TestTheElapsedClock:
    """An indeterminate bar says "something is happening" and no more.

    That is the whole doubt a Rietveld refinement or a batch of fifty
    spectra creates: a slow calculation and a hung one look identical.
    The clock lives on the shared base, so all five sections get it.
    """

    def test_the_clock_reads_as_a_stopwatch(self):
        from ramancarbon.gui.base import _clock

        assert _clock(3.2) == "3 s"
        assert _clock(59.4) == "59 s"
        assert _clock(143.0) == "2:23"
        assert _clock(3725.0) == "62:05"

    def test_a_background_run_starts_and_stops_it(self):
        """Through run_async and drain_queue, the real path."""
        import queue as _queue

        from ramancarbon.gui.base import SectionApp

        class _Root:
            def after(self, _ms, _fn=None):
                return "job"

            def after_cancel(self, _job):
                pass

        app = SectionApp.__new__(SectionApp)
        app.root = _Root()
        app.busy = False
        app.progress = None
        app.queue = _queue.Queue()
        app._started = None
        app._clock_job = None
        app.status_var = _Var()
        app.elapsed_var = _Var()

        app.run_async(lambda: 42, lambda _r: None, "trabajando…")
        assert app._started is not None
        assert app.elapsed_var.get() != ""

        # drain_queue re-arms itself through root.after, which the stub
        # answers without running anything, so this terminates.
        app.drain_queue()
        assert app._started is None
        assert app.busy is False


class _Var:
    """The two lines of tk.StringVar this needs."""

    def __init__(self):
        self._value = ""

    def set(self, value):
        self._value = value

    def get(self):
        return self._value


def test_the_two_raman_sections_share_the_preprocessing_settings():
    """Both sections edit ``session.preprocess_settings``, not a copy.

    The alternative is one file processed two different ways depending on
    which tab you were looking at, and a difference between the sections
    that nobody can account for.
    """
    text = source("tmd_app")
    assert "self.session.preprocess_settings" in text
    assert "_settings_from_widgets" in text and "_widgets_from_settings" in text


def test_the_dichalcogenide_section_reaches_the_general_phase_catalogue():
    """The oxide search only looks at oxides of the chalcogenide's own
    metals, which is what makes it precise and also what makes it blind to
    unreacted selenium. The section has to be able to run the general
    search too."""
    assert "find_tmd_phases_active" in source("tmd_app")
    assert "find_tmd_phases_active" in source("state")


# -- the parts of the widget code that are not widgets ------------------


def test_the_library_panel_writes_out_every_kind_of_entry():
    """``_describe`` is the catalogue browser's whole content, and it is a
    pure function of a catalogue entry, so it can be tested without a
    display. It has three kinds to handle and they share no fields."""
    from ramancarbon.analysis.heterostructure import load_oxides
    from ramancarbon.analysis.tmd import load_tmd_database, tmd_materials
    from ramancarbon.gui.tmd_app import _describe

    for material in tmd_materials():
        text = _describe("material", material)
        assert material.source in text
        assert material.confidence in text
        # A layered 2H of Mo or W says how it counts layers; everything else
        # says why it cannot.
        assert ("Separación E₂g–A₁g" in text) == material.counts_layers_by_separation
        assert ("Laminar      : NO" in text) != material.layered

    for oxide in load_oxides():
        text = _describe("oxide", oxide)
        assert oxide.source in text
        assert ("límite INFERIOR" in text) == (oxide.min_fwhm is not None)

    payload, _ = load_tmd_database()
    for entry in payload["_missing"]["entries"]:
        text = _describe("missing", entry)
        assert "NO está en la biblioteca" in text
        assert entry["reason"] in text


def test_the_modes_plot_and_table_survive_every_material():
    """Both read ``result.positions`` against the catalogue, and a mode the
    fit named but the catalogue does not have must not raise."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    from types import SimpleNamespace

    from ramancarbon.analysis.tmd import analyse_tmd, tmd_materials
    from ramancarbon.examples.demo_data import make_tmd_demo
    from ramancarbon.gui.theme import PALETTES
    from ramancarbon.gui.tmd_app import TMDApp, _reference_modes

    palette = next(iter(PALETTES.values()))
    for material in tmd_materials():
        result = analyse_tmd(make_tmd_demo(material.key, "bulk", seed=4))
        assert result.material == material.key
        assert set(result.positions) <= set(_reference_modes(material.key)) | {"x"}
        fake = SimpleNamespace(
            palette=palette, figure_palette=palette,
            session=SimpleNamespace(active=SimpleNamespace(tmd_result=result)),
        )
        figure = Figure()
        TMDApp._draw_modes(fake, figure)
        assert figure.axes and figure.axes[0].get_yticklabels()


def test_the_oxide_plot_draws_only_the_chemically_possible_oxides():
    """Drawing every oxide in the library would put tungsten oxide lines on
    a molybdenum spectrum, which is exactly what the analysis refuses to
    do. The plot has to restrict itself the same way."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    from types import SimpleNamespace

    from ramancarbon.analysis.tmd import analyse_tmd
    from ramancarbon.examples.demo_data import make_tmd_demo
    from ramancarbon.gui.theme import PALETTES
    from ramancarbon.gui.tmd_app import TMDApp

    palette = next(iter(PALETTES.values()))
    spectrum = make_tmd_demo("MoSe2", "bulk", high=1100.0,
                             oxide="MoO3_alpha", seed=5)
    result = analyse_tmd(spectrum)
    fake = SimpleNamespace(
        palette=palette, figure_palette=palette,
        session=SimpleNamespace(active=SimpleNamespace(
            tmd_result=result, display=spectrum, raw=spectrum)),
    )
    figure = Figure()
    TMDApp._draw_oxides(fake, figure)
    labels = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]
    assert labels and all("W" not in text for text in labels)
    assert any("MoO3" in text and "✓" in text for text in labels)

    # No spectrum at all: a placeholder, not a traceback.
    empty = SimpleNamespace(palette=palette, figure_palette=palette,
                            session=SimpleNamespace(active=None))
    TMDApp._draw_oxides(empty, Figure())


# -- theme: the parts that are data, not widgets ------------------------


@pytest.mark.parametrize("widget", ["TEntry", "TCombobox", "TSpinbox"])
def test_the_readonly_states_are_mapped_and_not_only_configured(widget):
    """clam ships its own state map for readonly and disabled, and a map
    beats a configure. Setting only the default state left every readonly
    combo box in the suite -- the figure preset, the baseline method, the
    normalisation, the deconvolution preset, the profile, the ratio basis
    -- drawing our light foreground on clam's pale grey field: invisible
    on the dark palette, which is the one this window opens on."""
    text = source("theme")
    assert f'style.map("{widget}"' in text, f"{widget} has no state map"


def test_the_combobox_dropdown_is_coloured_too():
    """The dropdown is a plain Tk listbox and ignores every ttk style."""
    text = source("theme")
    assert "*TCombobox*Listbox.background" in text
    assert "*TCombobox*Listbox.foreground" in text


def test_a_readonly_combobox_keeps_its_text_when_focused():
    """A readonly combobox draws its value as a SELECTION once it has
    focus. Without the selection colours the value disappears the moment
    you click it, which reads as the widget clearing itself."""
    text = source("theme")
    assert "selectbackground=[" in text and "selectforeground=[" in text


def test_the_figure_background_is_a_separate_choice_from_the_window():
    """A dark window is comfortable to work in; a white figure is what a
    manuscript prints. Choosing one should not choose the other."""
    from ramancarbon.gui.state import Session

    session = Session()
    session.figure_theme = "tema"
    assert session.figure_palette_name("oscuro") == "oscuro"
    assert session.figure_palette_name("claro") == "claro"
    session.figure_theme = "claro"
    assert session.figure_palette_name("oscuro") == "claro"
    session.figure_theme = "oscuro"
    assert session.figure_palette_name("claro") == "oscuro"


def test_drawing_code_uses_the_figure_palette_and_chrome_code_does_not():
    """The split only works if every _draw_ method draws with the figure
    palette. One left on the chrome palette is a line that stays dark on
    a white figure -- invisible, and only in the exported file."""
    for stem in ("app", "tmd_app", "xrd_app", "echem_app", "xps_app"):
        text = source(stem)
        for block in re.findall(r"\n    def _draw_\w+\(self.*?(?=\n    def |\Z)",
                                text, re.S):
            assert "self.palette" not in block, (
                f"{stem}: a _draw_ method still uses the chrome palette")
    # And the rc_context that sets the background, in both canvas layers.
    assert "matplotlib_style(self.figure_palette)" in source("base")
    assert "matplotlib_style(self.figure_palette)" in source("app")
    assert "matplotlib_style(self.palette)" not in source("base")


def test_every_canvas_gets_a_reset_zoom_button():
    """matplotlib's Home rewinds the view STACK, so after a redraw with new
    data it restores limits that belonged to the previous figure, and on an
    empty stack it does nothing at all — which reads as a dead button."""
    for stem in ("base", "app"):
        text = source(stem)
        assert "Restablecer zoom" in text, stem
    assert "def reset_zoom" in source("base")
    assert "def _reset_zoom" in source("app")


def test_plot_colours_are_roles_and_survive_a_restart():
    """Roles, not individual curves: the fitted curve has one colour
    throughout the application so the code is learnt once."""
    from ramancarbon.gui.state import Session
    from ramancarbon.gui.theme import DARK, LIGHT, PLOT_ROLES, with_colours

    assert {role for role, _ in PLOT_ROLES} == {
        "data", "fitted", "residual", "baseline",
        # The peak markers are roles too, and for the same reason plus
        # one: they used to borrow the theme's accent and warning, which
        # also colour buttons and status text, so recolouring the
        # unexplained peaks repainted the interface.
        "peak_carbon", "peak_phase", "peak_unknown",
    }
    for role, _ in PLOT_ROLES:
        assert hasattr(LIGHT, role) and hasattr(DARK, role), role
        assert with_colours(LIGHT, {role: "#123456"}).__getattribute__(
            role) == "#123456"
    recoloured = with_colours(DARK, {"data": "#ff0000",
                                     "components": ["#00ff00", "#0000ff"]})
    assert recoloured.data == "#ff0000"
    assert recoloured.components == ("#00ff00", "#0000ff")
    assert recoloured.fitted == DARK.fitted          # untouched roles stay

    # Nonsense is ignored, not raised on: these come from a preferences
    # file a user may have edited by hand.
    assert with_colours(LIGHT, {"data": "rojo"}).data == LIGHT.data
    assert with_colours(LIGHT, {}) is LIGHT

    session = Session()
    session.plot_colours = {"fitted": "#123456"}
    assert session.figure_palette("oscuro").fitted == "#123456"
    assert session.figure_palette("claro").fitted == "#123456"


def test_the_colour_dialog_is_wired_into_the_header():
    text = source("suite")
    assert "Colores…" in text
    assert "def _choose_colours" in text
    assert "def _apply_figure_palette" in text
    # Every place that rebuilds the figure palette goes through one method,
    # so a colour change and a theme change cannot diverge.
    assert text.count("self.session.figure_palette(") >= 1


def test_a_carbon_band_is_not_marked_as_an_unexplained_peak():
    """The spectrum plot split its markers two ways: explained by a
    catalogued PHASE, or unexplained. Carbon bands are assigned by a
    different path, so the D and the G of a carbon sample came out marked
    "sin explicar" — the best understood bands in the spectrum, called
    unknown, while the report named them on the next tab."""

    from ramancarbon.analysis.report import analyse
    from ramancarbon.examples.demo_data import demo_spectra
    from ramancarbon.gui.app import _carbon_bands, _explained_positions

    spectrum = next(s for s in demo_spectra() if "DWCNT" in s.name)
    result = analyse(spectrum)

    bands = _carbon_bands(result)
    assert bands, "no carbon bands came back at all"
    names = {name for _, name in bands}
    assert {"D", "G+"} & names or {"D", "G"} & names, names

    # Every detected peak that sits on an assigned band must be findable
    # as one, or the plot will mark it unexplained.
    phase = _explained_positions(result)
    for peak in result.peaks:
        nearest = min((abs(peak.position - p) for p, _ in bands), default=1e9)
        on_phase = any(abs(peak.position - p) <= 12.0 for p in phase)
        assert nearest <= 12.0 or on_phase, (
            f"the peak at {peak.position:.0f} cm-1 is neither a carbon band "
            f"nor a catalogued phase line, so it would be drawn as "
            f"unexplained; assigned bands are "
            f"{[(round(p), n) for p, n in bands]}")


def test_the_spectrum_plot_draws_three_groups_not_two():
    from ramancarbon.gui import plots

    source = inspect.getsource(plots.plot_spectrum)
    assert "bandas de carbono" in source
    assert "picos de una fase catalogada" in source
    assert "picos sin explicar" in source


def _parent_of(node: ast.FunctionDef) -> dict[str, str]:
    """Map each widget name in one build method to its parent's name.

    Only the forms this package actually uses: ``x = ttk.Frame(parent)``,
    ``outer, body = card(parent)`` (where ``body`` lives inside
    ``outer``), and ``paned, (a, b) = split_column(parent)``.
    """
    parents: dict[str, str] = {}

    def first_arg(call: ast.Call) -> str | None:
        if call.args and isinstance(call.args[0], ast.Name):
            return call.args[0].id
        if call.args and isinstance(call.args[0], ast.Attribute):
            return ast.unparse(call.args[0])
        return None

    for statement in ast.walk(node):
        if not isinstance(statement, ast.Assign):
            continue
        value = statement.value
        if not isinstance(value, ast.Call):
            continue
        parent = first_arg(value)
        if parent is None:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            parents[target.id] = parent
        elif isinstance(target, ast.Tuple):
            names = [e for e in target.elts if isinstance(e, ast.Name)]
            if names:
                parents[names[0].id] = parent
            # card() returns (outer, body): the body lives in the outer.
            if len(names) > 1:
                parents[names[1].id] = names[0].id
    return parents


@pytest.mark.parametrize("stem", sorted(SECTION_MODULES))
def test_nothing_that_expands_is_packed_before_a_fixed_card(stem):
    """Tk's packer gives each widget its requested size and only then
    divides what is left, so a card packed AFTER an expanding SIBLING
    gets whatever is over — which on a short window is nothing, and it is
    simply not drawn.

    This is not hypothetical. The Rietveld results card (convergence,
    evaluation count, weight fractions, and the two report buttons) sat
    after the parameter table, which expands, in the same pane; all four
    were below the fold, and the user reported every one of them as a
    missing feature when each was already computed. The electrochemistry
    companion figures were crushed to a strip against the bottom edge for
    the same reason.

    So within one notebook page, once something has been packed with
    ``expand=True``, nothing may be packed with ``fill="x"`` after it in
    the same parent.
    """
    text = source(stem)
    tree = ast.parse(text)
    offenders: list[str] = []

    for node in ast.walk(tree):
        # Tab builders only. A sidebar lives inside `scrollable_column`,
        # whose frame sizes itself to its contents, so there is no spare
        # height for `expand` to claim and a fixed card below one keeps
        # its natural size. A notebook page has a fixed height and does
        # not scroll, which is what makes the ordering matter there.
        if not (isinstance(node, ast.FunctionDef)
                and node.name.startswith("_build_tab")):
            continue
        parents = _parent_of(node)
        expanded: dict[str, list[str]] = {}
        for call in ast.walk(node):
            if not (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "pack"
                    and isinstance(call.func.value, ast.Name)):
                continue
            name = call.func.value.id
            parent = parents.get(name)
            if parent is None:
                continue
            keywords = {k.arg: k.value for k in call.keywords}
            expand = keywords.get("expand")
            expands = isinstance(expand, ast.Constant) and expand.value is True
            fill = keywords.get("fill")
            fill_value = fill.value if isinstance(fill, ast.Constant) else None
            if expands:
                expanded.setdefault(parent, []).append(name)
            elif fill_value == "x" and expanded.get(parent):
                offenders.append(
                    f"{stem}.{node.name}: {name}.pack(fill='x') comes after "
                    f"{expanded[parent]} in the same parent ({parent}), so it "
                    "gets whatever height is left over — often none"
                )

    assert not offenders, "\n".join(offenders)
