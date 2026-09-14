"""Headless smoke tests for the Tkinter GUI.

These are skipped when ``tkinter`` is missing (it is packaged separately on
many Linux distributions) or when no X display is reachable. On CI with
``xvfb-run`` they exercise the real widget tree: build on the worker
thread, redraw the 3D preview, and read back the info panel.
"""

from __future__ import annotations

import os

import pytest

tk = pytest.importorskip("tkinter", reason="tkinter is not installed")
pytest.importorskip("matplotlib", reason="matplotlib is not installed")


def test_importing_the_gui_leaves_the_matplotlib_backend_alone(tmp_path):
    """Importing the GUI must not reconfigure matplotlib process-wide.

    This one deliberately does *not* need a display -- it guards the case
    where there is none. The GUI embeds a bare Figure in a Tk canvas, which
    needs the Tk canvas class and not the global backend. An import-time
    ``matplotlib.use("TkAgg")`` used to switch the whole process over, so
    any later headless ``savefig`` died with "cannot load backend 'TkAgg'
    ... 'headless' is currently running" -- which is how it showed up: as
    an unrelated viz test failing only when run after the GUI tests.
    """
    import matplotlib

    from nanocarbon_lab.builders.fullerene import build_fullerene
    from nanocarbon_lab.viz import save_structure_png

    before = matplotlib.get_backend()
    import nanocarbon_lab.gui.app  # noqa: F401

    assert matplotlib.get_backend() == before

    # And the consequence that actually bit, not just the mechanism.
    out = save_structure_png(build_fullerene(freq=1, family="C60"),
                             tmp_path / "cage.png")
    assert out.stat().st_size > 0


def _make_root():
    """Return a hidden Tk root, or skip if no display is available."""
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        pytest.skip("no X display available")
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # display exists but is unusable
        pytest.skip(f"cannot open a Tk window: {exc}")
    root.withdraw()
    return root


def _pump_until_idle(root, app, timeout=180.0):
    """Run the event loop until the background build finishes."""
    import time

    deadline = time.time() + timeout
    root.update()
    while app._busy and time.time() < deadline:
        root.update()
        time.sleep(0.02)
    assert not app._busy, "GUI build did not finish within the timeout"


@pytest.fixture
def app():
    from nanocarbon_lab.gui.app import NanocarbonGUI

    root = _make_root()
    instance = NanocarbonGUI(root)
    _pump_until_idle(root, instance)
    yield instance
    # Close the matplotlib figure before tearing down Tk: otherwise the
    # canvas' Tk images are finalised after the interpreter has left the
    # main loop, which raises during garbage collection.
    import matplotlib.pyplot as plt

    plt.close(instance.figure)
    root.destroy()


def test_gui_builds_a_valid_structure_on_startup(app):
    assert app.atoms is not None
    counts = app.atoms.info["ring_counts"]
    assert counts[5] == 12
    assert sum((6 - s) * c for s, c in counts.items()) == 12


def test_gui_info_panel_reports_geometry(app):
    text = app.txt_info.get("1.0", "end")
    assert "atoms" in text and "rings" in text and "geometry" in text
    assert "OK" in text  # Euler + contact checks both pass


def test_radius_hint_tracks_freq(app):
    app.var_freq.set(3)
    app._update_radius_hint()
    hint3 = app.lbl_radius.cget("text")
    app.var_freq.set(5)
    app._update_radius_hint()
    hint5 = app.lbl_radius.cget("text")
    assert hint3 != hint5
    assert "5.8" in hint3 and "9.7" in hint5


def test_gui_rebuild_with_defects_and_bend(app):
    app.var_rings.set(8)
    app.var_freq.set(3)
    app.var_n_sw.set(1)
    app.var_n_dv.set(1)
    app.var_bend.set(0.3)
    app.on_build()
    _pump_until_idle(app.root, app)

    counts = app.atoms.info["ring_counts"]
    assert counts.get(7) == 2  # one Stone-Wales pair
    assert counts.get(8) == 1  # one divacancy
    geometry = app.atoms.info["geometry"]
    assert geometry["n_close_contacts"] == 0
    assert 1.30 < geometry["bond_min"] <= geometry["bond_max"] < 1.55


def test_preview_toggles_redraw_without_error(app):
    for bonds in (True, False):
        for colour in ("element", "ring", "plain"):
            app.var_show_bonds.set(bonds)
            app.var_colour_by.set(colour)
            app._redraw()
            app.root.update()


def test_element_colouring_actually_separates_the_elements(app):
    """The bug this replaced: every atom one colour, on every structure.

    Ring colouring paints a pristine tube entirely in the hexagon colour,
    which is correct and useless -- and it hid the dopant too.
    """
    app.var_mode_kind.set("capped tube")
    app.var_dopant.set("N")
    app.var_dopant_conc.set(0.08)
    app.on_build()
    _pump_until_idle(app.root, app)

    app.var_colour_by.set("element")
    element = set(app._atom_colours(app._ring_of_atom()))
    app.var_colour_by.set("ring")
    ring = set(app._atom_colours(app._ring_of_atom()))
    app.var_colour_by.set("plain")
    plain = set(app._atom_colours(app._ring_of_atom()))

    assert "N" in app.atoms.get_chemical_symbols(), "the doping did not happen"
    assert len(element) > 1, "a doped tube must not be one colour by element"
    assert len(plain) == 1, "plain is meant to be one colour"
    assert element != ring


def test_export_writes_bundle(app, tmp_path):
    from nanocarbon_lab.exports.xyz import write_render_bundle

    xyz_path, json_path = write_render_bundle(app.atoms, tmp_path / "gui")
    assert xyz_path.exists() and json_path.exists()
    assert int(xyz_path.read_text().splitlines()[0]) == len(app.atoms)


class TestBlenderDiscovery:
    """`find_blender` must work where `shutil.which` alone does not.

    The Windows installer does not put Blender on PATH and the macOS build
    hides inside an .app bundle, so PATH-only lookup reports "not found"
    on a perfectly normal install. These run on any host: the platform
    branches are exercised by patching `os.name` / `sys.platform`.
    """

    def _fake_windows_install(self, tmp_path, versions=("Blender 3.6", "Blender 4.2")):
        program_files = tmp_path / "Program Files"
        for version in versions:
            d = program_files / "Blender Foundation" / version
            d.mkdir(parents=True)
            (d / "blender.exe").write_text("")
        return program_files

    def test_finds_windows_install_and_prefers_newest(self, tmp_path, monkeypatch):
        from nanocarbon_lab.gui.app import find_blender

        program_files = self._fake_windows_install(tmp_path)
        monkeypatch.delenv("BLENDER", raising=False)
        monkeypatch.setattr("os.name", "nt")
        monkeypatch.setenv("ProgramFiles", str(program_files))
        monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "absent"))
        monkeypatch.setattr("shutil.which", lambda _name: None)

        found = find_blender()
        assert found is not None
        assert found.endswith("blender.exe")
        assert "4.2" in found  # newest install wins over 3.6

    def test_env_override_wins(self, tmp_path, monkeypatch):
        from nanocarbon_lab.gui.app import find_blender

        portable = tmp_path / "portable" / "blender.exe"
        portable.parent.mkdir(parents=True)
        portable.write_text("")
        monkeypatch.setenv("BLENDER", str(portable))
        assert find_blender() == str(portable)

    def test_returns_none_when_absent(self, tmp_path, monkeypatch):
        from nanocarbon_lab.gui.app import find_blender

        monkeypatch.delenv("BLENDER", raising=False)
        monkeypatch.setattr("os.name", "nt")
        monkeypatch.setenv("ProgramFiles", str(tmp_path / "empty"))
        monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "empty2"))
        monkeypatch.setattr("shutil.which", lambda _name: None)
        assert find_blender() is None


def _one_of_each_structure():
    """Smallest instance of every structure type the GUI can build.

    Kept in one place because the info panel is the thing most likely to
    break when a builder grows a new `info` key: each row is emitted from
    the presence of a field, and it is easy to key a row off one field
    while reading a sibling that only some builders record.
    """
    from nanocarbon_lab.builders import (
        build_bundle,
        build_capped_cnt,
        build_fullerene,
        build_junction,
        build_multiwall_cnt,
        build_nano_onion,
        build_schwarzite,
    )

    return {
        "capped tube": lambda: build_capped_cnt(n_body_rings=4, freq=2),
        "fullerene": lambda: build_fullerene(freq=1, family="C60"),
        "nano-onion": lambda: build_nano_onion(n_shells=2),
        "multi-wall": lambda: build_multiwall_cnt(
            n_shells=2, inner_freq=2, n_body_rings=4),
        "bundle": lambda: build_bundle(
            n_rings_across=1, freq=2, n_body_rings=4),
        "junction": lambda: build_junction(
            kind="L", tube_radius=5.0, arm_length=10.0),
        "schwarzite": lambda: build_schwarzite("primitive", cell=30.0),
    }


@pytest.mark.parametrize("kind", sorted(_one_of_each_structure()))
def test_info_panel_renders_every_structure_type(app, kind):
    """Two real bugs came from this: a fullerene has a radius but no
    length, and a nano-onion has a shell count but records its spacing as
    `shell_spacing`, not the multi-wall tube's `wall_spacing`. Both raised
    KeyError inside the queue poll and froze the window.
    """
    app.atoms = _one_of_each_structure()[kind]()
    app._update_info()
    text = app.txt_info.get("1.0", "end")
    assert "atoms" in text
    assert "Euler sum" in text
    assert "BROKEN" not in text.split("sp2 verdict")[0], (
        f"{kind}: Euler budget wrong in the panel"
    )


def _one_of_each_dichalcogenide():
    """One built structure per TMD mode, for the other info panel.

    The carbon panel above cannot cover these: a dichalcogenide has no
    rings to count and no Euler budget, so it renders through
    `_update_tmd_info` instead — a second place with the same failure
    mode, where a row keyed off one `info` field reads a sibling that
    only some builders record. A coil, for instance, has chiral indices
    and a roll strain but no single radius or diameter.
    """
    import warnings

    from nanocarbon_lab.tmd import (
        build_tmd_bulk,
        build_tmd_coil,
        build_tmd_layers,
        build_tmd_nanotube,
        build_tmd_ribbon,
    )

    def quiet(fn, **kwargs):
        def run():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return fn(**kwargs)
        return run

    return {
        "TMD layers": quiet(build_tmd_layers, n_layers=2, nx=2, ny=2),
        "TMD bulk": quiet(build_tmd_bulk, stacking="2H"),
        "TMD ribbon": quiet(build_tmd_ribbon, width=6, length=2),
        "TMD nanotube": quiet(build_tmd_nanotube, n=30, m=0),
        "TMD coil": quiet(build_tmd_coil, n=20, m=0, coil_radius=140.0,
                          pitch=60.0, turns=0.1),
    }


@pytest.mark.parametrize("kind", sorted(_one_of_each_dichalcogenide()))
def test_tmd_info_panel_renders_every_mode(app, kind):
    app.atoms = _one_of_each_dichalcogenide()[kind]()
    app._update_info()
    text = app.txt_info.get("1.0", "end")
    assert "material" in text
    assert "X/M ratio" in text
    assert "verdict" in text


def test_the_coil_panel_reports_both_strains(app):
    """Roll and bend are separate numbers with separate cures, and a
    panel showing only their sum would hide which one to fix."""
    app.atoms = _one_of_each_dichalcogenide()["TMD coil"]()
    app._update_info()
    text = app.txt_info.get("1.0", "end")
    assert "roll strain" in text
    assert "bend strain" in text
    assert "total strain" in text
    assert "coil radius" in text


def test_cage_panel_omits_tube_only_rows(app):
    from nanocarbon_lab.builders import build_fullerene

    app.atoms = build_fullerene(freq=1, family="C60")
    app._update_info()
    text = app.txt_info.get("1.0", "end")
    assert "C60" in text and "radius" in text
    # Tube-only rows must simply be absent, not blank or zero.
    assert "path strain" not in text
    assert "length" not in text


def test_a_display_error_does_not_stop_the_poll(app, monkeypatch):
    """The poll callback is the only thing keeping the window alive.

    If an exception escapes it, ``root.after`` is never re-armed and the
    app is stuck "building" forever with no way back. It must survive a
    broken redraw and keep polling.
    """
    def boom():
        raise KeyError("length")

    built = app.atoms
    monkeypatch.setattr(app, "_update_info", boom)
    monkeypatch.setattr(app, "_redraw", lambda: None)
    monkeypatch.setattr(app.worker, "poll", lambda: (1, "done", built))

    app._busy = True
    app._poll_worker()

    assert not app._busy, "the build must still be marked finished"
    assert "display failed" in app.status.cget("text").lower()

    # The poll survived, so a subsequent good result is still processed.
    monkeypatch.setattr(app, "_update_info", lambda: None)
    app._busy = True
    app._poll_worker()
    assert not app._busy
    assert "complete" in app.status.cget("text").lower()


def test_a_build_failure_is_shown_in_the_panel_not_a_dialog(app, monkeypatch):
    """A modal dialog blocks the event loop and throws away the
    parameters the user was about to fix, so errors go to a panel."""
    monkeypatch.setattr(
        app.worker, "poll",
        lambda: (1, "error", ("ValueError('cell too small')", "traceback here")),
    )
    app._busy = True
    app._poll_worker()

    assert not app._busy
    assert "failed" in app.status.cget("text").lower()
    assert "traceback here" in app.txt_error.get("1.0", "end")


def test_cancelling_releases_the_controls(app, monkeypatch):
    stopped = []
    monkeypatch.setattr(app.worker, "cancel", lambda: stopped.append(True))
    app._busy = True
    app.btn_build.config(state="disabled")
    app.on_cancel()
    assert stopped == [True]
    assert not app._busy
    assert str(app.btn_build.cget("state")) == "normal"


def test_typed_values_beat_the_slider_range(app):
    """"Control total": the slider is a convenience, the entry is the
    truth, and it accepts figures outside the slider's comfortable span.
    """
    app.var_coil_radius.set(37.5)
    assert float(app.var_coil_radius.get()) == pytest.approx(37.5)
    # Well beyond the slider's 200 Å top end, but a legitimate coil.
    app.var_coil_radius.set(640.0)
    assert float(app.var_coil_radius.get()) == pytest.approx(640.0)


def test_the_estimate_reacts_to_the_parameters(app):
    app.var_mode_kind.set("fullerene")
    app.var_cage_freq.set(1)
    app._update_estimate()
    small = app.lbl_estimate.cget("text")
    app.var_cage_freq.set(5)
    app._update_estimate()
    assert app.lbl_estimate.cget("text") != small
    assert "1500" in app.lbl_estimate.cget("text")  # 60 * 25


def test_current_job_round_trips_through_the_cli(app):
    """The GUI and the command line must describe the same structure."""
    import shlex

    from nanocarbon_lab.cli.main import build_parser
    from nanocarbon_lab.jobs import to_cli

    for mode in ("fullerene", "junction", "schwarzite", "bundle"):
        app.var_mode_kind.set(mode)
        command = to_cli(app.current_job(), out="out/x")
        build_parser().parse_args(shlex.split(command)[1:])


def test_every_preset_names_only_real_parameters(app):
    """A preset naming a parameter that no longer exists would silently
    do nothing, which is worse than failing."""
    from nanocarbon_lab.gui.app import PRESETS

    for name, values in PRESETS.items():
        unknown = set(values) - set(app._params)
        assert not unknown, f"preset {name!r} sets unknown parameters {unknown}"


def test_the_dopant_hint_tracks_the_element_and_the_amount(app):
    """Fifteen elements in one dropdown is fifteen chemistries, and the
    difference between 10% N and 10% Fe is the difference between a
    common material and one that does not exist. The hint has to say so
    before the build, not only in a warning after it."""
    from nanocarbon_lab.gui.app import MUTED, WARN_AMBER

    app.var_dopant.set("none")
    assert "Pure carbon" in app.lbl_dopant.cget("text")

    app.var_dopant.set("N")
    app.var_dopant_conc.set(0.10)
    # cget returns a Tk colour object, not a str.
    assert str(app.lbl_dopant.cget("foreground")) == MUTED

    app.var_dopant.set("Fe")
    assert str(app.lbl_dopant.cget("foreground")) == WARN_AMBER
    assert "physically meaningful" in app.lbl_dopant.cget("text")


def test_pentagon_placement_says_what_the_fraction_means(app):
    """The fraction counts against the pentagon sites there, which is a
    factor of four different on a capped tube."""
    app.var_dopant.set("N")
    app.var_dopant_site.set("pentagon")
    assert "pentagon sites" in app.lbl_dopant.cget("text")
    assert app.current_job().dopant_site == "pentagon"


def test_carbon_is_the_default_and_carries_no_dopant(app):
    """The host is always carbon; a structure is pure unless asked."""
    assert app.var_dopant.get() == "none"
    assert app.current_job().dopant is None


def test_the_metal_and_chalcogen_pickers_drive_the_formula(app):
    app.var_family.set("dichalcogenide")
    app.var_mode_kind.set("TMD layers")
    app.var_tmd_metal.set("W")
    app.var_tmd_chalcogen.set("Se")
    assert app.var_tmd_material.get() == "WSe2"
    assert app.current_job().params["material"] == "WSe2"


def test_an_impossible_pair_cannot_be_selected(app):
    """Sn has no tabulated telluride. Rather than let the pair be chosen
    and fail at build time, the chalcogen list renarrows and the
    selection falls back to one that exists."""
    app.var_family.set("dichalcogenide")
    app.var_mode_kind.set("TMD layers")
    app.var_tmd_metal.set("W")
    app.var_tmd_chalcogen.set("Te")
    assert app.var_tmd_material.get() == "WTe2"

    app.var_tmd_metal.set("Sn")
    assert "Te" not in app.cmb_tmd_chalcogen.cget("values")
    assert app.var_tmd_material.get() == "SnS2"


def test_a_preset_naming_a_compound_updates_both_pickers(app):
    """Presets name a compound, not a pair of elements, so the formula
    has to drive the pickers as well as follow them -- without the two
    traces feeding each other."""
    app.var_family.set("dichalcogenide")
    app.var_mode_kind.set("TMD layers")
    app.var_tmd_material.set("PtTe2")
    assert app.var_tmd_metal.get() == "Pt"
    assert app.var_tmd_chalcogen.get() == "Te"
    assert app.var_tmd_material.get() == "PtTe2"


def test_the_mx2_chemistry_panel_reaches_the_job(app):
    """Janus faces, alloys, vacancies and antisites existed in tmd/modify
    from the start and were reachable only from Python — no CLI flag, no
    GUI control, no mention in jobs.py."""
    app.var_family.set("dichalcogenide")
    app.var_mode_kind.set("TMD layers")
    app.var_tmd_edit.set("alloy")
    app.var_tmd_edit_element.set("W")
    app.var_tmd_edit_amount.set(0.4)

    job = app.current_job()
    assert job.tmd_edit == "alloy"
    assert job.tmd_edit_element == "W"
    assert job.tmd_edit_amount == pytest.approx(0.4)


def test_the_amount_hint_says_which_of_three_things_it_means(app):
    """One Amount field serves all four edits, so the hint below it is
    what keeps the number on screen unambiguous."""
    app.var_family.set("dichalcogenide")
    app.var_mode_kind.set("TMD layers")

    app.var_tmd_edit.set("alloy")
    assert "fraction" in app.lbl_tmd_chem.cget("text")

    app.var_tmd_edit.set("vacancies")
    assert "count" in app.lbl_tmd_chem.cget("text")

    app.var_tmd_edit.set("janus")
    assert "face" in app.lbl_tmd_chem.cget("text")


def test_the_chemistry_panel_is_dichalcogenide_only(app):
    """There is no Janus graphene; the panel must not follow the carbon
    modes the way the dopant panel does."""
    app.var_family.set("dichalcogenide")
    app.var_mode_kind.set("TMD layers")
    assert app.frame_tmd_chem.winfo_manager()

    app.var_family.set("carbon")
    app.var_mode_kind.set("capped tube")
    assert not app.frame_tmd_chem.winfo_manager()


def test_hint_labels_rewrap_when_the_column_widens(app):
    """The wraps were pixel values tuned to a 268 px column. After the
    dividers became draggable a fixed wrap is wrong twice: it clips a
    larger font, and it keeps the old width after the divider moves."""
    from nanocarbon_lab.gui.app import ScrollableColumn

    column = ScrollableColumn(app.root, width=200)
    label = tk.Label(column.interior, text="x" * 400, wraplength=230)
    label.pack()
    plain = tk.Label(column.interior, text="short")
    plain.pack()

    column._rewrap(500)
    assert int(label.cget("wraplength")) > 230
    # An ordinary label has wraplength 0, which is the flag that says it
    # is not an explanatory hint; it must be left alone.
    assert int(plain.cget("wraplength")) == 0


def test_the_network_panel_reaches_the_job(app):
    app.var_mode_kind.set("network")
    app.var_net_kind.set("cubic")
    app.var_net_cell.set(44.0)
    app.var_net_radius.set(6.5)

    job = app.current_job()
    assert job.mode == "network"
    assert job.params["kind"] == "cubic"
    assert job.params["cell"] == pytest.approx(44.0)
    assert job.params["tube_radius"] == pytest.approx(6.5)


def test_the_network_hint_warns_before_the_build_fails(app):
    """The failure is not visible in the numbers: shrink the cell and the
    nodes grow into each other until no tube is left. Saying so up front
    is the difference between a build that fails after two minutes and
    one that never started."""
    from nanocarbon_lab.gui.app import BAD_RED

    app.var_mode_kind.set("network")
    app.var_net_kind.set("cubic")
    app.var_net_cell.set(40.0)
    assert "free tube" in app.lbl_network.cget("text")

    app.var_net_cell.set(24.0)
    assert "Too small" in app.lbl_network.cget("text")
    assert str(app.lbl_network.cget("foreground")) == BAD_RED


def test_a_diamond_net_needs_a_bigger_cell_than_cubic(app):
    """Its struts are the quarter-diagonal, so the same edge buys 0.43
    of the tube length."""
    app.var_mode_kind.set("network")
    app.var_net_cell.set(40.0)
    app.var_net_kind.set("cubic")
    assert "Too small" not in app.lbl_network.cget("text")
    app.var_net_kind.set("diamond")
    assert "Too small" in app.lbl_network.cget("text")


def test_the_network_turns_annealing_off_on_entry(app):
    """Same reason as the schwarzite: at a node the 5-7 pairs are how a
    hexagonal net covers the curvature."""
    app.var_mode_kind.set("junction")
    app.var_anneal.set(80)
    app.var_mode_kind.set("network")
    assert app.var_anneal.get() == 0


def test_the_unit_cell_export_describes_what_it_wrote(app, tmp_path, monkeypatch):
    """A separate button from the render bundle because it answers a
    different question: the bundle is for looking at, this is for
    computing with."""
    from nanocarbon_lab.gui import app as app_module

    app.var_mode_kind.set("fullerene")
    app.var_cage_family.set("C60")
    app.var_cage_freq.set(1)
    app.on_build()
    _pump_until_idle(app.root, app)

    target = tmp_path / "cell.cif"
    monkeypatch.setattr(app_module.filedialog, "asksaveasfilename",
                        lambda **_: str(target))
    written = app.on_export_cell()
    assert written is not None and written.exists()
    assert "3D cell" in app.lbl_cell.cget("text")
    assert "across vacuum" in app.lbl_cell.cget("text")


def test_the_graft_panel_shows_for_every_family(app):
    """A group is added on top of a surface, not substituted into it.

    Carbon, MX2 and a vdW stack all have a surface, so the panel belongs
    to all three -- unlike the dopant panel, which is carbon's alone.
    """
    for mode in ("capped tube", "TMD layers", "twisted bilayer"):
        app.var_mode_kind.set(mode)
        assert app.frame_graft.winfo_manager(), mode


def test_the_graft_hint_lists_the_swaps_the_group_accepts(app):
    """The swap field is free text, so the panel has to say what fits.

    Which elements are offered depends on the group -- an -OH reaches
    S/Se/Te and an -NH2 reaches P/As/B -- so a fixed dropdown could not
    be right and the hint carries the information instead.
    """
    app.var_mode_kind.set("capped tube")
    app.var_graft.set("hydroxyl")
    hint = app.lbl_graft.cget("text")
    assert "S/Se/Te" in hint

    app.var_graft.set("amine")
    hint = app.lbl_graft.cget("text")
    assert "As/B/P" in hint


def test_an_impossible_swap_is_reported_before_building(app):
    """Catching it in the hint beats a traceback after a long build."""
    app.var_mode_kind.set("capped tube")
    app.var_graft.set("hydroxyl")
    app.var_graft_swap.set("O:N")
    hint = app.lbl_graft.cget("text")
    assert "bonds" in hint
    assert str(app.lbl_graft.cget("foreground")) == app_module_warn_colour()


def app_module_warn_colour():
    from nanocarbon_lab.gui import app as app_module
    return app_module.WARN_AMBER


def test_a_valid_swap_reports_the_rebuilt_formula(app):
    app.var_mode_kind.set("capped tube")
    app.var_graft.set("hydroxyl")
    app.var_graft_swap.set("O:S")
    assert "HS" in app.lbl_graft.cget("text")


def test_the_graft_reaches_the_job_for_every_family(app):
    """The panel must not be visibly present and silently ignored.

    Each family builds its Job on a different branch, so a field added
    to one of them alone is the exact failure this pins.
    """
    for mode in ("capped tube", "TMD layers", "twisted bilayer"):
        # The mode first: chemistry is chosen *for* a structure type, and
        # changing the type now clears it rather than carrying it over.
        app.var_mode_kind.set(mode)
        app.var_graft.set("carboxyl")
        app.var_graft_coverage.set(0.2)
        app.var_graft_swap.set("O:S")
        app.var_graft_where.set("edge")
        app.var_graft_face.set("both")
        job = app.current_job()
        assert job.graft == "carboxyl", mode
        assert job.graft_coverage == 0.2, mode
        assert job.graft_swap == "O:S", mode
        assert job.graft_where == "edge", mode
        assert job.graft_face == "both", mode


def test_no_graft_means_no_graft_in_the_job(app):
    app.var_mode_kind.set("capped tube")
    app.var_graft.set("none")
    assert app.current_job().graft is None


def test_the_command_line_reproduces_the_graft(app):
    """The copy-as-command-line button is only useful if it is complete."""
    from nanocarbon_lab.jobs import to_cli

    app.var_mode_kind.set("capped tube")
    app.var_graft.set("hydroxyl")
    app.var_graft_swap.set("O:S")
    app.var_graft_coverage.set(0.25)
    line = to_cli(app.current_job())
    assert "--graft hydroxyl" in line
    assert "--graft-swap O:S" in line
    assert "--graft-coverage 0.25" in line


def test_a_grafted_build_runs_end_to_end(app):
    """The panel is wired to a build, not only to a Job."""
    app.var_mode_kind.set("fullerene")
    app.var_cage_family.set("C60")
    app.var_cage_freq.set(1)
    app.var_graft.set("hydroxyl")
    app.var_graft_coverage.set(0.2)
    app.on_build()
    _pump_until_idle(app.root, app)

    atoms = app.atoms
    assert atoms is not None
    assert "O" in atoms.get_chemical_symbols()
    assert atoms.info["functionalization"][-1]["group"] == "hydroxyl"


def test_analysing_a_file_loads_it_as_the_current_structure(app, tmp_path,
                                                            monkeypatch):
    """A file from another code is a first-class structure, not a guest.

    Loading it as the current one is what lets every other button --
    export, unit cell, render, grafting -- apply to it exactly as it
    would to something built here.
    """
    from ase import io

    from nanocarbon_lab.builders import build_fullerene
    from nanocarbon_lab.gui import app as app_module

    cage = build_fullerene(freq=1, family="C60")
    bare = cage.copy()
    bare.info = {}
    target = tmp_path / "foreign.xyz"
    io.write(str(target), bare)

    monkeypatch.setattr(app_module.filedialog, "askopenfilename",
                        lambda **_: str(target))
    result = app.on_analyse_file()

    assert result is not None
    assert app.atoms is not None and len(app.atoms) == 60
    assert result["inferred"]["shape"]["shape"] == "cage"
    assert result["inferred"]["rings"]["counts"] == {5: 12, 6: 20}
    readout = app.txt_info.get("1.0", "end")
    assert "MEASURED" in readout and "INFERRED" in readout
    assert "cage" in app.lbl_analyse.cget("text")


def test_a_cancelled_file_dialog_changes_nothing(app, monkeypatch):
    from nanocarbon_lab.gui import app as app_module

    before = app.atoms
    monkeypatch.setattr(app_module.filedialog, "askopenfilename",
                        lambda **_: "")
    assert app.on_analyse_file() is None
    assert app.atoms is before


def test_an_unreadable_file_reports_instead_of_raising(app, tmp_path,
                                                       monkeypatch):
    """The GUI never opens a modal, so the message goes to a label."""
    from nanocarbon_lab.gui import app as app_module

    target = tmp_path / "junk.xyz"
    target.write_text("not a structure at all\n")
    monkeypatch.setattr(app_module.filedialog, "askopenfilename",
                        lambda **_: str(target))
    assert app.on_analyse_file() is None
    assert "Could not read" in app.lbl_analyse.cget("text")


class TestChemistryDoesNotOutliveItsStructure:
    """Doping and grafting are chosen for a material, not for a session.

    The panels were hidden on a change of mode and the values behind them
    were not, so a tube's 3% nitrogen went on reaching the builder after
    the structure became a dichalcogenide -- with nothing on screen to say
    so. That is the failure these pin.
    """

    def test_switching_structure_clears_the_dopant(self, app):
        app.var_mode_kind.set("capped tube")
        app.var_dopant.set("N")
        app.var_dopant_conc.set(0.05)

        app.var_mode_kind.set("fullerene")

        assert app.var_dopant.get() == "none"
        assert app.current_job().dopant is None

    def test_switching_structure_clears_the_graft(self, app):
        app.var_mode_kind.set("capped tube")
        app.var_graft.set("carboxyl")
        app.var_graft_where.set("edge")

        app.var_mode_kind.set("TMD layers")

        assert app.var_graft.get() == "none"
        assert app.var_graft_where.get() == "all"

    def test_reselecting_the_same_mode_keeps_the_chemistry(self, app):
        """Tk fires a trace on every set, including one that changes nothing."""
        app.var_mode_kind.set("capped tube")
        app.var_dopant.set("B")
        app.var_mode_kind.set("capped tube")
        assert app.var_dopant.get() == "B"

    def test_a_preset_keeps_the_chemistry_it_just_asked_for(self, app):
        """The regression the guard exists for.

        A preset writes mode_kind exactly as a user does. Without the
        guard the reset fires on that write and erases the doping the
        preset set a microsecond earlier, which would make every doped
        preset silently build a pristine structure.
        """
        app._apply_values({"mode_kind": "capped tube", "dopant": "N",
                           "dopant_conc": 0.04})
        assert app.var_dopant.get() == "N"
        assert app.current_job().dopant == "N"


class TestTheViewBar:
    """What replaced matplotlib's 2D toolbar on a 3D axes."""

    def test_each_viewpoint_actually_moves_the_camera(self, app):
        from nanocarbon_lab.gui.app import NanocarbonGUI

        seen = set()
        for _label, elev, azim in NanocarbonGUI.VIEWPOINTS:
            app._set_view(elev, azim)
            seen.add((round(app.ax.elev, 3), round(app.ax.azim, 3)))
        assert len(seen) == len(NanocarbonGUI.VIEWPOINTS), (
            "two viewpoints put the camera in the same place"
        )

    def test_zoom_narrows_and_widens_about_the_same_centre(self, app):
        def limits():
            return (app.ax.get_xlim3d(), app.ax.get_ylim3d(), app.ax.get_zlim3d())

        def centre(lim):
            return tuple(round((lo + hi) / 2.0, 6) for lo, hi in lim)

        app._zoom_fit()
        start = limits()
        app._zoom(1 / 1.25)
        closer = limits()
        assert closer[0][1] - closer[0][0] < start[0][1] - start[0][0]
        assert centre(closer) == centre(start), "zoom must not drift"

        app._zoom(1.25)
        assert limits()[0][1] - limits()[0][0] == pytest.approx(
            start[0][1] - start[0][0], rel=1e-9)

    def test_the_zoom_cannot_be_driven_to_a_degenerate_axis(self, app):
        for _ in range(40):
            app._zoom(1 / 1.25)
        low, high = app.ax.get_xlim3d()
        assert high > low, "the axis collapsed"

    def test_the_cell_is_drawn_only_when_asked(self, app):
        app.var_mode_kind.set("capped tube")
        app.on_build()
        _pump_until_idle(app.root, app)

        app.var_show_cell.set(False)
        app._redraw()
        without = len(app.ax.collections)
        app.var_show_cell.set(True)
        app._redraw()
        assert len(app.ax.collections) > without

    def test_viewing_the_unit_cell_replaces_the_structure(self, app):
        app.var_mode_kind.set("capped tube")
        app.on_build()
        _pump_until_idle(app.root, app)
        before = len(app.atoms)

        app.on_view_unit_cell()

        assert app.atoms is not None
        assert app.var_show_cell.get(), "the box is the point of the view"
        assert len(app.atoms) <= before


class TestTheBuildClock:
    """Whether a slow build is alive, which a bouncing bar cannot say."""

    def test_the_clock_reads_as_a_stopwatch(self):
        from nanocarbon_lab.gui.app import _clock

        assert _clock(3.2) == "3 s"
        assert _clock(59.4) == "59 s"
        assert _clock(143.0) == "2:23"
        assert _clock(3725.0) == "62:05"

    def test_a_finished_build_reports_what_it_took(self, app):
        app.var_mode_kind.set("capped tube")
        app.on_build()
        _pump_until_idle(app.root, app)

        assert "took" in app.lbl_elapsed.cget("text")
        assert "capped tube" in app._measured
        atoms, seconds = app._measured["capped tube"]
        assert atoms == len(app.atoms)
        assert seconds > 0.0

    def test_the_measurement_reaches_the_estimate(self, app):
        """A measurement beats a hand-written bucket, so it is shown."""
        app.var_mode_kind.set("capped tube")
        app.on_build()
        _pump_until_idle(app.root, app)
        app._update_estimate()
        assert "last here" in app.lbl_estimate.cget("text")

    def test_a_similar_size_is_not_dressed_up_as_a_prediction(self, app):
        app.var_mode_kind.set("capped tube")
        app.on_build()
        _pump_until_idle(app.root, app)
        atoms, _seconds = app._measured["capped tube"]
        assert "so ≈" not in app._scaled_prediction((atoms, 10.0))

    def test_a_much_bigger_job_is_scaled_by_the_square(self, app):
        """The profile puts the cost in a pairwise term, not a linear one."""
        app.var_mode_kind.set("capped tube")
        app.on_build()
        _pump_until_idle(app.root, app)
        atoms, _seconds = app._measured["capped tube"]
        text = app._scaled_prediction((atoms // 4, 10.0))
        assert "so ≈" in text
        # Four times the atoms, sixteen times the work: 160 s, not 40.
        assert "2:40" in text

    def test_the_clock_says_so_when_the_worker_has_gone(self, app):
        app._build_started = 0.0
        app.worker.is_alive = lambda: False
        try:
            app._tick_clock()
            assert "worker process is gone" in app.lbl_elapsed.cget("text")
        finally:
            if app._clock_job is not None:
                app.root.after_cancel(app._clock_job)
                app._clock_job = None
            app._build_started = None


class TestTheCoilHintSaysWhyItLooksWrong:
    """A coil can fail two ways and only one was ever reported.

    Too tight and the walls merge, which the builder refuses outright.
    Too loose and it builds perfectly and does not read as a coil at all
    — a gently curving tube — and that is the one people hit, because
    nothing about the numbers says so.
    """

    def _hint(self, app, radius, pitch, turns, tube):
        app.var_mode_kind.set("coil (relaxed)")
        app.var_coil_radius.set(radius)
        app.var_coil_pitch.set(pitch)
        app.var_coil_turns.set(turns)
        app.var_coil_tube_radius.set(tube)
        app._update_coil_hint()
        return app.lbl_coil.cget("text")

    def test_an_open_coil_is_called_open(self, app):
        text = self._hint(app, 90.0, 30.0, 3.0, 4.0)
        assert "open" in text

    def test_under_two_turns_is_called_out(self, app):
        text = self._hint(app, 22.0, 14.0, 1.5, 5.0)
        assert "two turns" in text

    def test_merging_walls_still_win_over_the_looks(self, app):
        """A hard failure must not be reported as a matter of taste."""
        text = self._hint(app, 22.0, 6.0, 3.0, 5.0)
        assert "merge" in text

    def test_a_good_coil_says_so(self, app):
        text = self._hint(app, 22.0, 14.0, 3.0, 5.0)
        assert "reads as a coil" in text

    def test_the_hint_names_the_faster_cleaner_route(self, app):
        """This mode meshes a surface; its wall cannot be a clean lattice."""
        text = self._hint(app, 22.0, 14.0, 3.0, 5.0)
        assert "helix" in text and "lattice" in text

    def test_the_shipped_defaults_read_as_a_coil(self, app):
        """The defaults were an open, 1.5-turn, 7800-atom quarter-hour."""
        app.var_mode_kind.set("coil (relaxed)")
        app._update_coil_hint()
        assert "reads as a coil" in app.lbl_coil.cget("text")
