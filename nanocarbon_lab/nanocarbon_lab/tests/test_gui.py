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
        for colour in (True, False):
            app.var_show_bonds.set(bonds)
            app.var_colour_rings.set(colour)
            app._redraw()
            app.root.update()


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
