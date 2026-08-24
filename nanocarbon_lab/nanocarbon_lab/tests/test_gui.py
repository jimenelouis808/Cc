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
