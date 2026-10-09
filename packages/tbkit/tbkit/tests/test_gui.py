"""The GUI: its Qt-free calculations always, the window when PySide6 and pyvista exist."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import graphene_nanoribbon, molecule

from tbkit.gui import actions


class TestActions:
    def test_suggested_model_covers_the_elements(self):
        assert actions.suggest_model(molecule("C6H6")) == "chn"
        assert actions.suggest_model(molecule("CH3COOH")) == "chno"
        assert actions.suggest_model(molecule("CH3SH")) == "chnos"
        assert actions.suggest_model(molecule("C60")) == "sp3"

    def test_check_model_refuses_missing_elements_and_cells_without_volume(self):
        assert actions.check_model(molecule("CH3SH"), actions.load_model("chno"))
        ribbon = graphene_nanoribbon(2, 1, type="zigzag", saturated=True, vacuum=5.0)
        chn = actions.load_model("chn")
        assert not actions.check_model(ribbon, chn)                  # Ewald, vacuum cell
        flat = ribbon.copy()
        flat.cell[0] = 0.0
        assert any("Ewald" in p for p in actions.check_model(flat, chn))
        assert actions.check_model(ribbon, actions.load_model("sp3"))  # no H in Xu

    def test_ground_state_follows_the_model_and_levels_are_consistent(self):
        atoms = molecule("C5H5N")
        state = actions.ground_state(atoms, actions.load_model("chn"))
        assert state.scc and state.converged
        rows = actions.levels_table(state, around=3)
        homo = next(r for r in rows if r[3] == "HOMO")
        lumo = next(r for r in rows if r[3] == "LUMO")
        assert homo[2] == pytest.approx(2.0) and lumo[2] == pytest.approx(0.0, abs=1e-6)
        assert lumo[1] - homo[1] == pytest.approx(state.info["gap"])
        assert sum(state.charges.values()) == pytest.approx(0.0, abs=1e-6)

    def test_pdos_adds_up_to_the_dos(self):
        state = actions.ground_state(molecule("CH3OH"), actions.load_model("chno"))
        curves = actions.dos_curves(state, 0.1, "element")
        assert np.allclose(sum(curves["projected"].values()), curves["total"], atol=1e-8)

    def test_orbital_grid_holds_the_orbital(self):
        # The model is orthogonal but the orbital is drawn with Slater functions
        # that overlap between neighbours: ∫ψ² = cᵀ S c, not exactly 1.
        state = actions.ground_state(molecule("C2H4"), actions.load_model("chn"))
        grid = actions.orbital_grid(state, 5, spacing=0.15)
        norm = np.sum(grid["values"] ** 2) * np.prod(grid["spacing"])
        assert 0.8 < norm < 1.6

    def test_sulfur_orbitals_can_be_drawn(self):
        state = actions.ground_state(molecule("CH3SH"), actions.load_model("chnos"))
        assert np.isfinite(actions.orbital_grid(state, 3)["values"]).all()

    def test_bands_of_a_ribbon(self):
        ribbon = graphene_nanoribbon(2, 1, type="armchair", saturated=True, vacuum=5.0)
        curves = actions.band_curves(ribbon, actions.load_model("pi"), npoints=20)
        assert curves["energies"].shape[0] == 20 and curves["labels"][0] in ("G", "Γ")

    def test_bands_refuse_finite_structures(self):
        with pytest.raises(ValueError, match="periódica"):
            actions.band_curves(molecule("C6H6"), actions.load_model("pi"))


def _qt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication

        import pyvista  # noqa: F401
    except ImportError as error:
        pytest.skip(f"GUI sin dependencias: {error}")
    return QApplication.instance() or QApplication([])


def _wait(window):
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(lambda: None if window.runner.running else loop.quit())
    timer.start(20)
    loop.exec()


def test_window_computes_levels_and_draws_an_orbital(tmp_path):
    _qt()
    from tbkit.gui.app import MainWindow

    window = MainWindow(interactive=False)
    window.set_atoms(molecule("C5H5N"), "piridina")
    assert window.model_name == "chn"
    electronic = window.pages["Electrónica"]
    electronic.compute.click()
    _wait(window)
    assert "gap 3." in electronic.summary.text()
    orbitals = window.pages["Orbitales"]
    orbitals.list.selectRow(4)
    _wait(window)
    assert orbitals.grid is not None and "E =" in orbitals.hint.text()
    window.colour.setCurrentIndex(1)                  # colour by charge
    window.view.screenshot(str(tmp_path / "view.png"))
    assert (tmp_path / "view.png").stat().st_size > 1000


def test_window_changing_charge_drops_the_ground_state():
    _qt()
    from tbkit.gui.app import MainWindow

    window = MainWindow(interactive=False)
    window.set_atoms(molecule("C6H6"), "benceno")
    window.pages["Electrónica"].compute.click()
    _wait(window)
    assert window.state is not None
    window.charge.setValue(1.0)
    assert window.state is None


def test_hubbard_zigzag_ribbon_is_antiferromagnetic_between_edges():
    ribbon = graphene_nanoribbon(4, 1, type="zigzag", saturated=False, vacuum=6.0)
    model = actions.load_model("pi")
    out = actions.hubbard_solution(ribbon, model, U=3.0, kmesh=40)
    assert out["converged"] and abs(out["magnetization"]) < 1e-6      # Lieb: S = 0
    assert out["moments"][0] * out["moments"][-1] < 0                  # opposite edges
    rows = {r["guess"]: r for r in actions.compare_guesses(ribbon, model, U=3.0, kmesh=40)}
    assert rows["antiferro"]["delta"] == 0.0 < rows["ferro"]["delta"]


def test_window_magnetism_page():
    _qt()
    from tbkit.gui.app import MainWindow

    window = MainWindow(interactive=False)
    window.set_model("pi")
    window.set_atoms(graphene_nanoribbon(3, 1, type="zigzag", saturated=False, vacuum=6.0),
                     "cinta")
    page = window.pages["Magnetismo"]
    page.use_model_u.setCurrentIndex(1)
    page.U.setValue(3.0)
    page.solve.click()
    _wait(window)
    assert "convergido" in page.summary.text() and page.moments.rowCount() == 6
    page.sweep_kind.setCurrentIndex(1)
    page.sweep_from.setValue(0.0)
    page.sweep_to.setValue(1.0)
    page.sweep_n.setValue(3)
    page.sweep_button.click()
    _wait(window)


def test_relax_and_modes_of_water():
    model = actions.load_model("chno")
    out = actions.relax_structure(molecule("H2O"), model, fmax=0.005)
    assert out["converged"] and out["energy"] <= out["energy_start"] + 1e-9
    modes = actions.vibration_modes(out["atoms"], model)
    frequencies = np.sort(modes["vibrations"].frequencies)[6:]
    assert len(frequencies) == 3 and frequencies.min() > 1000     # bend, two stretches
    assert "H" in modes["rows"][-1][2] and not modes["warnings"]


def test_pi_model_refuses_relaxation():
    with pytest.raises(ValueError, match="repulsiva"):
        actions.relax_structure(molecule("C6H6"), actions.load_model("pi"))


def test_window_geometry_page():
    _qt()
    from tbkit.gui.app import MainWindow

    window = MainWindow(interactive=False)
    window.set_atoms(molecule("H2O"), "agua")
    page = window.pages["Geometría y modos"]
    page.relax.click()
    _wait(window)
    assert "convergida" in page.summary.text() and page.undo.isEnabled()
    page.modes_button.click()
    _wait(window)
    assert page.table.rowCount() == 9
    page.table.selectRow(8)                      # animate the highest mode
    page.arrows()
    page.restore()
    assert not page.undo.isEnabled()


def test_window_sorts_modes_by_site_compares_scc_and_another_model():
    _qt()
    from tbkit.gui.app import MainWindow

    window = MainWindow(interactive=False)
    window.set_atoms(molecule("C5H5N"), "piridina")
    page = window.pages["Geometría y modos"]
    page.modes_button.click()
    _wait(window)
    names = [page.site.itemText(k) for k in range(page.site.count())]
    assert "N" in names and "anillos de 6" in names
    page.site.setCurrentIndex(names.index("N"))
    first = page.row_modes[0]
    assert page.table.item(0, 0).text() == str(first)
    page.table.selectRow(0)
    page.other_model.setCurrentIndex(page.other_model.findData("chno"))
    page.run_other()
    _wait(window)
    assert f"Modo {first}" in page.summary.text()
    electronic = window.pages["Electrónica"]
    electronic.scc_button.click()
    _wait(window)
    assert electronic.scc_summary.rowCount() == 5 and electronic.scc_charges.rowCount() == 11


def test_window_phonopy_page():
    _qt()
    pytest.importorskip("phonopy")
    from ase.build import graphene

    from tbkit.gui.app import MainWindow

    window = MainWindow(interactive=False)
    sheet = graphene(a=2.46, vacuum=6.0)
    sheet.pbc = [True, True, False]
    window.set_atoms(sheet, "grafeno")
    page = window.pages["Fonones (ZB)"]
    page.supercell[0].setValue(3)
    page.supercell[1].setValue(3)
    page.kmesh.setValue(6)
    page.run_button.click()
    _wait(window)
    assert "6/mmm" in page.summary.text()
    labels = [page.irreps.item(r, 2).text() for r in range(page.irreps.rowCount())]
    assert "E2g" in labels


def test_runner_collects_garbage_only_in_the_gui_thread():
    # The cyclic collector running in the worker destroyed Qt objects of the
    # GUI thread and crashed the window at random places: it is paused while a
    # job runs and resumed (with a collection) in the GUI thread afterwards.
    import gc

    _qt()
    from tbkit.gui.worker import Runner

    runner = Runner()
    seen = []
    runner.start("prueba", lambda: gc.isenabled(), on_done=seen.append)

    class _Window:
        pass

    window = _Window()
    window.runner = runner
    _wait(window)
    assert seen == [False] and gc.isenabled()
    for _ in range(3):                       # consecutive jobs, old threads released
        runner.start("otra", lambda: 1)
        _wait(window)
    assert len(runner._finished) == 1


def test_spectra_reuse_the_shared_modes():
    model = actions.load_model("chn")
    atoms = actions.relax_structure(molecule("C2H4"), model, fmax=0.005)["atoms"]
    vib = actions.vibration_modes(atoms, model)["vibrations"]
    phonons = (vib.frequencies, vib.modes)
    raman = actions.raman_spectrum(atoms, model, phonons=phonons)
    ir = actions.ir_spectrum_of(atoms, model, phonons=phonons)
    # Ethylene is centrosymmetric: Raman and IR modes are mutually exclusive.
    raman_freqs = {round(r[0]) for r in raman["rows"]}
    ir_freqs = {round(r[0]) for r in ir["rows"] if r[2] > 1e-3 * max(x[2] for x in ir["rows"])}
    assert raman_freqs and ir_freqs and not raman_freqs & ir_freqs
    resonant = actions.resonant_spectrum(atoms, model, [2.0, 3.0], phonons=phonons)
    assert resonant["activities"].shape == (2, len(vib.frequencies) - 6)


def test_write_csv(tmp_path):
    path = actions.write_csv(tmp_path / "s.csv", {"x": np.arange(3.0), "y": np.ones(3)})
    assert path.read_text().splitlines()[0] == "x,y"


def test_window_spectra_page():
    _qt()
    from tbkit.gui.app import MainWindow

    window = MainWindow(interactive=False)
    window.set_atoms(molecule("CO2"), "CO2")
    geometry, spectra = window.pages["Geometría y modos"], window.pages["Espectros"]
    geometry.relax.click()
    _wait(window)
    geometry.modes_button.click()
    _wait(window)
    assert window.phonons is not None
    spectra.raman_button.click()
    _wait(window)
    assert spectra.table.rowCount() >= 1
    spectra.ir_button.click()
    _wait(window)
    assert "μ =" in spectra.summary.text()
    spectra.lasers.setText("2.0 4.0")
    spectra.resonant_button.click()
    _wait(window)
    spectra.table.selectRow(0)


def test_record_keeps_numbers_and_drops_objects(tmp_path):
    import json

    model = actions.load_model("chn")
    atoms = molecule("C6H6")
    state = actions.ground_state(atoms, model)
    data = actions.record(atoms, model, "estado fundamental", {"charge": 0.0},
                          {"info": state.info, "state": state, "charges": state.charges})
    path = tmp_path / "r.json"
    path.write_text(json.dumps(data))
    again, model_again, back = actions.open_record(path)
    assert back["task"] == "estado fundamental" and "state" not in back["results"]
    assert len(again) == 12 and model_again.onsite == model.onsite


@pytest.mark.slow
def test_graphene_spectra_coarse():
    out = actions.graphene_spectra([2.0, 2.6], dk=0.05, dq=0.1)
    assert 1500 < out["rows"][0][1] < 1700                 # G
    assert out["dispersion_2d"] > 0                        # 2D moves up with the laser


def test_window_saves_and_reopens_a_record(tmp_path):
    _qt()
    from tbkit.gui.app import MainWindow

    window = MainWindow(interactive=False)
    window.set_atoms(molecule("C5H5N"), "piridina")
    window.pages["Electrónica"].compute.click()
    _wait(window)
    path = window.save_record(str(tmp_path / "registro.json"))
    assert path and window.last_record[2] == "estado fundamental"
    other = MainWindow(interactive=False)
    other.load_record(path)
    assert len(other.atoms) == 11 and "registro" in other.model_label.text()


def test_window_terminal_shows_commands_output_and_errors(tmp_path):
    _qt()
    from tbkit.gui.app import MainWindow

    window = MainWindow(interactive=False)
    window.set_atoms(molecule("H2O"), "agua")
    geometry = window.pages["Geometría y modos"]
    geometry.relax.click()
    _wait(window)
    _wait(window)                              # let the queued output arrive
    text = window.console.text()
    assert "estructura: agua" in text and "relax_structure(H2O" in text
    assert "BFGS" in text and "✓ Relajación" in text
    window.set_model("pi")
    window.set_atoms(molecule("C60"), "C60")
    assert window.model_name == "pi"
    geometry.relax.click()
    _wait(window)
    text = window.console.text()
    assert "✗ Relajación" in text and "Traceback" in text
    assert window.console.save(tmp_path / "t.log").read_text() == text


def test_every_control_and_panel_of_the_window_has_help():
    """No Qt needed: every label app.py gives a button, box or form row has a text."""
    import re
    from pathlib import Path

    source = (Path(actions.__file__).parent / "app.py").read_text(encoding="utf-8")
    labels = set(re.findall(r'Q(?:PushButton|CheckBox|GroupBox)\("([^"]+)"\)', source))
    labels |= set(re.findall(r'addRow\("([^"]+)"', source))
    assert labels and sorted(labels - set(actions.HELP)) == []
    titles = set(re.findall(r'title = "([^"]+)"', source))
    assert sorted(titles - set(actions.PANEL_HELP)) == []


def test_scaled_spectra_multiply_the_model_frequencies_only_when_asked():
    from tbkit.tasks import frequency_scale

    model = actions.load_model("chno")
    water = molecule("H2O")
    raw = actions.ir_spectrum_of(water, model)
    scaled = actions.ir_spectrum_of(water, model, scale=True)
    factor = frequency_scale(model)
    assert raw["frequency_scale"] is None and scaled["frequency_scale"] == factor
    assert [r[0] for r in scaled["rows"]] == pytest.approx([factor * r[0] for r in raw["rows"]])
    assert [r[2] for r in scaled["rows"]] == pytest.approx([r[2] for r in raw["rows"]])
    # A set without a factor is left alone even when asked.
    pi = actions.load_model("pi")
    assert actions.scaled(type("R", (), {})(), pi, True)[1] is None


def test_window_shows_panel_descriptions_and_tooltips():
    _qt()
    from tbkit.gui.app import MainWindow

    window = MainWindow(interactive=False)
    assert len(window.info_boxes) == len(window.pages)
    spectra = window.pages["Espectros"]
    assert spectra.raman_button.toolTip() == actions.HELP["Raman"]
    assert spectra.scale.toolTip() == actions.HELP["Escalar frecuencias"]
    assert spectra.fwhm.toolTip() == actions.HELP["Raman / IR"]
    window.show_descriptions(False)
    assert all(not box.isVisibleTo(window) for box in window.info_boxes)


def test_saved_spectra_open_from_csv_and_npz(tmp_path):
    grid = np.linspace(100, 1900, 50)
    curve = np.exp(-((grid - 1580) / 40) ** 2)
    csv = actions.write_csv(tmp_path / "s.csv", {"shift": grid, "intensity": curve})
    saved = actions.load_spectrum(csv)
    assert list(saved["curves"]) == ["intensity"] and np.allclose(saved["grid"], grid)
    np.savez(tmp_path / "spectra.npz", grid=grid, **{"2.33": curve, "1.96": 2 * curve},
             frequencies=np.array([1580.0, 1350.0]), activities=np.array([[1.0, 0.2], [0.5, 0.9]]))
    saved = actions.load_spectrum(tmp_path / "spectra.npz")
    assert set(saved["curves"]) == {"2.33 eV", "1.96 eV"}
    assert len(saved["sticks"]) == 2
    with pytest.raises(ValueError):
        actions.load_spectrum(tmp_path / "s.txt")
    _qt()
    from tbkit.gui.app import MainWindow

    window = MainWindow(interactive=False)
    page = window.pages["Espectros"]
    page.open_spectrum(str(tmp_path / "spectra.npz"))
    assert "2 curvas" in page.summary.text() and page.table.rowCount() == 2


def test_recipe_settings_overrides_and_command(tmp_path):
    recipes = {r["name"] for r in actions.recipe_list()}
    assert "coil_double_resonance" in recipes
    usage = actions.recipe_usage("coil_double_resonance")
    assert "pairs" in usage["steps"] and usage["parts"]
    rows = {r["name"]: r for r in actions.recipe_settings("coil_double_resonance")}
    edited = {name: row["value"] for name, row in rows.items()}
    assert actions.recipe_overrides("coil_double_resonance", edited) == {}
    edited["NK"] = "32"
    edited["GAMMA"] = "0.05"
    assert actions.recipe_overrides("coil_double_resonance", edited) == {"NK": 32, "GAMMA": 0.05}
    with pytest.raises(ValueError, match="NK"):
        actions.recipe_overrides("coil_double_resonance", {**edited, "NK": "2.5"})
    command = actions.launch_command("coil_double_resonance", "pairs 2.54", {"NK": 32},
                                     parts=4, workdir=tmp_path)
    assert command[3:5] == ["lanzar", "--partes"] and command[-4:-2] == ["pairs", "2.54"]
    saved = Path(command[-1])
    assert command[-2] == "--ajustes" and json.loads(saved.read_text()) == {"NK": 32}
    assert actions.launch_warnings(4, 1, 1, can_split=False)
    assert actions.launch_warnings(actions.cores() + 1, 1) and not actions.launch_warnings(1, 1)


def test_recipe_options_are_read_from_the_parser_without_running_it():
    names = {r["name"] for r in actions.recipe_list()}
    assert {"site_screening", "crystal_validation", "validation_report"} <= names
    usage = actions.recipe_usage("validation_report")       # main() without argv: not called
    assert usage["options"] == {"": []} and not usage["steps"]
    scaling = {r["dest"]: r for r in actions.recipe_usage("frequency_scaling")["options"][""]}
    assert scaling["solo_mostrar"]["boolean"] and scaling["min_cm1"]["default"] == "50.0"
    site = actions.recipe_usage("site_screening")
    rows = {r["dest"]: r for r in site["options"][""]}
    assert site["parts"] and rows["structure"]["required"] and rows["kmesh"]["default"] == "8"
    with pytest.raises(ValueError, match="structure"):
        actions.recipe_arguments(site["options"][""], {})
    words = actions.recipe_arguments(site["options"][""],
                                     {"structure": "coil.extxyz", "dopant": "N",
                                      "workdir": "out/sitios", "kmesh": "4", "top": "3"})
    assert words == ["coil.extxyz", "N", "out/sitios", "--kmesh", "4"]
    crystals = actions.recipe_usage("crystal_validation")
    assert crystals["steps"] == ["gpaw", "collect", "compare"]
    systems = {r["dest"]: r for r in crystals["options"]["gpaw"]}["systems"]
    assert actions.recipe_arguments([systems], {"systems": "graphene hBN"}) == \
        ["--systems", "graphene", "hBN"]
    raman = actions.recipe_usage("doped_raman")
    assert raman["step_dest"] == "step" and "raman" in raman["steps"]
    with pytest.raises(ValueError, match="no es una de"):
        actions.recipe_arguments(raman["options"][""], {"name": "boro"})


def test_recipe_status_lines(tmp_path):
    from tbkit.progress import Progress

    with Progress(4, "prueba", status=tmp_path / "a" / "progreso.json", quiet=True) as bar:
        bar.step()
    assert "prueba: 1/4" in actions.recipe_status(tmp_path)[0]
    bar.step(3)
    assert actions.recipe_status(tmp_path) == [] and actions.recipe_status(tmp_path, True)


def test_recipes_page_builds_a_launch_command():
    _qt()
    from tbkit.gui.app import MainWindow

    window = MainWindow(interactive=False)
    page = window.pages["Recetas"]
    page.recipe.setCurrentIndex(page.recipe.findData("coil_double_resonance"))
    page.step.setCurrentText("pairs")
    page.options.item(0, 1).setText("2.54")                  # the laser
    assert page.command()[-2:] == ["pairs", "2.54"]
    assert page.settings.rowCount() == len(actions.recipe_settings("coil_double_resonance"))
    page.recipe.setCurrentIndex(page.recipe.findData("site_screening"))     # no steps
    assert page.options.rowCount() == len(actions.recipe_usage("site_screening")["options"][""])
    page.recipe.setCurrentIndex(page.recipe.findData("crystal_validation"))
    page.step.setCurrentText("compare")
    assert [page.options.item(r, 0).text() for r in range(page.options.rowCount())] == \
        ["references (obligatorio)", "out (obligatorio)", "--systems"]
