"""The GUI: its Qt-free calculations always, the window when PySide6 and pyvista exist."""

from __future__ import annotations

import os

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

    def test_check_model_refuses_missing_elements_and_periodic_scc(self):
        assert actions.check_model(molecule("CH3SH"), actions.load_model("chno"))
        ribbon = graphene_nanoribbon(2, 1, type="zigzag", saturated=True, vacuum=5.0)
        chn = actions.load_model("chn")
        assert any("Ewald" in p for p in actions.check_model(ribbon, chn))
        assert not actions.check_model(ribbon, chn, scc=False)       # on purpose only
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
