"""Raman activities in vibspec: invariants, the workflow step, and its two methods.

No DFT here: the "field" method runs against a stand-in calculator whose
dipole responds to a uniform field through the bond-polarizability model,
so it must reproduce the "bond" result exactly -- which checks the finite-
field arithmetic, the units and the caching in one go.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from carbonforge.builders.nanoribbon import rebox
from carbonforge.tests.test_vibspec_workflow import (
    _WATER_CHARGES,
    SpringsAndCharges,
    _water,
    springs_factory,
)
from carbonforge.vibspec.core import CalcSpec, prepare, run
from carbonforge.vibspec.core.raman import (
    _COULOMB,
    bond_polarizability,
    field_polarizability,
    invariants,
    polarizability_derivatives,
    raman_activities,
    raman_progress,
)
from carbonforge.vibspec.core.workflow import collect


class FieldResponsive(SpringsAndCharges):
    """Springs and charges, plus an induced dipole alpha(r) E (bond model)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.field = np.zeros(3)

    def set_uniform_field(self, strength, axis):
        self.field = np.zeros(3)
        self.field[axis] = strength
        self.reset()

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        super().calculate(atoms, properties, system_changes or
                          ["positions", "numbers", "cell", "pbc"])
        alpha = bond_polarizability(self.atoms) / _COULOMB     # e Å² / V
        self.results["dipole"] = self.results["dipole"] + alpha @ self.field


class TestInvariants:
    def test_isotropic_and_anisotropic(self):
        assert invariants(np.eye(3)) == pytest.approx((1.0, 0.0))
        mean, gamma2 = invariants(np.diag([1.0, 0.0, 0.0]))
        assert mean == pytest.approx(1 / 3) and gamma2 == pytest.approx(1.0)

    def test_co2_selection_rule(self):
        """Centrosymmetric CO2: the symmetric stretch is Raman-active, the
        antisymmetric one is not (and vice versa in the IR)."""
        co2 = molecule("CO2")
        rebox(co2, 6.0)
        assert co2.get_chemical_symbols() == ["C", "O", "O"]
        masses = co2.get_masses()
        sym = np.zeros((3, 3))                 # oxygens out of phase, carbon still
        sym[1, 2], sym[2, 2] = -1.0, 1.0
        anti = np.zeros((3, 3))                # oxygens in phase, carbon against them
        anti[1, 2], anti[2, 2], anti[0, 2] = 1.0, 1.0, -2 * masses[1] / masses[0]
        modes = np.array([sym / np.sqrt(masses)[:, None], anti / np.sqrt(masses)[:, None]])
        derivative = polarizability_derivatives(co2, bond_polarizability,
                                                cache=_tmp(), save=False)
        activities, ratios = raman_activities(derivative, modes)
        assert activities[0] > 1e-3 and activities[1] < 1e-6 * activities[0]
        assert 0.0 <= ratios[0] < 0.75


def _tmp():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp()) / "raman"


def test_field_reproduces_the_model_it_is_given():
    water = molecule("H2O")
    rebox(water, 7.0)
    water.calc = FieldResponsive(water, _WATER_CHARGES)
    assert field_polarizability(water, 0.05) == pytest.approx(bond_polarizability(water),
                                                              rel=1e-6, abs=1e-9)
    assert not water.calc.field.any()                  # field switched off afterwards


def test_derivatives_are_cached_and_resumed(tmp_path):
    water = molecule("H2O")
    calls = []

    def counting(atoms: Atoms):
        calls.append(1)
        return bond_polarizability(atoms)

    first = polarizability_derivatives(water, counting, tmp_path / "raman")
    assert len(calls) == 18 and raman_progress(tmp_path / "raman", 3) == \
        "raman: 18/18 polarizabilidades"
    second = polarizability_derivatives(water, counting, tmp_path / "raman")
    assert len(calls) == 18 and np.allclose(first, second)
    assert np.allclose(water.positions, molecule("H2O").positions)   # left in place


class TestWorkflow:
    @pytest.mark.parametrize("method", ["bond", "field"])
    def test_water_ir_and_raman(self, tmp_path, method):
        reference, start = _water()
        spec = CalcSpec(raman=method, convergence={"density": 1e-7, "energy": 5e-5,
                                                   "eigenstates": 1e-8})
        prepare(start, spec, tmp_path)

        def factory(spec, txt, spinpol):
            return FieldResponsive(reference, _WATER_CHARGES)

        record = run(tmp_path, factory if method == "field" else
                     springs_factory(reference, _WATER_CHARGES))
        states = [entry["status"] for entry in record.history]
        assert states[-3:] == ["vibrations", "raman", "done"]
        results = record.results
        assert results["raman_method"] == method and results["raman_unit"] == "Å⁴/amu"
        assert len(results["raman_activity"]) == 3                 # water: 3 internal modes
        assert all(a > 0 for a in results["raman_activity"])       # all Raman-active
        assert all(0.0 <= r <= 0.75 + 1e-9 for r in results["depolarization"])
        spectrum = collect(tmp_path)
        assert spectrum.has_raman and spectrum.has_ir
        assert spectrum.activities("raman").shape == (3,)

    def test_both_methods_agree_when_the_physics_is_the_same(self, tmp_path):
        reference, start = _water()
        results = {}

        def factory(spec, txt, spinpol):
            return FieldResponsive(reference, _WATER_CHARGES)

        for method in ("bond", "field"):
            directory = tmp_path / method
            prepare(start, CalcSpec(raman=method), directory)
            results[method] = run(directory, factory).results["raman_activity"]
        assert results["field"] == pytest.approx(results["bond"], rel=1e-4)

    def test_off_by_default(self, tmp_path):
        reference, start = _water()
        prepare(start, CalcSpec(), tmp_path)
        record = run(tmp_path, springs_factory(reference, _WATER_CHARGES))
        assert "raman_activity" not in record.results
        assert not collect(tmp_path).has_raman


class TestValidation:
    def test_field_in_plane_waves_is_refused(self):
        assert any("PW" in e for e in CalcSpec(raman="field", mode="pw").validate().errors)

    def test_bond_model_warns_and_checks_elements(self):
        report = CalcSpec(raman="bond").validate()
        assert any("empírico" in w for w in report.warnings)
        cl = Atoms("HCl", positions=[[0, 0, 0], [0, 0, 1.27]])
        rebox(cl, 7.0)
        assert any("Cl" in e for e in CalcSpec(raman="bond").validate(cl).errors)

    def test_unknown_method(self):
        assert not CalcSpec(raman="resonant").validate().ok


@pytest.fixture(scope="module")
def raman_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("raman_water")
    reference, start = _water()
    prepare(start, CalcSpec(raman="bond"), directory)
    run(directory, springs_factory(reference, _WATER_CHARGES))
    return directory


class TestAnalysis:
    def test_raman_curve_with_and_without_factors(self, raman_run):
        from carbonforge.vibspec.core.analysis import RAMAN_WINDOW, computed_curve

        spectrum = collect(raman_run)
        grid = np.linspace(*RAMAN_WINDOW, 3000)
        bare = computed_curve(spectrum, grid, kind="raman")
        measured = computed_curve(spectrum, grid, kind="raman", laser_nm=532, temperature_k=300)
        assert bare.max() > 0 and measured.max() > 0
        assert not np.allclose(bare / bare.max(), measured / measured.max())

    def test_raman_experiment_is_used_as_intensity(self, tmp_path):
        from carbonforge.vibspec.core.analysis import prepare_experiment, read_raman, \
            to_absorbance

        path = tmp_path / "raman.txt"
        x = np.linspace(100, 3600, 500)
        path.write_text("\n".join(f"{a:.1f}\t{50 + 900 * np.exp(-((a - 1590) / 12) ** 2):.2f}"
                                  for a in x))
        measured = read_raman(path)
        assert measured.quantity == "raman"
        with pytest.raises(ValueError, match="absorbancia"):
            to_absorbance(measured)
        wx, wy = prepare_experiment(measured)
        assert wy.max() == pytest.approx(1.0) and abs(wx[np.argmax(wy)] - 1590) < 10

    def test_cli_plot_raman(self, raman_run, tmp_path, capsys):
        from carbonforge.cli.main import main

        out = tmp_path / "r.png"
        assert main(["vibspec", "plot", str(raman_run), "--kind", "raman", "-o", str(out)]) == 0
        assert out.exists()

    def test_cli_plot_refuses_raman_without_it(self, tmp_path, capsys):
        from carbonforge.cli.main import main

        reference, start = _water()
        prepare(start, CalcSpec(), tmp_path)
        run(tmp_path, springs_factory(reference, _WATER_CHARGES))
        assert main(["vibspec", "plot", str(tmp_path), "--kind", "raman"]) == 1
        assert "--raman" in capsys.readouterr().out


class TestQEConnection:
    def test_dynmat_out_is_a_computed_spectrum(self, tmp_path):
        from carbonforge.tests.test_results import DYNMAT_FULL
        from carbonforge.vibspec.core.analysis import load_computed

        (tmp_path / "dynmat.out").write_text(DYNMAT_FULL)
        full = len(__import__("carbonforge.results.spectra", fromlist=["read_dynmat"])
                   .read_dynmat(tmp_path / "dynmat.out").modes)
        spectrum, origin = load_computed(tmp_path)
        near_zero = sum(abs(m.frequency_cm1) < 100 for m in __import__(
            "carbonforge.results.spectra", fromlist=["read_dynmat"]).read_dynmat(
            tmp_path / "dynmat.out").modes)
        assert origin.startswith("QE") and len(spectrum.modes) == full - min(near_zero, 6)
        assert spectrum.has_raman and len(spectrum.modes) > 0

    def test_finite_model_gets_zero_dim_sum_rule(self, tmp_path):
        from carbonforge.builders.nanoribbon import build_finite_nanoribbon
        from carbonforge.calculations.spectroscopy import raman_setup
        from carbonforge.exports.qe import write_qe_spectroscopy

        flake = build_finite_nanoribbon(3, 2)
        written = write_qe_spectroscopy(flake, tmp_path, raman_setup(), force=True)
        assert "asr = 'zero-dim'" in written["dynmat"].read_text()
