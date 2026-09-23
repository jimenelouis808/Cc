"""Tests for vibspec phase 2: settings, records and the relax-then-IR workflow.

The workflow runs end to end on a spring-and-point-charge calculator whose
answer is known analytically, so no DFT code is needed. One test runs real
GPAW; it is marked ``gpaw`` and ``slow`` and skipped when GPAW is absent.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes

from carbonforge.builders import build_finite_nanoribbon
from carbonforge.builders.nanoribbon import rebox
from carbonforge.cli.main import main as cli_main
from carbonforge.vibspec.core import (
    CalcRecord,
    CalcSpec,
    VibspecError,
    apply_preset,
    collect,
    gpaw_available,
    index_records,
    prepare,
    run,
)
from carbonforge.vibspec.core.engines import gpaw_parameters
from carbonforge.vibspec.core.workflow import RIGID_MODE_TOLERANCE_CM1


class SpringsAndCharges(Calculator):
    """Harmonic springs between every pair closer than ``cutoff``, fixed charges.

    Pair potentials are invariant under rigid motion, so the six rigid-body
    modes are exactly zero, and the dipole is ``sum(q_i r_i)``.
    """

    implemented_properties = ["energy", "forces", "dipole"]

    def __init__(self, reference: Atoms, charges, k: float = 30.0, cutoff: float = 2.6):
        super().__init__()
        self.charges = np.asarray(charges, dtype=float)
        self.k = k
        d = reference.get_all_distances()
        self.pairs = [(i, j, d[i, j]) for i in range(len(reference))
                      for j in range(i + 1, len(reference)) if d[i, j] < cutoff]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = self.atoms.get_positions()
        energy, forces = 0.0, np.zeros_like(positions)
        for i, j, r0 in self.pairs:
            vector = positions[j] - positions[i]
            r = np.linalg.norm(vector)
            energy += 0.5 * self.k * (r - r0) ** 2
            f = self.k * (r - r0) * vector / r
            forces[i] += f
            forces[j] -= f
        self.results = {
            "energy": energy,
            "forces": forces,
            "dipole": (self.charges[:, None] * positions).sum(axis=0),
        }


def springs_factory(reference: Atoms, charges, k: float = 30.0):
    def factory(spec, txt, spinpol):
        return SpringsAndCharges(reference, charges, k=k)
    return factory


def _water(perturb: bool = True) -> tuple[Atoms, Atoms]:
    reference = molecule("H2O")
    rebox(reference, 7.0)
    start = reference.copy()
    if perturb:
        start.positions[1] += [0.0, 0.04, 0.03]
    return reference, start


_WATER_CHARGES = [-0.8, 0.4, 0.4]


@pytest.fixture(scope="class")
def water_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("water")
    reference, start = _water()
    prepare(start, CalcSpec(), directory)
    record = run(directory, springs_factory(reference, _WATER_CHARGES))
    return directory, record


class TestWaterWorkflow:
    def test_done_with_every_state_logged(self, water_run):
        _, record = water_run
        assert record.status == "done"
        states = [entry["status"] for entry in record.history]
        assert states == ["prepared", "relaxing", "relaxed", "vibrations", "done"]

    def test_relaxed_to_criterion(self, water_run):
        _, record = water_run
        assert record.relax["converged"]
        assert record.relax["fmax"] <= 0.01

    def test_rigid_modes_removed(self, water_run):
        _, record = water_run
        results = record.results
        assert results["n_rigid"] == 6
        assert len(results["frequencies_cm1"]) == 3
        # Not zero: residual forces up to fmax give the rotations a frequency.
        assert results["rigid_max_cm1"] < RIGID_MODE_TOLERANCE_CM1
        assert all(f > 0 for f in results["frequencies_cm1"])
        assert results["warnings"] == []

    def test_files_are_relative_and_present(self, water_run):
        directory, record = water_run
        for name in ("initial", "relaxed", "modes", "script"):
            assert not record.files[name].startswith("/")
            assert (directory / record.files[name]).exists()
        with np.load(directory / "modes.npz") as data:
            assert data["modes"].shape == (9, 3, 3)

    def test_record_reloads_with_versions(self, water_run):
        directory, record = water_run
        loaded = CalcRecord.load(directory)
        assert loaded.results == record.results
        assert loaded.versions["carbonforge"] and loaded.versions["ase"]
        assert loaded.versions["prepared_with"] == loaded.versions["carbonforge"]
        assert loaded.spec == CalcSpec().to_dict()

    def test_collect(self, water_run):
        directory, record = water_run
        spectrum = collect(directory)
        assert spectrum.has_ir and len(spectrum) == 3
        assert "⚠️" not in spectrum.summary()
        assert np.allclose(spectrum.frequencies, record.results["frequencies_cm1"])

    def test_rerun_is_a_no_op(self, water_run):
        directory, record = water_run
        again = run(directory, springs_factory(*_water()[:1], _WATER_CHARGES))
        assert again.history == record.history


def test_rigid_modes_vanish_at_a_true_minimum_with_small_steps(tmp_path):
    # The rotations' residual frequency is a finite-difference error that
    # grows linearly with delta (~28 cm^-1 at 0.01 Å for water, ~3 at 0.001).
    reference, start = _water()
    prepare(start, CalcSpec(fmax=1e-5, max_steps=2000, delta=0.001), tmp_path)
    record = run(tmp_path, springs_factory(reference, _WATER_CHARGES))
    assert record.results["rigid_max_cm1"] < 5.0


class TestAnalyticDiatomic:
    """A harmonic diatomic with charges ±q: frequency and intensity in closed form."""

    @pytest.mark.parametrize("method", ["frederiksen", "standard"])
    def test_frequency_and_intensity(self, tmp_path, method):
        k, q = 40.0, 0.3
        reference = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]])
        rebox(reference, 7.0)
        prepare(reference, CalcSpec(ir_method=method), tmp_path / "co")
        record = run(tmp_path / "co", springs_factory(reference, [q, -q], k=k))

        assert record.results["n_rigid"] == 5
        (frequency,) = record.results["frequencies_cm1"]
        (intensity,) = record.results["ir_intensity"]

        masses = reference.get_masses()
        mu = masses[0] * masses[1] / masses.sum()
        omega = np.sqrt(k / mu * units._e * 1e20 / units._amu)       # rad/s
        expected_cm1 = omega / (2 * np.pi * units._c * 100)
        assert frequency == pytest.approx(expected_cm1, rel=1e-3)

        e_angstrom_in_debye = 1.0 / units.Debye
        assert intensity == pytest.approx((q * e_angstrom_in_debye) ** 2 / mu, rel=1e-2)


class TestRefusals:
    def test_prepare_refuses_spin_paired_open_shell(self, tmp_path):
        flake = apply_preset(build_finite_nanoribbon(4, 4, edge="zigzag"), "graphitic")
        with pytest.raises(VibspecError, match="espín"):
            prepare(flake, CalcSpec(spinpol=False), tmp_path / "g")
        assert not (tmp_path / "g").exists()

    def test_forced_prepare_still_refused_by_run(self, tmp_path):
        flake = apply_preset(build_finite_nanoribbon(4, 4, edge="zigzag"), "graphitic")
        prepare(flake, CalcSpec(spinpol=False), tmp_path / "g", force=True)
        with pytest.raises(VibspecError):
            run(tmp_path / "g", springs_factory(flake, np.zeros(len(flake))))
        record = CalcRecord.load(tmp_path / "g")
        assert record.status == "error"
        assert "VibspecError" in record.error

    def test_existing_directory(self, tmp_path):
        _, water = _water(perturb=False)
        prepare(water, CalcSpec(), tmp_path)
        with pytest.raises(FileExistsError):
            prepare(water, CalcSpec(), tmp_path)
        prepare(water, CalcSpec(xc="BLYP"), tmp_path, overwrite=True)
        assert CalcRecord.load(tmp_path).spec["xc"] == "BLYP"

    def test_unconverged_relaxation_stops_before_vibrations(self, tmp_path):
        reference, start = _water()
        start.positions[1] += [0.0, 0.3, 0.0]     # stretch an O-H bond
        prepare(start, CalcSpec(max_steps=1), tmp_path)
        with pytest.raises(VibspecError, match="no convergió"):
            run(tmp_path, springs_factory(reference, _WATER_CHARGES))
        record = CalcRecord.load(tmp_path)
        assert record.status == "error"
        assert not (tmp_path / "modes.npz").exists()

    def test_unclosed_pyrrolic_ring_stops_before_vibrations(self, tmp_path):
        flake = apply_preset(build_finite_nanoribbon(4, 4, edge="zigzag"), "pyrrolic_precursor")
        prepare(flake, CalcSpec(), tmp_path)
        with pytest.raises(VibspecError, match="pentágono"):
            run(tmp_path, springs_factory(flake, np.zeros(len(flake))))
        record = CalcRecord.load(tmp_path)
        assert record.status == "error"
        assert record.relax["converged"]
        assert not (tmp_path / "ir").exists()


class TestCalcSpec:
    def test_defaults_are_valid(self):
        assert CalcSpec().validate().ok

    def test_round_trip(self):
        spec = CalcSpec(mode="fd", h=0.16, spinpol=True, scale_factor=0.98)
        assert CalcSpec.from_dict(spec.to_dict()) == spec

    def test_unknown_parameter(self):
        with pytest.raises(ValueError, match="desconocidos"):
            CalcSpec.from_dict({"kpts": [4, 4, 1]})

    def test_plane_waves_warn_about_images(self):
        report = CalcSpec(mode="pw").validate()
        assert report.ok
        assert any("imágenes" in w for w in report.warnings)
        _, water = _water(perturb=False)          # 7 Å per side, under the PW 8 Å
        assert any("vacío por lado" in w for w in CalcSpec(mode="pw").validate(water).warnings)
        assert any("ecut" in w for w in CalcSpec(mode="pw", ecut=300).validate().warnings)

    def test_ir_method(self):
        assert not CalcSpec(ir_method="magic").validate().ok
        assert CalcSpec().ir_method == "frederiksen"

    def test_unknown_mode(self):
        assert not CalcSpec(mode="grid").validate().ok

    def test_loose_density_warned(self):
        spec = CalcSpec(convergence={"energy": 5e-4, "density": 1e-4})
        assert any("densidad" in w for w in spec.validate().warnings)

    def test_coarse_lcao_grid_warned(self):
        assert any("huevera" in w for w in CalcSpec(h=0.22).validate().warnings)

    def test_spin_resolved_from_structure(self):
        _, water = _water(perturb=False)
        assert CalcSpec().resolved_spinpol(water) is False
        flake = apply_preset(build_finite_nanoribbon(4, 4, edge="zigzag"), "graphitic")
        assert CalcSpec().resolved_spinpol(flake) is True
        assert CalcSpec(spinpol=True).resolved_spinpol(water) is True

    def test_gpaw_parameters(self):
        lcao = gpaw_parameters(CalcSpec(), spinpol=True)
        assert lcao["symmetry"] == "off" and lcao["basis"] == "dzp" and lcao["spinpol"]
        assert "basis" not in gpaw_parameters(CalcSpec(mode="fd"), spinpol=False)
        pw = gpaw_parameters(CalcSpec(mode="pw", ecut=500), spinpol=False)
        assert pw["mode"] == {"name": "pw", "ecut": 500} and "h" not in pw


class TestIndexAndCli:
    def test_index(self, tmp_path):
        reference, start = _water()
        for name in ("a", "b"):
            prepare(start, CalcSpec(), tmp_path / "runs" / name)
        run(tmp_path / "runs" / "a", springs_factory(reference, _WATER_CHARGES))
        db = tmp_path / "vibspec.db"
        assert index_records(tmp_path / "runs", db) == 2
        assert index_records(tmp_path / "runs", db) == 2   # replaces, not duplicates
        with sqlite3.connect(db) as connection:
            (rows,) = connection.execute("select count(*) from systems").fetchone()
        assert rows == 2

    def test_prepare_show_index(self, tmp_path, capsys):
        structure = tmp_path / "amine.xyz"
        assert cli_main(["vibspec", "build", "--edge", "zigzag", "--width", "4",
                         "--length", "4", "--preset", "amine", "-o", str(structure)]) == 0
        directory = tmp_path / "runs" / "amine"
        assert cli_main(["vibspec", "prepare", str(structure), "-d", str(directory)]) == 0
        record = json.loads((directory / "record.json").read_text(encoding="utf-8"))
        assert record["status"] == "prepared"
        assert record["preset"]["key"] == "amine"
        assert cli_main(["vibspec", "show", str(tmp_path / "runs")]) == 0
        assert cli_main(["vibspec", "index", str(tmp_path / "runs"),
                         "--db", str(tmp_path / "v.db")]) == 0
        assert "prepared" in capsys.readouterr().out

    def test_prepare_refusal_exit_code(self, tmp_path):
        flake = apply_preset(build_finite_nanoribbon(4, 4, edge="zigzag"), "graphitic")
        path = tmp_path / "g.xyz"
        flake.write(path)
        assert cli_main(["vibspec", "prepare", str(path), "-d", str(tmp_path / "g"),
                         "--spinpol", "off"]) == 1

    def test_run_on_finished_directory_needs_no_gpaw(self, water_run_dir):
        assert cli_main(["vibspec", "run", str(water_run_dir)]) == 0


@pytest.fixture
def water_run_dir(tmp_path):
    reference, start = _water()
    prepare(start, CalcSpec(), tmp_path)
    run(tmp_path, springs_factory(reference, _WATER_CHARGES))
    return tmp_path


@pytest.mark.slow
@pytest.mark.gpaw
@pytest.mark.skipif(not gpaw_available(), reason="GPAW no está instalado")
def test_real_gpaw_n2(tmp_path):
    """N2 in LCAO: relax and IR with real GPAW. A few minutes, serial."""
    n2 = Atoms("N2", positions=[[0, 0, 0], [0, 0, 1.12]])
    rebox(n2, 6.5)
    prepare(n2, CalcSpec(h=0.18), tmp_path)
    record = run(tmp_path)
    assert record.status == "done"
    assert record.results["n_rigid"] == 5
    (frequency,) = record.results["frequencies_cm1"]
    # PBE puts the N2 stretch near 2350 cm^-1 (experiment: 2359 harmonic);
    # a generous window, since this is LCAO-dzp on a coarse grid.
    assert 2150 < frequency < 2550
    # A homonuclear diatomic has no dipole derivative: IR-inactive.
    assert record.results["ir_intensity"][0] < 1e-2
    # With the Frederiksen correction the translations are gone; what is
    # left of the rigid modes is the rotations on the LCAO grid.
    assert record.results["rigid_max_cm1"] < 150
