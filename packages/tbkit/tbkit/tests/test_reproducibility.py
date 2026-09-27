"""Parameter provenance, numerical checks, and reproducible runs.

From the development directive adopted for tbkit: every built-in number
has a unit and a source; H = H† and S > 0 are checked, never assumed;
eigenvectors satisfy C† S C = 1; a run is saved with everything needed to
redo it, and redoing it gives the same numbers.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.io import write

from tbkit import System, pi_model, solve, xu_carbon
from tbkit.cli import main
from tbkit.hamiltonian import check_hermitian
from tbkit.params import (
    PARAMETER_DIR,
    Constant,
    TBModel,
    load_parameters,
    model_from_dict,
    model_to_dict,
    read_parameter_file,
)
from tbkit.record import replay, run_simulation
from tbkit.tests.test_energy import nonorthogonal_model, ring


class TestProvenance:
    def test_every_file_has_reference_system_and_validity(self):
        for path in PARAMETER_DIR.glob("*.json"):
            data = read_parameter_file(path)
            for key in ("name", "reference", "system", "validity", "units"):
                assert data.get(key), f"{path.name}: falta '{key}'"

    def test_every_xu_number_has_unit_and_source(self):
        data = read_parameter_file("xu_carbon")
        entries = [v for table in data["onsite"].values() for v in table.values()]
        entries += data["hopping"] + [data["repulsive"]] + list(data["hubbard_u"].values())
        for entry in entries:
            assert entry.get("unit") and entry.get("source"), entry

    def test_pi_file_numbers_have_sources(self):
        data = read_parameter_file("pi_huckel")
        for key in ("t", "a_cc", "cutoff", "strain_beta", "hubbard_u_over_t"):
            assert data[key]["unit"] and data[key]["source"]
        assert all(entry["source"] for entry in data["heteroatoms"].values())

    def test_models_carry_their_provenance(self):
        assert "Xu" in xu_carbon().metadata["reference"]
        assert "Castro Neto" in pi_model().metadata["reference"]

    def test_round_trip_keeps_every_number(self):
        model = xu_carbon()
        again = model_from_dict(model_to_dict(model))
        diamond = bulk("C", "diamond", a=3.56)
        from tbkit.forces import energy_and_forces
        from tbkit.kpoints import mesh

        k, w = mesh(diamond, 3)
        e1 = energy_and_forces(solve(System.build(diamond, model), k, w), False)[0]
        e2 = energy_and_forces(solve(System.build(diamond, again), k, w), False)[0]
        assert e1 == pytest.approx(e2, abs=1e-12)

    def test_unknown_set(self):
        with pytest.raises(FileNotFoundError, match="xu_carbon"):
            load_parameters("no_existe")


class TestNumericalChecks:
    def test_hermiticity_is_enforced(self):
        with pytest.raises(ValueError, match="hermítica"):
            check_hermitian(np.array([[0.0, 1.0], [0.5, 0.0]]), "H")

    def test_overlap_must_be_positive_definite(self):
        model = TBModel("S enorme", orbitals={"H": ("s",)}, onsite={"H": {"s": 0.0}},
                        hopping={("H", "H", "sss"): Constant(-1.0, 2.0)},
                        overlap={("H", "H", "sss"): Constant(1.2, 2.0)}, valence={"H": 1.0})
        dimer = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]])
        with pytest.raises(ValueError, match="definida positiva"):
            solve(System.build(dimer, model))

    def test_eigenvectors_are_s_orthonormal(self):
        solution = solve(System.build(ring(6, seed=5), nonorthogonal_model()))
        c = solution.vectors[0, 0]
        s = solution.overlaps[0]
        assert c.T @ s @ c == pytest.approx(np.eye(len(c)), abs=1e-10)

    def test_zero_overlap_recovers_the_orthogonal_result(self):
        model = nonorthogonal_model()
        atoms = ring(6, seed=5)
        orthogonal = solve(System.build(atoms, TBModel(
            "sin S", model.orbitals, model.onsite, model.hopping, valence=model.valence)))
        zero = {key: Constant(0.0, 2.6) for key in model.overlap}
        with_zero = solve(System.build(atoms, TBModel(
            "S=0", model.orbitals, model.onsite, model.hopping, overlap=zero,
            valence=model.valence)))
        assert with_zero.energies == pytest.approx(orthogonal.energies, abs=1e-12)


class TestReproducibleRuns:
    def test_run_record_and_replay(self, tmp_path):
        write(tmp_path / "benceno.xyz", molecule("C6H6"))
        config = {"structure": "benceno.xyz", "model": "pi_huckel", "task": "levels",
                  "settings": {"kT": 0.001}, "output": "benceno.record.json"}
        (tmp_path / "sim.json").write_text(json.dumps(config))
        record = run_simulation(tmp_path / "sim.json")
        saved = json.loads((tmp_path / "benceno.record.json").read_text())
        for key in ("tbkit", "commit", "date", "platform", "settings", "model", "structure",
                    "results"):
            assert key in saved
        assert saved["results"]["gap"] == pytest.approx(5.4)
        again = replay(tmp_path / "benceno.record.json")
        assert again["levels"] == pytest.approx(saved["results"]["levels"], abs=1e-12)
        assert record["results"]["gap"] == saved["results"]["gap"]

    def test_relax_record_keeps_the_final_structure(self, tmp_path):
        cluster = ring(5, seed=6)
        write(tmp_path / "c5.xyz", cluster)
        config = {"structure": "c5.xyz", "model": "xu_carbon", "task": "relax",
                  "settings": {"fmax": 0.02}}
        (tmp_path / "relax.json").write_text(json.dumps(config))
        record = run_simulation(tmp_path / "relax.json")
        assert record["results"]["converged"] and record["results"]["max_force"] < 0.02
        assert "Lattice" in record["structure"] or "Properties" in record["structure"]

    def test_cli_run_relax_and_phonons(self, tmp_path, capsys):
        write(tmp_path / "c5.xyz", ring(5, seed=7))
        out = tmp_path / "c5_relajado.extxyz"
        assert main(["relax", str(tmp_path / "c5.xyz"), "--model", "sp3", "--fmax", "0.02",
                     "-o", str(out)]) == 0
        assert main(["phonons", str(out), "--model", "sp3"]) == 0
        text = capsys.readouterr().out
        assert "Frecuencias Γ" in text
        (tmp_path / "sim.json").write_text(json.dumps(
            {"structure": "c5_relajado.extxyz", "model": "xu_carbon", "task": "phonons"}))
        assert main(["run", str(tmp_path / "sim.json")]) == 0
        assert (tmp_path / "sim.record.json").exists()

    def test_pi_model_refuses_relaxation(self, tmp_path, capsys):
        write(tmp_path / "b.xyz", molecule("C6H6"))
        assert main(["relax", str(tmp_path / "b.xyz")]) == 1
        assert "repulsiva" in capsys.readouterr().out


def test_fit_report_says_what_changed():
    from tbkit.fit import Reference, fit
    from tbkit.tests.test_core import _acene

    naphthalene = _acene(2)
    truth = solve(System.build(naphthalene, pi_model(t=-2.9)), kT=1e-4)
    reference = Reference(naphthalene, truth.energies[0, 0], 5, 4, 4, label="naftaleno")
    result = fit(lambda x: pi_model(t=float(x[0]), heteroatoms=False), [-2.4], [reference],
                 names=["t"])
    report = result.summary()
    assert "t: -2.4000 → -2.9000" in report
    assert result.rms_initial["naftaleno"] > result.rms["naftaleno"]
    assert abs(result.gap_error["naftaleno"]) < 1e-6


def test_skf_spline_repulsive(tmp_path):
    """The Spline block: exponential head, cubic pieces, zero beyond the cutoff."""
    from ase.units import Bohr, Hartree

    from tbkit.repulsive import read_skf_spline

    lines = ["Spline", "2 4.0", "1.0 2.0 -0.5",
             "2.0 3.0 0.1 -0.2 0.05 0.0",
             "3.0 4.0 0.0125 -0.05 0.0 0.0 0.0 0.0"]
    spline = read_skf_spline(lines)
    assert spline.cutoff == pytest.approx(4.0 * Bohr)
    assert spline(1.0 * Bohr) == pytest.approx((np.exp(-1.0 + 2.0) - 0.5) * Hartree)
    assert spline(2.5 * Bohr) == pytest.approx((0.1 - 0.2 * 0.5 + 0.05 * 0.25) * Hartree)
    assert spline(4.5 * Bohr) == 0.0
