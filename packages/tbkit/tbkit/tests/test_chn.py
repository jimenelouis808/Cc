"""H and N on top of Xu's carbon: composite repulsion, the scc flag, the fit recipe.

The recipe is checked against synthetic references made by a known model
(so the answer is known exactly); the shipped parameter set is checked
against the GPAW references it was fitted to.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from ase.build import molecule

from tbkit.calculator import TBCalculator
from tbkit.params import (
    CutoffPolynomial,
    derivative,
    model_from_dict,
    model_to_dict,
    xu_carbon,
)
from tbkit.recipes import xu_chn
from tbkit.references import ReferenceStructure, distortions, load_references, save_references
from tbkit.repulsive import PairRepulsive, SumRepulsive


def _numeric_forces(atoms, energy, h=1e-4):
    forces = np.zeros((len(atoms), 3))
    for a in range(len(atoms)):
        for i in range(3):
            values = []
            for step in (h, -h):
                moved = atoms.copy()
                moved.positions[a, i] += step
                values.append(energy(moved))
            forces[a, i] = -(values[0] - values[1]) / (2 * h)
    return forces


KNOWN = {("C", "H"): (2.0, -1.0, 0.5, 0.0, 0.0), ("N", "H"): (3.0, -0.5, 0.0, 0.2, 0.0),
         ("C", "N"): (1.0, 0.3, 0.0, 0.0, 0.0), ("N", "N"): (0.8, 0.0, 0.1, 0.0, 0.0)}


@pytest.fixture(scope="module")
def true_model():
    laws = {pair: CutoffPolynomial(c, xu_chn.PAIRS[pair]["rc_rep"]) for pair, c in KNOWN.items()}
    electronic = xu_chn.build_model(xu_chn.initial_guess())
    repulsive = SumRepulsive((electronic.repulsive.terms[0], PairRepulsive(laws)))
    return xu_chn.build_model(xu_chn.initial_guess(), repulsive)


@pytest.fixture(scope="module")
def synthetic_refs(true_model):
    refs = []
    for name in ("CH4", "NH3", "HCN", "N2H4"):
        base = molecule(name)
        for tag, atoms in [("eq", base)] + distortions(base, n_random=2, seed=1):
            atoms = atoms.copy()
            atoms.calc = TBCalculator(true_model)
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            ref = ReferenceStructure(f"{name}/{tag}", name, atoms.copy(), energy, forces,
                                     np.zeros(1), 0)
            levels = xu_chn.model_levels(true_model, ref)
            electrons = sum({"C": 4, "N": 5, "H": 1}[s] for s in atoms.get_chemical_symbols())
            refs.append(dataclasses.replace(ref, levels=levels, n_occupied=electrons // 2))
    return refs


class TestPieces:
    def test_cutoff_polynomial_is_smooth_at_rc(self):
        law = CutoffPolynomial((1.0, -2.0, 0.5), rc=2.0)
        assert float(law(2.0)) == 0.0 and float(law(2.5)) == 0.0
        assert abs(float(derivative(law, 2.0 - 1e-4))) < 1e-6
        assert float(law(1.0)) == pytest.approx(1.0 - 2.0 + 0.5)

    def test_sum_repulsion_on_pure_carbon_is_xu(self):
        from ase.build import bulk

        diamond = bulk("C", "diamond", a=3.56)
        xu = xu_carbon().repulsive
        chn = xu_chn.build_model(xu_chn.initial_guess()).repulsive
        assert chn.energy_and_forces(diamond)[0] == pytest.approx(xu.energy_and_forces(diamond)[0])

    def test_sum_repulsion_forces(self, true_model):
        atoms = molecule("CH3CN")
        atoms.positions += np.random.default_rng(3).normal(0, 0.03, atoms.positions.shape)
        repulsive = true_model.repulsive
        _, forces = repulsive.energy_and_forces(atoms)
        numeric = _numeric_forces(atoms, lambda a: repulsive.energy_and_forces(a)[0])
        assert np.abs(forces - numeric).max() < 1e-6

    def test_embedded_alone_still_refuses_hydrogen(self):
        with pytest.raises(ValueError, match="solo describe"):
            xu_carbon().repulsive.energy_and_forces(molecule("CH4"))

    def test_scc_flag_round_trip_and_calculator_default(self, true_model):
        data = model_to_dict(true_model)
        assert data["scc"] is True
        again = model_from_dict(data)
        assert again.scc and TBCalculator(again).scc
        assert not TBCalculator(xu_carbon()).scc
        atoms = molecule("HCN")
        e1, e2 = [], []
        for model, out in ((true_model, e1), (again, e2)):
            atoms.calc = TBCalculator(model)
            out.append(atoms.get_potential_energy())
        assert e1[0] == pytest.approx(e2[0], abs=1e-9)

    def test_scc_forces_with_nitrogen(self, true_model):
        atoms = molecule("C5H5N")
        atoms.positions += np.random.default_rng(5).normal(0, 0.02, atoms.positions.shape)
        atoms.calc = TBCalculator(true_model)
        forces = atoms.get_forces()

        def energy(a):
            a = a.copy()
            a.calc = TBCalculator(true_model)
            return a.get_potential_energy()

        probe = [0, 5, 7]
        numeric = _numeric_forces(atoms, energy)[probe]
        assert np.abs(forces[probe] - numeric).max() < 1e-4

    def test_references_round_trip(self, synthetic_refs, tmp_path):
        path = save_references(tmp_path / "r.json", synthetic_refs[:3], {"code": "prueba"})
        again, settings = load_references(path)
        assert settings == {"code": "prueba"}
        assert again[1].label == synthetic_refs[1].label
        assert np.allclose(again[1].forces, synthetic_refs[1].forces, atol=1e-7)
        assert np.allclose(again[2].atoms.positions, synthetic_refs[2].atoms.positions)


class TestRecipe:
    def test_levels_exact_at_the_true_parameters_up_to_a_shift(self, synthetic_refs):
        shifted = [dataclasses.replace(r, levels=r.levels - 1.3) for r in synthetic_refs]
        residuals, shift, _ = xu_chn.level_residuals(
            xu_chn.build_model(xu_chn.initial_guess()), shifted)
        assert shift == pytest.approx(1.3, abs=1e-6)
        assert np.abs(residuals).max() < 1e-5

    def test_repulsion_fit_recovers_the_coefficients(self, synthetic_refs):
        electronic = xu_chn.build_model(xu_chn.initial_guess())
        repulsive, report = xu_chn.fit_repulsion(electronic, synthetic_refs, ridge=1e-12)
        laws = repulsive.terms[1].laws
        # The five powers are nearly collinear over the sampled bond lengths:
        # what is determined is V(r) and its slope there, not each coefficient.
        for pair, r in ((("C", "H"), np.linspace(1.02, 1.16, 8)),
                        (("N", "H"), np.linspace(0.96, 1.08, 8))):
            known = CutoffPolynomial(KNOWN[pair], xu_chn.PAIRS[pair]["rc_rep"])
            assert laws[pair](r) == pytest.approx(known(r), abs=2e-3)
            assert derivative(laws[pair], r) == pytest.approx(derivative(known, r), abs=2e-3)
        assert report["force_rms"] < 1e-4

    def test_parameter_names_match_the_vector(self):
        assert len(xu_chn.parameter_names()) == len(xu_chn.initial_guess())
