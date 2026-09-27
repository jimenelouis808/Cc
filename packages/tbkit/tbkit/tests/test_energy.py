"""Total energies, forces and phonons (the repulsive part).

Forces are checked against finite differences of the energy for every
kind of model the code supports: orthogonal periodic (Xu, with k-points
and smearing), non-orthogonal finite and periodic, and self-consistent
charges with overlap. Physics is checked against experiment for the Xu
carbon model, with tolerances that say how good an empirical model is
expected to be -- not tuned to what it happens to give.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, graphene

from tbkit import System, pi_model, solve, xu_carbon
from tbkit.calculator import TBCalculator
from tbkit.forces import energy_and_forces, entropy_term
from tbkit.kpoints import mesh
from tbkit.params import Exponential, TBModel, Tail
from tbkit.repulsive import EmbeddedRepulsive, PairRepulsive
from tbkit.scc import energy_and_forces as scc_energy_and_forces
from tbkit.scc import self_consistent


def _numeric_forces(atoms, make_calc, h=1e-4):
    out = np.zeros((len(atoms), 3))
    for a in range(len(atoms)):
        for x in range(3):
            energies = []
            for sign in (1, -1):
                probe = atoms.copy()
                probe.positions[a, x] += sign * h
                probe.calc = make_calc()
                energies.append(probe.get_potential_energy())
            out[a, x] = -(energies[0] - energies[1]) / (2 * h)
    return out


def _smooth(v0, beta, r1=2.2, rm=2.6):
    return Tail(Exponential(v0, 1.42, beta, rm), r1, rm)


def nonorthogonal_model(with_nitrogen: bool = False) -> TBModel:
    """A made-up but well-behaved s+p model with overlap and a pair repulsion."""
    elements = ["C", "N"] if with_nitrogen else ["C"]
    hopping, overlap, repulsion = {}, {}, {}
    for i, a in enumerate(elements):
        for b in elements[i:]:
            hopping.update({(a, b, "sss"): _smooth(-4.5, 2.0), (a, b, "pps"): _smooth(5.0, 2.2),
                            (a, b, "ppp"): _smooth(-2.0, 2.5)})
            hopping[(a, b, "sps")] = _smooth(4.2, 2.1)
            hopping[(b, a, "sps")] = _smooth(4.2 if a == b else 4.0, 2.1)
            overlap.update({(a, b, "sss"): _smooth(0.20, 2.0), (a, b, "pps"): _smooth(-0.25, 2.0),
                            (a, b, "ppp"): _smooth(0.12, 2.0)})
            overlap[(a, b, "sps")] = _smooth(-0.22, 2.0)
            overlap[(b, a, "sps")] = _smooth(-0.22 if a == b else -0.20, 2.0)
            repulsion[(a, b)] = _smooth(6.0, 3.5)
    orbitals = {el: ("s", "px", "py", "pz") for el in elements}
    onsite = {"C": {"s": -8.0, "p": -3.0}, "N": {"s": -11.0, "p": -5.0}}
    return TBModel("prueba no ortogonal", orbitals=orbitals,
                   onsite={el: onsite[el] for el in elements}, hopping=hopping,
                   overlap=overlap, valence={"C": 4.0, "N": 5.0},
                   hubbard_u={"C": 10.0, "N": 11.0}, repulsive=PairRepulsive(repulsion))


def ring(n=6, symbols=None, seed=0) -> Atoms:
    angles = 2 * np.pi * np.arange(n) / n
    radius = 1.42 / (2 * np.sin(np.pi / n))
    atoms = Atoms(symbols or f"C{n}", positions=np.c_[radius * np.cos(angles),
                                                      radius * np.sin(angles), np.zeros(n)])
    atoms.rattle(0.04, seed=seed)
    return atoms


class TestForces:
    def test_xu_periodic_with_kpoints(self):
        diamond = bulk("C", "diamond", a=3.56)
        diamond.rattle(0.05, seed=1)

        def calc():
            return TBCalculator(xu_carbon(), kpts=3, kT=0.1)
        diamond.calc = calc()
        analytic = diamond.get_forces()
        assert analytic == pytest.approx(_numeric_forces(diamond, calc), abs=1e-5)
        assert np.abs(analytic.sum(axis=0)).max() < 1e-10       # no net force

    def test_nonorthogonal_finite(self):
        molecule = ring(6, seed=2)

        def calc():
            return TBCalculator(nonorthogonal_model(), kT=0.05)
        molecule.calc = calc()
        assert molecule.get_forces() == pytest.approx(_numeric_forces(molecule, calc), abs=1e-5)

    def test_nonorthogonal_periodic(self):
        sheet = graphene(a=2.46, vacuum=6.0)
        sheet.pbc = (True, True, False)
        sheet.positions[0] += [0.03, -0.02, 0.05]

        def calc():
            return TBCalculator(nonorthogonal_model(), kpts=4, kT=0.1)
        sheet.calc = calc()
        assert sheet.get_forces() == pytest.approx(_numeric_forces(sheet, calc), abs=1e-5)

    def test_scc_with_overlap(self):
        molecule = ring(6, ["C"] * 5 + ["N"], seed=3)

        def calc():
            return TBCalculator(nonorthogonal_model(with_nitrogen=True), kT=0.05, scc=True)
        molecule.calc = calc()
        analytic = molecule.get_forces()
        assert analytic == pytest.approx(_numeric_forces(molecule, calc), abs=1e-4)
        charges = molecule.calc.results["charges"]
        assert charges[-1] < 0                  # N takes electrons

    def test_scc_energy_includes_the_second_order_term(self):
        system = System.build(ring(6, ["C"] * 5 + ["N"], seed=3),
                              nonorthogonal_model(with_nitrogen=True))
        result = self_consistent(system)
        energy, _, parts = scc_energy_and_forces(result, need_forces=False)
        second = 0.5 * float(result.dq @ result.gamma @ result.dq)
        assert second > 0 and energy == pytest.approx(
            parts["scc"] - parts["entropy_TS"] + parts["repulsive"])

    def test_models_without_repulsion_refuse(self):
        solution = solve(System.build(ring(6), pi_model()))
        with pytest.raises(ValueError, match="repulsiva"):
            energy_and_forces(solution)
        with pytest.raises(ValueError, match="repulsiva"):
            TBCalculator(pi_model())


class TestRepulsive:
    def test_tail_is_continuous_and_smooth(self):
        law = xu_carbon().hopping[("C", "C", "pps")]
        r1, rm = law.r1, law.rm
        for r in (r1, rm):
            left, right = float(law(r - 1e-7)), float(law(r + 1e-7))
            assert left == pytest.approx(right, abs=1e-5)
        assert float(law(rm)) == 0.0 and float(law(rm + 0.1)) == 0.0

    def test_embedded_forces_match_its_energy(self):
        repulsion = xu_carbon().repulsive
        atoms = ring(5, seed=4)
        energy, forces = repulsion.energy_and_forces(atoms)
        h = 1e-5
        probe = atoms.copy()
        probe.positions[2, 1] += h
        e_plus, _ = repulsion.energy_and_forces(probe)
        probe.positions[2, 1] -= 2 * h
        e_minus, _ = repulsion.energy_and_forces(probe)
        assert forces[2, 1] == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=1e-6)

    def test_embedded_refuses_other_elements(self):
        with pytest.raises(ValueError, match="H"):
            EmbeddedRepulsive(xu_carbon().repulsive.phi, (0.0, 1.0)).energy_and_forces(
                Atoms("CH", positions=[[0, 0, 0], [0, 0, 1.1]]))

    def test_entropy_vanishes_for_a_gapped_system(self):
        diamond = bulk("C", "diamond", a=3.56)
        solution = solve(System.build(diamond, xu_carbon()), *mesh(diamond, 3), kT=0.05)
        assert abs(entropy_term(solution)) < 1e-8


class TestXuCarbonPhysics:
    """Against experiment, with the accuracy an empirical TB model has."""

    @staticmethod
    def _energy_per_atom(atoms, k):
        solution = solve(System.build(atoms, xu_carbon()), *mesh(atoms, k), kT=0.05)
        return energy_and_forces(solution, need_forces=False)[0] / len(atoms)

    def test_diamond_lattice_constant(self):
        lattice = np.linspace(3.48, 3.64, 9)
        energies = [self._energy_per_atom(bulk("C", "diamond", a=a), 5) for a in lattice]
        fit = np.polyfit(lattice, energies, 2)
        a0 = -fit[1] / (2 * fit[0])
        assert a0 == pytest.approx(3.567, rel=0.01)          # experiment 3.567 Å

    def test_graphene_bond_length_and_stability(self):
        lattice = np.linspace(2.40, 2.52, 7)

        def sheet(a):
            g = graphene(a=a, vacuum=6.0)
            g.pbc = (True, True, False)
            return g
        energies = [self._energy_per_atom(sheet(a), 12) for a in lattice]
        fit = np.polyfit(lattice, energies, 2)
        a0 = -fit[1] / (2 * fit[0])
        assert a0 / np.sqrt(3) == pytest.approx(1.42, rel=0.01)   # C-C 1.42 Å
        diamond = self._energy_per_atom(bulk("C", "diamond", a=3.555), 5)
        assert min(energies) < diamond                       # graphite/graphene lowest

    def test_cohesive_energy(self):
        """Free atom: 2 s + 2 p electrons at the on-site energies, plus f(0)."""
        model = xu_carbon()
        atom = 2 * model.onsite["C"]["s"] + 2 * model.onsite["C"]["p"] + \
            model.repulsive.polynomial[0]
        cohesive = atom - self._energy_per_atom(bulk("C", "diamond", a=3.555), 5)
        assert cohesive == pytest.approx(7.37, rel=0.03)     # experiment 7.37 eV/atom

    @pytest.mark.parametrize("structure,expected,tolerance", [
        ("diamond", 1332.0, 0.10),      # Raman line of diamond; the model gives ~1224
        # G band: ~1666-1682 in the model once converged in k (Kohn anomaly:
        # the G mode couples to the Dirac cone, so it needs a dense mesh).
        ("graphene", 1582.0, 0.08),
    ])
    def test_raman_active_phonons(self, structure, expected, tolerance):
        from ase.vibrations import Vibrations

        if structure == "diamond":
            atoms, k = bulk("C", "diamond", a=3.555), 6
        else:
            atoms, k = graphene(a=2.455, vacuum=6.0), 36
            atoms.pbc = (True, True, False)
        atoms.calc = TBCalculator(xu_carbon(), kpts=k, kT=0.02)
        with tempfile.TemporaryDirectory() as directory:
            vibrations = Vibrations(atoms, name=os.path.join(directory, "v"), delta=0.005)
            vibrations.run()
            frequencies = np.real(vibrations.get_frequencies())
        highest = frequencies.max()
        assert np.abs(np.sort(np.abs(frequencies))[:3]).max() < 5.0   # acoustic ≈ 0 at Γ
        assert highest == pytest.approx(expected, rel=tolerance)
