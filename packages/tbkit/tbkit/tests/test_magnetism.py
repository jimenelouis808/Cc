"""Mean-field Hubbard and self-consistent charges.

Checks with a known answer: zigzag ribbon edges antiparallel with the
expected magnitude, graphene non-magnetic below U_c ≈ 2.2|t|, Lieb's
theorem on a triangulene (M = |N_A - N_B| = 2), m(E_F) = M, the
susceptibility of a paramagnet from a field sweep, and charge screening.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import graphene, graphene_nanoribbon, molecule

from tbkit import System, pi_model, solve
from tbkit.analysis import atomic_charges
from tbkit.hubbard import (
    MU_B,
    magnetization_vs_doping,
    magnetization_vs_energy,
    magnetization_vs_field,
    mean_field,
    sublattices,
    tesla_to_ev,
)
from tbkit.kpoints import mesh
from tbkit.scc import self_consistent

T = 2.7


@pytest.fixture(scope="module")
def ribbon_result():
    ribbon = graphene_nanoribbon(4, 1, type="zigzag", saturated=False, vacuum=6.0)
    system = System.build(ribbon, pi_model())
    k, w = mesh(ribbon, 48)
    return ribbon, mean_field(system, U=T, kpts=k, weights=w)


def _triangulene_explicit() -> Atoms:
    """Triangle of 6 hexagons (1+2+3) on a honeycomb lattice with zigzag edges."""
    a = 1.42
    hexagon = [np.array([a * np.cos(np.pi / 6 + k * np.pi / 3),
                         a * np.sin(np.pi / 6 + k * np.pi / 3), 0.0]) for k in range(6)]
    centres = []
    dx, dy = np.sqrt(3) * a, 1.5 * a
    for row in range(3):
        for col in range(3 - row):
            centres.append(np.array([col * dx + row * dx / 2, row * dy, 0.0]))
    points = []
    for c in centres:
        for h in hexagon:
            p = c + h
            if all(np.linalg.norm(p - q) > 0.1 for q in points):
                points.append(p)
    return Atoms(f"C{len(points)}", positions=points)


class TestHubbard:
    def test_zigzag_edges_are_antiparallel(self, ribbon_result):
        ribbon, result = ribbon_result
        assert result.converged
        assert result.magnetization == pytest.approx(0.0, abs=1e-6)
        x = ribbon.positions[:, 0]
        moments = result.moments
        left, right = int(np.argmin(x)), int(np.argmax(x))
        assert moments[left] == pytest.approx(-moments[right], abs=1e-6)
        assert 0.15 < abs(moments[left]) < 0.35        # ~0.24 μB for U = |t|, 4-ZGNR
        assert result.solution.gap() > 0.1             # AFM opens a gap

    def test_antiferro_beats_paramagnetic(self, ribbon_result):
        ribbon, afm = ribbon_result
        system = System.build(ribbon, pi_model())
        k, w = mesh(ribbon, 48)
        para = mean_field(system, U=T, kpts=k, weights=w, guess="paramagnetic")
        assert afm.energy < para.energy - 1e-4

    def test_graphene_stays_non_magnetic_below_uc(self):
        sheet = graphene(a=2.46, vacuum=5.0)
        sheet.pbc = (True, True, False)
        system = System.build(sheet, pi_model())
        result = mean_field(system, U=T, kpts=mesh(sheet, 24)[0], weights=mesh(sheet, 24)[1])
        assert max(abs(m) for m in result.moments.values()) < 1e-3

    def test_lieb_theorem_on_triangulene(self):
        flake = _triangulene_explicit()
        system = System.build(flake, pi_model())
        colours = sublattices(system)
        imbalance = abs(int(colours.sum()))
        assert len(flake) == 22 and imbalance == 2
        result = mean_field(system, U=T, guess="ferro")
        assert result.magnetization == pytest.approx(imbalance, abs=1e-3)

    def test_m_of_e_reaches_m_at_the_fermi_level(self):
        flake = _triangulene_explicit()
        result = mean_field(System.build(flake, pi_model()), U=T, guess="ferro")
        energy, m, dm = magnetization_vs_energy(result, sigma=0.02)
        at_fermi = m[np.argmin(np.abs(energy))]
        assert at_fermi == pytest.approx(result.magnetization, abs=0.05)
        assert m[-1] == pytest.approx(0.0, abs=1e-3)   # all states filled: no imbalance
        assert np.trapezoid(dm, energy) == pytest.approx(m[-1], abs=1e-3)

    def test_field_sweep_gives_a_paramagnetic_susceptibility(self):
        """Benzene has no moment; a Zeeman field polarises nothing until 2h
        beats the gap, so χ = 0 at small h. Beyond, electrons flip in pairs of
        levels: M jumps in steps of 2 (4 at h = 3 eV: two flips)."""
        system = System.build(molecule("C6H6"), pi_model())
        fields = np.array([0.0, 0.1, 0.2])
        h, m, chi, results = magnetization_vs_field(system, fields, U=0.5 * T,
                                                    guess="paramagnetic")
        assert np.allclose(m, 0.0, atol=1e-6) and np.allclose(chi, 0.0, atol=1e-5)
        _, m_big, _, _ = magnetization_vs_field(system, [3.0], U=0.5 * T,
                                                guess="paramagnetic")
        assert m_big[0] > 1.9 and round(m_big[0]) % 2 == 0
        assert m_big[0] == pytest.approx(round(m_big[0]), abs=1e-3)

    def test_doping_sweep(self):
        system = System.build(molecule("C6H6"), pi_model())
        x, m, _ = magnetization_vs_doping(system, charges=[0.0, 1.0], U=T, guess="ferro")
        assert m[0] == pytest.approx(0.0, abs=1e-6)
        assert abs(m[1]) == pytest.approx(1.0, abs=1e-3)   # one hole: a doublet

    def test_units(self):
        assert tesla_to_ev(1.0) == pytest.approx(MU_B)
        assert tesla_to_ev(10.0) == pytest.approx(5.788e-4, rel=1e-3)


class TestSCC:
    def test_symmetric_molecule_has_no_charges(self):
        result = self_consistent(System.build(molecule("C6H6"), pi_model()))
        assert result.converged and max(abs(q) for q in result.charges.values()) < 1e-8

    def test_charge_spreads_evenly_on_a_cation(self):
        result = self_consistent(System.build(molecule("C6H6"), pi_model()), charge=1.0)
        assert all(q == pytest.approx(1 / 6, abs=1e-6) for q in result.charges.values())

    def test_electronegative_atom_is_screened(self):
        pyridine = molecule("C5H5N")
        system = System.build(pyridine, pi_model())
        n = pyridine.get_chemical_symbols().index("N")
        bare = atomic_charges(solve(system))[n]
        screened = self_consistent(system).charges[n]
        assert bare < 0 and screened < 0 and abs(screened) < abs(bare)

    def test_periodic_is_refused(self):
        sheet = graphene(a=2.46, vacuum=5.0)
        sheet.pbc = (True, True, False)
        with pytest.raises(ValueError, match="Ewald"):
            self_consistent(System.build(sheet, pi_model()))
