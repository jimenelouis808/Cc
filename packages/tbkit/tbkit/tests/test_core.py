"""The engine against results known in closed form.

π model: graphene's bands (E = ±|t| |f(k)|, Dirac cone at K), benzene and
the acenes (Hückel levels), metallic versus semiconducting zigzag nanotubes.
sp³ Slater-Koster: rotational invariance of the blocks, a non-orthogonal
dimer solved by hand, diamond's gap. Analysis: electron counting, charges,
bond orders, DOS normalisation, the PDOS sum rule, orbitals on a grid.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, graphene, molecule, nanotube

from tbkit import System, pi_model, solve, xu_carbon
from tbkit.analysis import (
    atomic_charges,
    bands,
    bond_orders,
    coulson_bond_orders,
    dos,
    orbital_on_grid,
    pdos,
    populations,
    write_cube,
)
from tbkit.kpoints import band_path, mesh
from tbkit.params import Constant, TBModel
from tbkit.slater_koster import block

T = 2.7


def _graphene():
    sheet = graphene(a=2.46, vacuum=5.0)
    sheet.pbc = (True, True, False)
    return sheet


class TestPiModel:
    def test_graphene_dirac_point_and_gamma(self):
        system = System.build(_graphene(), pi_model())
        kpoints = {"G": (0, 0, 0), "K": (1 / 3, 1 / 3, 0), "M": (0.5, 0, 0)}
        energies = {name: np.sort(bands(system, np.array([k]))[0, 0])
                    for name, k in kpoints.items()}
        assert energies["G"] == pytest.approx([-3 * T, 3 * T])
        assert energies["K"] == pytest.approx([0.0, 0.0], abs=1e-9)   # Dirac point
        assert energies["M"] == pytest.approx([-T, T])

    def test_graphene_bands_follow_the_analytic_formula(self):
        sheet = _graphene()
        system = System.build(sheet, pi_model())
        rng = np.random.default_rng(1)
        k = rng.random((20, 3)) * [1, 1, 0]
        numeric = np.sort(bands(system, k)[0], axis=1)
        a1, a2 = sheet.cell[0], sheet.cell[1]
        kc = system.kpoint_cartesian(k)
        # ASE's cell puts the B neighbours of A at d, d - a1, d - a1 - a2.
        f = 1 + np.exp(-1j * kc @ a1) + np.exp(-1j * kc @ (a1 + a2))
        assert numeric[:, 1] == pytest.approx(T * np.abs(f), abs=1e-8)

    def test_graphene_is_half_filled_at_zero(self):
        system = System.build(_graphene(), pi_model())
        solution = solve(system, *mesh(system.atoms, 30))
        assert solution.fermi == pytest.approx(0.0, abs=1e-6)
        assert solution.electrons == pytest.approx(2.0)

    def test_benzene_levels_and_bond_orders(self):
        solution = solve(System.build(molecule("C6H6"), pi_model()))
        assert np.sort(solution.energies[0, 0]) == pytest.approx(
            [-2 * T, -T, -T, T, T, 2 * T])
        assert solution.gap() == pytest.approx(2 * T)
        coulson = coulson_bond_orders(solution)
        assert len(coulson) == 6 and all(v == pytest.approx(2 / 3) for v in coulson.values())
        wiberg = bond_orders(solution)
        assert max(wiberg.values()) == pytest.approx(4 / 9)

    @pytest.mark.parametrize("rings,gap_over_t", [(1, 2.0), (2, 1.2360680), (3, 0.8284271)])
    def test_acene_gaps(self, rings, gap_over_t):
        """Hückel HOMO-LUMO gaps: benzene 2|t|, naphthalene 1.236|t|, anthracene 0.828|t|."""
        solution = solve(System.build(_acene(rings), pi_model()))
        assert solution.gap() == pytest.approx(gap_over_t * T, rel=1e-6)

    @pytest.mark.parametrize("n,metallic", [(9, True), (10, False), (12, True)])
    def test_zigzag_nanotubes(self, n, metallic):
        tube = nanotube(n, 0, length=1, bond=1.42, vacuum=5.0)
        system = System.build(tube, pi_model())
        path = np.linspace([0, 0, 0], [0, 0, 0.5], 61)
        energies = bands(system, path)[0]
        gap = energies[energies > 1e-9].min() - energies[energies < -1e-9].max() \
            if (energies > 1e-9).any() else 0.0
        smallest = np.abs(energies).min()
        if metallic:
            assert smallest < 1e-6
        else:
            assert smallest > 0.2 and gap > 0.4

    def test_hydrogen_is_outside_a_pi_model(self):
        system = System.build(molecule("C6H6"), pi_model())
        assert system.basis.size == 6 and system.electrons == 6

    def test_graphitic_versus_pyridinic_nitrogen(self):
        pyridine = molecule("C5H5N")
        system = System.build(pyridine, pi_model())
        assert system.electrons == 6              # two-fold N brings one π electron


def _acene(rings: int) -> Atoms:
    """Polyacene carbon skeleton in the xy plane (H left out: π model)."""
    positions = []
    a = 1.42
    for r in range(rings):
        cx = r * a * np.sqrt(3)
        for angle in range(6):
            theta = np.pi / 6 + angle * np.pi / 3
            positions.append([cx + a * np.cos(theta), a * np.sin(theta), 0.0])
    unique = []
    for p in positions:
        if all(np.linalg.norm(np.subtract(p, q)) > 0.1 for q in unique):
            unique.append(p)
    return Atoms(f"C{len(unique)}", positions=unique)


class TestSlaterKoster:
    def test_blocks_rotate_like_the_orbitals(self):
        model = xu_carbon()
        rng = np.random.default_rng(3)
        vector = np.array([0.3, -1.1, 0.7])
        from scipy.spatial.transform import Rotation

        rotation = Rotation.random(random_state=rng).as_matrix()
        orbs = ("s", "px", "py", "pz")
        original = block(model, model.hopping, "C", orbs, "C", orbs, vector)
        rotated = block(model, model.hopping, "C", orbs, "C", orbs, rotation @ vector)
        big = np.eye(4)
        big[1:, 1:] = rotation
        assert rotated == pytest.approx(big @ original @ big.T, abs=1e-10)

    def test_block_is_the_transpose_of_the_reverse_bond(self):
        model = xu_carbon()
        orbs = ("s", "px", "py", "pz")
        v = np.array([1.0, 0.4, -0.2])
        forward = block(model, model.hopping, "C", orbs, "C", orbs, v)
        backward = block(model, model.hopping, "C", orbs, "C", orbs, -v)
        assert forward == pytest.approx(backward.T)

    def test_non_orthogonal_dimer_by_hand(self):
        """Two s orbitals, H = [[e, t], [t, e]], S = [[1, s], [s, 1]]:
        E = (e ± t) / (1 ± s)."""
        e, t, s = -1.0, -2.0, 0.2
        model = TBModel("dímero s", orbitals={"H": ("s",)}, onsite={"H": {"s": e}},
                        hopping={("H", "H", "sss"): Constant(t, 2.0)},
                        overlap={("H", "H", "sss"): Constant(s, 2.0)}, valence={"H": 1.0})
        dimer = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]])
        solution = solve(System.build(dimer, model))
        assert np.sort(solution.energies[0, 0]) == pytest.approx(
            sorted([(e + t) / (1 + s), (e - t) / (1 - s)]))
        mulliken = populations(solution).sum()
        assert mulliken == pytest.approx(2.0)
        assert populations(solution, "lowdin").sum() == pytest.approx(2.0)

    def test_diamond_has_a_gap(self):
        diamond = bulk("C", "diamond", a=3.567)
        system = System.build(diamond, xu_carbon())
        solution = solve(system, *mesh(diamond, 6))
        assert solution.gap() > 3.0          # the model gives ~5 eV (exp. 5.5 indirect)
        assert solution.electrons == pytest.approx(8.0)


class TestAnalysis:
    def test_charges_sum_to_the_net_charge(self):
        pyridine = molecule("C5H5N")
        solution = solve(System.build(pyridine, pi_model()), charge=1.0)
        charges = atomic_charges(solution)
        assert sum(charges.values()) == pytest.approx(1.0)
        n = pyridine.get_chemical_symbols().index("N")
        neutral = atomic_charges(solve(System.build(pyridine, pi_model())))
        assert neutral[n] < 0                # the electronegative N pulls π charge

    def test_dos_integrates_to_the_number_of_states(self):
        system = System.build(_graphene(), pi_model())
        solution = solve(system, *mesh(system.atoms, 24))
        grid, values = dos(solution, np.linspace(-12, 12, 6000), sigma=0.1)
        assert np.trapezoid(values, grid) == pytest.approx(4.0, rel=1e-3)   # 2 bands x 2 spins
        _, parts = pdos(solution, "atom", grid, sigma=0.1)
        assert sum(parts.values()) == pytest.approx(values, abs=1e-8)
        # Graphene: the DOS vanishes at the Dirac point (in the limit).
        at_zero = values[np.argmin(np.abs(grid))]
        assert at_zero < 0.1 * values.max()

    def test_band_path_on_graphene(self):
        sheet = _graphene()
        system = System.build(sheet, pi_model())
        path = band_path(sheet, "GKMG", npoints=90)
        energies = bands(system, path)[0]
        assert np.abs(energies).min() < 0.05       # passes through the Dirac point

    def test_orbital_on_grid_and_cube(self, tmp_path):
        benzene = molecule("C6H6")
        solution = solve(System.build(benzene, pi_model()))
        homo = 2                                  # 0-based: levels -2t, -t, -t | ...
        origin, steps, values = orbital_on_grid(solution, homo, spacing=0.3)
        # A pz orbital is odd in z: the grid values flip sign across the plane.
        middle = values.shape[2] // 2
        assert np.allclose(values[:, :, :middle].sum(), -values[:, :, -middle:].sum(),
                           rtol=0.05, atol=1e-6)
        path = write_cube(tmp_path / "homo.cube", benzene, origin, steps, values)
        header = path.read_text().splitlines()
        assert int(header[2].split()[0]) == len(benzene)
