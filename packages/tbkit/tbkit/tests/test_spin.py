"""Spin-polarised SCC (``tbkit.spin``).

Closed forms: a free carbon atom polarises to its Hund p² (M = 2) and gains exactly
2 W_pp; a π model with W = -U/2 and no charge SCC is the mean-field Hubbard model on a
half-filled bipartite flake (same moments, same energy), so the triangulene obeys Lieb's
theorem (M = 2). The spin constants of the table are GPAW's (published PBE values for C).
Forces of a spin-polarised, charge-SCC, non-orthogonal radical match finite differences.
"""

from __future__ import annotations

import pytest
from ase import Atoms

from tbkit.hamiltonian import System
from tbkit.params import pi_model, xu_carbon
from tbkit.spin import energy_and_forces, shells_of, spin_polarised, spin_table
from tbkit.tests.test_energy import nonorthogonal_model, ring
from tbkit.tests.test_magnetism import _triangulene_explicit


def test_carbon_constants_are_the_published_pbe_values():
    w = spin_table(xu_carbon())["C"]
    hartree = 27.211386
    assert w["ss"] / hartree == pytest.approx(-0.031, abs=0.001)
    assert w["sp"] / hartree == pytest.approx(-0.025, abs=0.001)
    assert w["pp"] / hartree == pytest.approx(-0.023, abs=0.001)


def test_free_carbon_atom_hund_closed_form():
    system = System.build(Atoms("C"), xu_carbon())
    polarised = spin_polarised(system, charges=False, guess="ferro", kT=0.001)
    flat = spin_polarised(system, charges=False, guess="none", kT=0.001)
    w_pp = spin_table(xu_carbon())["C"]["pp"]
    assert polarised.converged and flat.converged
    assert polarised.magnetization == pytest.approx(2.0, abs=1e-6)
    assert flat.magnetization == pytest.approx(0.0, abs=1e-9)
    assert polarised.energy - flat.energy == pytest.approx(2 * w_pp, abs=1e-6)


def test_shells_and_w_blocks():
    system = System.build(Atoms("C2", positions=[[0, 0, 0], [0, 0, 1.3]]), xu_carbon())
    shells = shells_of(system)
    assert list(shells.atom) == [0, 0, 1, 1]
    assert shells.w[0, 2] == 0.0 and shells.w[0, 1] == shells.w[1, 0] != 0.0
    assert list(shells.of_orbital) == [0, 1, 1, 1, 2, 3, 3, 3]


def test_pi_limit_is_mean_field_hubbard_and_obeys_lieb():
    from tbkit.hubbard import mean_field

    u = 2.7
    system = System.build(_triangulene_explicit(), pi_model())
    spin = spin_polarised(system, W=-u / 2, charges=False, guess="ferro", kT=0.005,
                          tol=1e-10)
    hubbard = mean_field(system, U=u, guess="ferro", kT=0.005, tol=1e-10)
    assert spin.converged and spin.magnetization == pytest.approx(2.0, abs=1e-6)
    for atom, m in hubbard.moments.items():
        assert spin.moments[atom] == pytest.approx(m, abs=1e-6)
    assert spin.energy == pytest.approx(hubbard.energy, abs=1e-6)


def test_fixed_magnetisation_two_fermi_levels():
    system = System.build(_triangulene_explicit(), pi_model())
    singlet = spin_polarised(system, W=-1.35, charges=False, magnetization=0.0,
                             guess="antiferro", kT=0.005)
    triplet = spin_polarised(system, W=-1.35, charges=False, magnetization=2.0, kT=0.005)
    assert singlet.magnetization == pytest.approx(0.0, abs=1e-8)
    assert triplet.magnetization == pytest.approx(2.0, abs=1e-8)
    assert triplet.energy < singlet.energy               # Lieb: the ground state has S = 1


def test_forces_match_finite_differences():
    model = nonorthogonal_model(with_nitrogen=True)
    atoms = ring(5, symbols="C2NC2", seed=2)              # 21 electrons: a doublet

    def energy(a, need=False):
        result = spin_polarised(System.build(a, model), W={"C": {"ss": -0.83, "sp": -0.68,
                                                                  "pp": -0.62},
                                                            "N": {"ss": -0.90, "sp": -0.75,
                                                                  "pp": -0.69}},
                                charges=True, kT=0.1, tol=1e-11, guess="ferro")
        assert result.converged
        return energy_and_forces(result, need_forces=need), result

    (_, forces, _), result = energy(atoms, True)
    assert abs(result.magnetization) > 0.5
    h = 1e-4
    for atom, axis in ((0, 0), (2, 1), (4, 2), (3, 0)):
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[atom, axis] += h
        minus.positions[atom, axis] -= h
        fd = -(energy(plus)[0][0] - energy(minus)[0][0]) / (2 * h)
        assert forces[atom, axis] == pytest.approx(fd, abs=2e-6)


def test_periodic_zigzag_ribbon_is_mean_field_hubbard():
    from ase.build import graphene_nanoribbon

    from tbkit.hubbard import mean_field
    from tbkit.kpoints import mesh

    u = 2.7
    ribbon = graphene_nanoribbon(4, 1, type="zigzag", saturated=False, vacuum=6.0)
    system = System.build(ribbon, pi_model())
    k, w = mesh(ribbon, 24)
    spin = spin_polarised(system, W=-u / 2, charges=False, guess="antiferro", kT=0.005,
                          kpts=k, weights=w, tol=1e-10)
    hubbard = mean_field(system, U=u, kpts=k, weights=w, kT=0.005, tol=1e-10)
    assert spin.magnetization == pytest.approx(0.0, abs=1e-6)
    for atom, m in hubbard.moments.items():
        assert spin.moments[atom] == pytest.approx(m, abs=1e-5)
    assert spin.energy == pytest.approx(hubbard.energy, abs=1e-5)


def test_periodic_forces_match_finite_differences():
    from tbkit.kpoints import mesh

    model = nonorthogonal_model(with_nitrogen=True)
    chain = Atoms("C2N", positions=[[0, 0, 0], [0.1, 0, 1.38], [0, 0.1, 2.75]],
                  cell=[8.0, 8.0, 4.15], pbc=True)
    table = {"C": {"ss": -0.83, "sp": -0.68, "pp": -0.62},
             "N": {"ss": -0.90, "sp": -0.75, "pp": -0.69}}

    def energy(a, need=False):
        system = System.build(a, model)
        k, w = mesh(system, 4)
        result = spin_polarised(system, W=table, charges=True, kT=0.1, tol=1e-11,
                                kpts=k, weights=w)
        assert result.converged
        return energy_and_forces(result, need_forces=need)

    _, forces, _ = energy(chain, True)
    h = 1e-4
    for atom, axis in ((0, 2), (1, 0), (2, 2)):
        plus, minus = chain.copy(), chain.copy()
        plus.positions[atom, axis] += h
        minus.positions[atom, axis] -= h
        fd = -(energy(plus)[0] - energy(minus)[0]) / (2 * h)
        assert forces[atom, axis] == pytest.approx(fd, abs=5e-6)
