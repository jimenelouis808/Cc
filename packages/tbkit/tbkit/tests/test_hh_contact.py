"""H···H contact repulsion: forces, what it leaves alone, and the case it is for."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from tbkit.recipes.hh_contact import term
from tbkit.repulsive import repulsive_from_dict


def _fd_forces(t, atoms, h=1e-5):
    out = np.zeros((len(atoms), 3))
    for a in range(len(atoms)):
        for k in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[a, k] += h
            minus.positions[a, k] -= h
            out[a, k] = -(t.energy_and_forces(plus)[0] - t.energy_and_forces(minus)[0]) / (2 * h)
    return out


def _crowded():
    """Two methanols with their hydrogens in contact, and a bond being stretched."""
    a = molecule("CH3OH")
    b = molecule("CH3OH")
    b.positions += [0.4, 2.6, 0.3]
    atoms = a + b
    atoms.positions[2] += [0.0, 0.0, 0.25]          # one C-H half-way out of its bond
    return atoms


def test_forces_are_the_derivative_of_the_energy():
    t = term()
    atoms = _crowded()
    energy, forces = t.energy_and_forces(atoms)
    assert energy > 1e-4
    assert np.abs(forces - _fd_forces(t, atoms)).max() < 1e-6


def test_zero_for_hydrogens_on_the_same_atom_and_for_h2():
    t = term()
    for name in ("CH4", "C2H6", "H2O", "NH3", "H2"):
        assert t.energy_and_forces(molecule(name))[0] == pytest.approx(0.0, abs=1e-12), name


def test_round_trip_through_the_parameter_file():
    t = term()
    back = repulsive_from_dict(t.to_dict())
    atoms = _crowded()
    assert back.energy_and_forces(atoms)[0] == pytest.approx(t.energy_and_forces(atoms)[0])


def test_gpaw_wall_is_reproduced_for_the_dimer_it_comes_from():
    t = term()
    for d, expected in ((1.5, 0.1219), (1.7, 0.0494)):
        dimer = Atoms("H4", positions=[[0, 0, 0], [0.75, 0, 0], [0.75 + d, 0, 0],
                                       [1.5 + d, 0, 0]])
        # The wall was taken from the nearest contact alone; the dimer's second
        # contacts (d + 0.75 Å) add a few per cent on top.
        assert t.energy_and_forces(dimer)[0] == pytest.approx(expected, rel=0.10)
