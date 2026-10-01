"""Environment-dependent TB (Tang et al. 1996): the paper's coordinations, forces, two-centre limit."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, graphene

from tbkit.calculator import TBCalculator
from tbkit.environment import _switch
from tbkit.params import load_parameters, model_from_dict, read_parameter_file

NAME = "tang1996_published"


def test_effective_coordinations_are_the_papers():
    """Tang et al. give g for six structures (text after Eq. 5); scale-free."""
    spec = load_parameters(NAME).environment
    d = 1.42
    chain = Atoms("C", positions=[[0, 0, 0]], cell=[30, 30, d], pbc=True)
    sheet = graphene("C2", a=d * np.sqrt(3), vacuum=15.0)
    sheet.pbc = True
    cases = {"chain": (chain, 2.08639), "graphite": (sheet, 3.17678),
             "diamond": (bulk("C", "diamond", a=d * 4 / np.sqrt(3)), 4.41022),
             "sc": (bulk("C", "sc", a=d), 6.23620),
             "bcc": (bulk("C", "bcc", a=d * 2 / np.sqrt(3)), 10.38529),
             "fcc": (bulk("C", "fcc", a=d * np.sqrt(2)), 11.89829)}
    for name, (atoms, paper) in cases.items():
        assert spec.evaluate(atoms).g[0] == pytest.approx(paper, abs=1e-3), name


def test_forces_are_the_derivative_of_the_energy():
    rng = np.random.default_rng(1)
    pos = np.array([[0, 0, 0], [1.4, 0.1, 0], [2.1, 1.3, 0.2], [0.7, 2.4, -0.1],
                    [-0.7, 1.3, 0.3], [3.5, 1.4, 0.6]]) + rng.normal(0, 0.05, (6, 3))
    atoms = Atoms("C6", positions=pos)
    model = load_parameters(NAME)

    def energy(a):
        a = a.copy()
        a.calc = TBCalculator(model, kT=0.05)
        return a.get_potential_energy()

    atoms.calc = TBCalculator(model, kT=0.05)
    forces = atoms.get_forces()
    h = 1e-4
    for i in range(len(atoms)):
        for k in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[i, k] += h
            minus.positions[i, k] -= h
            assert forces[i, k] == pytest.approx(-(energy(plus) - energy(minus)) / (2 * h),
                                                 abs=1e-5)


def test_periodic_forces_are_the_derivative_of_the_energy():
    atoms = bulk("C", "diamond", a=3.6)
    atoms.positions[0] += [0.07, -0.03, 0.05]
    model = load_parameters(NAME)

    def energy(a):
        a = a.copy()
        a.calc = TBCalculator(model, kpts=3, kT=0.05)
        return a.get_potential_energy()

    atoms.calc = TBCalculator(model, kpts=3, kT=0.05)
    forces = atoms.get_forces()
    h = 1e-4
    for k in range(3):
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[0, k] += h
        minus.positions[0, k] -= h
        assert forces[0, k] == pytest.approx(-(energy(plus) - energy(minus)) / (2 * h),
                                             abs=1e-5)


def test_without_screening_and_scaling_it_is_two_centre():
    """β1 = 0 and δ = 0: Eq. (1) is a1 r^-a2 exp(-a3 r^a4) times our cutoff (paper's remark)."""
    data = copy.deepcopy(read_parameter_file(NAME))
    for f in data["environment"]["functions"].values():
        f["b1"]["value"] = 0.0
        f["delta"]["value"] = 0.0
    spec = model_from_dict(data).environment
    atoms = Atoms("C3", positions=[[0, 0, 0], [1.4, 0, 0], [2.8, 0.1, 0]])
    env = spec.evaluate(atoms)
    f = spec.functions["pps"]
    t, _ = _switch(env.r, *spec.pair_cut)
    expected = f.a1 * env.r ** (-f.a2) * np.exp(-f.a3 * env.r ** f.a4) * t
    assert env.values["pps"] == pytest.approx(expected, rel=1e-12)
