"""Coupling between layers: far hopping (``TBModel.far_hopping``) and the layer dispersion
term (``repulsive.LayerDispersionTerm``).

Closed forms: a π model whose far law reaches only the dimer pair of AB bilayer graphene
gives, at K, the levels {-V, 0, 0, +V} with V the law at d (the in-plane terms vanish at K);
two isolated atoms give the pair law itself. Invariances: a far law leaves a monolayer's
bands untouched (no pair of graphene within 5.5 Å is more than six bonds apart), and the
dispersion term is exactly zero there. Forces of Xu + far law + dispersion match
finite differences of the energy on a distorted bilayer.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms

from tbkit.analysis import bands
from tbkit.hamiltonian import System
from tbkit.params import (Exponential, Onset, Sum, Tail, law_from_dict, law_to_dict,
                          model_from_dict, model_to_dict, pi_model, xu_carbon)
from tbkit.recipes.interlayer import geometry, stack
from tbkit.repulsive import LayerDispersionTerm, SumRepulsive

K = np.array([[1 / 3, -1 / 3, 0.0]])        # K of the 60° cell of recipes.interlayer


def _far(model, law, bonds=("ppp",)):
    for bond in bonds:
        model.far_hopping[("C", "C", bond)] = law
    return model


def test_laws_round_trip_and_values():
    inner = Tail(Exponential(0.4, 3.35, 7.0, 5.0), 4.5, 5.0)
    law = Sum((Tail(Exponential(-2.7, 1.42, 3.0, 2.6), 2.45, 2.6), Onset(inner, 2.9, 3.1)))
    again = law_from_dict(law_to_dict(law))
    d = np.linspace(1.3, 5.2, 400)
    assert np.allclose(again(d), law(d))
    assert law.cutoff == 5.0
    assert np.all(Onset(inner, 2.9, 3.1)(np.array([2.5, 2.89])) == 0.0)
    assert Onset(inner, 2.9, 3.1)(3.2) == pytest.approx(inner(3.2))


def test_dimer_splitting_at_k_closed_form():
    v = Tail(Exponential(0.38, 3.35, 7.0, 3.5), 3.4, 3.5)       # reaches the dimer only
    model = _far(pi_model(t=-2.7), v)
    system = System.build(geometry("AB@3.35"), model)
    levels = np.sort(bands(system, K)[0, 0])
    expected = np.sort([-float(v(3.35)), 0.0, 0.0, float(v(3.35))])
    assert np.allclose(levels, expected, atol=1e-10)
    assert sum(b.far for b in system.bonds) == 2          # the dimer, both directions


def test_far_law_leaves_monolayer_untouched():
    path = np.array([[0, 0, 0], [0.5, 0, 0], [1 / 3, 1 / 3, 0], [0.2, 0.1, 0]])
    mono = geometry("mono")
    plain = bands(System.build(mono, xu_carbon()), path)
    far = Tail(Exponential(0.4, 3.35, 7.0, 5.5), 5.0, 5.5)
    coupled = _far(xu_carbon(), far, ("pps", "ppp"))
    assert np.allclose(bands(System.build(mono, coupled), path), plain, atol=1e-12)


def test_far_bonds_count_by_topology():
    """A compressed bilayer still couples (distance alone could not tell it from a
    second neighbour), and nothing within six bonds of the same layer does."""
    law = Tail(Exponential(0.4, 3.35, 7.0, 3.0), 2.8, 3.0)
    system = System.build(geometry("AA@2.5"), _far(xu_carbon(), law, ("pps",)))
    far = [b for b in system.bonds if b.far]
    assert far and all(abs(b.vector[2]) > 2.4 for b in far)


def test_dispersion_pair_closed_form_and_zero_in_a_layer():
    term = LayerDispersionTerm(0.02, 3.0, 15.0, 3.0)
    atoms = Atoms("C2", positions=[[0, 0, 0], [0, 0, 3.6]])
    energy, forces = term.energy_and_forces(atoms)
    v, dv = term.pair(3.6)
    assert energy == pytest.approx(float(v))
    assert forces[1, 2] == pytest.approx(-float(dv))
    small = LayerDispersionTerm(0.02, 3.0, 15.0, 3.0, tail=(5.0, 5.5))
    e_mono, f_mono = small.energy_and_forces(geometry("mono"))
    assert e_mono == 0.0 and not f_mono.any()


def test_forces_match_finite_differences():
    from tbkit.forces import energy_and_forces
    from tbkit.kpoints import mesh
    from tbkit.solver import solve

    model = xu_carbon()
    far = Onset(Tail(Exponential(0.35, 3.35, 7.0, 5.0), 4.5, 5.0), 2.9, 3.1)
    _far(model, far, ("pps",))
    _far(model, Onset(Tail(Exponential(-0.05, 3.35, 7.0, 5.0), 4.5, 5.0), 2.9, 3.1), ("ppp",))
    model.repulsive = SumRepulsive((model.repulsive, LayerDispersionTerm(0.02, 3.0, 15.0, 3.0)))
    model = model_from_dict(model_to_dict(model))                 # through the file layout
    atoms = stack([0.0, 1.0], 3.3)
    atoms.rattle(0.04, seed=3)

    def energy(a, need=False):
        system = System.build(a, model)
        k, w = mesh(system, 3)
        return energy_and_forces(solve(system, k, w, kT=0.1), need_forces=need)

    _, forces, _ = energy(atoms, True)
    h = 1e-4
    for atom, axis in ((0, 0), (1, 2), (2, 2), (3, 1)):
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[atom, axis] += h
        minus.positions[atom, axis] -= h
        fd = -(energy(plus)[0] - energy(minus)[0]) / (2 * h)
        assert forces[atom, axis] == pytest.approx(fd, abs=2e-6)


@pytest.mark.parametrize("name", ["c_capas", "xu_capas"])
def test_shipped_sets_are_what_the_recipe_writes(name):
    from tbkit.params import read_parameter_file
    from tbkit.recipes import interlayer_fit as fit

    data = read_parameter_file(name)
    far = {e["bond"]: e["law"] for e in data["far_hopping"]}
    p = [far["pps"]["v0"], far["pps"]["beta"], far["ppp"]["v0"], far["ppp"]["beta"]]
    if name == "c_capas":
        again = fit.bands_dict(p, data["fit"])
    else:
        disp = data["repulsive"]["terms"][1]
        again = fit.energy_dict(p, [disp[k] for k in ("a", "alpha", "c6", "r0")], data["fit"])
    assert json.loads(json.dumps(again)) == data


@pytest.mark.parametrize("name, base", [("c_capas", "c_bandas"), ("xu_capas", "xu_carbon")])
def test_shipped_sets_keep_their_base_in_a_monolayer(name, base):
    from tbkit.params import load_parameters

    path = np.array([[0, 0, 0], [0.5, 0, 0], [1 / 3, -1 / 3, 0], [0.2, 0.1, 0]])
    mono = geometry("mono")
    assert np.allclose(bands(System.build(mono, load_parameters(name)), path),
                       bands(System.build(mono, load_parameters(base)), path), atol=1e-12)
