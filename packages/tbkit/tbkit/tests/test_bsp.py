"""B, S and P on a fixed xu_chno: the base stays exact, every pair is defined.

The free-atom numbers (U, ⟨ns|r|np⟩) are checked against DFTB's published
Hubbard values; the shipped sets, once installed, against their GPAW fits.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from ase.build import molecule

from tbkit.calculator import TBCalculator
from tbkit.hamiltonian import System
from tbkit.params import load_parameters
from tbkit.recipes import xu_family
from tbkit.recipes.xu_bsp import FAMILIES, HUBBARD_U
from tbkit.solver import solve


@pytest.mark.parametrize("element", sorted(FAMILIES))
def test_molecules_without_the_new_element_are_exactly_xu_chno(element):
    family = FAMILIES[element]
    chno = load_parameters("xu_chno")
    model = family.build_model(family.initial_guess(),
                               family.repulsion_from_coefficients(
                                   np.ones(len(family.pairs) * len(xu_family.POWERS))))
    atoms = molecule("CH3COOH")
    energies = []
    for m in (chno, model):
        a = atoms.copy()
        a.calc = TBCalculator(m)
        energies.append((a.get_potential_energy(), a.get_forces()))
    assert energies[0][0] == pytest.approx(energies[1][0], abs=1e-9)
    assert np.allclose(energies[0][1], energies[1][1], atol=1e-8)
    assert model.extra_polarizability == chno.extra_polarizability


@pytest.mark.parametrize("element", sorted(FAMILIES))
def test_every_pair_is_defined(element):
    family = FAMILIES[element]
    fewer = dict(family.pairs)
    fewer.pop(next(iter(fewer)))
    with pytest.raises(ValueError, match="pares sin definir"):
        dataclasses.replace(family, pairs=fewer)


def test_level_shift_is_the_base_one():
    import json

    from tbkit.params import PARAMETER_DIR

    data = json.loads((PARAMETER_DIR / "xu_chno.json").read_text(encoding="utf-8"))
    assert FAMILIES["S"].fixed_shift() == data["fit"]["level_shift_eV"]
    assert xu_family.XuFamily.fixed_shift(dataclasses.replace(FAMILIES["S"], base=None)) is None


def test_hubbard_u_equals_dftb():
    from ase.units import Hartree

    dftb = {"B": 0.2961, "P": 0.2894, "S": 0.3288}      # Ha, 3ob / matsci
    for element, value in dftb.items():             # Se: no published DFTB value to compare
        assert HUBBARD_U[element] / Hartree == pytest.approx(value, abs=2e-4)


def test_new_element_builds_a_hermitian_hamiltonian():
    family = FAMILIES["S"]
    model = family.build_model(family.initial_guess())
    solution = solve(System.build(molecule("CH3SH"), model))
    assert np.all(np.isfinite(solution.energies))


def _shipped(element):
    import json

    from tbkit.params import PARAMETER_DIR
    from tbkit.references import load_references

    path = PARAMETER_DIR / f"xu_chno{element.lower()}.json"
    if not path.exists():
        pytest.skip(f"{path.name} todavía no instalado")
    data = json.loads(path.read_text(encoding="utf-8"))
    names = data["fit"]["references"]
    refs = []
    for name in [names] if isinstance(names, str) else names:
        refs += load_references(PARAMETER_DIR / "references" / name)[0]
    return data, load_parameters(path.stem), refs


@pytest.mark.parametrize("element", sorted(FAMILIES))
def test_shipped_set_references_and_units(element):
    import hashlib

    from tbkit.params import PARAMETER_DIR

    data, model, _ = _shipped(element)
    names, shas = data["fit"]["references"], data["fit"]["references_sha256"]
    for name, digest in zip([names] if isinstance(names, str) else names,
                            [shas] if isinstance(shas, str) else shas, strict=True):
        assert hashlib.sha256((PARAMETER_DIR / "references" / name).read_bytes()
                              ).hexdigest() == digest
    entries = [v for table in data["onsite"].values() for v in table.values()]
    entries += data["hopping"] + list(data["hubbard_u"].values())
    for entry in entries:
        assert entry.get("unit") and entry.get("source"), entry
    assert model.scc and element in model.orbitals
    assert list(data["fit"]["parameters"]) == FAMILIES[element].parameter_names()


@pytest.mark.parametrize("element", sorted(FAMILIES))
def test_shipped_set_keeps_xu_chno(element):
    _, model, _ = _shipped(element)
    chno = load_parameters("xu_chno")
    for el in chno.onsite:
        assert model.onsite[el] == chno.onsite[el]
    atoms = molecule("CH3COOH")
    energies = []
    for m in (chno, model):
        a = atoms.copy()
        a.calc = TBCalculator(m)
        energies.append(a.get_potential_energy())
    assert energies[0] == pytest.approx(energies[1], abs=1e-9)


@pytest.mark.parametrize("element, label, limit", [
    ("B", "coronene_BN/eq", 0.06), ("B", "borazine/eq", 0.03),
    ("S", "coronene_SH/eq", 0.06), ("S", "thiophene/eq", 0.06),
    ("P", "coronene_PO3H2/eq", 0.10), ("P", "H3PO4/eq", 0.08),
    # the ester collapsed (methyl H onto O) before active learning
    ("P", "PO(OMe)3/eq", 0.10),
    ("Se", "coronene_SeH/eq", 0.03), ("Se", "selenophene/eq", 0.06)])
def test_geometry_close_to_gpaw(element, label, limit):
    _, model, refs = _shipped(element)
    ref = next(r for r in refs if r.label == label)
    errors = xu_family.relaxed_bond_errors(model, ref, fmax=0.02)
    assert max(errors.values()) < limit, errors


def _centred_angle():
    from tbkit.repulsive import CentredAngleTerm

    return CentredAngleTerm((0.8, -1.3, 2.1, 0.5), "P",
                            {"O": (1.75, 2.0), "C": (2.0, 2.25), "H": (1.55, 1.8)})


def test_centred_angle_forces_are_energy_derivatives():
    from ase import Atoms

    rng = np.random.default_rng(3)
    atoms = Atoms("POOCH", positions=[[0, 0, 0], [1.5, 0.1, 0], [-0.5, 1.4, 0.2],
                                      [-0.4, -0.6, 1.7], [0.3, -1.45, -0.2]])
    atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
    term = _centred_angle()
    _, forces = term.energy_and_forces(atoms)
    delta, fd = 1e-5, np.zeros_like(forces)
    for a in range(len(atoms)):
        for k in range(3):
            for sign in (1, -1):
                moved = atoms.copy()
                moved.positions[a, k] += sign * delta
                fd[a, k] -= sign * term.energy_and_forces(moved)[0] / (2 * delta)
    assert np.allclose(forces, fd, atol=1e-6)
    assert np.allclose(forces.sum(axis=0), 0, atol=1e-10)


def test_centred_angle_is_zero_without_its_centre_and_round_trips():
    from tbkit.repulsive import repulsive_from_dict

    term = _centred_angle()
    energy, forces = term.energy_and_forces(molecule("CH3OH"))
    assert energy == 0 and not forces.any()
    again = repulsive_from_dict(term.to_dict())
    atoms = molecule("PH3")
    assert again.energy_and_forces(atoms)[0] == pytest.approx(term.energy_and_forces(atoms)[0])


def test_centred_torsion_forces_are_energy_derivatives():
    from ase import Atoms

    from tbkit.repulsive import CentredTorsionTerm, repulsive_from_dict

    rng = np.random.default_rng(1)
    atoms = Atoms("SeOOHC", positions=[[0, 0, 0], [1.8, 0, 0], [-0.6, 1.5, 0],
                                       [2.1, 0.9, 0.3], [-0.5, -0.8, 1.6]])
    atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
    term = CentredTorsionTerm((0.3, -0.2, 0.15), "Se", {"O": (1.8, 2.05), "C": (2.1, 2.35)},
                              (1.05, 1.25))
    energy, forces = term.energy_and_forces(atoms)
    assert energy != 0
    delta, fd = 1e-5, np.zeros_like(forces)
    for a in range(len(atoms)):
        for k in range(3):
            for sign in (1, -1):
                moved = atoms.copy()
                moved.positions[a, k] += sign * delta
                fd[a, k] -= sign * term.energy_and_forces(moved)[0] / (2 * delta)
    assert np.allclose(forces, fd, atol=1e-7)
    assert np.allclose(forces.sum(axis=0), 0, atol=1e-10)
    assert repulsive_from_dict(term.to_dict()).energy_and_forces(atoms)[0] == pytest.approx(energy)
    assert term.energy_and_forces(molecule("CH3OH"))[0] == 0
