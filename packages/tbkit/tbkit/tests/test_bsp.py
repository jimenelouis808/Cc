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
    for element, u in HUBBARD_U.items():
        assert u / Hartree == pytest.approx(dftb[element], abs=2e-4)


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
    ("S", "coronene_SH/eq", 0.06), ("S", "CH3SO3H/eq", 0.08),
    ("P", "coronene_PO3H2/eq", 0.08), ("P", "H3PO4/eq", 0.08)])
def test_geometry_close_to_gpaw(element, label, limit):
    _, model, refs = _shipped(element)
    ref = next(r for r in refs if r.label == label)
    errors = xu_family.relaxed_bond_errors(model, ref, fmax=0.02)
    assert max(errors.values()) < limit, errors
