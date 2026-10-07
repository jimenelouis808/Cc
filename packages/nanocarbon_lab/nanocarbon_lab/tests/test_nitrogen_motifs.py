"""Pyridinic and pyrrolic N: composition, closed shell, coordination, periodic cells."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import graphene

from nanocarbon_lab.dopants.nitrogen import pyridinic_divacancy, pyrrolic_vacancy

VALENCE = {"C": 4, "N": 5, "H": 1}


def _sheet():
    atoms = graphene(a=2.46, size=(5, 5, 1), vacuum=6.0)
    atoms.pbc = (True, True, False)
    return atoms


def _neighbours(atoms, k, cutoff=1.8):
    d = atoms.get_distances(k, range(len(atoms)), mic=True)
    return np.flatnonzero((d > 0.1) & (d < cutoff))


def test_pyridinic_divacancy_is_n4v2_and_closed_shell():
    sheet = _sheet()
    a = 0
    b = int(_neighbours(sheet, a)[0])
    out = pyridinic_divacancy(sheet, (a, b))
    symbols = out.get_chemical_symbols()
    assert len(out) == len(sheet) - 2 and symbols.count("N") == 4
    assert sum(VALENCE[s] for s in symbols) % 2 == 0
    for n in out.info["nitrogen_motif"]["nitrogen"]:
        assert symbols[n] == "N" and len(_neighbours(out, n)) == 2   # pyridinic: two C


def test_pyrrolic_vacancy_places_nh_and_two_ch():
    sheet = _sheet()
    removed = 10
    ring_atom, *_ = (int(k) for k in _neighbours(sheet, removed))
    out = pyrrolic_vacancy(sheet, ring_atom, removed)
    symbols = out.get_chemical_symbols()
    assert symbols.count("N") == 1 and symbols.count("H") == 3
    assert len(out) == len(sheet) - 1 + 3
    assert sum(VALENCE[s] for s in symbols) % 2 == 0
    n = out.info["nitrogen_motif"]["nitrogen"][0]
    hydrogens = [k for k in _neighbours(out, n, 1.2) if symbols[k] == "H"]
    assert len(hydrogens) == 1
    assert out.get_distance(n, hydrogens[0], mic=True) == pytest.approx(1.01)


def test_refuses_bad_choices():
    sheet = _sheet()
    with pytest.raises(ValueError, match="not bonded"):
        pyridinic_divacancy(sheet, (0, 30))
    with pytest.raises(ValueError, match="not bonded"):
        pyrrolic_vacancy(sheet, 0, 30)
