"""The band-fitted carbon set (``c_bandas``, recipe ``band_fit``).

Closed form: in planar graphene p_z decouples, so the π levels at Γ are those of a
2×2 generalised problem written straight from the file's distance laws (first, second
and third neighbours: all inside the cutoff, none in the tail). The file must be what
the recipe writes from its own parameter vector (never edited by hand), it must be
electronic only (no repulsion), and graphene must stay gapless at K.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tbkit.params import load_parameters
from tbkit.recipes import band_fit
from tbkit.recipes.band_validation import geometry


@pytest.fixture(scope="module")
def data():
    from tbkit.params import read_parameter_file

    return read_parameter_file("c_bandas")


@pytest.fixture(scope="module")
def graphene_system():
    from tbkit.hamiltonian import System

    return System.build(geometry("graphene"), load_parameters("c_bandas"))


def _law(entry, d):
    law = entry["law"]
    assert d < entry["tail"][0]
    return law["v0"] * np.exp(-law["beta"] * (d / law["d0"] - 1.0))


def _vector(data):
    hop = {e["bond"]: e["law"] for e in data["hopping"]}
    ovl = {e["bond"]: e["law"] for e in data["overlap"]}
    return np.r_[[hop[b]["v0"] for b in band_fit.BONDS], [hop[b]["beta"] for b in band_fit.BONDS],
                 [ovl[b]["v0"] for b in band_fit.BONDS], ovl["sss"]["beta"]]


def test_file_is_what_the_recipe_writes(data):
    again = band_fit.parameter_dict(_vector(data), data["fit"])
    assert json.loads(json.dumps(again)) == data


def test_onsite_is_the_computed_atomic_value(data):
    assert data["onsite"]["C"]["s"]["value"] == band_fit.ES_ATOM
    assert data["onsite"]["C"]["p"]["value"] == 0.0


def test_pi_levels_at_gamma_closed_form(data, graphene_system):
    from tbkit.analysis import bands

    a = geometry("graphene").cell.lengths()[0]
    d1, d2, d3 = a / np.sqrt(3), a, 2 * a / np.sqrt(3)
    hop = {e["bond"]: e for e in data["hopping"]}["ppp"]
    ovl = {e["bond"]: e for e in data["overlap"]}["ppp"]
    h_aa, s_aa = 6 * _law(hop, d2), 1 + 6 * _law(ovl, d2)
    h_ab, s_ab = 3 * _law(hop, d1) + 3 * _law(hop, d3), 3 * _law(ovl, d1) + 3 * _law(ovl, d3)
    expected = [(h_aa + h_ab) / (s_aa + s_ab), (h_aa - h_ab) / (s_aa - s_ab)]
    levels = bands(graphene_system, np.zeros((1, 3)))[0, 0]
    for e in expected:
        assert np.min(np.abs(levels - e)) < 1e-8


def test_graphene_gapless_at_k(graphene_system):
    from tbkit.analysis import bands

    levels = bands(graphene_system, np.array([[1 / 3, 1 / 3, 0.0]]))[0, 0]
    n = int(round(graphene_system.electrons)) // 2
    assert abs(levels[n] - levels[n - 1]) < 1e-8


def test_electronic_only():
    model = load_parameters("c_bandas")
    assert model.repulsive is None or not model.repulsive
