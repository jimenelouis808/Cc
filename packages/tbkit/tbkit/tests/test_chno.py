"""The shipped ``xu_chno`` set against the GPAW references it was fitted to.

Carbon stays Xu's, the H/N/O parameters reproduce their own fit, the
acute-angle term closes the epoxide on graphene-like carbon without touching
anything with angles >= 80 degrees.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from ase.build import molecule

from tbkit.params import PARAMETER_DIR, load_parameters, xu_carbon
from tbkit.recipes import xu_family
from tbkit.recipes.xu_chno import CHNO
from tbkit.references import load_references

if not (PARAMETER_DIR / "xu_chno.json").exists():
    pytest.skip("xu_chno.json todavía no instalado", allow_module_level=True)


@pytest.fixture(scope="module")
def shipped():
    data = json.loads((PARAMETER_DIR / "xu_chno.json").read_text(encoding="utf-8"))
    refs = []
    for name in data["fit"]["references"]:
        refs += load_references(PARAMETER_DIR / "references" / name)[0]
    for ref in refs:
        if ref.group in data["fit"].get("held_out", {}):
            ref.role = "test"
    return data, load_parameters("xu_chno"), refs


def test_every_number_has_unit_and_source(shipped):
    data = shipped[0]
    entries = [v for table in data["onsite"].values() for v in table.values()]
    entries += data["hopping"] + [data["repulsive"]] + list(data["hubbard_u"].values())
    entries += [e for term in data["repulsive"]["terms"] if term["type"] == "pair"
                for e in term["pairs"]]
    for entry in entries:
        assert entry.get("unit") and entry.get("source"), entry


def test_references_are_the_ones_fitted(shipped):
    fit = shipped[0]["fit"]
    for name, digest in zip(fit["references"], fit["references_sha256"], strict=True):
        path = PARAMETER_DIR / "references" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_carbon_is_xu_untouched(shipped):
    model, xu = shipped[1], xu_carbon()
    assert model.onsite["C"] == xu.onsite["C"]
    d = np.linspace(1.2, 2.6, 15)
    for bond in ("sss", "sps", "pps", "ppp"):
        assert np.allclose(model.law(model.hopping, "C", "C", bond)(d),
                           xu.law(xu.hopping, "C", "C", bond)(d))


def test_reproduces_its_level_fit(shipped):
    data, model, refs = shipped
    train = [r for r in refs if r.role == "train"]
    _, shift, _ = xu_family.level_residuals(model, train)
    assert shift == pytest.approx(data["fit"]["level_shift_eV"], abs=1e-4)


def test_hubbard_u_of_oxygen_equals_dftb_mio(shipped):
    from ase.units import Hartree

    assert shipped[1].hubbard_u["O"] / Hartree == pytest.approx(0.4954, abs=2e-4)
    assert shipped[1].scc


def test_parameter_vector_matches_the_family(shipped):
    assert list(shipped[0]["fit"]["parameters"]) == CHNO.parameter_names()


def test_acute_term_is_zero_on_pure_carbon(shipped):
    from ase.build import bulk

    from tbkit.graphene import graphene_cell
    from tbkit.repulsive import AcuteAngleTerm

    terms = [t for t in shipped[1].repulsive.terms if isinstance(t, AcuteAngleTerm)]
    assert len(terms) == 1
    for atoms in (molecule("C6H6"), molecule("C60"), bulk("C", "diamond", a=3.56),
                  graphene_cell()):
        energy, forces = terms[0].energy_and_forces(atoms)
        assert energy == 0.0 and not forces.any()


def test_graphene_oxide_epoxide_stays_closed(shipped):
    _, model, refs = shipped
    epoxide = next(r for r in refs if r.label == "coronene_epoxide/eq")
    errors = xu_family.relaxed_bond_errors(model, epoxide, fmax=0.02)
    assert errors["C-C"] < 0.06          # without the ring scans it opened by 0.57 Å
    assert errors["C-O"] < 0.13          # 0.10 when fitted


def test_carboxylic_acid_geometry_close_to_gpaw(shipped):
    _, model, refs = shipped
    acid = next(r for r in refs if r.label == "CH3COOH/eq")
    errors = xu_family.relaxed_bond_errors(model, acid, fmax=0.02)
    assert max(errors.values()) < 0.12   # C-C 0.09 when fitted (single bond next to C=O)


def test_ring_opening_scans_are_in_the_training_data(shipped):
    from tbkit.recipes.chno_references import RING_CC, RING_SCANS

    labels = {r.label for r in shipped[2] if r.role == "train"}
    for name in RING_SCANS:
        for d in RING_CC:
            assert f"{name}/cc{d:.2f}" in labels
    assert not any(r.role == "train" for r in shipped[2] if r.group == "bicyclobutane")


def test_polarizability_references_are_the_ones_fitted(shipped):
    fit = shipped[0]["alpha_fit"]
    assert fit["fitted_elements"] == ["O"]
    for name, digest in zip(fit["references"], fit["references_sha256"], strict=True):
        path = PARAMETER_DIR / "references" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_shares_the_optics_of_xu_chn(shipped):
    chn = load_parameters("xu_chn")
    for element in ("H", "C", "N"):
        assert shipped[1].extra_polarizability[element] == chn.extra_polarizability[element]
        if element != "H":
            assert shipped[1].onsite_dipole[element] == chn.onsite_dipole[element]


def test_formic_acid_alpha_tensor_against_gpaw(shipped):
    from tbkit.hamiltonian import System
    from tbkit.optics import polarizability_linear_response

    _, model, refs = shipped
    data = json.loads((PARAMETER_DIR / "references" / "gpaw_chno_alpha.json")
                      .read_text(encoding="utf-8"))
    gpaw = next(e for e in data["polarizabilities"] if e["group"] == "HCOOH")["alpha"]
    atoms = next(r for r in refs if r.label == "HCOOH/eq").atoms
    tb = polarizability_linear_response(System.build(atoms, model))
    assert np.sort(np.linalg.eigvalsh(tb)) == pytest.approx(
        np.sort(np.linalg.eigvalsh(gpaw)), rel=0.10)            # ≤ 3 % when fitted


def test_hessians_are_the_ones_fitted(shipped):
    fit = shipped[0]["fit"]
    if "hessians" not in fit:
        pytest.skip("conjunto ajustado sin hessianas")
    path = PARAMETER_DIR / "references" / fit["hessians"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == fit["hessians_sha256"]


def test_methanol_frequencies_against_gpaw(shipped):
    # The C-H hopping tail once sat where the hydroxyl H meets its carbon's
    # neighbour: methanol relaxed onto its edge and got an O-H at 4950 cm⁻¹.
    from tbkit.recipes.xu_family import frequency_validation, load_hessians

    if "hessians" not in shipped[0]["fit"]:
        pytest.skip("conjunto ajustado sin hessianas")
    targets = [t for t in load_hessians(PARAMETER_DIR / "references" /
                                        shipped[0]["fit"]["hessians"]) if t.name == "CH3OH"]
    entry = frequency_validation(shipped[1], targets)["CH3OH"]
    assert entry["rms"] < 160 and not entry["tail_hits"]          # 120 when fitted
    assert abs(entry["tb"][-1] - entry["gpaw"][-1]) < 300        # the O-H stretch
