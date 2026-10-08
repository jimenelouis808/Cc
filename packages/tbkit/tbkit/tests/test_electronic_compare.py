"""recipes/electronic_compare: the path DOS counts states, a Dirac point has no gap,
and the report reads what the recipe writes."""

from __future__ import annotations

import json

import numpy as np

from tbkit import report as rp
from tbkit.recipes import electronic_compare as ec


def test_path_dos_integrates_to_two_states_per_band():
    # A single cosine band on an even path: trapezoid weights sum to 1, spin gives 2.
    k = np.linspace(0, 0.5, ec.NK)
    bands = [[-1.0 + np.cos(2 * np.pi * kk)] for kk in k]
    grid = np.linspace(-4, 2, 6001)
    assert abs(np.trapezoid(ec.dos(bands, grid, 0.05), grid) - 2.0) < 1e-6


def test_gap_of_a_semimetal_and_an_insulator():
    assert ec.gap([[-1.0, 0.001, 0.002]]) == 0.0          # Dirac point just above E_F
    assert ec.gap([[-0.3, 0.5], [-0.1, 0.4]]) == 0.5


def test_tb_graphene_has_its_dirac_point_at_k(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "WORK", tmp_path)
    data = json.loads(ec.run_tb("graphene").read_text())
    atoms = ec.geometry("graphene")
    _, x, labels = ec.kpath("graphene", atoms)
    k = int(np.argmin(np.abs(x - labels[2][0])))
    assert min(abs(e) for e in data["bands_minus_fermi"][k]) < 1e-3
    assert ec.gap(data["bands_minus_fermi"]) == 0.0


def test_report_adapter_reads_the_recipe_files(tmp_path):
    k = [[0, 0, z] for z in np.linspace(0, 0.5, 3)]
    for name in ("pristine", "N"):
        for side in ("gpaw", "tb"):
            (tmp_path / name).mkdir(exist_ok=True)
            (tmp_path / name / f"{side}.json").write_text(json.dumps(
                {"kpts": k, "bands_minus_fermi": [[-1.0, 0.5]] * 3, "settings": side}))
    grid = np.linspace(-8, 6, 11)
    np.savez(tmp_path / "dos.npz", grid=grid, **{f"{n}_{s}": grid * 0 for n in ("pristine", "N")
                                                for s in ("gpaw", "tb")})
    entry = {"gap_gpaw_eV": 1.5, "gap_tb_eV": 1.5, "dos_correlation_-2_+2": 1.0,
             "dos_correlation_-6_+4": 1.0, "states_within_1eV": {"gpaw": 2.0, "tb": 2.0},
             "settings": {}}
    (tmp_path / "report.json").write_text(json.dumps(
        {"what": "x (recipes/electronic_compare)", "sigma_eV": 0.1,
         "systems": {"pristine": entry, "N": {**entry, "delta_dos_correlation_-3_+3": 0.5}}}))
    rep = rp.recognise(tmp_path)
    assert [t.id for t in rep.tables()] == ["resumen_coil"]
    assert {f.id for f in rep.figures()} == {"dos_pristine", "dos_N", "delta_dos_N",
                                             "bandas_pristine", "bandas_N"}
