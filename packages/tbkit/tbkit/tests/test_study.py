"""tbkit.recipes.study: any set of systems, a change embedded in its host (small and fast)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from tbkit.recipes import study as st


def _toluene_from(benzene: Atoms) -> Atoms:
    """Benzene's H6 becomes the methyl C (same index), three H appended: host atoms first."""
    out = benzene.copy()
    c0, h6 = out.positions[0], out.positions[6]
    axis = (h6 - c0) / np.linalg.norm(h6 - c0)
    out[6].symbol = "C"
    out.positions[6] = c0 + 1.50 * axis
    perpendicular = np.cross(axis, [0, 0, 1.0])
    perpendicular /= np.linalg.norm(perpendicular)
    other = np.cross(axis, perpendicular)
    for angle in (0, 2 * np.pi / 3, 4 * np.pi / 3):
        direction = 0.34 * axis + 0.94 * (np.cos(angle) * perpendicular + np.sin(angle) * other)
        out.append("H")
        out.positions[-1] = out.positions[6] + 1.09 * direction / np.linalg.norm(direction)
    return out


@pytest.fixture(scope="module")
def small_study(tmp_path_factory):
    folder = tmp_path_factory.mktemp("estudio")
    benzene = molecule("C6H6")
    benzene.center(vacuum=5.0)
    benzene.write(folder / "benceno.extxyz")
    _toluene_from(benzene).write(folder / "tolueno.extxyz")
    path = folder / "estudio.json"
    path.write_text(json.dumps({
        "nombre": "prueba", "modelo": "xu_chn", "carpeta": str(folder / "out"),
        "sistemas": {"benceno": {"estructura": "benceno.extxyz"},
                     "tolueno": {"estructura": "tolueno.extxyz", "sobre": "benceno",
                                 "centros": [6]}},
        "relajar": {"fmax": 0.05},
        "hessiana": {"radio": 3.0},
        "raman": {"laseres_ev": [2.33], "min_cm1": 2900.0},
        "ir": {"min_cm1": 2900.0}}))
    study = st.Study(path)
    st.run_step(study, "todo", study.order())
    return study


def test_study_runs_any_systems_and_embeds_the_change(small_study):
    study = small_study
    assert study.order() == ["benceno", "tolueno"]
    assert study.kpts(study.relaxed("benceno")) == (1, 1, 1)          # a molecule: Γ only
    report = json.loads((study.work / "report.json").read_text())
    assert report["sistemas"] == ["benceno", "tolueno"] and report["tolueno"]["atoms"] == 15
    # embedded only around the change: fewer rows than atoms
    tol = study.relaxed("tolueno")
    assert len(study.region("tolueno", tol)) < len(tol)
    checks = json.loads((study.folder("tolueno") / "check.json").read_text())
    assert all(abs(c["full_cm1"] - c["embedded_cm1"]) < 25 for c in checks)
    assert "ir" in report["tolueno"]
    from tbkit.report import recognise

    rep = recognise(study.work)
    assert rep.title.startswith("Estudio prueba") and rep.figures()


def test_wrong_atom_order_is_refused(small_study, tmp_path):
    study = small_study
    tol = study.relaxed("tolueno")
    shuffled = tol[[1, 0] + list(range(2, len(tol)))]
    shuffled.write(tmp_path / "mal.extxyz")
    path = tmp_path / "estudio.json"
    path.write_text(json.dumps({
        "nombre": "mal", "carpeta": str(tmp_path / "out"),
        "sistemas": {"benceno": {"estructura": str(study.folder("benceno") / "relaxed.extxyz")},
                     "mal": {"estructura": "mal.extxyz", "sobre": "benceno",
                             "centros": [6]}}}))
    bad = st.Study(path)
    with pytest.raises(ValueError, match="orden"):
        bad.check_mapping("mal")
    with pytest.raises(ValueError, match="centros"):
        path.write_text(json.dumps({"sistemas": {"a": {"estructura": "x"},
                                                 "b": {"estructura": "y", "sobre": "a"}}}))
        st.Study(path)


def test_dft_request_picks_the_modes_that_matter(small_study):
    study = small_study
    folder = study.dft_request("tolueno", code="gpaw", relax=True)
    request = json.loads((folder / "solicitud.json").read_text())
    modes = json.loads((folder / "tbkit_modos.json").read_text())["modes"]
    assert request["tareas"][0]["tipo"] == "relajar" and modes
    vib = study.vibrations("tolueno")
    assert all(vib.frequencies[m] > 100 for m in modes)


def test_change_centres_and_study_file(tmp_path):
    from tbkit.gui import actions

    benzene = molecule("C6H6")
    toluene = _toluene_from(benzene)
    assert actions.change_centres(benzene, toluene) == [6, 12, 13, 14]
    swapped = toluene[[1, 0] + list(range(2, len(toluene)))]
    with pytest.raises(ValueError, match="orden"):
        actions.change_centres(benzene, swapped)
    path = actions.write_study(tmp_path / "e", "prueba gui", ("benceno", benzene),
                               [("tolueno", toluene)])
    data = json.loads(path.read_text())
    assert data["sistemas"]["tolueno"] == {"estructura": "tolueno.extxyz", "sobre": "benceno",
                                           "centros": [6, 12, 13, 14]}
    assert st.Study(path).order() == ["benceno", "tolueno"]
