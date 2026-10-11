"""carbonforge.exchange: DFT requests of other packages prepared and run (EMT, no DFT engine)."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms

from carbonforge.exchange import SCHEMA, prepare, read_request, status
from carbonforge.jobs.queue import ExchangeAdapter, adapter_for, engine_of


def _request(folder, tasks, code="emt", vectors=None):
    folder.mkdir(parents=True)
    atoms = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.25]], cell=[8, 8, 8])
    atoms.center()
    atoms.write(folder / "estructura.extxyz")
    if vectors is not None:
        np.save(folder / "vectores.npy", vectors)
    (folder / "solicitud.json").write_text(json.dumps(
        {"esquema": SCHEMA, "nombre": "co", "estructura": "estructura.extxyz", "codigo": code,
         "ajustes": {}, "tareas": tasks, "recursos": {"mpi": 4, "memoria_gb": 8}}))
    return folder


def _run(folder):
    return subprocess.run([sys.executable, "run_dft.py"], cwd=folder, capture_output=True,
                          text=True, timeout=300, check=False)



def test_bad_requests_are_refused(tmp_path):
    with pytest.raises(ValueError, match="falta solicitud"):
        read_request(tmp_path)
    folder = _request(tmp_path / "a", [{"id": "x", "tipo": "magia"}])
    with pytest.raises(ValueError, match="tipo"):
        read_request(folder)
    folder = _request(tmp_path / "b", [{"id": "sp", "tipo": "punto_simple", "sobre": "relax"}])
    with pytest.raises(ValueError, match="sobre"):
        read_request(folder)
    folder = _request(tmp_path / "c", [{"id": "sp", "tipo": "punto_simple"}], code="qe")
    with pytest.raises(ValueError, match="pseudos"):
        read_request(folder)


def test_emt_run_end_to_end_resumes_and_refuses_a_changed_request(tmp_path):
    vectors = np.zeros((1, 2, 3))
    vectors[0, :, 2] = [-1.0, 1.0]                  # the C-O stretch
    folder = _request(tmp_path / "co", [
        {"id": "relax", "tipo": "relajar", "fmax": 0.01},
        {"id": "sp", "tipo": "punto_simple", "sobre": "relax"},
        {"id": "modos", "tipo": "desplazamientos", "vectores": "vectores.npy",
         "amplitud": 0.01, "sobre": "relax"},
        {"id": "fc", "tipo": "hessiana", "delta": 0.005, "sobre": "relax"}], vectors=vectors)
    script = prepare(folder, slurm=True)
    assert script.exists() and "#SBATCH --ntasks=4" in (folder / "trabajo.slurm").read_text()
    assert isinstance(adapter_for(folder), ExchangeAdapter) and engine_of(folder) == "emt"
    run = _run(folder)
    assert run.returncode == 0, run.stderr
    result = json.loads((folder / "resultado.json").read_text())
    assert result["terminado"] and status(folder)["state"] == "terminado"
    assert ExchangeAdapter().succeeded(folder)
    relax, sp = result["tareas"]["relax"], result["tareas"]["sp"]
    assert relax["convergida"] and np.abs(sp["fuerzas"]).max() < 0.01
    hessian = np.load(folder / result["tareas"]["fc"]["matriz"])
    assert hessian.shape == (6, 6) and np.allclose(hessian, hessian.T, atol=0.05)
    # the stretch from the ± displacements equals the one from the Hessian
    disp = {(d["vector"], d["signo"]): d for d in result["tareas"]["modos"]["desplazamientos"]}
    s = result["tareas"]["modos"]["escalas"][0]
    k_u = -(np.array(disp[(0, 1)]["fuerzas"]) - np.array(disp[(0, -1)]["fuerzas"])) / (2 * s)
    assert np.sum(vectors[0] * k_u) == pytest.approx(
        vectors[0].ravel() @ hessian @ vectors[0].ravel(), rel=0.02)
    # a second run computes nothing again
    stamps = {p: p.stat().st_mtime_ns for p in (folder / "tareas").rglob("*.json")
              if p.name != "solicitud_usada.json"}
    assert _run(folder).returncode == 0
    assert all(p.stat().st_mtime_ns == t for p, t in stamps.items())
    # a changed request is refused: its results would be someone else's
    request = json.loads((folder / "solicitud.json").read_text())
    request["tareas"][0]["fmax"] = 0.05
    (folder / "solicitud.json").write_text(json.dumps(request))
    rerun = _run(folder)
    assert rerun.returncode != 0 and "cambió" in rerun.stderr + rerun.stdout
