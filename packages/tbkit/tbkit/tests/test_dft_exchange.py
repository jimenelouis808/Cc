"""tbkit.dft_exchange: requests written, results read and compared (no DFT engine needed)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase.build import graphene, molecule

from tbkit import dft_exchange as dx
from tbkit.modes import Vibrations


def _fake_run(folder, hessian, born=None):
    """What a driver writes for a harmonic system: F = -K u exactly, μ = Z·u."""
    request = json.loads((folder / "solicitud.json").read_text())
    task = next(t for t in request["tareas"] if t["tipo"] == "desplazamientos")
    vectors = np.load(folder / "vectores.npy")
    scales, rows = [], []
    for i, v in enumerate(vectors):
        s = task["amplitud"] / np.linalg.norm(v, axis=1).max()
        scales.append(s)
        for sign in (1, -1):
            u = sign * s * v
            forces = -(hessian @ u.ravel()).reshape(-1, 3)
            dipole = None if born is None else list(np.einsum("aij,aj->i", born, u))
            rows.append({"vector": i, "signo": sign, "energia": 0.0,
                         "fuerzas": forces.tolist(), "dipolo": dipole})
    zero = np.zeros((len(vectors[0]), 3)).tolist()
    result = {"esquema": dx.SCHEMA, "codigo": "prueba", "terminado": True,
              "unidades": {"energia": "eV", "fuerza": "eV/Å", "distancia": "Å"},
              "tareas": {"sp": {"energia": 0.0, "fuerzas": zero},
                         "modos": {"amplitud": task["amplitud"], "escalas": scales,
                                   "desplazamientos": rows}}}
    (folder / "resultado.json").write_text(json.dumps(result))


def test_rayleigh_frequency_of_an_eigenmode_is_exact(tmp_path):
    atoms = molecule("H2O")
    rng = np.random.default_rng(3)
    a = rng.normal(size=(9, 9))
    hessian = a @ a.T
    vib = Vibrations.from_hessian(atoms, hessian, source="prueba")
    folder = dx.mode_check_request(tmp_path / "agua", vib, [6, 7, 8], code="emt")
    born = np.repeat(np.eye(3)[None] * 0.3, 3, axis=0)        # an isotropic Born charge
    _fake_run(folder, hessian, born)
    rows = dx.mode_frequencies(dx.read_result(folder), folder)
    assert [r["mode"] for r in rows] == [6, 7, 8]
    for r in rows:                                    # exact: the patterns are eigenmodes
        assert r["dft_cm1"] == pytest.approx(r["tb_cm1"], rel=1e-9)
        assert r["dft_over_tb"] == pytest.approx(1.0, rel=1e-9)
        assert r["dft_ir_km_mol"] is not None
    # Along periodic axes the dipole is not used (it would need the Berry phase).
    assert all(r["dft_ir_km_mol"] is None
               for r in dx.mode_frequencies(dx.read_result(folder), folder, (True,) * 3))


def test_requests_kpoints_and_refusals(tmp_path):
    sheet = graphene(vacuum=6.0)
    assert dx.kpts_for(sheet) == [13, 13, 1]
    benzene = molecule("C6H6", vacuum=5.0)
    assert dx.kpts_for(benzene) == [1, 1, 1]
    folder = dx.write_request(tmp_path / "b", benzene, [{"id": "sp", "tipo": "punto_simple"}])
    request = json.loads((folder / "solicitud.json").read_text())
    assert request["esquema"] == dx.SCHEMA and request["ajustes"]["base"] == "dzp"
    with pytest.raises(FileExistsError):
        dx.write_request(folder, benzene, [{"id": "sp", "tipo": "punto_simple"}])
    with pytest.raises(FileNotFoundError):
        dx.read_result(folder)


def test_summary_compares_forces_with_the_model(tmp_path):
    from tbkit.calculator import TBCalculator
    from tbkit.params import xu_carbon

    atoms = graphene(vacuum=6.0)
    atoms.positions[0, 2] += 0.05
    model = xu_carbon()
    probe = atoms.copy()
    probe.calc = TBCalculator(model, kpts=4, kT=0.05)
    forces = probe.get_forces()
    folder = dx.write_request(tmp_path / "g", atoms, [{"id": "sp", "tipo": "punto_simple"}])
    (folder / "resultado.json").write_text(json.dumps({
        "esquema": dx.SCHEMA, "codigo": "prueba", "terminado": True,
        "unidades": {"energia": "eV", "fuerza": "eV/Å"},
        "tareas": {"sp": {"energia": -1.0, "fuerzas": forces.tolist()}}}))
    out = dx.summary(folder, model, kmesh=4, kT=0.05)
    # 1e-6: the structure goes through an extxyz file (positions to ~1e-8 Å)
    assert out["tareas"]["sp"]["fuerzas_tbkit_vs_dft"]["rms_eV_A"] == pytest.approx(0, abs=1e-6)
