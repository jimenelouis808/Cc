"""Charge corrections for IR: exact derivatives, neutrality, periodic = finite, the fit."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import molecule

from tbkit import charge_model as cm


def _dipole(model, atoms):
    _, dq = cm.correction(model, atoms)
    pos = atoms.get_positions()
    return (dq[:, None] * pos).sum(axis=0)


def _numerical_born(model, atoms, h=1e-4):
    out = np.zeros((len(atoms), 3, 3))
    for a in range(len(atoms)):
        for i in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[a, i] += h
            minus.positions[a, i] -= h
            out[a, i] = (_dipole(model, plus) - _dipole(model, minus)) / (2 * h)
    return out


@pytest.fixture
def stretched():
    """Acetonitrile with its bonds pulled into the switching region and out of symmetry."""
    atoms = molecule("CH3CN")
    rng = np.random.default_rng(1)
    atoms.positions += rng.normal(scale=0.08, size=atoms.positions.shape)
    return atoms


def test_bond_flux_born_charges_are_the_derivative_of_its_dipole(stretched):
    model = cm.BondFlux.for_elements(stretched.get_chemical_symbols())
    model.theta = np.random.default_rng(2).normal(size=model.n_params())
    dz, dq = cm.correction(model, stretched)
    assert abs(dq.sum()) < 1e-12                               # neutral
    assert np.abs(dz.sum(axis=0)).max() < 1e-10                 # translation sum rule
    assert np.allclose(dz, _numerical_born(model, stretched), atol=1e-6)


def test_bond_flux_leaves_pure_carbon_alone():
    atoms = molecule("C6H6")
    del atoms[[a.index for a in atoms if a.symbol == "H"]]
    model = cm.BondFlux.for_elements(["C", "H"])
    model.theta = np.ones(model.n_params())
    dz, dq = cm.correction(model, atoms)
    assert np.allclose(dq, 0) and np.allclose(dz, 0)


def test_wire_correction_equals_the_molecule_across_the_axis(stretched):
    model = cm.BondFlux.for_elements(stretched.get_chemical_symbols())
    model.theta = np.random.default_rng(3).normal(size=model.n_params())
    finite = stretched.copy()
    finite.center(vacuum=8.0)
    wire = finite.copy()
    wire.pbc = (False, False, True)
    dz_f, _ = cm.correction(model, finite)
    dz_w, _ = cm.correction(model, wire)
    assert np.allclose(dz_w[:, :, :2], dz_f[:, :, :2], atol=1e-12)
    assert np.allclose(dz_w[:, :, 2], 0)                        # no dipole along the axis


def test_environment_flux_born_charges_are_the_derivative_of_its_dipole(stretched):
    pytest.importorskip("dscribe")
    model = cm.EnvironmentFlux(("C", "H", "N"), r_cut=3.5, n_max=2, l_max=2)
    model.theta = np.random.default_rng(4).normal(size=model.n_params()) * 0.01
    dz, dq = cm.correction(model, stretched)
    assert abs(dq.sum()) < 1e-10
    assert np.abs(dz.sum(axis=0)).max() < 1e-6
    assert np.allclose(dz, _numerical_born(model, stretched, h=1e-4), atol=1e-5)


def test_fit_recovers_a_planted_correction():
    """Reference = TB + a known θ: the fit finds it, and the CV error is ~0."""
    rng = np.random.default_rng(5)
    model = cm.BondFlux.for_elements(["C", "H", "N"])
    truth = rng.normal(size=model.n_params()) * 0.1
    rows = []
    for name in ("CH3CN", "HCN", "N2H4", "NH3", "CH4", "C2H6", "C5H5N"):
        atoms = molecule(name)
        atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
        z_tb = rng.normal(scale=0.3, size=(len(atoms), 3, 3))
        A, _ = model.design(atoms)
        rows.append({"atoms": atoms, "group": name, "z_tb": z_tb,
                     "z_ref": z_tb + np.tensordot(truth, A, axes=1)})
    record = cm.fit(rows, model, ridges=(1e-10,))
    assert record["rms_tb_e"] > 0.01
    assert record["rms_train_e"] < 1e-6
    assert record["rms_cv_e"] < 1e-3


def test_save_and_load_round_trip(tmp_path):
    model = cm.BondFlux.for_elements(["C", "H"])
    model.theta = np.array([0.1, -0.2])
    loaded = cm.load(cm.save(model, tmp_path / "m.json"))
    assert loaded.pairs == [("C", "H")] and np.allclose(loaded.theta, model.theta)
