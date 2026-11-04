"""d orbitals in the Slater-Koster blocks (``tbkit.slater_koster``).

Closed forms: along z every block is diagonal in m, with σ for m = 0, π for |m| = 1 and δ
for |m| = 2 (dz2-dz2 = V_ddσ, dzx-dzx = V_ddπ, dxy-dxy = dx2-y2 = V_ddδ, s-dz2 = V_sdσ,
pz-dz2 = V_pdσ, px-dzx = V_pdπ). Covariance: a rotated bond gives the block rotated by
the real representations of p (R itself) and d (R acting on the traceless symmetric
matrices of the cubic harmonics), for every pair of shells: the test that catches a sign
or a factor of the table. Parity: H_ab(v) = H_ba(-v)ᵀ with heteronuclear laws. The
vectorised ``blocks`` equals ``block``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tbkit.params import BONDS, D_BONDS, D_ORBITALS, ORDERED_BONDS, Constant, TBModel
from tbkit.slater_koster import block, blocks

SPD = ("s", "px", "py", "pz") + D_ORBITALS


def _model() -> TBModel:
    rng = np.random.default_rng(4)
    hopping = {}
    for a in ("A", "B"):
        for b in ("A", "B"):
            for bond in BONDS + D_BONDS:
                if bond in ORDERED_BONDS or (b, a, bond) not in hopping:
                    hopping[(a, b, bond)] = Constant(float(rng.uniform(-3, 3)), 10.0)
                else:                    # symmetric integrals: one law for both orders
                    hopping[(a, b, bond)] = hopping[(b, a, bond)]
    return TBModel("prueba spd", orbitals={"A": SPD, "B": SPD},
                   onsite={"A": {"s": 0.0, "p": 0.0, "d": 0.0},
                           "B": {"s": 0.0, "p": 0.0, "d": 0.0}},
                   hopping=hopping, valence={"A": 1.0, "B": 1.0})


def _d_matrices():
    """Orthonormal traceless symmetric matrices of dxy, dyz, dzx, dx2-y2, dz2."""
    def sym(i, j):
        m = np.zeros((3, 3))
        m[i, j] = m[j, i] = 1.0
        return m / np.sqrt(2)
    xx = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2)
    zz = np.diag([-1.0, -1.0, 2.0]) / np.sqrt(6)
    return [sym(0, 1), sym(1, 2), sym(2, 0), xx, zz]


def _representation(rotation):
    q = _d_matrices()
    d = np.array([[np.sum(q[k] * (rotation @ q[j] @ rotation.T)) for j in range(5)]
                  for k in range(5)])
    rep = np.zeros((9, 9))
    rep[0, 0] = 1.0
    rep[1:4, 1:4] = rotation
    rep[4:, 4:] = d
    return rep


def _rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def test_bond_along_z_is_sigma_pi_delta():
    model = _model()
    h = block(model, model.hopping, "A", SPD, "A", SPD, np.array([0.0, 0.0, 2.0]))
    i = {o: k for k, o in enumerate(SPD)}
    law = {b: model.hopping[("A", "A", b)](2.0) for b in BONDS + D_BONDS}
    assert h[i["dz2"], i["dz2"]] == pytest.approx(law["dds"])
    assert h[i["dzx"], i["dzx"]] == pytest.approx(law["ddp"])
    assert h[i["dyz"], i["dyz"]] == pytest.approx(law["ddp"])
    assert h[i["dxy"], i["dxy"]] == pytest.approx(law["ddd"])
    assert h[i["dx2-y2"], i["dx2-y2"]] == pytest.approx(law["ddd"])
    assert h[i["s"], i["dz2"]] == pytest.approx(law["sds"])
    assert h[i["pz"], i["dz2"]] == pytest.approx(law["pds"])
    assert h[i["px"], i["dzx"]] == pytest.approx(law["pdp"])
    off = h.copy()
    for o in SPD:                                  # nothing else couples along z except
        off[i[o], i[o]] = 0.0                      # the pairs with the same m
    for a, b in (("s", "pz"), ("s", "dz2"), ("pz", "dz2"), ("px", "dzx"), ("py", "dyz"),
                 ("pz", "s"), ("dz2", "s"), ("dz2", "pz"), ("dzx", "px"), ("dyz", "py")):
        off[i[a], i[b]] = 0.0
    assert np.abs(off).max() < 1e-12


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rotational_covariance(seed):
    model = _model()
    vector = np.random.default_rng(10 + seed).normal(size=3) * 1.3
    rotation = _rotation(seed)
    rep = _representation(rotation)
    for pair in (("A", "A"), ("A", "B"), ("B", "A")):
        h = block(model, model.hopping, pair[0], SPD, pair[1], SPD, vector)
        turned = block(model, model.hopping, pair[0], SPD, pair[1], SPD, rotation @ vector)
        assert turned == pytest.approx(rep @ h @ rep.T, abs=1e-10)


def test_parity_and_hermiticity_of_the_pair():
    model = _model()
    vector = np.array([0.7, -1.1, 1.6])
    ab = block(model, model.hopping, "A", SPD, "B", SPD, vector)
    ba = block(model, model.hopping, "B", SPD, "A", SPD, -vector)
    assert ab == pytest.approx(ba.T, abs=1e-12)


def test_vectorised_blocks_agree():
    model = _model()
    vectors = np.random.default_rng(7).normal(size=(5, 3)) * 1.5
    many = blocks(model, model.hopping, "A", SPD, "B", SPD, vectors)
    for k, v in enumerate(vectors):
        assert many[k] == pytest.approx(block(model, model.hopping, "A", SPD, "B", SPD, v),
                                        abs=1e-12)


def test_skf_with_d_columns(tmp_path):
    """A synthetic homonuclear .skf with d columns: the laws reproduce the table and a
    bond along z gives its σ, π, δ values; the d on-site energy and occupation are read."""
    from ase.units import Bohr, Hartree

    from tbkit.skf import load_skf_set
    from tbkit.tests.test_parameters import _write_skf

    grid = 0.05
    r = np.arange(1, 200) * grid
    decay = np.exp(-r / 2.0)
    columns = {"Hss0": -0.30 * decay, "Hsp0": 0.25 * decay, "Hpp0": 0.30 * decay,
               "Hpp1": -0.10 * decay, "Hsd0": -0.12 * decay, "Hpd0": -0.15 * decay,
               "Hpd1": 0.06 * decay, "Hdd0": -0.08 * decay, "Hdd1": 0.04 * decay,
               "Hdd2": -0.01 * decay, "Sss0": 0.2 * decay, "Sdd0": 0.05 * decay}
    onsite = (-0.20, -0.10, -0.30, 0.0, 0.3, 0.25, 0.28, 6.0, 0.0, 2.0)   # Ed Ep Es ... fd fp fs
    _write_skf(tmp_path / "X-X.skf", grid, columns, onsite, repeat_zeros=False)
    model = load_skf_set(tmp_path, {"X": SPD})
    assert model.onsite["X"]["d"] == pytest.approx(-0.20 * Hartree)
    assert model.valence["X"] == pytest.approx(8.0)
    d = 40 * grid * Bohr
    h = block(model, model.hopping, "X", SPD, "X", SPD, np.array([0.0, 0.0, d]))
    i = {o: k for k, o in enumerate(SPD)}
    k = 39
    assert h[i["dz2"], i["dz2"]] == pytest.approx(columns["Hdd0"][k] * Hartree, rel=1e-6)
    assert h[i["dzx"], i["dzx"]] == pytest.approx(columns["Hdd1"][k] * Hartree, rel=1e-6)
    assert h[i["dxy"], i["dxy"]] == pytest.approx(columns["Hdd2"][k] * Hartree, rel=1e-6)
    assert h[i["s"], i["dz2"]] == pytest.approx(columns["Hsd0"][k] * Hartree, rel=1e-6)
    assert h[i["pz"], i["dz2"]] == pytest.approx(columns["Hpd0"][k] * Hartree, rel=1e-6)
    assert h[i["px"], i["dzx"]] == pytest.approx(columns["Hpd1"][k] * Hartree, rel=1e-6)


def _smooth_spd_model():
    """A made-up spd model with smooth laws, overlap and a pair repulsion (element Ti)."""
    from tbkit.params import Exponential, Tail
    from tbkit.repulsive import PairRepulsive

    def law(v0, beta=2.0):
        return Tail(Exponential(v0, 2.5, beta, 4.0), 3.6, 4.0)

    values = {"sss": -1.2, "sps": 1.5, "pps": 2.0, "ppp": -0.6, "sds": -0.8, "pds": -1.0,
              "pdp": 0.5, "dds": -0.9, "ddp": 0.5, "ddd": -0.1}
    hopping = {("Ti", "Ti", b): law(v) for b, v in values.items()}
    overlap = {("Ti", "Ti", b): law(0.04 * np.sign(-v) * min(abs(v), 1.0), 2.5)
               for b, v in values.items()}
    return TBModel("prueba spd suave", orbitals={"Ti": SPD},
                   onsite={"Ti": {"s": -3.0, "p": 1.0, "d": -4.0}}, hopping=hopping,
                   overlap=overlap, valence={"Ti": 4.0}, hubbard_u={"Ti": 4.0},
                   repulsive=PairRepulsive({("Ti", "Ti"): law(3.0, 4.0)}))


def test_forces_with_d_orbitals_match_finite_differences():
    from ase import Atoms

    from tbkit.forces import energy_and_forces
    from tbkit.hamiltonian import System
    from tbkit.solver import solve

    model = _smooth_spd_model()
    atoms = Atoms("Ti4", positions=[[0, 0, 0], [2.4, 0.2, 0], [1.1, 2.1, 0.3],
                                    [1.0, 0.8, 2.0]])

    def energy(a, need=False):
        return energy_and_forces(solve(System.build(a, model), kT=0.1), need_forces=need)

    _, forces, _ = energy(atoms, True)
    h = 1e-4
    for atom, axis in ((0, 0), (1, 1), (2, 2), (3, 0)):
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[atom, axis] += h
        minus.positions[atom, axis] -= h
        fd = -(energy(plus)[0] - energy(minus)[0]) / (2 * h)
        assert forces[atom, axis] == pytest.approx(fd, abs=2e-6)


def test_spin_polarised_d_shell_hund_closed_form():
    """Four electrons in a free atom's degenerate d shell (s and p far above): they all
    align, M = 4, and gain exactly ½ W_dd M² = 8 W_dd."""
    from ase import Atoms

    from tbkit.hamiltonian import System
    from tbkit.spin import spin_polarised

    model = _smooth_spd_model()
    model.onsite["Ti"] = {"s": 5.0, "p": 9.0, "d": -4.0}
    w = {"Ti": {"ss": -0.4, "sp": -0.3, "pp": -0.5, "sd": -0.1, "pd": -0.03, "dd": -0.6}}
    system = System.build(Atoms("Ti"), model)
    polarised = spin_polarised(system, W=w, charges=False, guess="ferro", kT=0.001)
    flat = spin_polarised(system, W=w, charges=False, guess="none", kT=0.001)
    assert polarised.magnetization == pytest.approx(4.0, abs=1e-6)
    assert polarised.energy - flat.energy == pytest.approx(8 * w["Ti"]["dd"], abs=1e-6)


def test_d_family_from_a_built_family_gets_the_d_integrals():
    """xu_bsp --d-orbitals derives the pair integrals again: a family already built without
    d keeps its list in the pair specs (a fit once ran with no d coupling at all)."""
    import dataclasses

    from tbkit.recipes.xu_bsp import FAMILIES

    base = FAMILIES["Se"]
    pairs = {k: {kk: vv for kk, vv in s.items() if kk != "bonds"} for k, s in base.pairs.items()}
    with_d = dataclasses.replace(base, pairs=pairs, d_elements=("Se",))
    names = with_d.parameter_names()
    assert "onsite Se d" in names
    assert any("sds(H,Se)" in n for n in names) and any("dds" in n for n in names)
    assert len(names) > len(base.parameter_names()) + 10
