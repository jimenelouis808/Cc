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
