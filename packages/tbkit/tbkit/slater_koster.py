"""Two-centre Slater-Koster blocks for s, p and d orbitals.

For a bond from atom A to atom B with direction cosines (l, m, n), the matrix elements
between the real orbitals are those of Slater & Koster, Phys. Rev. 94, 1498 (1954),
Table I, written here for orbital pairs with the angular momentum of A not above that of
B (s-s, s-p, s-d, p-p, p-d, d-d). A pair in the other order is the parity image with the
elements exchanged:

    <a_A | b_B>(l, m, n) = (-1)^(l_a + l_b) <b_B | a_A>(l, m, n)   (laws with B first),

so ``<p_A|s_B> = -c V_spσ(s on B)``, ``<d_A|p_B> = -<p|d>(p on B)``, ``<d_A|s_B> = <s|d>``.

Integrals (keys of the model's hopping and overlap tables): ``sss``, ``sps``, ``pps``,
``ppp``, ``sds``, ``pds``, ``pdp``, ``dds``, ``ddp``, ``ddd``. The mixed ones (``sps``,
``sds``, ``pds``, ``pdp``) are ordered: the lower angular momentum on the first element.
Real d orbitals: ``dxy``, ``dyz``, ``dzx``, ``dx2-y2``, ``dz2`` (3z² - r²).

The same formulas give overlap blocks from overlap integrals. The local π orbital of a π
model (``"pi"``) couples only to another ``"pi"``, through ``V_ppπ(d)`` with no angular
factor. Rotational covariance of every block (s, p and d) is tested.
"""

from __future__ import annotations

import numpy as np

from .params import TBModel, angular

_AXIS = {"px": 0, "py": 1, "pz": 2}
_LORDER = {"s": 0, "p": 1, "d": 2}
_R3 = np.sqrt(3.0)


def _sd(ob, x, y, z, sds):
    if ob == "dxy":
        return _R3 * x * y * sds
    if ob == "dyz":
        return _R3 * y * z * sds
    if ob == "dzx":
        return _R3 * z * x * sds
    if ob == "dx2-y2":
        return 0.5 * _R3 * (x * x - y * y) * sds
    return (z * z - 0.5 * (x * x + y * y)) * sds                     # dz2


def _pd(oa, ob, x, y, z, pds, pdp):
    c = {"px": x, "py": y, "pz": z}
    if ob in ("dxy", "dyz", "dzx"):
        # d_{ab}: x-y, y-z or z-x; the p axis is one of the two, or the third
        first, second = {"dxy": ("px", "py"), "dyz": ("py", "pz"), "dzx": ("pz", "px")}[ob]
        if oa == first:
            ca, cb = c[first], c[second]
            return _R3 * ca * ca * cb * pds + cb * (1 - 2 * ca * ca) * pdp
        if oa == second:
            ca, cb = c[second], c[first]
            return _R3 * ca * ca * cb * pds + cb * (1 - 2 * ca * ca) * pdp
        return (_R3 * pds - 2 * pdp) * x * y * z
    if ob == "dx2-y2":
        lm = x * x - y * y
        if oa == "px":
            return 0.5 * _R3 * x * lm * pds + x * (1 - lm) * pdp
        if oa == "py":
            return 0.5 * _R3 * y * lm * pds - y * (1 + lm) * pdp
        return 0.5 * _R3 * z * lm * pds - z * lm * pdp
    q = z * z - 0.5 * (x * x + y * y)                                 # dz2
    if oa == "px":
        return x * q * pds - _R3 * x * z * z * pdp
    if oa == "py":
        return y * q * pds - _R3 * y * z * z * pdp
    return z * q * pds + _R3 * z * (x * x + y * y) * pdp


_DD = ("dxy", "dyz", "dzx", "dx2-y2", "dz2")


def _dd(oa, ob, x, y, z, s, p, d):
    i, j = _DD.index(oa), _DD.index(ob)
    if i > j:                                    # d-d is symmetric in the pair
        oa, ob = ob, oa
    x2, y2, z2 = x * x, y * y, z * z
    key = (oa, ob)
    if key == ("dxy", "dxy"):
        return 3 * x2 * y2 * s + (x2 + y2 - 4 * x2 * y2) * p + (z2 + x2 * y2) * d
    if key == ("dyz", "dyz"):
        return 3 * y2 * z2 * s + (y2 + z2 - 4 * y2 * z2) * p + (x2 + y2 * z2) * d
    if key == ("dzx", "dzx"):
        return 3 * z2 * x2 * s + (z2 + x2 - 4 * z2 * x2) * p + (y2 + z2 * x2) * d
    if key == ("dxy", "dyz"):
        return 3 * x * y2 * z * s + x * z * (1 - 4 * y2) * p + x * z * (y2 - 1) * d
    if key == ("dxy", "dzx"):
        return 3 * x2 * y * z * s + y * z * (1 - 4 * x2) * p + y * z * (x2 - 1) * d
    if key == ("dyz", "dzx"):
        return 3 * x * y * z2 * s + x * y * (1 - 4 * z2) * p + x * y * (z2 - 1) * d
    lm = x2 - y2
    q = z2 - 0.5 * (x2 + y2)
    if key == ("dxy", "dx2-y2"):
        return 1.5 * x * y * lm * s - 2 * x * y * lm * p + 0.5 * x * y * lm * d
    if key == ("dyz", "dx2-y2"):
        return 1.5 * y * z * lm * s - y * z * (1 + 2 * lm) * p + y * z * (1 + 0.5 * lm) * d
    if key == ("dzx", "dx2-y2"):
        return 1.5 * z * x * lm * s + z * x * (1 - 2 * lm) * p - z * x * (1 - 0.5 * lm) * d
    if key == ("dxy", "dz2"):
        return (_R3 * x * y * q * s - 2 * _R3 * x * y * z2 * p
                + 0.5 * _R3 * x * y * (1 + z2) * d)
    if key == ("dyz", "dz2"):
        return (_R3 * y * z * q * s + _R3 * y * z * (x2 + y2 - z2) * p
                - 0.5 * _R3 * y * z * (x2 + y2) * d)
    if key == ("dzx", "dz2"):
        return (_R3 * x * z * q * s + _R3 * x * z * (x2 + y2 - z2) * p
                - 0.5 * _R3 * x * z * (x2 + y2) * d)
    if key == ("dx2-y2", "dx2-y2"):
        return 0.75 * lm * lm * s + (x2 + y2 - lm * lm) * p + (z2 + 0.25 * lm * lm) * d
    if key == ("dx2-y2", "dz2"):
        return (0.5 * _R3 * lm * q * s - _R3 * z2 * lm * p
                + 0.25 * _R3 * (1 + z2) * lm * d)
    return q * q * s + 3 * z2 * (x2 + y2) * p + 0.75 * (x2 + y2) ** 2 * d   # dz2, dz2


def _element(oa, ob, la, lb, c, v, el_a, el_b):
    """One matrix element (scalar or array over bonds) for ``la <= lb``."""
    x, y, z = c[..., 0], c[..., 1], c[..., 2]
    if la == "s" and lb == "s":
        return v("sss", el_a, el_b)
    if la == "s" and lb == "p":
        return c[..., _AXIS[ob]] * v("sps", el_a, el_b)
    if la == "s" and lb == "d":
        return _sd(ob, x, y, z, v("sds", el_a, el_b))
    if la == "p" and lb == "p":
        ca, cb = c[..., _AXIS[oa]], c[..., _AXIS[ob]]
        delta = 1.0 if oa == ob else 0.0
        return ca * cb * v("pps", el_a, el_b) + (delta - ca * cb) * v("ppp", el_a, el_b)
    if la == "p" and lb == "d":
        return _pd(oa, ob, x, y, z, v("pds", el_a, el_b), v("pdp", el_a, el_b))
    return _dd(oa, ob, x, y, z, v("dds", el_a, el_b), v("ddp", el_a, el_b),
               v("ddd", el_a, el_b))


def _fill(out, orbs_a, orbs_b, el_a, el_b, c, v):
    for i, oa in enumerate(orbs_a):
        la = angular(oa)
        for j, ob in enumerate(orbs_b):
            lb = angular(ob)
            if oa == "pi" or ob == "pi":
                if oa != ob:
                    raise ValueError("Un orbital 'pi' solo se acopla con otro 'pi' (modelo π).")
                out[..., i, j] = v("ppp", el_a, el_b)  # local π orbitals: no angular factor
            elif _LORDER[la] <= _LORDER[lb]:
                out[..., i, j] = _element(oa, ob, la, lb, c, v, el_a, el_b)
            else:                                       # parity image, elements exchanged
                sign = -1.0 if (_LORDER[la] + _LORDER[lb]) % 2 else 1.0
                out[..., i, j] = sign * _element(ob, oa, lb, la, c, v, el_b, el_a)
    return out


def block(model: TBModel, table: dict, el_a: str, orbs_a: tuple[str, ...], el_b: str,
          orbs_b: tuple[str, ...], vector: np.ndarray) -> np.ndarray:
    """The ``len(orbs_a) x len(orbs_b)`` block for the bond ``A -> B`` (vector B - A, Å)."""
    vector = np.asarray(vector, dtype=float)
    d = float(np.linalg.norm(vector))
    c = vector / d
    values: dict[tuple, float] = {}

    def v(bond: str, first: str, second: str) -> float:
        key = (bond, first, second)
        if key not in values:
            law = model.law(table, first, second, bond)
            values[key] = float(law(d)) if law is not None else 0.0
        return values[key]

    return _fill(np.zeros((len(orbs_a), len(orbs_b))), orbs_a, orbs_b, el_a, el_b, c, v)


def blocks(model: TBModel, table: dict, el_a: str, orbs_a: tuple[str, ...], el_b: str,
           orbs_b: tuple[str, ...], vectors: np.ndarray) -> np.ndarray:
    """:func:`block` for many bonds of one element pair at once: ``(n_bonds, n_a, n_b)``.

    The same formulas with the direction cosines and the distance laws as arrays; the band
    forces call it instead of looping :func:`block` bond by bond (tested to agree).
    """
    vectors = np.asarray(vectors, dtype=float)
    d = np.linalg.norm(vectors, axis=1)
    c = vectors / d[:, None]
    values: dict[tuple, np.ndarray] = {}

    def v(bond: str, first: str, second: str) -> np.ndarray:
        key = (bond, first, second)
        if key not in values:
            law = model.law(table, first, second, bond)
            values[key] = np.asarray(law(d), dtype=float) if law is not None else np.zeros_like(d)
        return values[key]

    return _fill(np.zeros((len(d), len(orbs_a), len(orbs_b))), orbs_a, orbs_b, el_a, el_b,
                 c, v)
