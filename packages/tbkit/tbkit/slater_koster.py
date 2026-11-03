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


def _sd(ob, l, m, n, sds):
    if ob == "dxy":
        return _R3 * l * m * sds
    if ob == "dyz":
        return _R3 * m * n * sds
    if ob == "dzx":
        return _R3 * n * l * sds
    if ob == "dx2-y2":
        return 0.5 * _R3 * (l * l - m * m) * sds
    return (n * n - 0.5 * (l * l + m * m)) * sds                     # dz2


def _pd(oa, ob, l, m, n, pds, pdp):
    c = {"px": l, "py": m, "pz": n}
    if ob in ("dxy", "dyz", "dzx"):
        # d_{ab}: x-y, y-z or z-x; the p axis is one of the two, or the third
        first, second = {"dxy": ("px", "py"), "dyz": ("py", "pz"), "dzx": ("pz", "px")}[ob]
        if oa == first:
            ca, cb = c[first], c[second]
            return _R3 * ca * ca * cb * pds + cb * (1 - 2 * ca * ca) * pdp
        if oa == second:
            ca, cb = c[second], c[first]
            return _R3 * ca * ca * cb * pds + cb * (1 - 2 * ca * ca) * pdp
        return (_R3 * pds - 2 * pdp) * l * m * n
    if ob == "dx2-y2":
        lm = l * l - m * m
        if oa == "px":
            return 0.5 * _R3 * l * lm * pds + l * (1 - lm) * pdp
        if oa == "py":
            return 0.5 * _R3 * m * lm * pds - m * (1 + lm) * pdp
        return 0.5 * _R3 * n * lm * pds - n * lm * pdp
    q = n * n - 0.5 * (l * l + m * m)                                 # dz2
    if oa == "px":
        return l * q * pds - _R3 * l * n * n * pdp
    if oa == "py":
        return m * q * pds - _R3 * m * n * n * pdp
    return n * q * pds + _R3 * n * (l * l + m * m) * pdp


_DD = ("dxy", "dyz", "dzx", "dx2-y2", "dz2")


def _dd(oa, ob, l, m, n, s, p, d):
    i, j = _DD.index(oa), _DD.index(ob)
    if i > j:                                    # d-d is symmetric in the pair
        oa, ob = ob, oa
    l2, m2, n2 = l * l, m * m, n * n
    key = (oa, ob)
    if key == ("dxy", "dxy"):
        return 3 * l2 * m2 * s + (l2 + m2 - 4 * l2 * m2) * p + (n2 + l2 * m2) * d
    if key == ("dyz", "dyz"):
        return 3 * m2 * n2 * s + (m2 + n2 - 4 * m2 * n2) * p + (l2 + m2 * n2) * d
    if key == ("dzx", "dzx"):
        return 3 * n2 * l2 * s + (n2 + l2 - 4 * n2 * l2) * p + (m2 + n2 * l2) * d
    if key == ("dxy", "dyz"):
        return 3 * l * m2 * n * s + l * n * (1 - 4 * m2) * p + l * n * (m2 - 1) * d
    if key == ("dxy", "dzx"):
        return 3 * l2 * m * n * s + m * n * (1 - 4 * l2) * p + m * n * (l2 - 1) * d
    if key == ("dyz", "dzx"):
        return 3 * l * m * n2 * s + l * m * (1 - 4 * n2) * p + l * m * (n2 - 1) * d
    lm = l2 - m2
    q = n2 - 0.5 * (l2 + m2)
    if key == ("dxy", "dx2-y2"):
        return 1.5 * l * m * lm * s - 2 * l * m * lm * p + 0.5 * l * m * lm * d
    if key == ("dyz", "dx2-y2"):
        return 1.5 * m * n * lm * s - m * n * (1 + 2 * lm) * p + m * n * (1 + 0.5 * lm) * d
    if key == ("dzx", "dx2-y2"):
        return 1.5 * n * l * lm * s + n * l * (1 - 2 * lm) * p - n * l * (1 - 0.5 * lm) * d
    if key == ("dxy", "dz2"):
        return (_R3 * l * m * q * s - 2 * _R3 * l * m * n2 * p
                + 0.5 * _R3 * l * m * (1 + n2) * d)
    if key == ("dyz", "dz2"):
        return (_R3 * m * n * q * s + _R3 * m * n * (l2 + m2 - n2) * p
                - 0.5 * _R3 * m * n * (l2 + m2) * d)
    if key == ("dzx", "dz2"):
        return (_R3 * l * n * q * s + _R3 * l * n * (l2 + m2 - n2) * p
                - 0.5 * _R3 * l * n * (l2 + m2) * d)
    if key == ("dx2-y2", "dx2-y2"):
        return 0.75 * lm * lm * s + (l2 + m2 - lm * lm) * p + (n2 + 0.25 * lm * lm) * d
    if key == ("dx2-y2", "dz2"):
        return (0.5 * _R3 * lm * q * s - _R3 * n2 * lm * p
                + 0.25 * _R3 * (1 + n2) * lm * d)
    return q * q * s + 3 * n2 * (l2 + m2) * p + 0.75 * (l2 + m2) ** 2 * d   # dz2, dz2


def _element(oa, ob, la, lb, c, v, el_a, el_b):
    """One matrix element (scalar or array over bonds) for ``la <= lb``."""
    l, m, n = c[..., 0], c[..., 1], c[..., 2]
    if la == "s" and lb == "s":
        return v("sss", el_a, el_b)
    if la == "s" and lb == "p":
        return c[..., _AXIS[ob]] * v("sps", el_a, el_b)
    if la == "s" and lb == "d":
        return _sd(ob, l, m, n, v("sds", el_a, el_b))
    if la == "p" and lb == "p":
        ca, cb = c[..., _AXIS[oa]], c[..., _AXIS[ob]]
        delta = 1.0 if oa == ob else 0.0
        return ca * cb * v("pps", el_a, el_b) + (delta - ca * cb) * v("ppp", el_a, el_b)
    if la == "p" and lb == "d":
        return _pd(oa, ob, l, m, n, v("pds", el_a, el_b), v("pdp", el_a, el_b))
    return _dd(oa, ob, l, m, n, v("dds", el_a, el_b), v("ddp", el_a, el_b),
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
