"""Two-centre Slater-Koster blocks for s and p orbitals.

For a bond from atom A to atom B with direction cosines (l, m, n), the
matrix elements between the real orbitals are (Slater & Koster, Phys. Rev.
94, 1498 (1954), Table I):

* <s_A|s_B>   = V_ssσ
* <s_A|p_B,α> = c_α V_spσ(s on A, p on B)
* <p_A,α|s_B> = -c_α V_spσ(s on B, p on A)
* <p_A,α|p_B,β> = c_α c_β V_ppσ + (δ_αβ - c_α c_β) V_ppπ

The same formulas give overlap blocks from overlap integrals. The local π
orbital of a π model (``"pi"``) couples only to another ``"pi"``, through
``V_ppπ(d)`` with no angular factor.
"""

from __future__ import annotations

import numpy as np

from .params import TBModel, angular

_AXIS = {"px": 0, "py": 1, "pz": 2}


def block(model: TBModel, table: dict, el_a: str, orbs_a: tuple[str, ...], el_b: str,
          orbs_b: tuple[str, ...], vector: np.ndarray) -> np.ndarray:
    """The ``len(orbs_a) x len(orbs_b)`` block for the bond ``A -> B`` (vector B - A, Å)."""
    d = float(np.linalg.norm(vector))
    c = vector / d
    values: dict[str, float] = {}

    def v(bond: str, first: str, second: str) -> float:
        key = (bond, first, second)
        if key not in values:
            law = model.law(table, first, second, bond)
            values[key] = float(law(d)) if law is not None else 0.0
        return values[key]

    out = np.zeros((len(orbs_a), len(orbs_b)))
    for i, oa in enumerate(orbs_a):
        la = angular(oa)
        for j, ob in enumerate(orbs_b):
            lb = angular(ob)
            if oa == "pi" or ob == "pi":
                if oa != ob:
                    raise ValueError("Un orbital 'pi' solo se acopla con otro 'pi' (modelo π).")
                out[i, j] = v("ppp", el_a, el_b)      # local π orbitals: no angular factor
            elif la == "s" and lb == "s":
                out[i, j] = v("sss", el_a, el_b)
            elif la == "s" and lb == "p":
                out[i, j] = c[_AXIS[ob]] * v("sps", el_a, el_b)
            elif la == "p" and lb == "s":
                out[i, j] = -c[_AXIS[oa]] * v("sps", el_b, el_a)
            else:
                ca, cb = c[_AXIS[oa]], c[_AXIS[ob]]
                delta = 1.0 if oa == ob else 0.0
                out[i, j] = ca * cb * v("pps", el_a, el_b) + (delta - ca * cb) * v(
                    "ppp", el_a, el_b)
    return out
