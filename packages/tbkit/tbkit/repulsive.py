"""Short-range repulsion: what turns a tight-binding model into total energies.

The band-structure energy alone binds atoms without limit; a repulsive term,
fitted together with the hopping, balances it. Two forms are provided:

``PairRepulsive``
    ``E_rep = ½ Σ_{i≠j} V_{ab}(r_ij)``, one law per element pair. The form of
    DFTB (the ``Spline`` block of a ``.skf`` file, :class:`SkfSpline`) and of
    most fitted models.
``EmbeddedRepulsive``
    ``E_rep = Σ_i f(Σ_j φ(r_ij))`` with a polynomial ``f``: the form of Xu et
    al.'s carbon model, which lets the repulsion per bond depend on
    coordination (diamond, graphite and chains with one set of numbers).

Both return energies (eV) and forces (eV/Å), with ``F = -∂E/∂r`` from the
chain rule and the laws' derivatives. Periodic images are included through
ASE's neighbour list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.units import Bohr, Hartree

from .params import derivative, law_from_dict, law_to_dict


@dataclass
class PairRepulsive:
    """``E = ½ Σ_{i≠j} V_ab(r_ij)``; ``laws[(a, b)]`` in either element order."""

    laws: dict

    def _law(self, a: str, b: str):
        return self.laws.get((a, b)) or self.laws.get((b, a))

    def cutoff(self) -> float:
        return max(law.cutoff for law in self.laws.values())

    def energy_and_forces(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        symbols = atoms.get_chemical_symbols()
        forces = np.zeros((len(atoms), 3))
        energy = 0.0
        ii, jj, dd, vv = neighbor_list("ijdD", atoms, self.cutoff())
        for i, j, d, vec in zip(ii, jj, dd, vv, strict=True):
            law = self._law(symbols[i], symbols[j])
            if law is None:
                continue
            energy += 0.5 * float(law(d))
            push = 0.5 * float(derivative(law, d)) * vec / d
            forces[i] += push
            forces[j] -= push
        return energy, forces

    def to_dict(self) -> dict:
        return {"type": "pair",
                "pairs": [{"pair": [a, b], "law": law_to_dict(law)}
                          for (a, b), law in self.laws.items()]}


@dataclass
class EmbeddedRepulsive:
    """``E = Σ_i f(x_i)``, ``x_i = Σ_j φ(r_ij)``, ``f(x) = Σ_n c_n x^n`` (Xu et al.).

    Alone, it refuses atoms outside ``elements``. Inside a
    :class:`SumRepulsive` (``others="ignore"``) it acts on those elements
    only: other atoms neither embed nor count as neighbours, and their
    repulsion comes from the other terms.
    """

    phi: object
    polynomial: tuple[float, ...]
    elements: tuple[str, ...] = ("C",)
    others: str = "error"

    def cutoff(self) -> float:
        return self.phi.cutoff

    def _f(self, x):
        return sum(c * x ** n for n, c in enumerate(self.polynomial))

    def _df(self, x):
        return sum(n * c * x ** (n - 1) for n, c in enumerate(self.polynomial) if n)

    def energy_and_forces(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        symbols = atoms.get_chemical_symbols()
        unknown = sorted(set(symbols) - set(self.elements))
        if unknown and self.others != "ignore":
            raise ValueError(f"La repulsión embebida solo describe {self.elements}; "
                             f"no {', '.join(unknown)}.")
        ii, jj, dd, vv = neighbor_list("ijdD", atoms, self.cutoff())
        if unknown:
            inside = np.isin(symbols, self.elements)
            keep = inside[ii] & inside[jj]
            ii, jj, dd, vv = ii[keep], jj[keep], dd[keep], vv[keep]
        x = np.zeros(len(atoms))
        np.add.at(x, ii, self.phi(dd))
        energy = float(np.sum(self._f(x)))
        forces = np.zeros((len(atoms), 3))
        weight = self._df(x)[ii] * derivative(self.phi, dd)
        push = (weight / dd)[:, None] * vv
        np.add.at(forces, ii, push)
        np.add.at(forces, jj, -push)
        if unknown:
            # f(0) is a per-atom constant: it belongs to the embedded elements only.
            energy -= float(np.sum(~np.isin(symbols, self.elements))) * self._f(0.0)
        return energy, forces

    def to_dict(self) -> dict:
        return {"type": "embedded", "elements": list(self.elements),
                "phi": law_to_dict(self.phi), "polynomial": list(self.polynomial)}


@dataclass
class SumRepulsive:
    """The sum of several repulsive terms (e.g. Xu's embedded C-C plus C-H pairs)."""

    terms: tuple

    def cutoff(self) -> float:
        return max(term.cutoff() for term in self.terms)

    def energy_and_forces(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        energy, forces = 0.0, np.zeros((len(atoms), 3))
        for term in self.terms:
            e, f = term.energy_and_forces(atoms)
            energy += e
            forces += f
        return energy, forces

    def to_dict(self) -> dict:
        return {"type": "sum", "terms": [term.to_dict() for term in self.terms]}


@dataclass(frozen=True)
class SkfSpline:
    """The repulsive ``Spline`` block of a ``.skf`` file, as a distance law (eV, Å).

    Below the first knot ``exp(-a1 r + a2) + a3``; then cubic pieces
    ``c0 + c1 x + c2 x² + c3 x³`` (the last one quintic), ``x = r - r0``;
    zero beyond the cutoff. Stored in the file's atomic units.
    """

    a: tuple[float, float, float]
    knots: tuple[tuple[float, ...], ...]      # (r0, r1, c0..c3[, c4, c5]) in Bohr/Hartree
    cutoff_bohr: float

    @property
    def cutoff(self) -> float:
        return self.cutoff_bohr * Bohr

    def __call__(self, d):
        r = np.atleast_1d(np.asarray(d, dtype=float)) / Bohr
        out = np.zeros_like(r)
        first = self.knots[0][0]
        a1, a2, a3 = self.a
        low = r < first
        out[low] = np.exp(-a1 * r[low] + a2) + a3
        for knot in self.knots:
            r0, r1, coefficients = knot[0], knot[1], knot[2:]
            inside = (r >= r0) & (r < r1)
            x = r[inside] - r0
            out[inside] = sum(c * x ** n for n, c in enumerate(coefficients))
        out = out * Hartree
        return out if np.ndim(d) else float(out[0])

    def to_dict(self) -> dict:
        return {"type": "SkfSpline", "a": list(self.a), "knots": [list(k) for k in self.knots],
                "cutoff_bohr": self.cutoff_bohr}

    @classmethod
    def from_dict(cls, data: dict) -> "SkfSpline":
        return cls(tuple(data["a"]), tuple(tuple(k) for k in data["knots"]),
                   float(data["cutoff_bohr"]))


def read_skf_spline(lines: list[str]) -> Optional[SkfSpline]:
    """Parse the ``Spline`` block (lines of a ``.skf`` from 'Spline' on)."""
    start = next((i for i, ln in enumerate(lines) if ln.strip().lower().startswith("spline")),
                 None)
    if start is None:
        return None
    n_int, cutoff = lines[start + 1].split()[:2]
    a = tuple(float(v) for v in lines[start + 2].split()[:3])
    knots = []
    for line in lines[start + 3:start + 3 + int(n_int)]:
        values = [float(v) for v in line.split()]
        knots.append(tuple(values))
    return SkfSpline(a, tuple(knots), float(cutoff))


def repulsive_from_dict(data: dict):
    """A repulsive term from a parameter file's ``repulsive`` entry."""
    from .params import Tail

    kind = data["type"]
    if kind == "sum":
        terms = []
        for entry in data["terms"]:
            term = repulsive_from_dict(entry)
            if isinstance(term, EmbeddedRepulsive):
                term.others = "ignore"
            terms.append(term)
        return SumRepulsive(tuple(terms))
    if kind == "embedded":
        phi = law_from_dict(data["phi"])
        if data.get("tail"):
            phi = Tail(phi, *map(float, data["tail"]))
        return EmbeddedRepulsive(phi, tuple(float(c) for c in data["polynomial"]),
                                 tuple(data.get("elements", ["C"])))
    if kind == "pair":
        laws = {}
        for entry in data["pairs"]:
            law = law_from_dict(entry["law"])
            if entry.get("tail"):
                law = Tail(law, *map(float, entry["tail"]))
            laws[tuple(entry["pair"])] = law
        return PairRepulsive(laws)
    raise ValueError(f"Tipo de repulsión desconocido: {kind!r}.")
