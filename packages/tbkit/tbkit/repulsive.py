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
    if kind == "acute_angle":
        return AcuteAngleTerm(tuple(float(c) for c in data["coefficients"]),
                              float(data.get("theta0_degrees", 80.0)), float(data.get("r1", 1.7)),
                              float(data.get("rm", 2.0)), tuple(data.get("elements", ("C", "N", "O"))))
    if kind == "centred_angle":
        return CentredAngleTerm(tuple(float(c) for c in data["coefficients"]), data["centre"],
                                {k: (float(v[0]), float(v[1])) for k, v in data["bonds"].items()},
                                float(data.get("cos0", -1 / 3)))
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


@dataclass
class AcuteAngleTerm:
    """A correction for bond angles far below any in Xu's training: three-membered rings.

    ``E = Σ_{j-i-k} s(θ_jik) f(r_ij) f(r_ik)`` over triplets of heavy atoms
    (``elements``) with both bonds shorter than ``rm``, where

    * ``s(θ) = Σ_n c_n (cos θ - cos θ0)^(n+2)`` for θ < θ0, 0 otherwise (value
      and slope vanish at θ0);
    * ``f(r) = 1`` below ``r1`` and a cubic to zero (value and slope) at ``rm``.

    With θ0 = 80°, no angle of graphene, diamond, nanotubes, fullerenes or
    aromatic rings (all ≥ 100°) is touched: those results are exactly those
    of the uncorrected model. It exists because an orthogonal minimal-basis
    model cannot describe the bent bonds of cyclopropane-like rings (Xu's
    C-C gives cyclopropane a 1.72 Å bond, measured 1.51 Å, and opens the
    epoxide of graphene oxide into an ether). Its coefficients are fitted,
    and declared as such, never assumed.
    """

    coefficients: tuple[float, ...]
    theta0_degrees: float = 80.0
    r1: float = 1.7
    rm: float = 2.0
    elements: tuple[str, ...] = ("C", "N", "O")

    def cutoff(self) -> float:
        return self.rm

    def _f(self, r):
        x = np.clip((r - self.r1) / (self.rm - self.r1), 0.0, 1.0)
        value = 1 - 3 * x ** 2 + 2 * x ** 3
        slope = (-6 * x + 6 * x ** 2) / (self.rm - self.r1)
        return value, slope

    def basis(self, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
        """Energy (n,) and forces (n, N, 3) of each power with unit coefficient."""
        n_terms = len(self.coefficients)
        energies = np.zeros(n_terms)
        forces = np.zeros((n_terms, len(atoms), 3))
        symbols = np.array(atoms.get_chemical_symbols())
        heavy = np.isin(symbols, self.elements)
        ii, jj, dd, vv = neighbor_list("ijdD", atoms, self.rm)
        keep = heavy[ii] & heavy[jj]
        ii, jj, dd, vv = ii[keep], jj[keep], dd[keep], vv[keep]
        c0 = np.cos(np.radians(self.theta0_degrees))
        for centre in np.unique(ii):
            mine = np.flatnonzero(ii == centre)
            if len(mine) < 2:
                continue
            for a_index, a in enumerate(mine):
                for b in mine[a_index + 1:]:
                    ra, rb = vv[a], vv[b]
                    da, db = dd[a], dd[b]
                    cos = float(ra @ rb / (da * db))
                    if cos <= c0:
                        continue
                    fa, dfa = self._f(da)
                    fb, dfb = self._f(db)
                    # ∂cos/∂r_a and ∂cos/∂r_b (bond vectors from the centre)
                    dcos_a = rb / (da * db) - cos * ra / da ** 2
                    dcos_b = ra / (da * db) - cos * rb / db ** 2
                    u = cos - c0
                    for n in range(n_terms):
                        p = n + 2
                        s, ds = u ** p, p * u ** (p - 1)
                        energies[n] += s * fa * fb
                        grad_a = ds * dcos_a * fa * fb + s * dfa * fb * ra / da
                        grad_b = ds * dcos_b * fa * fb + s * fa * dfb * rb / db
                        # E depends on r_a = x_j - x_i and r_b = x_k - x_i
                        forces[n, jj[a]] -= grad_a
                        forces[n, jj[b]] -= grad_b
                        forces[n, centre] += grad_a + grad_b
        return energies, forces

    def energy_and_forces(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        energies, forces = self.basis(atoms)
        c = np.asarray(self.coefficients, dtype=float)
        return float(c @ energies), np.einsum("n,nax->ax", c, forces)

    def to_dict(self) -> dict:
        return {"type": "acute_angle", "coefficients": list(self.coefficients),
                "theta0_degrees": self.theta0_degrees, "r1": self.r1, "rm": self.rm,
                "elements": list(self.elements)}


@dataclass
class CentredAngleTerm:
    """Bond-angle stiffness at the atoms of one element (``centre``).

    ``E = Σ_{j-X-k} s(θ_jXk) f_j(r_Xj) f_k(r_Xk)`` over pairs of bonds of every
    ``centre`` atom X, with ``s(θ) = Σ_n c_n (cos θ - cos0)^n`` (n = 1..N; cos0
    tetrahedral by default) and ``f`` 1 below ``bonds[element][0]`` and a cubic
    to zero (value and slope) at ``bonds[element][1]``: only bonded neighbours
    count, never second neighbours.

    It exists because a minimal sp basis has no d orbitals: at hypervalent
    centres (Se(IV), P(V)) it leaves the angles too soft, and seleninic and
    phosphonic acids closed O-X-O by 15° and turned the hydroxyl onto the other
    O, a geometry GPAW puts 0.3 eV higher (the problem DFTB's 3ob set has with
    hypervalent P and S). Linear in its coefficients, which are fitted with the
    pair repulsion; zero in any structure without the centre element, so the
    set it is added to is unchanged there. The polynomial is only determined
    over the angles of the training data.
    """

    coefficients: tuple[float, ...]
    centre: str
    bonds: dict                               # neighbour element -> (r1, rm), Å
    cos0: float = -1 / 3

    def cutoff(self) -> float:
        return max(rm for _, rm in self.bonds.values())

    @staticmethod
    def _f(r, r1, rm):
        x = min(max((r - r1) / (rm - r1), 0.0), 1.0)
        return 1 - 3 * x ** 2 + 2 * x ** 3, (-6 * x + 6 * x ** 2) / (rm - r1)

    def basis(self, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
        """Energy (n,) and forces (n, N, 3) of each power with unit coefficient."""
        n_terms = len(self.coefficients)
        energies = np.zeros(n_terms)
        forces = np.zeros((n_terms, len(atoms), 3))
        symbols = atoms.get_chemical_symbols()
        if self.centre not in symbols:
            return energies, forces
        ii, jj, dd, vv = neighbor_list("ijdD", atoms, self.cutoff())
        for centre in {i for i, s in enumerate(symbols) if s == self.centre}:
            mine = []
            for a in np.flatnonzero(ii == centre):
                window = self.bonds.get(symbols[jj[a]])
                if window is not None and dd[a] < window[1]:
                    mine.append((a, *self._f(dd[a], *window)))
            for index, (a, fa, dfa) in enumerate(mine):
                for b, fb, dfb in mine[index + 1:]:
                    ra, rb, da, db = vv[a], vv[b], dd[a], dd[b]
                    cos = float(ra @ rb / (da * db))
                    dcos_a = rb / (da * db) - cos * ra / da ** 2
                    dcos_b = ra / (da * db) - cos * rb / db ** 2
                    u = cos - self.cos0
                    for n in range(n_terms):
                        p = n + 1
                        s, ds = u ** p, p * u ** (p - 1)
                        energies[n] += s * fa * fb
                        grad_a = ds * dcos_a * fa * fb + s * dfa * fb * ra / da
                        grad_b = ds * dcos_b * fa * fb + s * fa * dfb * rb / db
                        forces[n, jj[a]] -= grad_a
                        forces[n, jj[b]] -= grad_b
                        forces[n, centre] += grad_a + grad_b
        return energies, forces

    def energy_and_forces(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        energies, forces = self.basis(atoms)
        c = np.asarray(self.coefficients, dtype=float)
        return float(c @ energies), np.einsum("n,nax->ax", c, forces)

    def to_dict(self) -> dict:
        return {"type": "centred_angle", "coefficients": list(self.coefficients),
                "centre": self.centre, "bonds": {k: list(v) for k, v in self.bonds.items()},
                "cos0": self.cos0}
