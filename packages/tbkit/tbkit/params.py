"""Tight-binding parameters: orbitals, on-site energies and distance laws.

A :class:`TBModel` says, per element, which valence orbitals it carries
(``"pi"`` for a π model, ``"s", "px", "py", "pz"`` for sp³), their on-site
energies, and, per element pair, the two-centre Slater-Koster integrals as
functions of distance:

====== ==========================================
``sss`` ssσ   (s on the first element, s on the second)
``sps`` spσ   (s on the first element, p on the second)
``pps`` ppσ
``ppp`` ppπ
====== ==========================================

``"pi"`` is the π orbital of a π model: the p orbital normal to the carbon
surface at each atom, whatever its orientation in space. Between two ``pi``
orbitals the hopping is ``ppp(d)`` with no angular factor -- the standard
π-only approximation, which is what makes curved systems (nanotubes,
fullerenes) work. A Cartesian ``pz`` would lose the bonds along z of a tube.

The same keys, in :attr:`TBModel.overlap`, give the overlap integrals of a
non-orthogonal model (``H c = E S c``); an empty table means orthogonal.

Units throughout: eV and Å.

Built-in parameter sets are named after their source, which is also stated
in their docstring: no number here is unexplained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

BONDS = ("sss", "sps", "pps", "ppp")
_L = {"s": "s", "px": "p", "py": "p", "pz": "p", "pi": "p"}


def angular(orbital: str) -> str:
    """``"s"`` or ``"p"`` for an orbital name."""
    return _L[orbital]


class DistanceLaw(Protocol):
    cutoff: float

    def __call__(self, d: np.ndarray | float) -> np.ndarray | float: ...


@dataclass(frozen=True)
class Constant:
    """The same value up to ``cutoff`` (nearest-neighbour models)."""

    value: float
    cutoff: float

    def __call__(self, d):
        return np.where(np.asarray(d) <= self.cutoff, self.value, 0.0)


@dataclass(frozen=True)
class Exponential:
    """``v0 exp(-beta (d/d0 - 1))``: strained graphene (Pereira et al. 2009, beta ≈ 3.37)."""

    v0: float
    d0: float
    beta: float
    cutoff: float

    def __call__(self, d):
        d = np.asarray(d, dtype=float)
        return np.where(d <= self.cutoff, self.v0 * np.exp(-self.beta * (d / self.d0 - 1.0)), 0.0)


@dataclass(frozen=True)
class Harrison:
    """``v0 (d0/d)^n``; Harrison's universal scaling uses n = 2."""

    v0: float
    d0: float
    cutoff: float
    n: float = 2.0

    def __call__(self, d):
        d = np.asarray(d, dtype=float)
        return np.where(d <= self.cutoff, self.v0 * (self.d0 / d) ** self.n, 0.0)


@dataclass(frozen=True)
class GSP:
    """Goodwin-Skinner-Pettifor: ``v0 (r0/d)^n exp{n[-(d/rc)^nc + (r0/rc)^nc]}``.

    Smoothly decaying, so it works beyond first neighbours; the form of the
    Xu-Wang-Chan-Ho carbon model.
    """

    v0: float
    r0: float
    n: float
    nc: float
    rc: float
    cutoff: float

    def __call__(self, d):
        d = np.asarray(d, dtype=float)
        value = self.v0 * (self.r0 / d) ** self.n * np.exp(
            self.n * (-(d / self.rc) ** self.nc + (self.r0 / self.rc) ** self.nc))
        return np.where(d <= self.cutoff, value, 0.0)


@dataclass(frozen=True)
class Table:
    """Tabulated values interpolated with a cubic spline (e.g. from a ``.skf``)."""

    r: tuple[float, ...]
    values: tuple[float, ...]
    cutoff: float

    def __call__(self, d):
        from scipy.interpolate import CubicSpline

        spline = CubicSpline(np.asarray(self.r), np.asarray(self.values), extrapolate=False)
        d = np.asarray(d, dtype=float)
        out = np.nan_to_num(spline(np.clip(d, self.r[0], None)))
        return np.where(d <= self.cutoff, out, 0.0)


ValenceRule = Callable[[str, int], float]


@dataclass
class TBModel:
    """A complete, parametrised tight-binding model.

    Attributes
    ----------
    name
        Label (and source) of the parameter set.
    orbitals
        Per element, its orbitals. Elements absent here are left out of the
        model (a π model ignores H: the σ skeleton and the C-H bonds are
        assumed, not computed).
    onsite
        ``onsite[element][l]`` with ``l`` in ``"s"``, ``"p"``, eV.
    hopping
        ``hopping[(el1, el2, bond)]`` -> distance law, eV.
    overlap
        Same keys, dimensionless; empty for an orthogonal model.
    valence
        Electrons each element brings to the orbitals of the model. A
        callable ``rule(element, coordination)`` may refine it (graphitic
        versus pyridinic N in a π model).
    hubbard_u
        On-site Coulomb U per element, eV (self-consistent charges, Hubbard).
    """

    name: str
    orbitals: dict[str, tuple[str, ...]]
    onsite: dict[str, dict[str, float]]
    hopping: dict[tuple[str, str, str], DistanceLaw]
    overlap: dict[tuple[str, str, str], DistanceLaw] = field(default_factory=dict)
    valence: dict[str, float] = field(default_factory=dict)
    valence_rule: Optional[ValenceRule] = None
    hubbard_u: dict[str, float] = field(default_factory=dict)

    @property
    def orthogonal(self) -> bool:
        return not self.overlap

    def elements(self) -> set[str]:
        return set(self.orbitals)

    def law(self, table: dict, el1: str, el2: str, bond: str) -> Optional[DistanceLaw]:
        """The law for ``bond`` between ``el1`` and ``el2``.

        ``sss``, ``pps`` and ``ppp`` are symmetric in the elements; ``sps`` is
        ordered (s on ``el1``), and its mirror is ``(el2, el1, "sps")``.
        """
        found = table.get((el1, el2, bond))
        if found is None and bond != "sps":
            found = table.get((el2, el1, bond))
        return found

    def cutoff(self) -> float:
        laws = list(self.hopping.values()) + list(self.overlap.values())
        return max((law.cutoff for law in laws), default=0.0)

    def electrons_of(self, element: str, coordination: int) -> float:
        if self.valence_rule is not None:
            return float(self.valence_rule(element, coordination))
        return float(self.valence[element])

    def check(self) -> list[str]:
        """Problems that make the model unusable (missing on-site energies...)."""
        problems = []
        for element, orbitals in self.orbitals.items():
            for orbital in orbitals:
                if angular(orbital) not in self.onsite.get(element, {}):
                    problems.append(f"{element}: falta la energía on-site '{angular(orbital)}'.")
            if element not in self.valence and self.valence_rule is None:
                problems.append(f"{element}: falta su número de electrones de valencia.")
        return problems


# --------------------------------------------------------------------------
# Built-in parameter sets
# --------------------------------------------------------------------------

#: C-C distance of graphene, Å.
A_CC = 1.42

#: Graphene nearest-neighbour hopping, eV. 2.7 eV is the standard value
#: (Castro Neto et al., Rev. Mod. Phys. 81, 109 (2009)); DFT fits give 2.5-2.8.
T_GRAPHENE = -2.7


def _pi_valence(element: str, coordination: int) -> float:
    """π electrons: C 1, B 0, N 2 when three-fold (graphitic/pyrrolic), 1 when two-fold
    (pyridinic, lone pair in the plane), O 2 (ether/hydroxyl) or 1 (carbonyl)."""
    if element == "C":
        return 1.0
    if element == "B":
        return 0.0
    if element == "N":
        return 2.0 if coordination >= 3 else 1.0
    if element == "O":
        return 2.0 if coordination >= 2 else 1.0
    raise KeyError(element)


def pi_model(t: float = T_GRAPHENE, cutoff: float = 1.75, strain_beta: float | None = None,
             heteroatoms: bool = True) -> TBModel:
    """Orthogonal π model: graphene, nanotubes, ribbons, flakes, fullerenes.

    Carbon on-site 0 (the energy reference), nearest-neighbour hopping ``t``
    within ``cutoff``. With ``strain_beta`` the hopping follows
    ``t exp(-beta (d/1.42 - 1))`` instead of being constant.

    Heteroatoms use Hückel parameters (Streitwieser, *Molecular Orbital
    Theory for Organic Chemists*, 1961): on-site ``h |t|`` below carbon and
    hopping ``k t``: N (two-fold, pyridinic) h 0.5, N (three-fold) h 1.5,
    B h -1.0, O h 1.0 (carbonyl) / 2.0 (ether); k_CN 1.0, k_CB 0.7,
    k_CO 1.0 (0.8 for ether-like). The three-fold N value is used for all N
    here; the two-fold correction is the valence rule (1 π electron). These
    are textbook π parameters, good for trends, not for quantitative levels.
    """
    law: Callable[[float], DistanceLaw]
    if strain_beta is None:
        def law(v): return Constant(v, cutoff)
    else:
        def law(v): return Exponential(v, A_CC, strain_beta, cutoff)
    orbitals = {"C": ("pi",)}
    onsite = {"C": {"p": 0.0}}
    hopping = {("C", "C", "ppp"): law(t)}
    if heteroatoms:
        a = abs(t)
        orbitals.update({"N": ("pi",), "B": ("pi",), "O": ("pi",)})
        # Below carbon = more negative energy (more electronegative).
        onsite.update({"N": {"p": -1.5 * a}, "B": {"p": 1.0 * a}, "O": {"p": -1.0 * a}})
        hopping.update({("C", "N", "ppp"): law(1.0 * t), ("C", "B", "ppp"): law(0.7 * t),
                        ("C", "O", "ppp"): law(1.0 * t), ("N", "N", "ppp"): law(1.0 * t),
                        ("B", "N", "ppp"): law(0.7 * t)})
    return TBModel(
        name=f"π Hückel (t = {t} eV)", orbitals=orbitals, onsite=onsite, hopping=hopping,
        valence_rule=_pi_valence, hubbard_u={"C": 1.0 * abs(t), "N": 1.0 * abs(t),
                                             "B": 1.0 * abs(t), "O": 1.0 * abs(t)},
    )


def xu_carbon() -> TBModel:
    """Orthogonal sp³ model of carbon: Xu, Wang, Chan and Ho,
    J. Phys.: Condens. Matter 4, 6047 (1992).

    E_s = -2.99, E_p = 3.71 eV; ssσ -5.0, spσ 4.7, ppσ 5.5, ppπ -1.55 eV at
    r0 = 1.536 Å, GSP scaling with n = 2, nc = 6.5, rc = 2.18 Å, cut off at
    2.6 Å. Fitted to diamond, graphite, the C2 dimer and linear chains; the
    repulsive part of the original model is not included (no total
    energies or forces here).
    """
    r0, n, nc, rc, cut = 1.536, 2.0, 6.5, 2.18, 2.6

    def gsp(v):
        return GSP(v, r0, n, nc, rc, cut)

    return TBModel(
        name="sp3 carbono (Xu, Wang, Chan, Ho 1992)",
        orbitals={"C": ("s", "px", "py", "pz")},
        onsite={"C": {"s": -2.99, "p": 3.71}},
        hopping={("C", "C", "sss"): gsp(-5.0), ("C", "C", "sps"): gsp(4.7),
                 ("C", "C", "pps"): gsp(5.5), ("C", "C", "ppp"): gsp(-1.55)},
        valence={"C": 4.0},
        hubbard_u={"C": 10.0},
    )
