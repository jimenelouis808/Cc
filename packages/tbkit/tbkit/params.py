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

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
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


@dataclass(frozen=True)
class Tail:
    """``inner`` up to ``r1``, then a cubic to zero at ``rm``: a smooth cutoff.

    The cubic matches the value and slope of ``inner`` at ``r1`` and reaches
    zero with zero slope at ``rm``, so energies and forces are continuous
    through the cutoff (Xu et al. use this form between 2.45 and 2.6 Å).
    """

    inner: DistanceLaw
    r1: float
    rm: float

    @property
    def cutoff(self) -> float:
        return self.rm

    @cached_property
    def _coefficients(self):
        h = 1e-6
        f = float(self.inner(self.r1))
        slope = float((self.inner(self.r1 + h) - self.inner(self.r1 - h)) / (2 * h))
        delta = self.rm - self.r1
        a2 = (-3 * f - 2 * slope * delta) / delta ** 2
        a3 = (2 * f + slope * delta) / delta ** 3
        return f, slope, a2, a3

    def __call__(self, d):
        d = np.asarray(d, dtype=float)
        f, slope, a2, a3 = self._coefficients
        x = d - self.r1
        tail = f + slope * x + a2 * x ** 2 + a3 * x ** 3
        inner = self.inner(np.minimum(d, self.r1))
        return np.where(d < self.r1, inner, np.where(d < self.rm, tail, 0.0))


def derivative(law: DistanceLaw, d, h: float = 1e-5):
    """dV/dd by central differences (exact to O(h²); every law is smooth)."""
    d = np.asarray(d, dtype=float)
    return (law(d + h) - law(d - h)) / (2 * h)


_LAWS = {cls.__name__: cls for cls in (Constant, Exponential, Harrison, GSP, Table)}


def law_to_dict(law) -> dict:
    """A JSON-ready description of a distance law (for records and files)."""
    from dataclasses import asdict

    if isinstance(law, Tail):
        return {"type": "Tail", "inner": law_to_dict(law.inner), "r1": law.r1, "rm": law.rm}
    if hasattr(law, "to_dict"):
        return law.to_dict()
    data = asdict(law)
    data = {k: (list(v) if isinstance(v, tuple) else v) for k, v in data.items()}
    return {"type": type(law).__name__, **data}


def law_from_dict(data: dict):
    data = dict(data)
    kind = data.pop("type")
    if kind == "Tail":
        return Tail(law_from_dict(data["inner"]), float(data["r1"]), float(data["rm"]))
    if kind == "SkfSpline":
        from .repulsive import SkfSpline

        return SkfSpline.from_dict(data)
    if kind not in _LAWS:
        raise ValueError(f"Ley de distancia desconocida: {kind!r}.")
    if kind == "Table":
        data["r"], data["values"] = tuple(data["r"]), tuple(data["values"])
    return _LAWS[kind](**data)


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
    repulsive
        Pair or embedded repulsion, needed for total energies and forces.
    metadata
        Provenance of the parameters (reference, validity...).
    """

    name: str
    orbitals: dict[str, tuple[str, ...]]
    onsite: dict[str, dict[str, float]]
    hopping: dict[tuple[str, str, str], DistanceLaw]
    overlap: dict[tuple[str, str, str], DistanceLaw] = field(default_factory=dict)
    valence: dict[str, float] = field(default_factory=dict)
    valence_rule: Optional[ValenceRule] = None
    hubbard_u: dict[str, float] = field(default_factory=dict)
    #: Short-range repulsion for total energies and forces
    #: (:mod:`tbkit.repulsive`); None means electronic structure only.
    repulsive: Optional[object] = None
    #: Where the numbers come from: reference, system, validity, notes, and
    #: the per-parameter descriptions of the file they were read from.
    metadata: dict = field(default_factory=dict)

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
# Parameter files
# --------------------------------------------------------------------------

PARAMETER_DIR = Path(__file__).with_name("parameters")


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


#: Valence rules a parameter file may name.
VALENCE_RULES: dict[str, ValenceRule] = {"pi_huckel": _pi_valence}


def read_parameter_file(name_or_path: str | Path) -> dict:
    """The raw content of a parameter file: a built-in name or a path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = PARAMETER_DIR / f"{name_or_path}.json"
    if not path.exists():
        available = ", ".join(p.stem for p in sorted(PARAMETER_DIR.glob("*.json")))
        raise FileNotFoundError(f"No existe el conjunto de parámetros {name_or_path!r}. "
                                f"Incluidos: {available}.")
    return json.loads(path.read_text(encoding="utf-8"))


def _value(entry):
    return entry["value"] if isinstance(entry, dict) else entry


def model_from_dict(data: dict) -> TBModel:
    """A model from the generic parameter-file layout (see ``parameters/*.json``)."""
    def laws(entries):
        out = {}
        for entry in entries:
            law = law_from_dict(entry["law"])
            if entry.get("tail"):
                law = Tail(law, *map(float, entry["tail"]))
            out[(entry["pair"][0], entry["pair"][1], entry["bond"])] = law
        return out

    repulsive = None
    if data.get("repulsive"):
        from .repulsive import repulsive_from_dict

        repulsive = repulsive_from_dict(data["repulsive"])
    rule = data.get("valence_rule")
    metadata = {k: data[k] for k in ("reference", "system", "validity", "notes", "units")
                if k in data}
    metadata["parameters"] = data
    return TBModel(
        name=data["name"],
        orbitals={el: tuple(orbs) for el, orbs in data["orbitals"].items()},
        onsite={el: {shell: float(_value(v)) for shell, v in table.items()}
                for el, table in data["onsite"].items()},
        hopping=laws(data.get("hopping", [])),
        overlap=laws(data.get("overlap", [])),
        valence={el: float(_value(v)) for el, v in data.get("valence", {}).items()},
        valence_rule=VALENCE_RULES[rule] if rule else None,
        hubbard_u={el: float(_value(v)) for el, v in data.get("hubbard_u", {}).items()},
        repulsive=repulsive,
        metadata=metadata,
    )


def model_to_dict(model: TBModel) -> dict:
    """A parameter-file dictionary for ``model`` (records, sharing a fitted set).

    Laws are written in full; the per-parameter descriptions of the source
    file are kept when the model was read from one.
    """
    def entries(table):
        return [{"pair": [a, b], "bond": bond, "law": law_to_dict(law)}
                for (a, b, bond), law in table.items()]

    rule = next((name for name, fn in VALENCE_RULES.items() if fn is model.valence_rule), None)
    data = {
        "name": model.name,
        "units": {"energy": "eV", "length": "Å"},
        "orbitals": {el: list(orbs) for el, orbs in model.orbitals.items()},
        "onsite": model.onsite,
        "valence": model.valence,
        "hubbard_u": model.hubbard_u,
        "hopping": entries(model.hopping),
        "overlap": entries(model.overlap),
    }
    if rule:
        data["valence_rule"] = rule
    if model.repulsive is not None:
        data["repulsive"] = model.repulsive.to_dict()
    for key in ("reference", "system", "validity", "notes"):
        if key in model.metadata:
            data[key] = model.metadata[key]
    return data


def load_parameters(name_or_path: str | Path) -> TBModel:
    """A model from a parameter file (``"xu_carbon"`` or a path to a JSON file)."""
    data = read_parameter_file(name_or_path)
    if "t" in data and "orbitals" not in data:
        raise ValueError("Ese archivo describe el modelo π: usa pi_model().")
    return model_from_dict(data)


def pi_model(t: Optional[float] = None, cutoff: Optional[float] = None,
             strain_beta: Optional[float] = None, heteroatoms: bool = True) -> TBModel:
    """Orthogonal π model: graphene, nanotubes, ribbons, flakes, fullerenes.

    Numbers from ``parameters/pi_huckel.json`` (each with its source):
    carbon on-site 0 (the energy reference), nearest-neighbour hopping
    ``t = -2.7 eV`` within 1.75 Å; with ``strain_beta`` the hopping follows
    ``t exp(-beta (d/1.42 - 1))``. Heteroatoms (N, B, O) use Hückel's
    ``α_X = α + h β`` and ``β_CX = k β``. Textbook π parameters: good for
    trends, not for quantitative levels (fit them with :mod:`tbkit.fit`).
    Arguments override the file's values.
    """
    data = read_parameter_file("pi_huckel")
    t = float(data["t"]["value"]) if t is None else float(t)
    cutoff = float(data["cutoff"]["value"]) if cutoff is None else float(cutoff)
    a_cc = float(data["a_cc"]["value"])

    def law(v):
        return Constant(v, cutoff) if strain_beta is None else Exponential(
            v, a_cc, strain_beta, cutoff)

    orbitals = {"C": ("pi",)}
    onsite = {"C": {"p": 0.0}}
    hopping = {("C", "C", "ppp"): law(t)}
    if heteroatoms:
        for element, entry in data["heteroatoms"].items():
            orbitals[element] = ("pi",)
            onsite[element] = {"p": float(entry["h"]) * t}      # α + h β with β = t
        for pair, entry in data["bond_factors"].items():
            a, b = pair.split("-")
            hopping[(a, b, "ppp")] = law(float(entry["k"]) * t)
    u = float(data["hubbard_u_over_t"]["value"]) * abs(t)
    metadata = {k: data[k] for k in ("reference", "system", "validity", "notes") if k in data}
    metadata["parameters"] = data
    return TBModel(name=f"π Hückel (t = {t} eV)", orbitals=orbitals, onsite=onsite,
                   hopping=hopping, valence_rule=_pi_valence,
                   hubbard_u={el: u for el in orbitals}, metadata=metadata)


def xu_carbon() -> TBModel:
    """sp³ carbon of Xu, Wang, Chan and Ho (1992), with its repulsive term.

    All numbers in ``parameters/xu_carbon.json``: orthogonal s+p basis, GSP
    hopping with smooth tails between 2.45 and 2.6 Å, and the embedded
    repulsion ``E_rep = Σ_i f(Σ_j φ(r_ij))`` that makes total energies and
    forces possible. Fitted to diamond, graphite, C2 and chains.
    """
    return load_parameters("xu_carbon")
