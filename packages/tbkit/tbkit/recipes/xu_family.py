"""Xu's carbon plus heteroatoms fitted to GPAW: the recipe behind ``xu_chn``, ``xu_chno``...

Carbon-carbon is Xu, Wang, Chan and Ho's model, untouched (on-site energies,
hoppings, embedded repulsion): it sets the energy zero and the heteroatoms
are fitted around it. A :class:`XuFamily` names what is added:

* on-site energies of each heteroatom (s for H; s and p otherwise);
* Slater-Koster hoppings of every listed pair, in Xu's GSP form
  ``v0 (r0/d)^n exp{n[-(d/rc)^nc + (r0/rc)^nc]}`` with nc = 6.5, rc and the
  smooth tail scaled from Xu's by ``r0_pair / r0_CC``; one v0 per bond type
  and one n per pair are fitted;
* Hubbard U and intra-atomic dipoles: computed, never fitted (GPAW's
  all-electron atom, PBE: U = dε/dn of the valence level, the DFTB
  definition; d = |⟨2s|r|2p⟩|);
* pair repulsions ``Σ_k c_k (rc - d)^k`` (k = 3...7).

The ground state is self-consistent (DFTB2-like). The fit (Kohn-Sham levels
with one common shift; then levels, forces and energies together with the
repulsion solved exactly at each step, as in NRL-TB's joint fit) is the one
documented in :mod:`tbkit.recipes.xu_chn`, which is an instance of this.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from ase.neighborlist import neighbor_list
from scipy.optimize import least_squares

from ..hamiltonian import System
from ..params import GSP, TBModel, Tail, model_from_dict, model_to_dict, read_parameter_file
from ..references import ReferenceStructure, load_references
from ..repulsive import PairRepulsive, SumRepulsive
from ..scc import self_consistent

#: Xu's C-C scale: GSP r0, rc and tail in Å.
XU_R0, XU_RC, XU_TAIL = 1.536329, 2.18, (2.45, 2.6)
NC = 6.5
POWERS = (3, 4, 5, 6, 7)

#: Harrison's universal η (Harrison, Electronic Structure, 1980) for the
#: starting guesses, V = η ħ²/(m d²) with ħ²/m = 7.62 eV Å².
_ETA = {"sss": -1.40, "sps": 1.84, "pps": 3.24, "ppp": -0.81}

#: Free-atom valence levels (eV), GPAW all-electron atom, PBE, spin-paired:
#: starting guesses only, shifted so that C 2p lands on Xu's E_p.
FREE_ATOM_LEVELS = {"H": {"s": -6.492}, "C": {"s": -13.738, "p": -5.289},
                    "N": {"s": -18.4, "p": -7.095}, "O": {"s": -23.912, "p": -9.038},
                    "B": {"s": -9.438, "p": -3.609}, "P": {"s": -13.894, "p": -5.518},
                    "S": {"s": -17.142, "p": -7.021}}
VALENCE = {"H": 1.0, "C": 4.0, "N": 5.0, "O": 6.0, "B": 3.0, "P": 5.0, "S": 6.0}
SP = ("s", "px", "py", "pz")


def pair_bonds(a: str, b: str) -> tuple:
    """The Slater-Koster integrals a pair has (H carries only s)."""
    if b == "H" and a != "H":
        a, b = b, a
    if a == "H":
        return ((a, b, "sss"), (a, b, "sps"))
    if a == b:
        return ((a, a, "sss"), (a, a, "sps"), (a, a, "pps"), (a, a, "ppp"))
    return ((a, b, "sss"), (a, b, "sps"), (b, a, "sps"), (a, b, "pps"), (a, b, "ppp"))


@dataclass
class XuFamily:
    """What a parameter set of the family adds to Xu's carbon."""

    name: str                                  # model name
    recipe: str                                # module that runs it
    heteroatoms: tuple[str, ...]               # e.g. ("H", "N")
    pairs: dict                                # (a, b) -> {"r0", "rc_rep"}; bonds derived
    hubbard_u: dict                            # eV, all elements
    onsite_dipole: dict                        # Å, elements with p
    system: str = ""
    validity_notes: str = ""
    experimental_alpha: dict = field(default_factory=dict)
    #: Acute-angle correction (:class:`tbkit.repulsive.AcuteAngleTerm`): number
    #: of powers fitted, with its fixed shape; None = no correction (xu_chn).
    acute: Optional[dict] = None
    #: Reference groups computed as training data but kept out of the fit
    #: (validated only), with the reason recorded in the parameter file.
    held_out: dict = field(default_factory=dict)
    #: A shipped parameter set kept fixed (e.g. "xu_chno"): only the
    #: heteroatoms and pairs listed here are fitted, on top of it, with its
    #: level shift. Every pair between elements must then be defined
    #: (checked): a missing hopping law would silently be zero.
    base: Optional[str] = None
    #: For pairs with H: hopping tail at ``r0 + h_tail`` (Å) and repulsion cutoff
    #: at ``r0 + h_tail[1]``, instead of Xu's tail scaled by r0/r0_CC. The scaled
    #: C-H tail (1.74-1.84 Å) sits where a hydroxyl H meets its carbon's
    #: neighbour (1.85-2.2 Å): alcohols relaxed onto the tail's edge and got an
    #: O-H mode at 4950 cm⁻¹ (GPAW 3618). None keeps the scaled tail (xu_chn).
    h_tail: Optional[tuple] = None
    #: The same for pairs of heavy atoms other than C-C (C-O, C-N, N-O, O-O...):
    #: their scaled tails (2.2-2.5 Å) overlap second-neighbour distances (O...O
    #: in CO2 or a carboxyl, C...O across a ring). None keeps the scaled tail.
    heavy_tail: Optional[tuple] = None

    def acute_term(self, coefficients=None):
        from ..repulsive import AcuteAngleTerm

        spec = self.acute
        c = tuple(coefficients) if coefficients is not None else (1.0,) * spec["powers"]
        return AcuteAngleTerm(c, spec.get("theta0_degrees", 80.0), spec.get("r1", 1.7),
                              spec.get("rm", 2.0), tuple(spec.get("elements", ("C", "N", "O"))))

    def __post_init__(self):
        for (a, b), spec in self.pairs.items():
            spec.setdefault("bonds", pair_bonds(a, b))
            offsets = self.h_tail if "H" in (a, b) else \
                self.heavy_tail if (a, b) != ("C", "C") else None
            if offsets is not None:
                r1, rm = (round(spec["r0"] + d, 3) for d in offsets)
                spec.setdefault("tail", (r1, rm))
                spec["rc_rep"] = min(spec["rc_rep"], rm)
        if self.base is not None:
            known = {frozenset(key[:2]) for key in self.base_model().hopping}
            known |= {frozenset(pair) for pair in self.pairs}
            elements = self.elements()
            missing = [f"{a}-{b}" for i, a in enumerate(elements) for b in elements[i:]
                       if frozenset((a, b)) not in known and not a == b == "H"]
            if missing:
                raise ValueError(f"{self.name}: pares sin definir: {', '.join(missing)}")

    def base_model(self):
        """The fixed model underneath: Xu's carbon, or the ``base`` set."""
        return _base_model(self.base or "xu_carbon")

    def elements(self) -> list[str]:
        base = self.base_model()
        return list(dict.fromkeys(list(base.orbitals) + list(self.heteroatoms)))

    def fixed_shift(self) -> Optional[float]:
        """The base set's level shift (Xu zero vs GPAW vacuum), None without a base."""
        if self.base is None:
            return None
        return float(read_parameter_file(self.base)["fit"]["level_shift_eV"])

    # --- parameters ---------------------------------------------------------

    @property
    def onsite_parameters(self) -> list[tuple[str, str]]:
        out = []
        for element in self.heteroatoms:
            out += [(element, "s")] if element == "H" else [(element, "s"), (element, "p")]
        return out

    def parameter_names(self) -> list[str]:
        names = [f"onsite {el} {shell}" for el, shell in self.onsite_parameters]
        for (a, b), spec in self.pairs.items():
            for first, second, bond in spec["bonds"]:
                names.append(f"{a}-{b} {bond}({first},{second}) v0")
            names.append(f"{a}-{b} n")
        return names

    def initial_guess(self) -> np.ndarray:
        shift = 3.71 - FREE_ATOM_LEVELS["C"]["p"]      # Xu's C p minus the free atom's
        x = [FREE_ATOM_LEVELS[el][shell] + shift for el, shell in self.onsite_parameters]
        for spec in self.pairs.values():
            for _, _, bond in spec["bonds"]:
                x.append(_ETA[bond] * 7.62 / spec["r0"] ** 2)
            x.append(2.0)
        return np.array(x)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        names = self.parameter_names()
        lower = np.full(len(names), -np.inf)
        upper = np.full(len(names), np.inf)
        for i, name in enumerate(names):
            if name.endswith(" n"):
                lower[i], upper[i] = 0.5, 6.0
        return lower, upper

    @staticmethod
    def _law(v0: float, n: float, r0: float, tail: Optional[tuple] = None):
        scale = r0 / XU_R0
        r1, rm = tail if tail is not None else (XU_TAIL[0] * scale, XU_TAIL[1] * scale)
        return Tail(GSP(v0, r0, n, NC, XU_RC * scale, rm), r1, rm)

    def build_model(self, x, repulsive: Optional[object] = None) -> TBModel:
        base = self.base_model()
        x = list(map(float, x))
        onsite = {el: dict(table) for el, table in base.onsite.items()}
        k = 0
        for element, shell in self.onsite_parameters:
            onsite.setdefault(element, {})[shell] = x[k]
            k += 1
        hopping = dict(base.hopping)
        for spec in self.pairs.values():
            values = x[k:k + len(spec["bonds"])]
            n = x[k + len(spec["bonds"])]
            for (first, second, bond), v0 in zip(spec["bonds"], values, strict=True):
                hopping[(first, second, bond)] = self._law(v0, n, spec["r0"], spec.get("tail"))
            k += len(spec["bonds"]) + 1
        if repulsive is None:
            repulsive = SumRepulsive(self.base_terms())
        elements = self.elements()
        return TBModel(name=self.name,
                       orbitals={el: ("s",) if el == "H" else SP for el in elements},
                       onsite=onsite, hopping=hopping,
                       valence={el: VALENCE[el] for el in elements},
                       hubbard_u={**base.hubbard_u, **self.hubbard_u}, repulsive=repulsive,
                       scc=True, onsite_dipole={**base.onsite_dipole, **self.onsite_dipole},
                       extra_polarizability=dict(base.extra_polarizability)
                       if self.base else {})

    def base_terms(self) -> tuple:
        """Repulsive terms that stay fixed: Xu's embedded C-C, or all of the base's."""
        base = self.base_model()
        if self.base is not None:
            return tuple(base.repulsive.terms)
        xu = copy.copy(base.repulsive)
        xu.others = "ignore"
        return (xu,)

    def repulsion_from_coefficients(self, coefficients) -> SumRepulsive:
        from ..params import CutoffPolynomial

        laws = {}
        n = len(POWERS)
        for p, (pair, spec) in enumerate(self.pairs.items()):
            c = tuple(float(v) for v in coefficients[p * n:(p + 1) * n])
            laws[pair] = CutoffPolynomial(c, spec["rc_rep"], POWERS[0])
        terms = [*self.base_terms(), PairRepulsive(laws)]
        if self.acute:
            start = len(self.pairs) * n
            terms.append(self.acute_term([float(v) for v in coefficients[start:]]))
        return SumRepulsive(tuple(terms))


_BASES: dict = {}


def _base_model(name: str) -> TBModel:
    if name not in _BASES:
        _BASES[name] = model_from_dict(read_parameter_file(name))
    return _BASES[name]


# --------------------------------------------------------------------------
# Levels
# --------------------------------------------------------------------------

def selection(ref: ReferenceStructure) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the compared levels and their weights."""
    n = ref.n_occupied
    indices = np.arange(n + 2)
    weights = np.ones(n + 2)
    weights[n - 1] = weights[n] = 2.0       # HOMO, LUMO
    weights[n + 1] = 0.5                    # LUMO+1: basis-limited in LCAO, lightly
    return indices, weights


def model_levels(model: TBModel, ref: ReferenceStructure, kT: float = 0.01) -> np.ndarray:
    result = self_consistent(System.build(ref.atoms, model), kT=kT, tol=1e-7)
    if not result.converged:
        raise RuntimeError(f"{ref.label}: SCC sin converger")
    return np.sort(result.solution.energies[0, 0])


def level_residuals(model: TBModel, refs: list[ReferenceStructure],
                    shift: Optional[float] = None) -> tuple[np.ndarray, float, dict]:
    """Weighted residuals after the best common shift (or the one given), the shift,
    and per-label RMS."""
    differences, weights, labels = [], [], []
    for ref in refs:
        indices, w = selection(ref)
        tb = model_levels(model, ref)[indices]
        differences.append(tb - ref.levels[indices])
        weights.append(w)
        labels.append(ref.label)
    d = np.concatenate(differences)
    w = np.concatenate(weights)
    if shift is None:
        shift = float(np.sum(w ** 2 * d) / np.sum(w ** 2))
    rms = {label: float(np.sqrt(np.mean((diff - shift) ** 2)))
           for label, diff in zip(labels, differences, strict=True)}
    return w * (d - shift), shift, rms


def fit_levels(family: XuFamily, refs: list[ReferenceStructure],
               x0: Optional[np.ndarray] = None):
    x0 = family.initial_guess() if x0 is None else np.asarray(x0, dtype=float)
    fixed = family.fixed_shift()

    def residuals(x):
        try:
            return level_residuals(family.build_model(x), refs, fixed)[0]
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            return np.full(sum(r.n_occupied + 2 for r in refs), 10.0)

    return least_squares(residuals, x0, bounds=family.bounds(), x_scale="jac")


# --------------------------------------------------------------------------
# Repulsion
# --------------------------------------------------------------------------

def pair_basis(atoms, pair: tuple[str, str], rc: float):
    """Energy (k,) and forces (k, N, 3) of each ``(rc - d)^p`` term for one pair."""
    symbols = np.array(atoms.get_chemical_symbols())
    ii, jj, dd, vv = neighbor_list("ijdD", atoms, rc)
    a, b = pair
    keep = ((symbols[ii] == a) & (symbols[jj] == b)) | ((symbols[ii] == b) & (symbols[jj] == a))
    ii, jj, dd, vv = ii[keep], jj[keep], dd[keep], vv[keep]
    energies = np.zeros(len(POWERS))
    forces = np.zeros((len(POWERS), len(atoms), 3))
    x = rc - dd
    for k, p in enumerate(POWERS):
        energies[k] = 0.5 * np.sum(x ** p)             # each pair appears twice
        slope = -p * x ** (p - 1)                      # dV/dd
        push = 0.5 * (slope / dd)[:, None] * vv        # vv = r_j - r_i
        np.add.at(forces[k], ii, push)
        np.add.at(forces[k], jj, -push)
    return energies, forces


def electronic_energy_and_forces(model: TBModel, ref: ReferenceStructure):
    """SCC energy and forces with only Xu's C-C repulsion."""
    from ..scc import energy_and_forces

    result = self_consistent(System.build(ref.atoms, model), kT=0.01, tol=1e-10)
    energy, forces, _ = energy_and_forces(result)
    return energy, forces


def repulsion_design(family: XuFamily, refs):
    """Per reference: the pair-repulsion basis energies (n,) and forces (n, N, 3)."""
    out = []
    for ref in refs:
        blocks_e, blocks_f = [], []
        for pair, spec in family.pairs.items():
            e, f = pair_basis(ref.atoms, pair, spec["rc_rep"])
            blocks_e.append(e)
            blocks_f.append(f)
        if family.acute:
            e, f = family.acute_term().basis(ref.atoms)
            blocks_e.append(e)
            blocks_f.append(f)
        out.append((np.concatenate(blocks_e), np.concatenate(blocks_f)))
    return out


def solve_repulsion(refs, design, results, groups, wf=1.0, we=3.0, ridge=1e-6,
                    hessians=(), h_design=(), h_electronic=(), wh=0.1):
    """Linear least squares for the pair coefficients given electronic E and F
    (and, with ``hessians``, the curvature: GPAW Hessian minus the electronic one,
    upper triangle, weight ``wh`` per eV/Å²)."""
    rows, targets, e_rows, e_targets = [], [], [], []
    for ref, (basis_e, basis_f), (_, e_tb, f_tb) in zip(refs, design, results, strict=True):
        rows.append(basis_f.reshape(len(basis_e), -1).T)
        targets.append((ref.forces - f_tb).ravel())
        e_rows.append(basis_e)
        e_targets.append(ref.energy - e_tb)
    n_basis = len(design[0][0]) if design else len(h_design[0]) if h_design else 0
    e_rows = np.array(e_rows).reshape(-1, n_basis)
    e_targets = np.array(e_targets, dtype=float)
    for members in groups.values():             # remove each molecule's constant
        e_rows[members] -= e_rows[members].mean(axis=0)
        e_targets[members] -= e_targets[members].mean()
    h_rows, h_targets = [], []
    for target, basis, electronic in zip(hessians, h_design, h_electronic, strict=True):
        upper = np.triu_indices(basis.shape[1])
        h_rows.append(np.stack([b[upper] for b in basis], axis=1))
        h_targets.append((target.hessian - electronic)[upper])
    a = np.vstack([wf * r for r in rows] + [we * e_rows] + [wh * r for r in h_rows]
                  ).reshape(-1, n_basis)
    y = np.concatenate([wf * t for t in targets] + [we * e_targets]
                       + [wh * t for t in h_targets])
    active = np.abs(a).sum(axis=0) > 0
    lhs = a[:, active].T @ a[:, active] + ridge * np.eye(int(active.sum()))
    c = np.zeros(a.shape[1])
    c[active] = np.linalg.solve(lhs, a[:, active].T @ y)
    n_force = sum(len(t) for t in targets)
    n_energy = len(e_targets)
    residual = y - a @ c
    if hessians:
        return (c, residual[:n_force], residual[n_force:n_force + n_energy],
                residual[n_force + n_energy:])
    return c, residual[:n_force], residual[n_force:]


def fit_repulsion(family: XuFamily, model: TBModel, refs, energy_weight: float = 3.0,
                  ridge: float = 1e-8):
    """Pair coefficients for a fixed electronic part; returns (repulsion, report)."""
    results = [(None,) + electronic_energy_and_forces(model, ref) for ref in refs]
    groups: dict[str, list[int]] = {}
    for i, ref in enumerate(refs):
        groups.setdefault(ref.group, []).append(i)
    c, r_force, r_energy = solve_repulsion(refs, repulsion_design(family, refs), results,
                                           groups, 1.0, energy_weight, ridge)
    report = {"force_rms": float(np.sqrt(np.mean(r_force ** 2))),
              "energy_rms": float(np.sqrt(np.mean((r_energy / energy_weight) ** 2)))}
    return family.repulsion_from_coefficients(c), report


# --------------------------------------------------------------------------
# Levels, forces and energies together
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Hessians (curvature) as fit targets
# --------------------------------------------------------------------------

@dataclass
class HessianTarget:
    """GPAW's Cartesian Hessian of a molecule at its own minimum (eV/Å²)."""

    name: str
    atoms: object
    hessian: np.ndarray
    frequencies: np.ndarray


def load_hessians(path: Path) -> list[HessianTarget]:
    """Targets from a file of :mod:`tbkit.recipes.frequency_references`."""
    from ase import Atoms

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [HessianTarget(name, Atoms(e["symbols"], positions=e["positions"]),
                          np.array(e["hessian"], dtype=float),
                          np.array(e["frequencies"], dtype=float))
            for name, e in data["molecules"].items()]


def _symmetric(columns: np.ndarray) -> np.ndarray:
    return 0.5 * (columns + columns.T)


def electronic_hessian(model: TBModel, atoms, delta: float = 0.005) -> np.ndarray:
    """(3N, 3N) Hessian of the SCC electronic energy (no pair repulsion added by the
    fit), central differences of its forces."""
    from ..scc import energy_and_forces

    positions = atoms.get_positions()
    n = len(atoms)
    columns = np.zeros((3 * n, 3 * n))
    for k in range(3 * n):
        forces = []
        for sign in (1.0, -1.0):
            moved = atoms.copy()
            p = positions.copy()
            p[k // 3, k % 3] += sign * delta
            moved.set_positions(p)
            result = self_consistent(System.build(moved, model), kT=0.01, tol=1e-10)
            if not result.converged:
                raise RuntimeError("SCC sin converger en la hessiana")
            forces.append(energy_and_forces(result)[1].ravel())
        columns[:, k] = -(forces[0] - forces[1]) / (2 * delta)
    return _symmetric(columns)


def basis_hessians(family: "XuFamily", atoms, delta: float = 1e-4) -> np.ndarray:
    """(n_basis, 3N, 3N) Hessians of the repulsion basis (pair powers and acute
    terms): independent of the electronic parameters, computed once."""
    positions = atoms.get_positions()
    n = len(atoms)

    def basis_forces(p):
        moved = atoms.copy()
        moved.set_positions(p)
        blocks = [pair_basis(moved, pair, spec["rc_rep"])[1]
                  for pair, spec in family.pairs.items()]
        if family.acute:
            blocks.append(family.acute_term().basis(moved)[1])
        return np.concatenate(blocks).reshape(-1, 3 * n)

    size = len(basis_forces(positions))
    out = np.zeros((size, 3 * n, 3 * n))
    for k in range(3 * n):
        plus, minus = positions.copy(), positions.copy()
        plus[k // 3, k % 3] += delta
        minus[k // 3, k % 3] -= delta
        out[:, :, k] = -(basis_forces(plus) - basis_forces(minus)) / (2 * delta)
    return np.array([_symmetric(h) for h in out])


_WORKER: dict = {}


def _init_worker(family, refs, hessians=()):
    _WORKER["family"] = family
    _WORKER["refs"] = refs
    _WORKER["hessians"] = list(hessians)


def _evaluate_hessian(args):
    x, index = args
    target = _WORKER["hessians"][index]
    return electronic_hessian(_WORKER["family"].build_model(x), target.atoms)


def _evaluate_one(args):
    from ..scc import energy_and_forces

    x, index = args
    ref = _WORKER["refs"][index]
    model = _WORKER["family"].build_model(x)
    result = self_consistent(System.build(ref.atoms, model), kT=0.01, tol=1e-10)
    if not result.converged:
        raise RuntimeError(f"{ref.label}: SCC sin converger")
    energy, forces, _ = energy_and_forces(result)
    return np.sort(result.solution.energies[0, 0]), energy, forces


class JointObjective:
    """Residuals of levels, forces and energies, with the repulsion solved exactly.

    For given electronic parameters the best pair repulsion is a linear
    least-squares problem (variable projection); what is minimised over the
    electronic parameters is what remains after it -- the spirit of NRL-TB's
    joint fit of eigenvalues and total energies (Papaconstantopoulos et al.
    2024, their eq. 11).
    """

    def __init__(self, family: XuFamily, refs, level_weight=1.0, force_weight=1.0,
                 energy_weight=3.0, ridge=1e-6, workers=4, hessians=(), hessian_weight=0.1):
        from concurrent.futures import ProcessPoolExecutor

        self.family = family
        self.refs = refs
        self.wl, self.wf, self.we = level_weight, force_weight, energy_weight
        self.ridge = ridge
        self.fixed_shift = family.fixed_shift()
        self.design = repulsion_design(family, refs)
        self.groups: dict[str, list[int]] = {}
        for i, ref in enumerate(refs):
            self.groups.setdefault(ref.group, []).append(i)
        self.hessians = list(hessians)
        self.wh = hessian_weight
        self.h_design = [basis_hessians(family, t.atoms) for t in self.hessians]
        self.pool = ProcessPoolExecutor(workers, initializer=_init_worker,
                                        initargs=(family, refs, self.hessians))
        self.size = None

    def close(self):
        self.pool.shutdown()

    def evaluate(self, x):
        x = np.asarray(x)
        hessians = [self.pool.submit(_evaluate_hessian, (x, i))
                    for i in range(len(self.hessians))]          # the slow ones first
        results = list(self.pool.map(_evaluate_one, [(x, i) for i in range(len(self.refs))],
                                     chunksize=4))
        self._h_electronic = [h.result() for h in hessians]
        return results

    def residuals(self, x, full: bool = False):
        try:
            results = self.evaluate(x)
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            if full or self.size is None:
                raise
            return np.full(self.size, 30.0)
        differences, weights = [], []
        for ref, (levels, _, _) in zip(self.refs, results, strict=True):
            indices, w = selection(ref)
            differences.append(levels[indices] - ref.levels[indices])
            weights.append(w)
        d, w = np.concatenate(differences), np.concatenate(weights)
        shift = self.fixed_shift if self.fixed_shift is not None else \
            float(np.sum(w ** 2 * d) / np.sum(w ** 2))
        solved = solve_repulsion(self.refs, self.design, results, self.groups,
                                 self.wf, self.we, self.ridge, self.hessians, self.h_design,
                                 self._h_electronic, self.wh)
        c, r_force, r_energy = solved[:3]
        r_hessian = solved[3] if self.hessians else np.zeros(0)
        out = np.concatenate([self.wl * w * (d - shift), r_force, r_energy, r_hessian])
        self.size = len(out)
        if full:
            return out, {"shift": shift, "coefficients": c,
                         "level_rms": float(np.sqrt(np.mean((d - shift) ** 2))),
                         "force_rms": float(np.sqrt(np.mean((r_force / self.wf) ** 2))),
                         "energy_rms": float(np.sqrt(np.mean((r_energy / self.we) ** 2))),
                         "hessian_rms": float(np.sqrt(np.mean((r_hessian / self.wh) ** 2)))
                         if self.hessians else None}
        return out


def fit_joint(family: XuFamily, refs, x0, workers: int = 4, max_nfev: int = 400,
              verbose: bool = True, **weights):
    """Least squares over the electronic parameters on levels + forces + energies."""
    objective = JointObjective(family, refs, workers=workers, **weights)
    lower, upper = family.bounds()
    try:
        _, info0 = objective.residuals(x0, full=True)
        if verbose:
            extra = f", hessiana {info0['hessian_rms']:.2f} eV/Å²" \
                if info0.get("hessian_rms") is not None else ""
            print(f"conjunto, inicio: niveles {info0['level_rms']:.3f} eV, fuerzas "
                  f"{info0['force_rms']:.3f} eV/Å, energías {info0['energy_rms']:.3f} eV"
                  f"{extra}", flush=True)
        result = least_squares(objective.residuals, np.clip(x0, lower + 1e-9, upper - 1e-9),
                               bounds=(lower, upper), x_scale="jac", max_nfev=max_nfev)
        _, info = objective.residuals(result.x, full=True)
    finally:
        objective.close()
    if verbose:
        extra = f", hessiana {info['hessian_rms']:.2f} eV/Å²" \
            if info.get("hessian_rms") is not None else ""
        print(f"conjunto, final ({result.message}): niveles {info['level_rms']:.3f} eV, "
              f"fuerzas {info['force_rms']:.3f} eV/Å, energías {info['energy_rms']:.3f} eV"
              f"{extra}", flush=True)
    return result, info, info0


# --------------------------------------------------------------------------
# Validation and the parameter file
# --------------------------------------------------------------------------

def bond_lengths(atoms, factor: float = 1.2) -> dict[tuple[int, int], float]:
    """Bonded pairs (i < j): closer than ``factor`` times the sum of the covalent
    radii (Cordero et al. 2008, via ASE): C-C 1.82, C-H 1.28, O-H 1.16,
    S-H 1.63, C-S 2.17, S-S 2.52, P-C 2.24 Å."""
    from ase.data import covalent_radii

    symbols = atoms.get_chemical_symbols()
    radii = covalent_radii[atoms.numbers]
    ii, jj, dd = neighbor_list("ijd", atoms, factor * 2 * radii.max())
    out = {}
    for i, j, d in zip(ii, jj, dd, strict=True):
        if i < j and d < factor * (radii[i] + radii[j]) and \
                not symbols[i] == symbols[j] == "H":
            out[(int(i), int(j))] = float(d)
    return out


def relaxed_bond_errors(model: TBModel, ref: ReferenceStructure, fmax: float = 0.01) -> dict:
    """Relax ``ref`` with the model; the largest bond-length error per bond type (Å)."""
    from ase.optimize import BFGS

    from ..calculator import TBCalculator

    atoms = ref.atoms.copy()
    atoms.calc = TBCalculator(model)
    BFGS(atoms, logfile=None).run(fmax=fmax, steps=300)
    symbols = atoms.get_chemical_symbols()
    errors: dict[str, list[float]] = {}
    for (i, j), d_ref in bond_lengths(ref.atoms).items():
        kind = "-".join(sorted((symbols[i], symbols[j])))
        errors.setdefault(kind, []).append(atoms.get_distance(i, j) - d_ref)
    return {kind: float(np.max(np.abs(v))) for kind, v in errors.items()}


def validate(family: XuFamily, model: TBModel, structures, shift: float,
             relax: bool = True) -> dict:
    """Levels, relaxed bonds and polarizabilities for every relaxed ("eq") reference."""
    from ..optics import polarizability_linear_response

    report = {}
    for ref in structures:
        if not ref.label.endswith("/eq"):
            continue
        indices, _ = selection(ref)
        levels = model_levels(model, ref)[indices] - shift
        n = ref.n_occupied
        entry = {"role": ref.role,
                 "level_rms": float(np.sqrt(np.mean((levels - ref.levels[indices]) ** 2))),
                 "gap_tb": float(levels[n] - levels[n - 1]),
                 "gap_dft": float(ref.levels[n] - ref.levels[n - 1])}
        if relax:
            entry["bond_error_max"] = relaxed_bond_errors(model, ref)
        if ref.group in family.experimental_alpha:
            alpha = polarizability_linear_response(System.build(ref.atoms, model))
            entry["alpha_mean"] = float(np.trace(alpha) / 3)
            entry["alpha_experiment"] = family.experimental_alpha[ref.group]
        report[ref.group] = entry
    return report


def tail_hits(model: TBModel, atoms, margin: float = 0.02) -> list[str]:
    """Atom pairs whose distance lies inside a hopping law's switch-off window
    (``Tail`` r1-rm, widened by ``margin``): a minimum there has a curvature the
    switch, not the physics, decides."""
    from ase.neighborlist import neighbor_list

    windows = {}
    for (a, b, _), law in model.hopping.items():
        if isinstance(law, Tail):
            windows[frozenset((a, b))] = (law.r1, law.rm)
    if not windows:
        return []
    symbols = atoms.get_chemical_symbols()
    reach = max(rm for _, rm in windows.values()) + margin
    ii, jj, dd = neighbor_list("ijd", atoms, reach)
    hits = []
    for i, j, d in zip(ii, jj, dd, strict=True):
        window = windows.get(frozenset((symbols[i], symbols[j])))
        if i < j and window and window[0] - margin <= d <= window[1] + margin:
            hits.append(f"{symbols[i]}{i}-{symbols[j]}{j} {d:.3f} Å "
                        f"(cola {window[0]:.2f}-{window[1]:.2f})")
    return hits


def frequency_validation(model: TBModel, targets, fmax: float = 0.001) -> dict:
    """Model frequencies at its own minimum (relaxed from GPAW's) against GPAW's,
    per molecule: RMS over the internal modes and both lists (cm⁻¹)."""
    from ase.optimize import BFGS

    from ..calculator import TBCalculator
    from ..tasks import phonons

    out = {}
    for target in targets:
        atoms = target.atoms.copy()
        atoms.calc = TBCalculator(model)
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=500)
        tb = np.sort(phonons(atoms, model)[0]["frequencies_cm1"])
        rigid = 5 if _linear(target.atoms) else 6
        gpaw = np.sort(target.frequencies)[rigid:]
        tb = tb[rigid:]
        out[target.name] = {"tail_hits": tail_hits(model, atoms),
                            "rms": float(np.sqrt(np.mean((tb - gpaw) ** 2))),
                            "max_error": float(np.max(np.abs(tb - gpaw))),
                            "gpaw": gpaw.round(1).tolist(), "tb": tb.round(1).tolist()}
    return out


def _linear(atoms) -> bool:
    if len(atoms) < 3:
        return True
    centred = atoms.get_positions() - atoms.get_positions().mean(axis=0)
    return bool(np.linalg.svd(centred, compute_uv=False)[1] < 1e-3)


def validation_table(report: dict) -> str:
    lines = [f"{'molécula':16s} {'rol':5s} {'RMS niv.':>8s} {'gap TB':>7s} {'gap DFT':>7s} "
             f"{'α TB':>6s} {'α exp':>6s}  error máx. de enlace (Å)"]
    for name, e in report.items():
        alpha = f"{e['alpha_mean']:6.2f} {e['alpha_experiment']:6.2f}" if "alpha_mean" in e \
            else f"{'':6s} {'':6s}"
        bonds = ", ".join(f"{k} {v:.3f}" for k, v in e.get("bond_error_max", {}).items())
        lines.append(f"{name:16s} {e['role']:5s} {e['level_rms']:8.3f} {e['gap_tb']:7.2f} "
                     f"{e['gap_dft']:7.2f} {alpha}  {bonds}")
    return "\n".join(lines)


def bond_ranges(structures) -> dict[str, list[float]]:
    """Shortest and longest bond of each element pair in the structures (Å)."""
    ranges: dict[str, list[float]] = {}
    for ref in structures:
        symbols = ref.atoms.get_chemical_symbols()
        for (i, j), d in bond_lengths(ref.atoms).items():
            kind = "-".join(sorted((symbols[i], symbols[j])))
            low, high = ranges.get(kind, [d, d])
            ranges[kind] = [round(min(low, d), 3), round(max(high, d), 3)]
    return ranges


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_file(family: XuFamily, model: TBModel, x, shift: float, report: dict,
                   references: Sequence[Path], settings: dict) -> dict:
    data = model_to_dict(model)
    source = (f"ajuste a GPAW ({settings.get('xc')}, {settings.get('mode')}/"
              f"{settings.get('basis')}, GPAW {settings.get('gpaw_version')}); receta "
              f"{family.recipe}")
    hetero = ", ".join(family.heteroatoms)
    data["reference"] = ("C-C: C. H. Xu, C. Z. Wang, C. T. Chan, K. M. Ho, J. Phys.: Condens. "
                         f"Matter 4, 6047 (1992). {hetero}: " + source)
    data["system"] = family.system
    ranges = ", ".join(f"{k} {v[0]:.2f}-{v[1]:.2f} Å" for k, v in
                       sorted(report.get("bond_ranges", {}).items())
                       if k != "C-C" and (family.base is None
                                          or set(k.split("-")) & set(family.heteroatoms)))
    if family.base is not None:
        ranges += f"; el resto, como {family.base}"
    data["validity"] = ("sistemas finitos de capa cerrada (SCC sin Ewald); enlaces dentro de lo "
                        f"muestreado ({ranges}); energías relativas solo dentro de una misma "
                        "composición (no se ajustaron energías de atomización); C-C como en "
                        "xu_carbon" + (f"; {family.validity_notes}" if family.validity_notes
                                       else ""))
    data["notes"] = ("C-C idéntico a xu_carbon. Niveles: todos los ocupados, el LUMO y el "
                     f"LUMO+1, con un desplazamiento común de {shift:.4f} eV entre el cero de Xu "
                     "y el vacío de GPAW. Sin interacción H-H. U de Hubbard: dε/dn del átomo "
                     "libre con GPAW (PBE), igual a DFTB mio.")
    xu_source = "Xu 1992 (idéntico a xu_carbon)"
    fit_source = f"ajustado ({source})"
    base_elements = set()
    if family.base is not None:
        base_data = read_parameter_file(family.base)
        base_elements = set(base_data["onsite"])
        data["reference"] += f". Resto: {family.base} (fijo)"
        data["notes"] = (f"Todo lo de {family.base} sin cambios; se ajustan solo "
                         f"{hetero} y sus pares, con el desplazamiento de niveles de "
                         f"{family.base} ({shift:.4f} eV). U de Hubbard y dipolos: átomo libre "
                         "con GPAW (PBE).")

    def origin(elements):
        if set(elements) == {"C"}:
            return xu_source
        if set(elements) <= base_elements:
            return f"{family.base} (fijo)"
        return fit_source

    for element, table in data["onsite"].items():
        for shell, value in table.items():
            table[shell] = {"value": value, "unit": "eV", "source": origin([element])}
    for entry in data["hopping"]:
        entry["unit"] = "eV"
        entry["source"] = origin(entry["pair"])
    data["repulsive"]["unit"] = "eV"
    pairs = ", ".join(f"{a}-{b}" for a, b in family.pairs)
    data["repulsive"]["source"] = f"C-C: Xu 1992 (embebida); pares {pairs}: " + fit_source
    fixed_terms = len(family.base_terms()) if family.base is not None else 0
    if fixed_terms:
        data["repulsive"]["terms"][:fixed_terms] = base_data["repulsive"]["terms"]
        data["repulsive"]["source"] = (f"términos de {family.base} (fijos); pares {pairs}: "
                                       + fit_source)
    for term in data["repulsive"]["terms"][fixed_terms:]:
        if term["type"] == "acute_angle":
            term["unit"] = "eV"
            term["source"] = ("corrección de ángulos agudos (anillos de tres miembros), " +
                              fit_source)
        if term["type"] == "pair":
            for entry in term["pairs"]:
                entry["unit"] = "eV"
                entry["source"] = fit_source
                pair = "-".join(sorted(entry["pair"]))
                if pair in report.get("bond_ranges", {}):
                    entry["fitted_range_angstrom"] = report["bond_ranges"][pair]
    refs = [Path(r) for r in references]
    data["fit"] = {"references": refs[0].name if len(refs) == 1 else [r.name for r in refs],
                   "references_sha256": _sha256(refs[0]) if len(refs) == 1
                   else [_sha256(r) for r in refs],
                   "parameters": dict(zip(family.parameter_names(), map(float, x), strict=True)),
                   "level_shift_eV": shift, **report}
    if family.held_out:
        data["fit"]["held_out"] = dict(family.held_out)
    data["onsite_dipole"] = {el: {"value": d, "unit": "Å",
                                  "source": "GPAW aeatom PBE, |⟨ns|r|np⟩| del átomo libre"}
                             for el, d in model.onsite_dipole.items()}
    data["hubbard_u"] = {el: {"value": u, "unit": "eV",
                              "source": "GPAW aeatom PBE, dε/dn del nivel de valencia"}
                         for el, u in model.hubbard_u.items()}
    if family.base is not None and base_data.get("extra_polarizability"):
        data["extra_polarizability"] = {
            el: dict(v, source=f"de {family.base} (no reajustada)")
            for el, v in base_data["extra_polarizability"].items()}
    return data


def run(family: XuFamily, references: Sequence[Path], out: Path, verbose: bool = True,
        workers: int = 4, x0: Optional[np.ndarray] = None, hessians: Optional[Path] = None,
        hessian_weight: float = 0.1) -> dict:
    """Fit ``family`` to the references (one or several files) and write the set.

    ``hessians``: a file of :mod:`tbkit.recipes.frequency_references`; its
    Hessians join forces and energies in the joint fit (curvature, i.e. the
    frequencies), weighted by ``hessian_weight`` per eV/Å²."""
    targets = load_hessians(hessians) if hessians else []
    structures, settings = [], {}
    for path in references:
        items, settings = load_references(path)
        structures += items
    for s in structures:
        if s.group in family.held_out:
            s.role = "test"
    train = [s for s in structures if s.role == "train"]
    result = fit_levels(family, train, x0)
    electronic = family.build_model(result.x)
    _, shift, rms = level_residuals(electronic, train, family.fixed_shift())
    if verbose:
        print(f"Niveles: {result.message}; desplazamiento {shift:.3f} eV", flush=True)
    joint, info, info0 = fit_joint(family, train, result.x, workers=workers, verbose=verbose,
                                   hessians=targets, hessian_weight=hessian_weight)
    x = joint.x
    shift = info["shift"]
    model = family.build_model(x, family.repulsion_from_coefficients(info["coefficients"]))
    report = {"stage1_level_rms": float(np.sqrt(np.mean(np.square(list(rms.values()))))),
              "joint_start": {k: info0[k] for k in ("level_rms", "force_rms", "energy_rms")},
              "level_rms_train": info["level_rms"], "force_rms": info["force_rms"],
              "energy_rms": info["energy_rms"], "optimiser": str(joint.message),
              "hessian_rms": info.get("hessian_rms"),
              "bond_ranges": bond_ranges(train)}
    if verbose:
        for name, before, after in zip(family.parameter_names(), result.x, x, strict=True):
            print(f"  {name:30s} {before:9.4f} → {after:9.4f}")
    validation = validate(family, model, structures, shift)
    if verbose:
        print(validation_table(validation))
    report["validation"] = validation
    if targets:
        report["frequencies"] = frequency_validation(model, targets)
        report["hessian_weight"] = hessian_weight
        if verbose:
            for name, entry in report["frequencies"].items():
                hits = f"  ⚠ en cola: {'; '.join(entry['tail_hits'])}" if entry["tail_hits"] \
                    else ""
                print(f"  {name:12s} RMS {entry['rms']:6.0f} cm⁻¹, máx {entry['max_error']:6.0f}"
                      f"{hits}")
    data = parameter_file(family, model, x, shift, report, references, settings)
    if hessians:
        data["fit"]["hessians"] = Path(hessians).name
        data["fit"]["hessians_sha256"] = _sha256(hessians)
    Path(out).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return data
