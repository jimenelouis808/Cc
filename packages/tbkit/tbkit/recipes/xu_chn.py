"""Fit H and N to GPAW on top of Xu's carbon: the ``xu_chn`` parameter set.

Carbon-carbon is Xu, Wang, Chan and Ho's model, untouched (on-site
energies, hoppings and embedded repulsion): it sets the energy zero and
everything else is fitted around it. The fit adds

* on-site energies of H (s) and N (s, p);
* Slater-Koster hoppings C-H, N-H, C-N and N-N, in Xu's own GSP form
  ``v0 (r0/d)^n exp{n[-(d/rc)^nc + (r0/rc)^nc]}`` with ``nc = 6.5`` and ``rc``
  and the smooth tail scaled from Xu's by ``r0_pair / r0_CC``; ``v0`` per bond
  type and one ``n`` per pair are fitted;
* Hubbard U of H, C and N: not fitted, computed as ``dε/dn`` of the free
  atom's valence level with GPAW's all-electron atom (PBE, spin-paired) --
  the DFTB definition; the values agree with DFTB's mio set to 1e-4 Ha;
* pair repulsions C-H, N-H, C-N, N-N, ``Σ_k c_k (rc - d)^k`` (k = 3...7).

The ground state is self-consistent (DFTB2-like, ``scc: true``): charge
transfer between C and N depends on the environment (graphitic, pyridinic,
pyrrolic), which a fixed on-site energy cannot follow.

Stage 1 (electronic): least squares on the Kohn-Sham levels of the
training structures -- all occupied valence levels and the LUMO, the
frontier ones weighted double -- with one common shift between model and
DFT levels (both are absolute for finite molecules; the shift is Xu's
arbitrary zero). Stage 2 (repulsion): with the electronic part fixed, the
repulsion must supply ``F_DFT - F_TB`` and, within each molecule's set of
geometries, ``E_DFT - E_TB`` up to a constant. Both are linear in the
polynomial coefficients: one linear least-squares solve.

Run::

    python -m tbkit.recipes.xu_chn REFERENCES.json OUT.json

``REFERENCES.json`` from :mod:`tbkit.recipes.chn_references`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Optional

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

#: Per pair: reference length r0 (a typical single bond, Å), the bonds it
#: has, and the repulsion cutoff (beyond the longest stretched bond of the
#: training data, short of second neighbours).
PAIRS = {
    ("C", "H"): {"r0": 1.09, "bonds": (("H", "C", "sss"), ("H", "C", "sps")), "rc_rep": 1.75},
    ("N", "H"): {"r0": 1.01, "bonds": (("H", "N", "sss"), ("H", "N", "sps")), "rc_rep": 1.65},
    ("C", "N"): {"r0": 1.47, "bonds": (("C", "N", "sss"), ("C", "N", "sps"), ("N", "C", "sps"),
                                       ("C", "N", "pps"), ("C", "N", "ppp")), "rc_rep": 2.2},
    ("N", "N"): {"r0": 1.45, "bonds": (("N", "N", "sss"), ("N", "N", "sps"), ("N", "N", "pps"),
                                       ("N", "N", "ppp")), "rc_rep": 2.1},
}
POWERS = (3, 4, 5, 6, 7)

#: Hubbard U (eV) from GPAW's all-electron atom, PBE, spin-paired,
#: U = dε/dn of the valence level by central differences (±0.05 e);
#: GPAW 25.7. DFTB mio: H 0.4195, C 0.3647, N 0.4309 Ha -- the same.
HUBBARD_U = {"H": 11.4154, "C": 9.9228, "N": 11.7247}

#: Harrison's universal η (Harrison, Electronic Structure, 1980) for the
#: starting guesses, V = η ħ²/(m d²) with ħ²/m = 7.62 eV Å².
_ETA = {"sss": -1.40, "sps": 1.84, "pps": 3.24, "ppp": -0.81}


def parameter_names() -> list[str]:
    names = ["onsite H s", "onsite N s", "onsite N p"]
    for (a, b), spec in PAIRS.items():
        for first, second, bond in spec["bonds"]:
            names.append(f"{a}-{b} {bond}({first},{second}) v0")
        names.append(f"{a}-{b} n")
    return names


def initial_guess() -> np.ndarray:
    """Free-atom level spacings (GPAW PBE) shifted onto Xu's C 2p; Harrison hoppings."""
    shift = 3.71 - (-5.289)             # Xu's C p minus GPAW's free-atom C 2p
    x = [-6.492 + shift, -18.4 + shift, -7.095 + shift]
    for spec in PAIRS.values():
        for _, _, bond in spec["bonds"]:
            x.append(_ETA[bond] * 7.62 / spec["r0"] ** 2)
        x.append(2.0)
    return np.array(x)


def _law(v0: float, n: float, r0: float):
    scale = r0 / XU_R0
    rm = XU_TAIL[1] * scale
    return Tail(GSP(v0, r0, n, NC, XU_RC * scale, rm), XU_TAIL[0] * scale, rm)


def build_model(x: np.ndarray, repulsive: Optional[object] = None) -> TBModel:
    """Xu's carbon plus H and N from the parameter vector ``x``."""
    base = model_from_dict(read_parameter_file("xu_carbon"))
    x = list(map(float, x))
    onsite = dict(base.onsite)
    onsite["H"] = {"s": x[0]}
    onsite["N"] = {"s": x[1], "p": x[2]}
    hopping = dict(base.hopping)
    k = 3
    for spec in PAIRS.values():
        values = x[k:k + len(spec["bonds"])]
        n = x[k + len(spec["bonds"])]
        for (first, second, bond), v0 in zip(spec["bonds"], values, strict=True):
            hopping[(first, second, bond)] = _law(v0, n, spec["r0"])
        k += len(spec["bonds"]) + 1
    xu = copy.copy(base.repulsive)
    xu.others = "ignore"
    if repulsive is None:
        repulsive = SumRepulsive((xu,))
    return TBModel(name="C/H/N: Xu (C-C) + H, N ajustados a GPAW",
                   orbitals={"C": ("s", "px", "py", "pz"), "N": ("s", "px", "py", "pz"),
                             "H": ("s",)},
                   onsite=onsite, hopping=hopping, valence={"C": 4.0, "N": 5.0, "H": 1.0},
                   hubbard_u=dict(HUBBARD_U), repulsive=repulsive, scc=True)


# --------------------------------------------------------------------------
# Stage 1: levels
# --------------------------------------------------------------------------

def _selection(ref: ReferenceStructure) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the compared levels and their weights."""
    n = ref.n_occupied
    indices = np.arange(n + 1)
    weights = np.ones(n + 1)
    weights[n - 1] = weights[n] = 2.0       # HOMO, LUMO
    return indices, weights


def model_levels(model: TBModel, ref: ReferenceStructure, kT: float = 0.01) -> np.ndarray:
    result = self_consistent(System.build(ref.atoms, model), kT=kT, tol=1e-7)
    if not result.converged:
        raise RuntimeError(f"{ref.label}: SCC sin converger")
    return np.sort(result.solution.energies[0, 0])


def level_residuals(model: TBModel, refs: list[ReferenceStructure]
                    ) -> tuple[np.ndarray, float, dict]:
    """Weighted residuals after the best common shift, the shift, and per-label RMS."""
    differences, weights, labels = [], [], []
    for ref in refs:
        indices, w = _selection(ref)
        tb = model_levels(model, ref)[indices]
        differences.append(tb - ref.levels[indices])
        weights.append(w)
        labels.append(ref.label)
    d = np.concatenate(differences)
    w = np.concatenate(weights)
    shift = float(np.sum(w ** 2 * d) / np.sum(w ** 2))
    rms = {label: float(np.sqrt(np.mean((diff - shift) ** 2)))
           for label, diff in zip(labels, differences, strict=True)}
    return w * (d - shift), shift, rms


def fit_levels(refs: list[ReferenceStructure], x0: Optional[np.ndarray] = None):
    x0 = initial_guess() if x0 is None else np.asarray(x0, dtype=float)

    def residuals(x):
        try:
            return level_residuals(build_model(x), refs)[0]
        except (RuntimeError, np.linalg.LinAlgError, ValueError):
            return np.full(sum(r.n_occupied + 1 for r in refs), 10.0)

    lower = np.full(len(x0), -np.inf)
    upper = np.full(len(x0), np.inf)
    names = parameter_names()
    for i, name in enumerate(names):
        if name.endswith(" n"):
            lower[i], upper[i] = 0.5, 6.0
    return least_squares(residuals, x0, bounds=(lower, upper), x_scale="jac")


# --------------------------------------------------------------------------
# Stage 2: repulsion
# --------------------------------------------------------------------------

def _pair_basis(atoms, pair: tuple[str, str], rc: float):
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


def fit_repulsion(model: TBModel, refs: list[ReferenceStructure], energy_weight: float = 3.0,
                  ridge: float = 1e-8) -> tuple[SumRepulsive, dict]:
    """Linear least squares for the pair coefficients (electronic part fixed)."""
    pairs = list(PAIRS)
    n_coef = len(POWERS)
    rows, targets = [], []
    groups: dict[str, list[int]] = {}
    energy_rows, energy_targets = [], []
    for ref in refs:
        e_tb, f_tb = electronic_energy_and_forces(model, ref)
        basis_e = np.zeros(len(pairs) * n_coef)
        basis_f = np.zeros((len(pairs) * n_coef, len(ref.atoms), 3))
        for p, pair in enumerate(pairs):
            e, f = _pair_basis(ref.atoms, pair, PAIRS[pair]["rc_rep"])
            basis_e[p * n_coef:(p + 1) * n_coef] = e
            basis_f[p * n_coef:(p + 1) * n_coef] = f
        rows.append(basis_f.reshape(len(basis_e), -1).T)
        targets.append((ref.forces - f_tb).ravel())
        groups.setdefault(ref.group, []).append(len(energy_rows))
        energy_rows.append(basis_e)
        energy_targets.append(ref.energy - e_tb)
    energy_rows = np.array(energy_rows)
    energy_targets = np.array(energy_targets)
    for members in groups.values():            # remove each molecule's constant
        energy_rows[members] -= energy_rows[members].mean(axis=0)
        energy_targets[members] -= energy_targets[members].mean()
    a = np.vstack(rows + [energy_weight * energy_rows])
    y = np.concatenate(targets + [energy_weight * energy_targets])
    active = np.abs(a).sum(axis=0) > 0
    lhs = a[:, active].T @ a[:, active] + ridge * np.eye(int(active.sum()))
    coefficients = np.zeros(a.shape[1])
    coefficients[active] = np.linalg.solve(lhs, a[:, active].T @ y)
    laws = {}
    from ..params import CutoffPolynomial

    for p, pair in enumerate(pairs):
        c = tuple(float(v) for v in coefficients[p * n_coef:(p + 1) * n_coef])
        laws[pair] = CutoffPolynomial(c, PAIRS[pair]["rc_rep"], POWERS[0])
    xu = copy.copy(model.repulsive.terms[0])
    residual = y - a @ coefficients
    n_force = sum(len(t) for t in targets)
    report = {"force_rms": float(np.sqrt(np.mean(residual[:n_force] ** 2))),
              "energy_rms": float(np.sqrt(np.mean((residual[n_force:] / energy_weight) ** 2)))}
    return SumRepulsive((xu, PairRepulsive(laws))), report


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

#: Mean static polarizabilities (Å³), experiment: T. M. Miller, "Atomic and
#: molecular polarizabilities", CRC Handbook of Chemistry and Physics. Only
#: molecules with a well-established value; a comparison, not a fit target.
EXPERIMENTAL_ALPHA = {"CH4": 2.593, "C2H6": 4.47, "C2H4": 4.252, "C6H6": 10.32, "N2": 1.7403}


def bond_lengths(atoms, cutoff: float = 1.75) -> dict[tuple[int, int], float]:
    """Bonded pairs (i < j) of a molecule: heavy-heavy within ``cutoff``, X-H within 1.3 Å."""
    symbols = atoms.get_chemical_symbols()
    ii, jj, dd = neighbor_list("ijd", atoms, cutoff)
    out = {}
    for i, j, d in zip(ii, jj, dd, strict=True):
        if i < j and (d < 1.3 or "H" not in (symbols[i], symbols[j])):
            out[(int(i), int(j))] = float(d)
    return out


def relaxed_bond_errors(model: TBModel, ref: ReferenceStructure, fmax: float = 0.01) -> dict:
    """Relax ``ref``'s geometry with the model; bond-length errors against DFT by bond type."""
    from ase.optimize import BFGS

    from ..calculator import TBCalculator

    atoms = ref.atoms.copy()
    atoms.calc = TBCalculator(model)
    BFGS(atoms, logfile=None).run(fmax=fmax, steps=300)
    symbols = atoms.get_chemical_symbols()
    reference = bond_lengths(ref.atoms)
    errors: dict[str, list[float]] = {}
    for (i, j), d_ref in reference.items():
        kind = "-".join(sorted((symbols[i], symbols[j])))
        errors.setdefault(kind, []).append(atoms.get_distance(i, j) - d_ref)
    return {kind: float(np.max(np.abs(v))) for kind, v in errors.items()}


def validate(model: TBModel, structures: list[ReferenceStructure], shift: float,
             relax: bool = True) -> dict:
    """Levels, relaxed bonds and polarizabilities for every relaxed ("eq") reference."""
    from ..optics import polarizability_linear_response

    report = {}
    for ref in structures:
        if not ref.label.endswith("/eq"):
            continue
        indices, _ = _selection(ref)
        levels = model_levels(model, ref)[indices] - shift
        entry = {"role": ref.role,
                 "level_rms": float(np.sqrt(np.mean((levels - ref.levels[indices]) ** 2))),
                 "gap_tb": float(levels[-1] - levels[-2]),
                 "gap_dft": float(ref.levels[ref.n_occupied] - ref.levels[ref.n_occupied - 1])}
        if relax:
            entry["bond_error_max"] = relaxed_bond_errors(model, ref)
        if ref.group in EXPERIMENTAL_ALPHA:
            alpha = polarizability_linear_response(System.build(ref.atoms, model))
            entry["alpha_mean"] = float(np.trace(alpha) / 3)
            entry["alpha_experiment"] = EXPERIMENTAL_ALPHA[ref.group]
        report[ref.group] = entry
    return report


def validation_table(report: dict) -> str:
    lines = [f"{'molécula':12s} {'rol':5s} {'RMS niv.':>8s} {'gap TB':>7s} {'gap DFT':>7s} "
             f"{'α TB':>6s} {'α exp':>6s}  error máx. de enlace (Å)"]
    for name, e in report.items():
        alpha = f"{e['alpha_mean']:6.2f} {e['alpha_experiment']:6.2f}" if "alpha_mean" in e \
            else f"{'':6s} {'':6s}"
        bonds = ", ".join(f"{k} {v:.3f}" for k, v in e.get("bond_error_max", {}).items())
        lines.append(f"{name:12s} {e['role']:5s} {e['level_rms']:8.3f} {e['gap_tb']:7.2f} "
                     f"{e['gap_dft']:7.2f} {alpha}  {bonds}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The parameter file
# --------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_file(model: TBModel, x: np.ndarray, shift: float, report: dict,
                   references: Path, settings: dict) -> dict:
    data = model_to_dict(model)
    source = (f"ajuste a GPAW ({settings.get('xc')}, {settings.get('mode')}/"
              f"{settings.get('basis')}, GPAW {settings.get('gpaw_version')}); receta "
              "tbkit.recipes.xu_chn")
    data["reference"] = ("C-C: C. H. Xu, C. Z. Wang, C. T. Chan, K. M. Ho, J. Phys.: Condens. "
                         "Matter 4, 6047 (1992). H y N: " + source)
    data["system"] = ("moléculas C/H/N de capa cerrada: hidrocarburos saturados, insaturados "
                      "y aromáticos; aminas, iminas, nitrilos, N piridínico y pirrólico")
    data["validity"] = ("sistemas finitos (SCC sin Ewald); enlaces C-H, N-H de 0.95 a 1.2 Å, "
                        "C-N y N-N de 1.1 a 1.6 Å; energías relativas dentro de una misma "
                        "composición (no se ajustaron energías de atomización)")
    data["notes"] = ("C-C idéntico a xu_carbon. Niveles: todos los ocupados y el LUMO, con un "
                     f"desplazamiento común de {shift:.4f} eV entre el cero de Xu y el vacío "
                     "de GPAW. Sin interacción H-H. U de Hubbard: dε/dn del átomo libre con "
                     "GPAW (PBE), igual a DFTB mio.")
    names = parameter_names()
    data["fit"] = {"references": references.name, "references_sha256": _sha256(references),
                   "parameters": dict(zip(names, map(float, x), strict=True)),
                   "level_shift_eV": shift, **report}
    data["hubbard_u"] = {el: {"value": u, "unit": "eV",
                              "source": "GPAW aeatom PBE, dε/dn del nivel de valencia"}
                         for el, u in HUBBARD_U.items()}
    return data


def run(references: Path, out: Path, verbose: bool = True) -> dict:
    structures, settings = load_references(references)
    train = [s for s in structures if s.role == "train"]
    result = fit_levels(train)
    electronic = build_model(result.x)
    _, shift, rms = level_residuals(electronic, train)
    if verbose:
        print(f"Niveles: {result.message}; desplazamiento {shift:.3f} eV")
        for name, value in zip(parameter_names(), result.x, strict=True):
            print(f"  {name:28s} {value:9.4f}")
    repulsive, report = fit_repulsion(electronic, train)
    model = build_model(result.x, repulsive)
    report["level_rms_train"] = float(np.sqrt(np.mean(np.square(list(rms.values())))))
    if verbose:
        print(f"RMS niveles {report['level_rms_train']:.3f} eV; fuerzas "
              f"{report['force_rms']:.3f} eV/Å; energías {report['energy_rms']:.3f} eV")
    validation = validate(model, structures, shift)
    if verbose:
        print(validation_table(validation))
    report["validation"] = validation
    data = parameter_file(model, result.x, shift, report, Path(references), settings)
    Path(out).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return data


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Ajusta H y N a GPAW sobre el C de Xu.")
    parser.add_argument("references", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    run(args.references, args.out)


if __name__ == "__main__":
    main()
