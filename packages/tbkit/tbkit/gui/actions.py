"""What the GUI computes, as plain functions without Qt.

Every page of the window calls one of these in a worker thread and draws what
comes back (arrays, dicts, tables). Keeping them free of Qt means they are
tested like the rest of tbkit, and the window only lays out and draws.

The ground state is computed once per (structure, model, charge, SCC) and
shared: levels, DOS, PDOS, charges and orbitals all come from the same
solution, self-consistent when the model says so (``TBModel.scc``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms

from ..analysis import atomic_charges, bond_orders, dos, orbital_on_grid, pdos
from ..hamiltonian import System
from ..kpoints import band_path, gamma, mesh
from ..params import TBModel, load_parameters, pi_model, xu_carbon
from ..solver import Solution, solve

#: Built-in models the window offers, name -> (label, loader).
MODELS = {
    "pi": ("π (Hückel, t = −2,7 eV)", lambda: pi_model(t=-2.7)),
    "sp3": ("Xu: carbono sp³ con repulsión", xu_carbon),
    "chn": ("C/H/N (xu_chn, SCC)", lambda: load_parameters("xu_chn")),
    "chno": ("C/H/N/O (xu_chno, SCC)", lambda: load_parameters("xu_chno")),
    "chnob": ("C/H/N/O + B (xu_chnob, SCC)", lambda: load_parameters("xu_chnob")),
    "chnos": ("C/H/N/O + S (xu_chnos, SCC)", lambda: load_parameters("xu_chnos")),
    "chnop": ("C/H/N/O + P (xu_chnop, SCC)", lambda: load_parameters("xu_chnop")),
}


def load_model(name_or_path: str) -> TBModel:
    """A built-in model by name, or any parameter file by path."""
    if name_or_path in MODELS:
        return MODELS[name_or_path][1]()
    return load_parameters(name_or_path)


def suggest_model(atoms: Atoms) -> str:
    """The smallest built-in set that covers the elements of ``atoms``."""
    elements = set(atoms.get_chemical_symbols())
    for name, covers in (("sp3", {"C"}), ("chn", {"C", "H", "N"}),
                         ("chno", {"C", "H", "N", "O"}), ("chnob", {"C", "H", "N", "O", "B"}),
                         ("chnos", {"C", "H", "N", "O", "S"}),
                         ("chnop", {"C", "H", "N", "O", "P"})):
        if elements <= covers:
            return name
    return "pi"


def check_model(atoms: Atoms, model: TBModel, scc: Optional[bool] = None) -> list[str]:
    """Reasons the model cannot treat ``atoms`` (empty list: it can).

    ``scc`` overrides the model's choice, as the window's selector does: a set
    fitted with SCC may be run without it on purpose (a periodic structure),
    never by default."""
    problems = []
    missing = sorted(set(atoms.get_chemical_symbols()) - set(model.orbitals))
    if missing:
        problems.append(f"El modelo no tiene parámetros para: {', '.join(missing)}.")
    if (model.scc if scc is None else scc) and any(atoms.get_pbc()):
        problems.append("SCC y estructura periódica: SCC necesita un sistema finito (no hay "
                        "suma de Ewald). Pon SCC en «no» a sabiendas (el modelo se ajustó "
                        "con SCC) o usa un fragmento finito.")
    return problems


def structure_summary(atoms: Atoms) -> dict:
    pbc = atoms.get_pbc()
    return {"formula": atoms.get_chemical_formula(), "atoms": len(atoms),
            "periodic": "no" if not pbc.any() else "".join("xyz"[i] for i in range(3) if pbc[i]),
            "elements": sorted(set(atoms.get_chemical_symbols()))}


# --------------------------------------------------------------------------
# Ground state
# --------------------------------------------------------------------------

@dataclass
class GroundState:
    solution: Solution
    charges: dict[int, float]
    scc: bool
    iterations: int = 0
    converged: bool = True
    info: dict = field(default_factory=dict)


def ground_state(atoms: Atoms, model: TBModel, charge: float = 0.0, kT: float = 0.01,
                 kmesh: int = 24, scc: Optional[bool] = None) -> GroundState:
    """Solve once: SCC when the model (or ``scc``) asks for it, else one diagonalisation."""
    problems = check_model(atoms, model, scc)
    if problems:
        raise ValueError(" ".join(problems))
    system = System.build(atoms, model)
    use_scc = model.scc if scc is None else scc
    if use_scc:
        from ..scc import self_consistent

        result = self_consistent(system, charge=charge, kT=kT)
        state = GroundState(result.solution, result.charges, True, result.iterations,
                            result.converged)
    else:
        kpts = mesh(atoms, kmesh) if system.periodic else gamma()
        solution = solve(system, *kpts, charge=charge, kT=kT)
        state = GroundState(solution, atomic_charges(solution), False)
    solution = state.solution
    homo, lumo = solution.homo_lumo()
    state.info = {"homo": homo, "lumo": lumo, "gap": solution.gap(), "fermi": solution.fermi,
                  "electrons": solution.electrons, "orbitals": solution.system.basis.size
                  if hasattr(solution.system.basis, "size") else None}
    return state


def levels_table(state: GroundState, around: int = 10) -> list[tuple[int, float, float, str]]:
    """``(index, energy, occupation, label)`` for the levels around the gap (Γ, spin 0)."""
    solution = state.solution
    energies = solution.energies[0, 0]
    occupations = solution.occupations[0, 0]           # 2 per level when spin-paired
    homo = int(np.flatnonzero(occupations > 1e-3).max()) if (occupations > 1e-3).any() else -1
    low, high = max(0, homo - around + 1), min(len(energies), homo + around + 1)
    rows = []
    for i in range(low, high):
        label = "HOMO" if i == homo else "LUMO" if i == homo + 1 else \
            f"HOMO−{homo - i}" if i < homo else f"LUMO+{i - homo - 1}"
        rows.append((i, float(energies[i]), float(occupations[i]), label))
    return rows


def dos_curves(state: GroundState, sigma: float = 0.1, by: Optional[str] = "element",
               window: tuple[float, float] = (-15.0, 10.0), points: int = 1500) -> dict:
    """Total DOS and (optionally) PDOS against E − E_F."""
    solution = state.solution
    grid = np.linspace(solution.fermi + window[0], solution.fermi + window[1], points)
    _, total = dos(solution, grid, sigma)
    out = {"energy": grid - solution.fermi, "total": total, "projected": {}}
    if by:
        _, out["projected"] = pdos(solution, by, grid, sigma)
    return out


def band_curves(atoms: Atoms, model: TBModel, path: Optional[str] = None,
                npoints: int = 200, kT: float = 0.01, kmesh: int = 24) -> dict:
    """Bands along a path of a periodic structure, relative to E_F of a mesh."""
    from ..analysis import bands

    system = System.build(atoms, model)
    if not system.periodic:
        raise ValueError("Las bandas necesitan una estructura periódica.")
    bandpath = band_path(atoms, path, npoints)
    energies = bands(system, bandpath)
    fermi = solve(system, *mesh(atoms, kmesh), kT=kT).fermi
    x, xticks, labels = bandpath.get_linear_kpoint_axis()
    return {"x": x, "energies": energies[0] - fermi, "ticks": xticks, "labels": list(labels)}


def orbital_grid(state: GroundState, band: int, spacing: float = 0.25) -> dict:
    """One eigenstate on a grid (for an isosurface)."""
    origin, steps, values = orbital_on_grid(state.solution, band, spacing=spacing)
    return {"origin": [float(v) for v in origin], "spacing": [float(v) for v in np.diag(steps)],
            "values": values,
            "energy": float(state.solution.energies[0, 0, band])}


def bonds_of(atoms: Atoms, factor: float = 1.2) -> list[tuple[int, int]]:
    """Bonds to draw: closer than ``factor`` times the covalent radii."""
    from ase.data import covalent_radii
    from ase.neighborlist import neighbor_list

    radii = covalent_radii[atoms.numbers]
    ii, jj, dd = neighbor_list("ijd", atoms, factor * 2 * radii.max(), self_interaction=False)
    return [(int(i), int(j)) for i, j, d in zip(ii, jj, dd, strict=True)
            if i < j and d < factor * (radii[i] + radii[j])]


def bond_order_table(state: GroundState, threshold: float = 0.1):
    return sorted(bond_orders(state.solution, threshold).items())


def read_structure(path: str | Path) -> Atoms:
    from ase.io import read

    atoms = read(str(path))
    if not any(atoms.get_pbc()) and atoms.cell.rank == 0:
        atoms.pbc = False
    return atoms


# --------------------------------------------------------------------------
# Magnetism (mean-field Hubbard)
# --------------------------------------------------------------------------

GUESSES = {"antiferro": "antiferro (subredes)", "ferro": "ferro", "random": "aleatorio",
           "paramagnetic": "paramagnético"}


def _hubbard_system(atoms, model, kmesh):
    system = System.build(atoms, model)
    kpts, weights = mesh(atoms, kmesh) if system.periodic else gamma()
    return system, kpts, weights


def hubbard_solution(atoms: Atoms, model: TBModel, U: Optional[float] = None,
                     charge: float = 0.0, kT: float = 0.005, field_tesla: float = 0.0,
                     guess: str = "antiferro", kmesh: int = 24, sigma: float = 0.05) -> dict:
    """One mean-field solution: moments, M, energy, m(E) and the spin-resolved DOS.

    ``U`` None takes the model's ``hubbard_u``. Moments are an order parameter
    of the approximation, not a correlated ground state (Lieb's theorem is the
    check for bipartite lattices)."""
    from ..hubbard import magnetization_vs_energy, mean_field, tesla_to_ev

    problems = [p for p in check_model(atoms, model, scc=False)]
    if problems:
        raise ValueError(" ".join(problems))
    system, kpts, weights = _hubbard_system(atoms, model, kmesh)
    result = mean_field(system, U=U, charge=charge, kpts=kpts, weights=weights, kT=kT,
                        field=tesla_to_ev(field_tesla), guess=guess)
    grid = np.linspace(-6.0, 6.0, 1200)
    energy, m, dm = magnetization_vs_energy(result, grid, sigma)
    absolute = grid + result.solution.fermi
    _, up = dos(result.solution, absolute, sigma, spin=0)
    _, down = dos(result.solution, absolute, sigma, spin=1)
    moments = result.moments
    return {"magnetization": result.magnetization, "energy": result.energy,
            "converged": result.converged, "iterations": result.iterations,
            "fermi": result.solution.fermi, "gap": result.solution.gap(),
            "moments": np.array([moments.get(i, 0.0) for i in range(len(atoms))]),
            "grid": grid, "m_of_E": m, "dm_dE": dm, "dos_up": up, "dos_down": down,
            "U": float(np.max(result.U)), "guess": guess}


def compare_guesses(atoms: Atoms, model: TBModel, guesses=("antiferro", "ferro", "paramagnetic"),
                    **kwargs) -> list[dict]:
    """Solve from several starting patterns: the lowest energy is the mean-field state."""
    rows = []
    for guess in guesses:
        try:
            out = hubbard_solution(atoms, model, guess=guess, **kwargs)
            rows.append({"guess": guess, "energy": out["energy"], "M": out["magnetization"],
                         "max_moment": float(np.abs(out["moments"]).max()),
                         "converged": out["converged"], "note": ""})
        except ValueError as error:        # e.g. antiferro on a non-bipartite lattice
            rows.append({"guess": guess, "energy": np.nan, "M": np.nan, "max_moment": np.nan,
                         "converged": False, "note": str(error)})
    lowest = np.nanmin([r["energy"] for r in rows]) if rows else np.nan
    for r in rows:
        r["delta"] = r["energy"] - lowest
    return rows


def field_sweep(atoms: Atoms, model: TBModel, tesla, U: Optional[float] = None,
                charge: float = 0.0, kT: float = 0.005, guess: str = "antiferro",
                kmesh: int = 24) -> dict:
    """M against the Zeeman field; each point starts from the previous one (one branch)."""
    from ..hubbard import magnetization_vs_field, tesla_to_ev

    system, kpts, weights = _hubbard_system(atoms, model, kmesh)
    tesla = np.asarray(tesla, dtype=float)
    _, m, _, _ = magnetization_vs_field(system, [tesla_to_ev(t) for t in tesla], U=U,
                                        charge=charge, kpts=kpts, weights=weights, kT=kT,
                                        guess=guess)
    chi = np.gradient(m, tesla) if len(tesla) > 1 else np.full(1, np.nan)
    return {"tesla": tesla, "M": m, "chi_per_tesla": chi}


def doping_sweep(atoms: Atoms, model: TBModel, charges, U: Optional[float] = None,
                 kT: float = 0.005, guess: str = "antiferro", kmesh: int = 24) -> dict:
    """M against added charge (e; positive removes electrons)."""
    from ..hubbard import magnetization_vs_doping

    system, kpts, weights = _hubbard_system(atoms, model, kmesh)
    x, m, _ = magnetization_vs_doping(system, charges=charges, U=U, kpts=kpts, weights=weights,
                                      kT=kT, guess=guess)
    return {"charge": x, "M": m}


# --------------------------------------------------------------------------
# Geometry and vibrations
# --------------------------------------------------------------------------

def _needs_repulsion(model: TBModel):
    if model.repulsive is None:
        raise ValueError(f"«{model.name}» no tiene parte repulsiva: sin energías ni fuerzas "
                         "(el modelo π no relaja ni da fonones). Usa sp3, chn, chno…")


def relax_structure(atoms: Atoms, model: TBModel, fmax: float = 0.02, steps: int = 500,
                    kT: float = 0.02, kmesh: int = 8, scc: Optional[bool] = None) -> dict:
    """BFGS with the model's forces (cell fixed); returns the new structure too."""
    from ase.optimize import BFGS

    from ..calculator import TBCalculator

    _needs_repulsion(model)
    problems = check_model(atoms, model, scc)
    if problems:
        raise ValueError(" ".join(problems))
    moved = atoms.copy()
    moved.calc = TBCalculator(model, kpts=kmesh, kT=kT, scc=scc)
    start = float(moved.get_potential_energy())
    trajectory = [start]
    optimizer = BFGS(moved, logfile="-")            # steps to stdout: the window's terminal
    optimizer.attach(lambda: trajectory.append(float(moved.get_potential_energy())))
    converged = bool(optimizer.run(fmax=fmax, steps=steps))
    forces = moved.get_forces()
    result = moved.copy()
    result.calc = None
    shift = np.linalg.norm(result.get_positions() - atoms.get_positions(), axis=1)
    return {"atoms": result, "converged": converged, "steps": optimizer.get_number_of_steps(),
            "energy": float(moved.get_potential_energy()), "energy_start": start,
            "max_force": float(np.linalg.norm(forces, axis=1).max()),
            "max_displacement": float(shift.max()), "trajectory": np.array(trajectory)}


def vibration_modes(atoms: Atoms, model: TBModel, kT: float = 0.02, kmesh: int = 12,
                    scc: Optional[bool] = None, sigma: float = 10.0) -> dict:
    """Γ modes with the Hessian (``tbkit.modes``), a table and the vibrational DOS."""
    from ..modes import participation, vibrational_dos, vibrations

    _needs_repulsion(model)
    problems = check_model(atoms, model, scc)
    if problems:
        raise ValueError(" ".join(problems))
    vib = vibrations(atoms, model, kmesh=kmesh, kT=kT, scc=scc)
    shares = participation(vib)
    rows = []
    for index, frequency in enumerate(vib.frequencies):
        parts = ", ".join(f"{name} {share[index]:.0%}" for name, share in shares.items()
                          if share[index] >= 0.05)
        rows.append((index, float(frequency), parts))
    return {"vibrations": vib, "rows": rows, "vdos": vibrational_dos(vib, sigma=sigma),
            "warnings": list(vib.warnings)}


# --------------------------------------------------------------------------
# Spectra: Raman, resonant Raman, IR
# --------------------------------------------------------------------------

def qe_phonons(atoms: Atoms, path: str, kind: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """``(frequencies, L)`` from a dynmat.x/matdyn.x mode file, for ``atoms`` (same order)."""
    from ..qe import modes_for_raman, read_qe_modes

    modes = read_qe_modes(path)
    return modes.frequencies, modes_for_raman(modes, atoms.get_masses(), kind)


def _groups_rows(groups, key="activity"):
    return [(float(g["frequency_cm1"]), int(g["degeneracy"]), float(g[key]),
             float(g.get("depolarization", np.nan))) for g in groups]


def raman_spectrum(atoms: Atoms, model: TBModel, phonons=None, laser_nm: float = 532.0,
                   temperature_k: float = 300.0, fwhm: float = 8.0, kT: float = 0.01,
                   kmesh: int = 12) -> dict:
    """Non-resonant Raman: active sets and the broadened spectrum (cross-section factors)."""
    from ..raman import raman, spectrum

    result = raman(atoms, model, kmesh=kmesh, kT=kT, phonons=phonons)
    grid = np.arange(0.0, max(float(result.frequencies.max()), 100.0) + 200.0, 0.5)
    grid, intensity = spectrum(result, grid, fwhm, laser_nm, temperature_k)
    return {"rows": _groups_rows(result.groups()), "grid": grid, "intensity": intensity,
            "alpha": result.alpha, "method": result.method, "warnings": list(result.warnings)}


def resonant_spectrum(atoms: Atoms, model: TBModel, lasers_ev, eta: float = 0.1, phonons=None,
                      kT: float = 0.01, kmesh: int = 24) -> dict:
    """Resonant Raman at each laser energy: activities per mode and laser."""
    from ..resonance import resonant_raman

    result = resonant_raman(atoms, model, np.asarray(lasers_ev, dtype=float), eta=eta,
                            kmesh=kmesh, kT=kT, phonons=phonons)
    strongest = result.activities.max() or 1.0
    keep = [k for k in range(len(result.frequencies))
            if result.activities[:, k].max() >= 1e-3 * strongest]
    return {"result": result, "lasers": result.lasers, "frequencies": result.frequencies,
            "activities": result.activities, "active": keep, "method": result.method,
            "warnings": list(result.warnings)}


def ir_spectrum_of(atoms: Atoms, model: TBModel, phonons=None, fwhm: float = 10.0,
                   kT: float = 0.01) -> dict:
    """IR intensities (km/mol) with the model's dipole, and the broadened absorption."""
    from ..infrared import infrared, ir_spectrum

    result = infrared(atoms, model, kT=kT, phonons=phonons)
    grid, absorption = ir_spectrum(result, fwhm=fwhm)
    rows = [(float(g["frequency_cm1"]), int(g["degeneracy"]), float(g["intensity_km_mol"]),
             np.nan) for g in result.groups()]
    return {"rows": rows, "grid": grid, "absorption": absorption,
            "dipole_debye": float(np.linalg.norm(result.dipole) / 0.20819434),
            "warnings": list(result.warnings)}


def write_csv(path: str | Path, columns: dict[str, np.ndarray]) -> Path:
    """A plain CSV (header + columns), for spectra and tables."""
    path = Path(path)
    names = list(columns)
    data = np.column_stack([np.asarray(columns[n], dtype=float) for n in names])
    np.savetxt(path, data, delimiter=",", header=",".join(names), comments="")
    return path


# --------------------------------------------------------------------------
# Graphene: G, 2D, 2D' by (double) resonance
# --------------------------------------------------------------------------

GRAPHENE_PHONONS = {"gpaw": "GPAW (PBE, incluidas)", "xu": "modelo de Xu (se calculan)"}


def graphene_spectra(lasers_ev, phonons: str = "gpaw", gamma: float = 0.1, dk: float = 0.01,
                     dq: float = 0.03, workers: int = 1, fwhm: float = 10.0) -> dict:
    """G, 2D and 2D′ of pristine graphene per laser, and the 2D dispersion (cm⁻¹/eV).

    π electrons with t(d) and the chosen force constants; the resonance
    energies are the π model's. ``phonons``: ``gpaw``, ``xu`` or a JSON path."""
    from ..graphene import graphene_raman, load_phonons

    lasers = np.atleast_1d(np.asarray(lasers_ev, dtype=float))
    results = graphene_raman(lasers, load_phonons(phonons), gamma=gamma, dk=dk, dq=dq,
                             workers=workers, fwhm=fwhm)
    positions = np.array([r["2D_position"] for r in results])
    slope = float(np.polyfit(lasers, positions, 1)[0]) if len(lasers) > 1 else float("nan")
    return {"results": results, "lasers": lasers, "dispersion_2d": slope,
            "rows": [(r["laser_ev"], r["g_frequency"], r["2D_position"], r["2D'_position"],
                      r["2D_intensity"] / r["g_intensity"]) for r in results]}


# --------------------------------------------------------------------------
# Reproducible records
# --------------------------------------------------------------------------

def plain(value):
    """Only what a JSON record can hold: numbers, text, arrays, lists and dicts of them
    (solutions, Vibrations and other objects are dropped)."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items() if _keep(v)}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value if _keep(v)]
    if isinstance(value, np.ndarray):
        return value.real.tolist() if np.iscomplexobj(value) else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _keep(value) -> bool:
    return isinstance(value, (dict, list, tuple, np.ndarray, np.generic, int, float, str,
                              bool, type(None)))


def record(atoms: Atoms, model: TBModel, task: str, settings: dict, results: dict) -> dict:
    """The record of one calculation made in the window (``tbkit.record`` format)."""
    from ..record import make_record

    return make_record(atoms, model, task, plain(settings), plain(results))


def open_record(path: str | Path) -> tuple[Atoms, TBModel, dict]:
    """Structure, model and the whole record back from a record file."""
    import io
    import json

    from ase.io import read

    from ..params import model_from_dict

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    atoms = read(io.StringIO(data["structure"]), format="extxyz")
    return atoms, model_from_dict(data["model"]), data
