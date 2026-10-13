"""What the GUI computes, as plain functions without Qt.

Every page of the window calls one of these in a worker thread and draws what
comes back (arrays, dicts, tables). Keeping them free of Qt means they are
tested like the rest of tbkit, and the window only lays out and draws.

The ground state is computed once per (structure, model, charge, SCC) and
shared: levels, DOS, PDOS, charges and orbitals all come from the same
solution, self-consistent when the model says so (``TBModel.scc``).
"""

from __future__ import annotations

import json
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
    "chnose": ("C/H/N/O + Se (xu_chnose, SCC)", lambda: load_parameters("xu_chnose")),
    "tang": ("Carbono dependiente del entorno (Tang, ajustado a GPAW)",
             lambda: load_parameters("tang_carbon")),
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
                         ("chnop", {"C", "H", "N", "O", "P"}),
                         ("chnose", {"C", "H", "N", "O", "Se"})):
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
    if (model.scc if scc is None else scc) and any(atoms.get_pbc()) and \
            abs(np.linalg.det(np.asarray(atoms.get_cell()))) < 1e-6:
        problems.append("SCC periódico (Ewald) necesita una celda con volumen: da vacío a "
                        "las direcciones no periódicas.")
    return problems


def structure_summary(atoms: Atoms) -> dict:
    pbc = atoms.get_pbc()
    return {"formula": atoms.get_chemical_formula(), "atoms": len(atoms),
            "periodic": "no" if not pbc.any() else "".join("xyz"[i] for i in range(3) if pbc[i]),
            "elements": sorted(set(atoms.get_chemical_symbols()))}


# --------------------------------------------------------------------------
# Ground state
# --------------------------------------------------------------------------

def kmesh_note(atoms: Atoms, model: TBModel, kmesh) -> str:
    """One line for the job's log: the mesh, how many points, what their vectors weigh.

    The k count is per periodic AXIS, so a 3D cell at 24 is 24³ = 13 824 points; printed
    first, it is what the Procesos log shows while those points are diagonalised.
    """
    from ..solver import k_memory_bytes

    pbc = atoms.get_pbc()
    if not pbc.any():
        return "malla k: solo Γ (estructura finita)"
    axes = System.build(atoms, model).k_axes
    counts = [kmesh if a else 1 for a in axes]
    nk = int(np.prod(counts))
    norb = sum(len(model.orbitals[el]) for el in atoms.get_chemical_symbols()
               if el in model.orbitals)
    need = k_memory_bytes(nk, norb, not model.orthogonal)
    vacuum = [ax for ax, p, a in zip("xyz", pbc, axes, strict=True) if p and not a]
    note = (f" · {', '.join(vacuum)} periódico{'s' if len(vacuum) > 1 else ''} pero sin "
            "enlaces que crucen la celda: un solo punto k ahí (exacto)") if vacuum else ""
    return (f"malla k: {'×'.join(map(str, counts))} = {nk} puntos · {norb} orbitales · "
            f"~{need / 1e9:.2f} GB en autovectores{note}")


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
    print(kmesh_note(atoms, model, kmesh), flush=True)
    kpts = mesh(system, kmesh) if system.periodic else gamma()
    if use_scc:
        from ..progress import Progress
        from ..scc import self_consistent

        # The window's k mesh, not the SCC default: it used to be left out here, so a
        # periodic SCC ran on 12 points per axis whatever the control said.
        with Progress(None, f"SCC ({system.basis.size} orbitales)") as bar:
            result = self_consistent(
                system, charge=charge, kT=kT,
                kpts=kpts[0] if system.periodic else None,
                weights=kpts[1] if system.periodic else None,
                on_iteration=lambda n, change: bar.step(note=f"máx |Δq| {change:.1e} e"))
        state = GroundState(result.solution, result.charges, True, result.iterations,
                            result.converged)
    else:
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
    fermi = solve(system, *mesh(system, kmesh), kT=kT).fermi
    x, xticks, labels = bandpath.get_linear_kpoint_axis()
    return {"x": x, "energies": energies[0] - fermi, "ticks": xticks, "labels": list(labels)}


def orbital_grid(state: GroundState, band: int, spacing: float = 0.25) -> dict:
    """One eigenstate on a grid (for an isosurface)."""
    origin, steps, values = orbital_on_grid(state.solution, band, spacing=spacing)
    return {"origin": [float(v) for v in origin], "spacing": [float(v) for v in np.diag(steps)],
            "values": values,
            "energy": float(state.solution.energies[0, 0, band])}


def bonds_with_images(atoms: Atoms, factor: float = 1.2):
    """Bonds as ``(i, j, offset)``: j in the cell ``offset`` (integers along the cell
    vectors) from i. A bond with a non-zero offset crosses the cell boundary and must be
    drawn to the image of j, not across the cell to j itself."""
    from ase.data import covalent_radii
    from ase.neighborlist import neighbor_list

    radii = covalent_radii[atoms.numbers]
    ii, jj, dd, ss = neighbor_list("ijdS", atoms, factor * 2 * radii.max(),
                                   self_interaction=False)
    out = []
    for i, j, d, shift in zip(ii, jj, dd, ss, strict=True):
        if d >= factor * (radii[i] + radii[j]):
            continue
        # each bond appears twice (i→j with S, j→i with -S): keep one
        if i < j or (i == j and tuple(shift) > (0, 0, 0)):
            out.append((int(i), int(j), tuple(int(x) for x in shift)))
    return out


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


#: Set in the window's job process: called with the current positions while a relaxation
#: runs (``tbkit.gui.jobs``), so the window can draw them. None elsewhere.
FRAME_SINK = None


def orbital_count(atoms: Atoms, model: TBModel) -> int:
    """Orbitals of the model on this structure (the size of H)."""
    return sum(len(model.orbitals.get(s, ())) for s in atoms.get_chemical_symbols())


def memory_estimate_gb(atoms: Atoms, model: TBModel) -> float:
    """Peak memory of one dense diagonalisation, GB: H, S, the eigenvectors and LAPACK's
    workspace, complex (k ≠ 0), about six n × n matrices of 16 bytes. Forces, SCC and
    phonons keep a few more; this is the floor, not a ceiling."""
    n = orbital_count(atoms, model)
    return 6 * 16 * n * n / 1e9


def supercell(atoms: Atoms, repeat) -> Atoms:
    """The structure repeated ``(n1, n2, n3)`` times along its cell vectors (only along
    periodic axes: a non-periodic axis has no neighbour cell to repeat into)."""
    repeat = tuple(int(n) for n in repeat)
    if any(n < 1 for n in repeat):
        raise ValueError("Las repeticiones deben ser ≥ 1.")
    for axis, n in enumerate(repeat):
        if n > 1 and not atoms.pbc[axis]:
            raise ValueError(f"El eje {'abc'[axis]} no es periódico: no se puede repetir.")
    out = atoms.repeat(repeat)
    out.info.pop("periodicity", None)
    return out


def read_structure(path: str | Path) -> Atoms:
    """Any structure file (.xtl included); periodic axes deduced when a cell is declared
    non-periodic but bonds cross it (``tbkit.structure_io``)."""
    from ..structure_io import read_structure as read

    return read(path)


# --------------------------------------------------------------------------
# Magnetism (mean-field Hubbard)
# --------------------------------------------------------------------------

GUESSES = {"antiferro": "antiferro (subredes)", "ferro": "ferro", "random": "aleatorio",
           "paramagnetic": "paramagnético"}


def _hubbard_system(atoms, model, kmesh):
    system = System.build(atoms, model)
    kpts, weights = mesh(system, kmesh) if system.periodic else gamma()
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
    from ..progress import Progress

    print(kmesh_note(atoms, model, kmesh), flush=True)
    moved = atoms.copy()
    moved.calc = TBCalculator(model, kpts=kmesh, kT=kT, scc=scc)
    start = float(moved.get_potential_energy())
    trajectory = [start]
    optimizer = BFGS(moved, logfile="-")            # steps to stdout: the window's terminal
    optimizer.attach(lambda: trajectory.append(float(moved.get_potential_energy())))
    if FRAME_SINK is not None:                      # the window draws the geometry as it moves
        optimizer.attach(lambda: FRAME_SINK(moved.get_positions()))
    # A counter per BFGS step, so Procesos shows the step, the time per step and the force
    # left (the step limit is a ceiling, not an estimate: it converges sooner or not at all).
    bar = Progress(steps, "relajación")

    def report():
        force = float(np.linalg.norm(moved.get_forces(), axis=1).max())
        bar.step(note=f"E {trajectory[-1]:.4f} eV · Fmáx {force:.3f} eV/Å (meta {fmax:g})")

    optimizer.attach(report)
    with bar:
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
    from ..modes import vibrations

    _needs_repulsion(model)
    problems = check_model(atoms, model, scc)
    if problems:
        raise ValueError(" ".join(problems))

    vib = vibrations(atoms, model, kmesh=kmesh, kT=kT, scc=scc)
    return _modes_result(vib, sigma)


def modes_from_file(path: str | Path, sigma: float = 10.0) -> dict:
    """Modes saved before (``modes.npz``, or a ``hessian.npy`` with its structure, or their
    folder), as :func:`vibration_modes` returns them: nothing is recomputed but the
    diagonalisation."""
    from ..modes import Vibrations

    return _modes_result(Vibrations.load(path), sigma)


def modes_from_hessian(atoms: Atoms, hessian, masses=None, source: str = "",
                       sigma: float = 10.0) -> dict:
    """Modes from a Hessian in memory (a reopened project), as :func:`vibration_modes`."""
    from ..modes import Vibrations

    return _modes_result(Vibrations.from_hessian(atoms, np.asarray(hessian), masses, source),
                         sigma)


def _modes_result(vib, sigma: float = 10.0) -> dict:
    from ..modes import participation, vibrational_dos
    from ..sites import site_groups

    atoms = vib.atoms
    shares = participation(vib)
    groups = site_groups(atoms)
    on_sites = participation(vib, groups) if groups else {}
    rows = []
    for index, frequency in enumerate(vib.frequencies):
        parts = ", ".join(f"{name} {share[index]:.0%}" for name, share in shares.items()
                          if share[index] >= 0.05)
        rows.append((index, float(frequency), parts))
    return {"vibrations": vib, "rows": rows, "vdos": vibrational_dos(vib, sigma=sigma),
            "warnings": list(vib.warnings), "site_weights": on_sites,
            "site_sizes": {name: len(members) for name, members in groups.items()}}


def modes_by_site(result: dict, site: Optional[str]) -> list[tuple]:
    """Rows ``(mode, ω, share on the site, enrichment, participation)`` sorted by the share
    on ``site`` (largest first); ``site=None`` keeps the frequency order.

    The enrichment is the share over the site's fraction of atoms: 1 for a mode
    spread evenly, N/N_site for a mode entirely on the site."""
    vib = result["vibrations"]
    rows = result["rows"]
    if site is None:
        return [(index, freq, float("nan"), float("nan"), parts) for index, freq, parts in rows]
    share = result["site_weights"][site]
    fraction = result["site_sizes"][site] / len(vib.atoms)
    order = np.argsort(-share)
    return [(int(k), float(vib.frequencies[k]), float(share[k]), float(share[k] / fraction),
             rows[k][2]) for k in order]


def mode_with_other_model(atoms: Atoms, vib, index: int, other: TBModel, kmesh: int = 8,
                          kT: float = 0.02, scc: Optional[bool] = None) -> dict:
    """Frequency of mode ``index`` (its displacement pattern) under ``other``: the
    Rayleigh quotient of that model's force constants (``tbkit.sites``), two force
    calls. Same pattern, another model's springs: a cross-check of a mode (e.g. Xu
    against Tang on a 5-7 mode) without recomputing every mode."""
    from ..calculator import TBCalculator
    from ..sites import projected_frequency

    _needs_repulsion(other)
    problems = check_model(atoms, other, scc)
    if problems:
        raise ValueError(" ".join(problems))
    out = projected_frequency(vib.atoms, vib.modes[index],
                              lambda: TBCalculator(other, kpts=kmesh, kT=kT, scc=scc))
    return {"index": int(index), "frequency_cm1": float(vib.frequencies[index]),
            "other_cm1": out["frequency_cm1"], "other": other.name, "step_A": out["step_A"]}


# --------------------------------------------------------------------------
# SCC on / off
# --------------------------------------------------------------------------

def compare_scc(atoms: Atoms, model: TBModel, charge: float = 0.0, kT: float = 0.01,
                kmesh: int = 24) -> dict:
    """The ground state with and without self-consistent charges, side by side.

    SCC lets charge transfer cost electrostatic energy (Hubbard U, γ), so it
    reduces the charges a bare TB puts on polar bonds and moves levels with
    them. For a set fitted with SCC the «con» column is the valid one; the
    other shows how much the result leans on SCC."""
    missing = sorted(set(atoms.get_chemical_symbols()) - set(model.hubbard_u or {}))
    if missing:
        raise ValueError(f"«{model.name}» no tiene U para {', '.join(missing)}: sin U no hay SCC "
                         "(usa un conjunto xu_ch*).")
    states = {flag: ground_state(atoms, model, charge=charge, kT=kT, kmesh=kmesh, scc=flag)
              for flag in (False, True)}
    rows = []
    for key, label in (("gap", "gap (eV)"), ("homo", "HOMO (eV)"), ("lumo", "LUMO (eV)"),
                       ("fermi", "E_F (eV)")):
        off, on = states[False].info[key], states[True].info[key]
        off = float(off) if off is not None else float("nan")
        on = float(on) if on is not None else float("nan")
        rows.append((label, off, on, on - off))
    q = {flag: np.array([states[flag].charges[i] for i in range(len(atoms))]) for flag in states}
    rows.append(("|q| máx. (e)", float(np.abs(q[False]).max()), float(np.abs(q[True]).max()),
                 float(np.abs(q[True]).max() - np.abs(q[False]).max())))
    symbols = atoms.get_chemical_symbols()
    charges = [(i, symbols[i], float(q[False][i]), float(q[True][i]),
                float(q[True][i] - q[False][i])) for i in range(len(atoms))]
    return {"rows": rows, "charges": charges, "iterations": states[True].iterations,
            "converged": states[True].converged, "states": states}


# --------------------------------------------------------------------------
# Phonons in the whole Brillouin zone (phonopy)
# --------------------------------------------------------------------------

def phonopy_available() -> bool:
    try:
        import phonopy  # noqa: F401
    except ImportError:
        return False
    return True


def bz_phonons(atoms: Atoms, model: TBModel, supercell=(4, 4, 4), delta: float = 0.01,
               kmesh: int = 12, kT: float = 0.02, scc: Optional[bool] = None,
               temperature: float = 300.0, cache_dir: Optional[str] = None) -> dict:
    """Dispersion, DOS per element, Γ symmetry labels and harmonic thermodynamics."""
    from .. import phonopy_bridge as pb

    _needs_repulsion(model)
    problems = check_model(atoms, model, scc)
    if problems:
        raise ValueError(" ".join(problems))
    result = pb.phonopy_phonons(atoms, model, supercell=supercell, delta=delta, kmesh=kmesh,
                                kT=kT, scc=scc, cache_dir=cache_dir)
    return {"dispersion": pb.dispersion(result, atoms), "dos": pb.phonon_dos(result, atoms),
            "irreps": pb.gamma_irreps(result), "point_group": pb.point_group(result),
            "thermal": pb.thermal(result, atoms, [temperature])[0],
            "supercell": result.supercell, "displacements": result.displacements,
            "warnings": result.warnings}


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
                   kmesh: int = 12, scale: bool = False) -> dict:
    """Non-resonant Raman: active sets and the broadened spectrum (cross-section factors)."""
    from ..raman import raman, spectrum

    result, factor = scaled(raman(atoms, model, kmesh=kmesh, kT=kT, phonons=phonons),
                            model, scale)
    grid = np.arange(0.0, max(float(result.frequencies.max()), 100.0) + 200.0, 0.5)
    grid, intensity = spectrum(result, grid, fwhm, laser_nm, temperature_k)
    return {"rows": _groups_rows(result.groups()), "grid": grid, "intensity": intensity,
            "alpha": result.alpha, "method": result.method, "warnings": list(result.warnings),
            "frequency_scale": factor}


def scaled(result, model: TBModel, scale: bool):
    """``(result, λ)``: frequencies times the set's scale factor when asked and the
    set has one (``frequency_scale``, against GPAW); otherwise unchanged, λ None.

    Applied before broadening, so the Raman cross-section factors (ω-dependent)
    and the Bose factor use the scaled frequencies too. The factor belongs to
    the model's own phonons: callers pass ``scale=False`` for imported ones (QE).
    """
    from dataclasses import replace

    from ..tasks import frequency_scale

    factor = frequency_scale(model) if scale else None
    if factor is None:
        return result, None
    return replace(result, frequencies=factor * np.asarray(result.frequencies)), factor


def resonant_spectrum(atoms: Atoms, model: TBModel, lasers_ev, eta: float = 0.1, phonons=None,
                      kT: float = 0.01, kmesh: int = 24, scale: bool = False) -> dict:
    """Resonant Raman at each laser energy: activities per mode and laser."""
    from ..progress import Progress
    from ..resonance import resonant_raman

    # The window's terminal receives the job's stderr: one line every 30 s with the
    # count of modes done and the time left.
    with Progress(0, "Raman resonante (tensores por modo)") as bar:
        result, factor = scaled(resonant_raman(atoms, model, np.asarray(lasers_ev, dtype=float),
                                               eta=eta, kmesh=kmesh, kT=kT, phonons=phonons,
                                               progress=bar.callback()),
                                model, scale)
    strongest = result.activities.max() or 1.0
    keep = [k for k in range(len(result.frequencies))
            if result.activities[:, k].max() >= 1e-3 * strongest]
    return {"result": result, "lasers": result.lasers, "frequencies": result.frequencies,
            "activities": result.activities, "active": keep, "method": result.method,
            "warnings": list(result.warnings), "frequency_scale": factor}


def ir_spectrum_of(atoms: Atoms, model: TBModel, phonons=None, fwhm: float = 10.0,
                   kT: float = 0.01, scale: bool = False) -> dict:
    """IR intensities (km/mol) with the model's dipole, and the broadened absorption."""
    from ..infrared import infrared, ir_spectrum

    result, factor = scaled(infrared(atoms, model, kT=kT, phonons=phonons), model,
                            scale)
    grid, absorption = ir_spectrum(result, fwhm=fwhm)
    rows = [(float(g["frequency_cm1"]), int(g["degeneracy"]), float(g["intensity_km_mol"]),
             np.nan) for g in result.groups()]
    return {"rows": rows, "grid": grid, "absorption": absorption,
            "dipole_debye": float(np.linalg.norm(result.dipole) / 0.20819434),
            "warnings": list(result.warnings), "frequency_scale": factor}


def export_report(path: str | Path, width: str = "simple") -> dict:
    """``tbkit report`` from the window: ``path`` is a recipe's ``report.json`` (its
    folder is reported) or a spectrum (.npz/.csv). Writes the HTML page, the CSV folder
    and the journal figures next to it; returns what was written."""
    from ..report import build

    path = Path(path)
    source = path.parent if path.name == "report.json" else path
    return build(source, width=width)


def open_result(path: str | Path) -> Path:
    """«Abrir resultado…»: a recipe's ``report.json`` (its folder is shown), a validation
    ``.json`` or a spectrum, laid out as a page in a temporary folder and opened in the
    browser; nothing is written next to the data."""
    from ..report import open_result as _open

    path = Path(path)
    source = path.parent if path.name == "report.json" else path
    return _open(source)


def load_spectrum(path: str | Path) -> dict:
    """A spectrum computed earlier, to look at or compare: ``{"grid", "curves", "sticks",
    "source"}``.

    Reads what tbkit writes: the CSV of «Exportar CSV…» (first column the shift,
    the rest curves) and ``.npz`` files with a ``grid`` and curves of its length
    (e.g. ``raman_*/spectra.npz`` of ``recipes/nanocoil``), plus, when present,
    ``frequencies`` and ``activities`` (one row per laser) as sticks."""
    path = Path(path)
    curves, sticks = {}, {}
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8") as handle:
            header = handle.readline().strip().lstrip("#").split(",")
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        grid = data[:, 0]
        for k, name in enumerate(header[1:], start=1):
            curves[name.strip() or f"columna {k}"] = data[:, k]
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        if "grid" not in data:
            raise ValueError(f"{path.name} no tiene 'grid': no es un espectro de tbkit.")
        grid = data["grid"]
        for name in data.files:
            values = data[name]
            if name != "grid" and values.ndim == 1 and len(values) == len(grid):
                label = f"{name} eV" if name.replace(".", "", 1).isdigit() else name
                curves[label] = values
        if "frequencies" in data.files and "activities" in data.files:
            activities = np.atleast_2d(data["activities"])
            labels = list(curves) if len(curves) == len(activities) else \
                [f"serie {k + 1}" for k in range(len(activities))]
            for label, row in zip(labels, activities, strict=True):
                sticks[label] = (data["frequencies"], row)
    else:
        raise ValueError("Formatos: .csv (Exportar CSV) o .npz con 'grid'.")
    if not curves:
        raise ValueError(f"{path.name} no trae ninguna curva de la longitud de 'grid'.")
    return {"grid": np.asarray(grid, dtype=float), "curves": curves, "sticks": sticks,
            "source": path.name}


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
# Recipes: choose one, edit its settings, launch it in parts, watch its progress
# --------------------------------------------------------------------------

def recipe_list() -> list[dict]:
    """Every recipe with a command line: name, first line of its note, and how many
    settings (declared ``PARAMS``) and command-line options it has."""
    import importlib
    import pkgutil

    from .. import recipes

    out = []
    for info in pkgutil.iter_modules(recipes.__path__):
        module = importlib.import_module(f"tbkit.recipes.{info.name}")
        if not hasattr(module, "main"):
            continue
        usage = recipe_usage(info.name)
        options = {o["dest"] for opts in usage["options"].values() for o in opts}
        out.append({"name": info.name,
                    "summary": (module.__doc__ or "").strip().splitlines()[0],
                    "count": len(getattr(module, "PARAMS", [])), "options": len(options)})
    return sorted(out, key=lambda r: (-r["count"], r["name"]))


class _Parsed(Exception):
    """Raised in place of parsing, once the recipe has built its parser."""


#: Options the window handles itself (the settings table, tbkit lanzar) or never shows.
_HANDLED = {"help", "part", "ajuste", "ajustes", "ver_ajustes"}


def _option_rows(parser) -> list[dict]:
    import argparse
    import json

    rows = []
    for action in parser._actions:
        if action.dest in _HANDLED or isinstance(action, argparse._SubParsersAction):
            continue
        flag = next((o for o in action.option_strings if o.startswith("--")),
                    action.option_strings[0] if action.option_strings else None)
        boolean = isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction,
                                      argparse.BooleanOptionalAction))
        default = action.default
        if default is None or default is argparse.SUPPRESS:
            text = ""
        elif boolean:
            text = "sí" if default else "no"
        elif isinstance(default, (list, tuple)):
            text = " ".join(str(v) for v in default)
        elif isinstance(default, (int, float, str, Path)):
            text = str(default)
        else:
            text = json.dumps(default, default=str)
        rows.append({"dest": action.dest, "flag": flag, "default": text,
                     "help": (action.help or "") % {"default": default}
                     if action.help and "%(" in action.help else (action.help or ""),
                     "choices": [str(c) for c in action.choices] if action.choices else [],
                     "nargs": action.nargs, "boolean": boolean,
                     "negative": isinstance(action, argparse._StoreFalseAction),
                     "required": flag is None and action.nargs not in ("?", "*")})
    return rows


def recipe_usage(name: str) -> dict:
    """What the recipe's own command line accepts, read from its argument parser:
    ``steps`` (sub-commands, or the choices of its first argument), ``options`` per step
    ("" for the ones every step takes), whether it splits in parts (``--part r/N``) and
    its ``--help`` text."""
    import argparse
    import contextlib
    import importlib
    import inspect
    import io

    module = importlib.import_module(f"tbkit.recipes.{name}")
    captured = []
    original = argparse.ArgumentParser.parse_args

    def grab(self, *args, **kwargs):
        captured.append(self)
        raise _Parsed

    # A main() that takes no argv has no command line: calling it would run the recipe.
    if inspect.signature(module.main).parameters:
        argparse.ArgumentParser.parse_args = grab
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.suppress(_Parsed, SystemExit):
                module.main([])
        finally:
            argparse.ArgumentParser.parse_args = original
    if not captured:
        return {"steps": [], "step_dest": None, "options": {"": []}, "parts": False,
                "parallel_steps": (), "help": (module.__doc__ or "").strip()}
    parser = captured[0]
    help_text = parser.format_help()
    options = {"": _option_rows(parser)}
    steps, step_dest = [], None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            steps = list(action.choices)
            for step, sub in action.choices.items():
                options[step] = _option_rows(sub)
                help_text += f"\n\n── {step} ──\n" + sub.format_help()
            break
    else:
        first = next((r for r in options[""] if r["flag"] is None and r["choices"]), None)
        if first is not None and options[""][0] is first:
            steps, step_dest = first["choices"], first["dest"]
            options[""] = options[""][1:]
    parts = any("--part" in a.option_strings for a in parser._actions) or any(
        "--part" in a.option_strings for opts in captured[1:] for a in opts._actions)
    if not parts:
        parts = "--part" in help_text
    # Which steps split in parts: the recipe says (PARALLEL_STEPS); without the list, every
    # step of a recipe with --part does (site_screening, structure_screening: no steps).
    parallel = getattr(module, "PARALLEL_STEPS", None)
    if not parts:
        parallel = ()
    elif parallel is None:
        parallel = tuple(steps) if steps else None
    return {"steps": steps, "step_dest": step_dest, "options": options, "parts": parts,
            "parallel_steps": parallel, "help": help_text}


def step_splits(usage: dict, step: str) -> bool:
    """Whether ``step`` of a recipe (its :func:`recipe_usage`) can run in several parts."""
    if not usage["parts"]:
        return False
    parallel = usage.get("parallel_steps")
    return parallel is None or not step or step in parallel


def recipe_arguments(rows: list[dict], values: dict[str, str]) -> list[str]:
    """Command-line words for the options whose value (text, as edited) is not the
    default: positionals in order, then ``--flag value``. A required positional left
    empty raises ``ValueError``."""
    import shlex

    words, flags = [], []
    for row in rows:
        text = values.get(row["dest"], row["default"]).strip()
        many = row["nargs"] in ("*", "+") or isinstance(row["nargs"], int)
        if row["choices"] and text and text != row["default"] and \
                any(w not in row["choices"] for w in (shlex.split(text) if many else [text])):
            raise ValueError(f"{row['flag'] or row['dest']}: «{text}» no es una de "
                             f"{row['choices']}")
        if row["flag"] is None:
            if not text:
                if row["required"]:
                    raise ValueError(f"falta «{row['dest']}»")
                continue
            words += shlex.split(text) if many else [text]
            continue
        if text == row["default"]:
            continue
        if row["boolean"]:
            on = text.lower() in ("sí", "si", "true", "1", "yes")
            if row["negative"]:
                on = not on
            if on:
                flags.append(row["flag"])
            elif row["flag"].startswith("--") and not row["negative"]:
                flags.append("--no-" + row["flag"][2:])
            continue
        if not text:
            continue
        flags += [row["flag"], *(shlex.split(text) if many else [text])]
    return words + flags


def recipe_settings(name: str) -> list[dict]:
    """Every declared setting with its current value as JSON text (what the table edits)."""
    import importlib
    import json

    from ..settings import _jsonable

    module = importlib.import_module(f"tbkit.recipes.{name}")
    return [{"name": p.name, "value": json.dumps(_jsonable(getattr(module, p.name)),
                                                 ensure_ascii=False),
             "unit": p.unit, "description": p.description, "why": p.why, "group": p.group}
            for p in getattr(module, "PARAMS", [])]


def recipe_overrides(name: str, edited: dict[str, str]) -> dict:
    """The edited values that differ from the recipe's, parsed and checked against the
    type of each default; a bad value raises ``ValueError`` naming the setting."""
    import importlib
    import json

    from ..settings import _convert, _jsonable

    module = importlib.import_module(f"tbkit.recipes.{name}")
    defaults = {row["name"]: row["value"] for row in recipe_settings(name)}
    out = {}
    for key, text in edited.items():
        if key not in defaults:
            raise ValueError(f"'{key}' no es un ajuste de {name}")
        if text.strip() == defaults[key]:
            continue
        try:
            value = _convert(text.strip(), getattr(module, key))
        except ValueError as error:
            raise ValueError(f"{key}: {error}") from error
        if json.dumps(_jsonable(value), ensure_ascii=False) != defaults[key]:
            out[key] = _jsonable(value)
    return out


def cores() -> int:
    """Processors this process may use (the affinity mask, not the machine's total)."""
    import os

    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                      # Windows, macOS
        return os.cpu_count() or 1


def launch_warnings(parts: int, threads: int, mpi: int = 1, can_split: bool = True) -> list[str]:
    """What is wrong or wasteful in a choice of parts × threads × MPI processes."""
    out = []
    used = parts * threads * mpi
    if parts > 1 and not can_split:
        out.append("esta receta no se divide en partes (no tiene --part): usa 1 parte")
    if used > cores():
        out.append(f"{parts} partes × {threads} hilos × {mpi} MPI = {used} > {cores()} "
                   "núcleos: los procesos se estorban y todo va más lento")
    if threads > 1 and parts > 1:
        out.append("con matrices pequeñas rinden más partes de 1 hilo que pocas de muchos")
    return out


def launch_command(name: str, arguments: str, overrides: dict, parts: int = 1,
                   threads: int = 1, mpi: int = 1, retries: int = 1,
                   workdir: str | Path = ".", logs: str = "out/registros") -> list[str]:
    """The ``tbkit lanzar`` command for a recipe step, with the edited settings written
    to a JSON next to the logs (``--ajustes``), so the run records them."""
    import json
    import shlex
    import sys
    import time

    command = [sys.executable, "-m", "tbkit.cli", "lanzar", "--partes", str(parts),
               "--hilos", str(threads), "--mpi", str(mpi), "--reintentos", str(retries),
               "--registros", logs, "--carpeta", "out", name, *shlex.split(arguments)]
    if overrides:
        folder = Path(workdir) / logs
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}_ajustes_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(overrides, indent=1, ensure_ascii=False), encoding="utf-8")
        command += ["--ajustes", str(path.resolve())]
    return command


def launch_recipe(command: list[str], workdir: str | Path = ".", log: str | Path | None = None):
    """Start ``command`` in its own session (closing the window does not stop it); its
    output goes to ``log``. Returns the process."""
    import subprocess

    workdir = Path(workdir)
    log = Path(log) if log else workdir / "out" / "registros" / "lanzador.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as handle:
        return subprocess.Popen(command, cwd=workdir, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)


def recipe_status(folder: str | Path, finished: bool = False) -> list[str]:
    """One line per calculation with a counter under ``folder`` (``tbkit estado``)."""
    from ..progress import status_files

    def where(path):
        parent = Path(path).parent
        return parent.relative_to(folder) if parent.is_relative_to(folder) else parent

    return [f"{row['line']}   [{where(row['path'])}]"
            for row in status_files(folder) if finished or not row["finished"]]


# --------------------------------------------------------------------------
# The Guía tab: where the work stands and what applies to this system
# --------------------------------------------------------------------------

def nitrogen_kinds(atoms: Atoms) -> dict[str, int]:
    """Count N by environment: graphitic (3 C), pyridinic (2 C, no H), pyrrolic (2 C + H in a
    5-ring) and amine (N with 2 H), by bonds within 1.2 × the covalent radii."""
    symbols = atoms.get_chemical_symbols()
    if "N" not in symbols:
        return {}
    neighbours: dict[int, list[int]] = {i: [] for i in range(len(atoms))}
    for i, j, _ in bonds_with_images(atoms):
        neighbours[i].append(j)
        neighbours[j].append(i)
    from ..sites import rings

    in_five = {a for ring in rings(atoms) if len(ring) == 5 for a in ring}
    out: dict[str, int] = {}
    for i, s in enumerate(symbols):
        if s != "N":
            continue
        carbons = sum(symbols[j] == "C" for j in neighbours[i])
        hydrogens = sum(symbols[j] == "H" for j in neighbours[i])
        if hydrogens >= 2:
            kind = "amina"
        elif carbons == 3:
            kind = "grafítico"
        elif carbons == 2 and hydrogens == 1 and i in in_five:
            kind = "pirrólico"
        elif carbons == 2 and hydrogens == 0:
            kind = "piridínico"
        else:
            kind = "otro"
        out[kind] = out.get(kind, 0) + 1
    return out


def system_caveats(atoms: Atoms, model: TBModel | None, state=None) -> list[str]:
    """What applies to this structure with this model, from what tbkit measured (the coil's
    GPAW checks, the validation files): short sentences, worst first."""
    out = []
    if model is not None:
        out += check_model(atoms, model)
    n = len(atoms)
    periodic = int(atoms.pbc.sum())
    if not periodic and atoms.cell.rank == 3:
        out.append("Tiene celda pero ningún eje periódico: si es periódica, «Celda…» marca "
                   "los ejes.")
    if model is not None:
        need = memory_estimate_gb(atoms, model)
        if n > 2000:
            out.append(f"{n} átomos: una diagonalización necesita ≈ {need:.1f} GB y relajar "
                       "todo es lento; para barrer sitios usa el modelo π o un recorte "
                       "alrededor del sitio.")
    electrons = None
    if state is not None:
        electrons = state.info.get("electrons")
        gap = state.info.get("gap")
        if not gap:
            out.append("Sin gap: el Raman no resonante no aplica (usa el resonante) y los "
                       "fonones dependen de la malla k (convérgela).")
    if electrons is not None and round(electrons) % 2 == 1 and not periodic:
        out.append("Número impar de electrones: capa abierta; el Raman no resonante no aplica "
                   "y el magnetismo (Hubbard) puede importar.")
    kinds = nitrogen_kinds(atoms)
    if kinds.get("piridínico") or kinds.get("pirrólico"):
        out.append("N piridínico o pirrólico: lo menos validado de tbkit (en la coil, las "
                   "fuerzas PBE en las geometrías de tbkit fueron de 2.6–7.9 eV/Å y el orden de "
                   "sitios no se confirmó): valida con DFT antes de usarlo.")
    if kinds.get("grafítico"):
        out.append("N grafítico: GPAW confirmó el mejor sitio de tbkit en la coil; su banda en "
                   "el nivel de Fermi sale 3.6 veces más ancha que en PBE (el N menos "
                   "localizado).")
    if periodic:
        out.append(f"Periódica en {periodic} eje(s): el IR solo usa los ejes no periódicos (a lo "
                   "largo de uno periódico el dipolo necesita la fase de Berry).")
    if model is not None and model.repulsive is not None:
        out.append("Frecuencias del modelo sin escalar: en la coil, los modos de G salen ~5 % "
                   "altos frente a PBE (cociente PBE/tbkit 0.95).")
    return out


def guide(info: dict) -> dict:
    """The steps of the work with their state. ``info``: atoms, model, model_problems,
    relaxed, state, modes, dft_checked, project_saved (what the window knows)."""
    atoms, model = info.get("atoms"), info.get("model")
    steps = [
        ("estructura", "Abrir la estructura (o un proyecto)", atoms is not None,
         "xyz, extxyz, xtl, cif, POSCAR…; Archivo → «Abrir proyecto…» retoma una sesión.",
         None),
        ("celda", "Revisar la celda y los ejes periódicos",
         atoms is not None and (bool(atoms.pbc.any()) or atoms.cell.rank < 3),
         "Una estructura periódica sin ejes marcados se calcula como molécula: «Celda…».",
         None),
        ("modelo", "Elegir el modelo", model is not None and not info.get("model_problems"),
         "El que cubre los elementos (se sugiere solo); los xu_ch* van con SCC.", None),
        ("relajar", "Relajar la geometría", bool(info.get("relaxed")),
         "Fuera del mínimo las frecuencias no son armónicas.", "Geometría y modos"),
        ("estado", "Estado fundamental: gap, DOS, cargas", info.get("state") is not None,
         "Dice si hay gap (qué Raman aplica) y dónde están los estados.", "Electrónica"),
        ("modos", "Modos en Γ", bool(info.get("modes")),
         "Frecuencias y patrones; base de los espectros.", "Geometría y modos"),
        ("espectros", "Raman e IR", bool(info.get("spectra")),
         "Con gap: no resonante; sin gap: resonante. IR: ejes no periódicos.", "Espectros"),
        ("dft", "Validar con DFT los modos que importan", bool(info.get("dft_checked")),
         ("Selecciona los modos en la tabla y «Validar con DFT…»: antes de publicar un "
          "número."), "Geometría y modos"),
        ("proyecto", "Guardar el proyecto", bool(info.get("project_saved")),
         "Ctrl+S: geometrías, ajustes y modos para seguir otro día.", None),
    ]
    out, next_set = [], False
    for key, title, done, why, tab in steps:
        if done:
            status = "hecho"
        elif not next_set:
            status, next_set = "siguiente", True
        else:
            status = "pendiente"
        out.append({"key": key, "title": title, "status": status, "why": why, "tab": tab})
    caveats = system_caveats(atoms, model, info.get("state")) if atoms is not None else []
    validity = (model.metadata.get("validity") or "") if model is not None else ""
    return {"steps": out, "caveats": caveats, "validity": str(validity)}


# --------------------------------------------------------------------------
# Studies (tbkit.recipes.study): any set of systems, changes embedded in their host
# --------------------------------------------------------------------------

def change_centres(base: Atoms, changed: Atoms, tolerance: float = 0.5) -> list[int]:
    """The atoms of ``changed`` that are not ``base``: a different element at the same index
    (substitution) or atoms beyond the base's count (added). ``ValueError`` if the base's
    atoms are not first and in order (an index far from its pair in the base)."""
    n = len(base)
    if len(changed) < n:
        raise ValueError("La estructura cambiada tiene menos átomos que la base: no es una "
                         "sustitución ni una adición sobre ella.")
    probe = changed[:n].copy()
    probe.positions -= base.positions
    if changed.pbc.any() and changed.cell.rank == 3:
        frac = (probe.cell.scaled_positions(probe.positions) + 0.5) % 1.0 - 0.5
        shift = np.linalg.norm(probe.cell.cartesian_positions(frac), axis=1)
    else:
        shift = np.linalg.norm(probe.positions, axis=1)
    substituted = [i for i in range(n) if base[i].symbol != changed[i].symbol]
    lost = [i for i in range(n) if shift[i] > tolerance and i not in substituted]
    if lost:
        raise ValueError(f"Los átomos {lost[:8]} están a más de {tolerance} Å de su par en la "
                         "base: la base debe ir primero y en su orden (sustituir conserva el "
                         "índice; los átomos añadidos van al final).")
    return substituted + list(range(n, len(changed)))


def write_study(folder: str | Path, name: str, base: tuple[str, Atoms],
                changes: list[tuple[str, Atoms]], model: str = "xu_chn",
                lasers=(1.96, 2.33, 2.54), radius: float = 6.0) -> Path:
    """A study file (``tbkit.recipes.study``) with its structures, from geometries of the
    session; the changed atoms of each change are found by :func:`change_centres`."""
    import re

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    def safe(text):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "sistema"

    systems = {}
    base_key = safe(base[0])
    base[1].write(folder / f"{base_key}.extxyz", format="extxyz")
    systems[base_key] = {"estructura": f"{base_key}.extxyz"}
    for label, atoms in changes:
        key = safe(label)
        if key in systems:
            key = f"{key}_{len(systems)}"
        atoms.write(folder / f"{key}.extxyz", format="extxyz")
        systems[key] = {"estructura": f"{key}.extxyz", "sobre": base_key,
                        "centros": change_centres(base[1], atoms)}
    path = folder / "estudio.json"
    path.write_text(json.dumps({"nombre": safe(name), "modelo": model, "kpts": "auto",
                                "kT": 0.05, "sistemas": systems,
                                "hessiana": {"delta": 0.01, "radio": radius},
                                "raman": {"laseres_ev": list(lasers), "eta": 0.1,
                                          "min_cm1": 900.0, "delta": 0.005}},
                               indent=1, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# DFT checks through carbonforge (nanocarbon-dft/1, files only)
# --------------------------------------------------------------------------

DFT_CODES = {"gpaw": "GPAW (PBE, LCAO dzp)", "qe": "Quantum ESPRESSO (pw.x)",
             "dftb+": "DFTB+ (.skf propios)"}


def dft_request(folder: str | Path, atoms: Atoms, code: str = "gpaw", relax: bool = False,
                single_point: bool = True, dos: bool = False, vibrations=None,
                modes=(), mpi: int = 1, settings: dict | None = None) -> str:
    """Write a DFT request for the active geometry (``tbkit.dft_exchange``); with ``modes``
    (indices of ``vibrations``), ± along each mode pattern as well. Returns the folder."""
    from .. import dft_exchange as dx

    tasks = []
    source = "entrada"
    if relax:
        tasks.append({"id": "relax", "tipo": "relajar", "fmax": 0.05, "pasos": 300})
        source = "relax"
    if single_point or not tasks:
        tasks.append({"id": "sp", "tipo": "punto_simple", "sobre": source})
    if dos:
        tasks.append({"id": "dos", "tipo": "dos", "sobre": source, "sigma_ev": 0.1})
    resources = {"mpi": int(mpi), "hilos": 1}
    if modes:
        folder = dx.mode_check_request(folder, vibrations, list(modes), code, relax_first=False,
                                       settings=settings, resources=resources)
        # the request's own tasks were written by mode_check_request: add ours around them
        request_path = Path(folder) / "solicitud.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        displacements = [t for t in request["tareas"] if t["tipo"] == "desplazamientos"]
        for t in displacements:
            t["sobre"] = source
        request["tareas"] = tasks + displacements
        request_path.write_text(json.dumps(request, indent=1, ensure_ascii=False),
                                encoding="utf-8")
        return str(folder)
    return str(dx.write_request(folder, atoms, tasks, code, settings=settings,
                                resources=resources, why="comprobar con DFT"))


def dft_summary(folder: str | Path, model: TBModel, kmesh: int = 4, kT: float = 0.05) -> dict:
    """The DFT result of a request folder against the model (``dft_exchange.summary``), with
    the DFT-relaxed structure if there is one (``relaxed``: Atoms or None)."""
    from ase.io import read

    from .. import dft_exchange as dx

    out = dx.summary(folder, model, kmesh=kmesh, kT=kT)
    result = dx.read_result(folder)
    relaxed = next((read(Path(folder) / t["estructura"]) for t in result["tareas"].values()
                    if isinstance(t, dict) and "estructura" in t), None)
    return {"summary": out, "relaxed": relaxed}


def dft_summary_lines(summary: dict) -> list[str]:
    """The summary as short lines (terminal and dialog)."""
    lines = [f"{summary['codigo']} {summary.get('version_codigo') or ''}".strip()
             + ("" if summary["terminado"] else " · SIN TERMINAR")]
    for task_id, t in summary["tareas"].items():
        if t.get("estado") == "pendiente":
            lines.append(f"{task_id}: pendiente")
            continue
        if "fuerzas_tbkit_vs_dft" in t:
            f = t["fuerzas_tbkit_vs_dft"]
            text = (f"{task_id}: fuerzas tbkit − DFT RMS {f['rms_eV_A']:.3f} eV/Å (máx. "
                    f"{f['max_eV_A']:.3f} en el átomo {f['max_atom']}; DFT RMS "
                    f"{f['rms_dft_eV_A']:.3f})")
            if t["tipo"] == "relajar":
                text += (f" · relajación {'convergida' if t.get('convergida') else 'SIN converger'}"
                         f", el DFT movió hasta {t['desplazamiento_max_A']:.3f} Å")
            lines.append(text)
        for m in t.get("modos", []):
            ir = f", IR {m['dft_ir_km_mol']:.3g} km/mol" if m["dft_ir_km_mol"] is not None else ""
            lines.append(f"  modo {m['mode']}: tbkit {m['tb_cm1']:.0f} · DFT {m['dft_cm1']:.0f} "
                         f"cm⁻¹ (×{m['dft_over_tb']:.3f}){ir}")
        if t.get("tipo") == "dos":
            gap = t.get("gap_eV")
            lines.append(f"{task_id}: gap DFT {gap:.3f} eV" if gap is not None
                         else f"{task_id}: sin gap (metálico)")
    return lines


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


# --------------------------------------------------------------------------
# Help shown in the window: what each panel is for and what each control does.
# Plain text here (no Qt) so a test can check that every control has one.
# --------------------------------------------------------------------------

PANEL_HELP = {
    "Guía": "Dónde va el trabajo con la geometría activa y qué sigue (doble clic en un paso, o "
            "«Ir al paso siguiente»), y lo que aplica a este sistema con este modelo según lo "
            "que tbkit midió: qué Raman e IR aplican, tamaño y memoria, tipos de N y qué tan "
            "validados están, cuánto se desvían las frecuencias.",
    "Electrónica": "Estado fundamental del modelo: niveles, DOS/PDOS, bandas y cargas. "
                   "«Calcular estado fundamental» es el paso que usan Orbitales, Magnetismo "
                   "y los espectros; se hace sobre la geometría activa (relájala antes en "
                   "«Geometría y modos»).",
    "Orbitales": "Orbitales moleculares (o de Bloch en Γ) como isosuperficies sobre la "
                 "estructura; sirve para ver dónde vive el HOMO/LUMO o un estado de borde.",
    "Magnetismo": "Hubbard de campo medio: momentos locales, m(E) y barridos de U, campo "
                  "o dopaje. Los momentos son un parámetro de orden, no un estado "
                  "correlacionado; el teorema de Lieb es la comprobación.",
    "Geometría y modos": "El primer paso: relaja las posiciones con las fuerzas del modelo "
                         "(la geometría se mueve en la vista mientras tanto) y calcula los "
                         "modos en Γ. La geometría relajada se añade a la lista «Geometría» "
                         "del panel izquierdo y queda activa: las demás pestañas calculan "
                         "sobre ella. Relaja antes de los modos: fuera del mínimo las "
                         "frecuencias no son armónicas.",
    "Fonones (ZB)": "Fonones en toda la zona de Brillouin con phonopy y las fuerzas del "
                    "modelo: dispersión por el camino de puntos especiales, DOS por "
                    "elemento, representación irreducible de cada modo en Γ (identifica "
                    "modos por simetría, no por frecuencia) y F, S, Cv armónicos. Solo "
                    "periódicos y relajados; los ejes no periódicos llevan supercelda 1.",
    "Espectros": "Raman (no resonante y resonante) e IR con los fonones del modelo o de "
                 "Quantum ESPRESSO. Raman solo en sistemas con gap y capa cerrada. "
                 "«Escalar frecuencias» multiplica por el factor del conjunto (frente a "
                 "GPAW, ~3 %); las crudas siguen siendo las del modelo.",
    "Grafeno": "Bandas G, 2D y 2D′ del grafeno por doble resonancia, con fonones de GPAW "
               "o de Xu; dispersión de la 2D con la energía del láser.",
    "Procesos": "Todo lo que corre: el cálculo de la ventana (y su cola), las recetas lanzadas "
                "desde aquí y cualquier receta con contador en la carpeta de trabajo, con su "
                "CPU, RAM, núcleos y avance. Selecciona una fila para ver el final de su "
                "registro; «Detener seleccionado» la mata con sus procesos hijos.",
    "Recetas": "Los cálculos largos de las recetas (doble resonancia de la coil, Raman de "
               "los dopados, referencias GPAW…) con todos sus números a la vista y "
               "editables: valor, unidad y por qué tiene ese valor (pasa el ratón). Elige "
               "partes en paralelo e hilos, lanza, y sigue el avance con el tiempo "
               "restante. El cálculo corre fuera de la ventana: cerrarla no lo detiene. Los "
               "ajustes usados quedan en ajustes_usados.json junto a los resultados.",
}

HELP = {
    # left column
    "Estructura": "La estructura abierta: xyz, extxyz, cif, POSCAR… Periódica si el "
                  "archivo trae celda.",
    "Abrir…": "Abre una estructura. Al cambiarla se borran los resultados anteriores.",
    "Modelo": "El conjunto de parámetros y cómo se resuelve (carga, SCC, temperatura).",
    "Parámetros": "Conjunto de parámetros: π (solo bandas), Xu (carbono), xu_ch* "
                  "(carbono con H, N, O, B, S, P, Se), Tang (carbono con defectos, "
                  "amorfo, curvatura 5-7: saltos que dependen del entorno). Cada uno "
                  "dice en su «validity» para qué sirve y qué error tiene.",
    "Carga (e)": "Carga total del sistema en electrones (+1 = un electrón menos).",
    "SCC": "Cargas autoconsistentes. Por defecto, lo que pide el conjunto: los xu_ch* "
           "se ajustaron con SCC y no deben usarse sin ella.",
    "kT": "Temperatura electrónica (eV): ensancha la ocupación cerca del nivel de Fermi. "
          "Pequeña para moléculas; algo mayor ayuda a converger metales.",
    "Colorear átomos": "Colorea los átomos por carga, momento magnético o elemento.",
    "Cancelar cálculo": "Detiene el cálculo en curso al instante: su proceso se mata (nada "
                        "sigue calculando) y se vacía la cola.",
    "Terminal": "Lo que hace cada cálculo mientras corre: el comando, lo que imprime y su "
                "avance (una línea cada 5 s), y cuánto tardó.",
    "Guardar…": "Guarda la geometría activa: extxyz (o .xyz, que aquí se escribe como "
                "extxyz y conserva la celda), .xtl, CIF o POSCAR.",
    "Celda…": "Fija la celda (longitudes y ángulos) y los ejes periódicos de la geometría "
              "activa, para un archivo que no los trae. «Detectar por enlaces» marca los "
              "ejes por los que hay enlaces que cruzan la celda.",
    "Geometría": "Las geometrías de la sesión: la abierta, las relajadas, con celda editada, "
                 "las supercelda… La elegida es la que usan todas las pestañas.",
    "Repetir": "Copias de la celda a lo largo de a, b, c (solo ejes periódicos). «Ver» "
                     "las dibuja sin cambiar el cálculo; «Crear» hace una supercelda nueva "
                     "(p. ej. 1×1×4) y la deja activa.",
    "Ver": "Dibuja las copias de la celda; el cálculo sigue con la celda original.",
    "Crear": "Crea la supercelda como geometría nueva y la deja activa para calcular.",
    "Recursos del cálculo": "Núcleos y memoria de los cálculos de la ventana. Cada cálculo "
                            "corre en su propio proceso: se puede detener al instante y se "
                            "detiene solo si pasa el límite de RAM.",
    "Núcleos": "Procesadores para los cálculos de la ventana (afinidad e hilos de álgebra "
               "lineal). Deja libres los que usen las recetas lanzadas.",
    "RAM máx.": "Si el cálculo pasa de esta memoria se detiene (antes de que el sistema mate "
                "procesos por falta de RAM). Antes de empezar se estima la memoria de una "
                "diagonalización y se avisa si no cabe.",
    "Longitudes a, b, c": "Longitudes de los vectores de celda (Å).",
    "Ángulos α, β, γ": "Ángulos entre los vectores de celda (grados).",
    "Periódica en": "Ejes que se repiten. Una molécula en una caja no tiene ninguno; un tubo, "
                    "uno; una lámina, dos.",
    "eje a": "Periódica a lo largo del vector a.",
    "eje b": "Periódica a lo largo del vector b.",
    "eje c": "Periódica a lo largo del vector c.",
    "Escalar las posiciones con la celda": "Si se marca, los átomos se mueven con la celda "
                                           "(mismas coordenadas fraccionarias); si no, "
                                           "conservan sus posiciones en Å.",
    "Detectar por enlaces": "Marca como periódicos los ejes por los que hay enlaces que "
                            "cruzan la celda escrita arriba.",
    "Exportar geometría…": "Guarda la geometría activa (la relajada, tras relajar).",
    "Usar geometría activa": "Guarda la geometría activa en geometrias/ de la carpeta de "
                             "trabajo y pone su ruta en la fila seleccionada (una opción como "
                             "«structure» o un ajuste como SOURCE).",
    "Nueva receta…": "Una receta propia: una geometría de la sesión, un modelo y pasos en "
                     "orden (relajar, niveles, DOS, bandas, modos, Raman, Hubbard). Se escribe "
                     "en recetas/ y se corre con «Lanzar», con contador y un registro "
                     "reproducible por paso.",
    "Ir al paso siguiente": "Lleva a la pestaña o al diálogo del siguiente paso pendiente.",
    "Muchos sistemas: nuevo estudio": "Para varias estructuras (una base y sus dopantes o "
                                      "grupos): arma un estudio en Recetas.",
    "Nuevo estudio…": "Un estudio (receta study) con geometrías de la sesión: una base y sus "
                      "cambios (dopantes, grupos). Relaja, calcula modos (los de los cambios solo "
                      "cerca del cambio), Raman resonante e IR, y escribe la solicitud DFT de "
                      "los modos que importan. Sirve para cualquier sistema 0D–3D.",
    "Nombre del estudio": "Nombre del estudio y de su carpeta (estudios/NOMBRE).",
    "Sistema base": "La estructura sin cambiar; sus modos se calculan completos.",
    "Cambios sobre la base": "Estructuras con un dopante o grupo sobre la base: los átomos de la "
                             "base primero y en su orden.",
    "Modelo del estudio": "Conjunto de parámetros de todo el estudio (xu_chn para C, H, N).",
    "Radio del cambio": "Las filas de la hessiana y las cargas de Born se recalculan a menos de "
                        "este radio de los átomos cambiados; el resto se toma de la base.",
    "Nombre": "Nombre de la receta (también el de su carpeta de resultados).",
    "Geometría de partida": "La geometría de la sesión de la que parte la receta.",
    "Modelo de la receta": "Conjunto de parámetros de la receta.",
    "Pasos, en este orden": "Tareas que se hacen una tras otra; la relajación pasa su "
                            "geometría a las siguientes.",
    "Malla k de la receta": "Puntos k por eje periódico en todos los pasos.",
    "kT de la receta": "Temperatura electrónica (eV) en todos los pasos.",
    "fmax de la relajación": "Criterio de convergencia de la relajación (eV/Å).",
    "Detener seleccionado": "Detiene el proceso elegido y sus hijos (las partes de una "
                            "receta). En el cálculo de la ventana equivale a «Cancelar».",
    "Actualizar lista": "Vuelve a leer los procesos ahora (se actualiza solo cada 2 s).",
    # Electrónica / Orbitales
    "Calcular estado fundamental": "Diagonaliza (con SCC si toca): niveles, gap, DOS y "
                                   "cargas. Lo usan las demás pestañas.",
    "Comparar SCC sí/no": "Resuelve con y sin cargas autoconsistentes y compara gap, HOMO, "
                          "LUMO, E_F y cargas; colorea Δq en la vista. La SCC cobra la "
                          "transferencia de carga (U, γ) y suele reducir las cargas polares. "
                          "Necesita U en el conjunto (xu_ch*).",
    "Calcular bandas": "Bandas a lo largo del camino de puntos especiales de la celda.",
    "Cargar niveles": "Superpone niveles de otro cálculo (p. ej. DFT) para comparar.",
    "Quitar": "Quita los niveles superpuestos.",
    # Magnetismo
    "Hubbard U": "Repulsión en el sitio (eV) del Hubbard de campo medio.",
    "Punto de partida": "Configuración inicial de espines; cambiarla puede llevar a otra "
                        "solución (compáralas).",
    "Campo (Zeeman)": "Campo magnético como desdoblamiento Zeeman (eV).",
    "Malla k (periódicos)": "Puntos k por eje periódico; más puntos, más precisión y más "
                            "tiempo.",
    "Resolver": "Resuelve el Hubbard de campo medio desde el punto de partida elegido.",
    "Comparar puntos de partida": "Resuelve desde varios puntos de partida y compara sus "
                                  "energías: la menor es la solución.",
    "Barrer": "Barre U, campo o dopaje y dibuja la magnetización.",
    # Geometría y modos
    "fmax": "Criterio de convergencia de la relajación: fuerza máxima (eV/Å).",
    "pasos máx.": "Número máximo de pasos de la relajación.",
    "malla k (periódicos)": "Puntos k por eje periódico para fuerzas y modos.",
    "Relajar": "Relaja las posiciones (celda fija) con las fuerzas del modelo.",
    "Volver a la original": "Recupera la estructura tal como se abrió.",
    "Modos en Γ": "Frecuencias y vectores de los modos en Γ por diferencias finitas.",
    "Validar con DFT…": "Escribe una solicitud DFT (formato nanocarbon-dft/1) para la geometría "
                        "activa: fuerzas, relajación, DOS y la frecuencia de los modos "
                        "seleccionados en la tabla. carbonforge la corre (aquí, en la cola, o en "
                        "un clúster con trabajo.slurm); nada se calcula en la ventana.",
    "Ver resultado DFT…": "Lee el resultado de una solicitud DFT y lo compara con el modelo: "
                          "fuerzas tbkit − DFT, frecuencia DFT de cada modo de tbkit (cociente "
                          "de Rayleigh) e IR, gap. Una geometría relajada por DFT se añade a la "
                          "lista «Geometría».",
    "Carpeta de la solicitud": "Carpeta nueva donde se escribe la solicitud (y donde quedará el "
                               "resultado). Una carpeta con solicitud no se reutiliza.",
    "Código DFT": "GPAW (con los ajustes de las referencias de tbkit), Quantum ESPRESSO (pw.x, "
                  "necesita pseudopotenciales) o DFTB+ (necesita .skf).",
    "Relajar con DFT primero": "El DFT relaja las posiciones (celda fija) y las demás tareas "
                               "parten de esa geometría.",
    "Fuerzas en la geometría": "Un punto simple: energía y fuerzas, comparadas con las del "
                               "modelo en la misma geometría.",
    "DOS y gap": "La DOS y el gap del DFT en esa geometría.",
    "Procesos MPI": "Procesos MPI del DFT (en la laptop, 4 = los núcleos físicos).",
    "Ajustes del código (JSON)": "Ajustes extra del código en JSON: QE necesita pseudos y "
                                 "pseudo_dir; DFTB+, skf_dir; GPAW acepta modo, base, h, kpts…",
    "Correrla ahora con carbonforge": "Lanza «carbonforge dft correr» en segundo plano (se ve "
                                      "en Procesos). Sin marcar, solo escribe la carpeta.",
    "Abrir modos…": "Abre modos calculados antes, sin recalcular: un modes.npz (de «Guardar "
                    "modos…») o una hessiana hessian.npy con su estructura al lado. Su estructura "
                    "queda como geometría activa.",
    "Guardar modos…": "Guarda los modos (frecuencias y vectores) para reutilizarlos.",
    "Flechas": "Muestra los desplazamientos del modo seleccionado como flechas.",
    "Parar": "Detiene la animación del modo.",
    "Ordenar modos por sitio": "Ordena los modos por su peso en un grupo de átomos: anillos "
                               "de 5, 6 o 7, cada heteroátomo, los C unidos a un heteroátomo, "
                               "H. «Enriquecimiento» = peso / fracción de átomos del grupo "
                               "(1: modo repartido; alto: localizado ahí).",
    "Comparar el modo con": "Frecuencia del modo seleccionado (mismo patrón de "
                            "desplazamiento) con otro conjunto de parámetros: cociente de "
                            "Rayleigh con dos llamadas de fuerzas. Exacta si el modo también "
                            "es propio del otro modelo; si no, cota superior.",
    "Frecuencia con el otro modelo": "Calcula la frecuencia del modo seleccionado con el "
                                     "modelo elegido (p. ej. Xu frente a Tang en un modo 5-7).",
    # Fonones (ZB)
    "Supercelda": "Repeticiones de la celda en x, y, z para las constantes de fuerza (1 en "
                  "los ejes no periódicos). Más grande: dispersión más fiel lejos de Γ.",
    "Desplazamiento": "Amplitud de los desplazamientos de phonopy (Å).",
    "Malla k de la celda": "Puntos k por eje de la celda unidad; en la supercelda se divide "
                           "por su tamaño (mismo muestreo electrónico).",
    "Temperatura": "Temperatura de F, S y Cv armónicos.",
    "Carpeta (reanudable)": "Si se da, cada supercelda desplazada guarda sus fuerzas ahí y un "
                            "cálculo interrumpido se reanuda (no mezcla estructuras: lo "
                            "comprueba).",
    "Calcular fonones en la ZB": "Fuerzas en las superceldas desplazadas, constantes de "
                                 "fuerza, dispersión, DOS, simetrías en Γ y termodinámica.",
    # Espectros
    "Fonones": "De dónde salen los modos: los del modelo (Geometría y modos) o un archivo "
               "de Quantum ESPRESSO; en Grafeno, GPAW o Xu.",
    "Raman / IR": "Láser, temperatura (factor de Bose) y anchura de línea del espectro.",
    "Resonante (eV)": "Energías de láser del Raman resonante y su amortiguamiento η. Las "
                      "resonancias son las del modelo, no energías ópticas.",
    "Escalar frecuencias": "Multiplica las frecuencias del modelo por el factor de su "
                           "conjunto (ajustado frente a GPAW, ~3 %; ver README). No se "
                           "aplica a modos importados de Quantum ESPRESSO.",
    "Raman": "Raman no resonante: actividades, despolarización y espectro.",
    "Raman resonante": "Actividades a cada energía de láser y perfiles de excitación.",
    "IR": "Intensidades IR (km/mol) con el dipolo del modelo; semicuantitativas.",
    "Exportar CSV…": "Guarda el espectro y la tabla mostrados como CSV.",
    "Exportar reporte…": "Escribe un reporte de una carpeta de resultados (elige su report.json) "
                         "o de un espectro guardado (.npz/.csv): una página HTML que se abre sin "
                         "internet, los datos de cada gráfica y tabla en CSV (Origin, Excel…) y "
                         "las figuras para revista en PDF, SVG y PNG a 600 dpi, a una columna.",
    "Abrir resultado…": "Abre en el navegador cualquier resultado o validación: el report.json "
                        "de una receta (Raman de la coil, comparación con GPAW, estructura "
                        "electrónica…), un .json de la carpeta validation/ o un espectro "
                        "(.npz/.csv). Las tablas y gráficas se ven sin internet; para "
                        "CSV y figuras de revista, «Exportar reporte…».",
    "Abrir espectro guardado…": "Muestra un espectro calculado antes (CSV de «Exportar CSV…» o "
                                "un .npz de las recetas, p. ej. raman_*/spectra.npz de la "
                                "coil) sin recalcular. Cada curva se normaliza a su máximo; "
                                "si hay un espectro calculado en la pestaña, se superpone para "
                                "comparar.",
    # Grafeno
    "Láseres (eV)": "Energías de láser para G, 2D y 2D′.",
    "γ electrónico": "Ensanchamiento electrónico (eV) de la doble resonancia.",
    "malla dk": "Paso de la malla en k (y en q) de la doble resonancia: menor, más "
                "preciso y más lento.",
    "procesos": "Procesos en paralelo para el cálculo de doble resonancia.",
    "Calcular G, 2D y 2D′": "Posiciones e intensidades de G, 2D y 2D′ para cada láser.",
    # Recetas
    "Receta": "La receta a correr; entre paréntesis, cuántos ajustes internos declara "
              "(tabla «Ajustes de la receta») y cuántas opciones de línea de comandos "
              "tiene (tabla «Opciones del paso»). Ambas tablas se editan con doble clic.",
    "Paso": "La etapa de la receta (p. ej. fc, dband, pairs, report en la doble "
            "resonancia; relax, hessian, raman en los dopados). «Ayuda de la receta» "
            "explica cada una.",
    "Otros argumentos": "Texto que se añade tal cual al final del comando (como en la "
                        "terminal). Casi nunca hace falta: cada opción de la receta está en "
                        "la tabla «Opciones del paso», con sus valores posibles.",
    "Carpeta de trabajo": "Donde se corre la receta: sus resultados van a out/ dentro de "
                          "ella y el avance se lee de ahí.",
    "Restaurar valores": "Vuelve a los valores de la receta (deshace lo editado en la tabla).",
    "Ayuda de la receta": "El --help de la receta: pasos, argumentos y opciones.",
    "Recursos": "Cuántos procesos y núcleos usa el cálculo. Partes × hilos × MPI no debe "
                "pasar de los núcleos disponibles.",
    "Partes en paralelo": "Procesos independientes que se reparten el trabajo (--part r/N); "
                          "con matrices pequeñas es lo que más acelera. Solo en recetas "
                          "que lo admiten.",
    "Hilos por parte": "Hilos de álgebra lineal (OMP_NUM_THREADS) de cada parte. Con las "
                       "matrices de tbkit, 1 suele ser lo mejor: los hilos compiten entre sí.",
    "Procesos MPI por parte": "Para los pasos de GPAW: procesos MPI de cada parte (necesita "
                              "mpiexec). 1 = sin MPI.",
    "Reintentos": "Veces que se relanza una parte que termina con error (las recetas "
                  "retoman donde quedaron).",
    "Lanzar": "Escribe los ajustes editados en un JSON (--ajustes) y lanza tbkit lanzar "
              "fuera de la ventana; el registro queda en out/registros/.",
    "Avance": "Los cálculos con contador bajo out/ de la carpeta de trabajo: hechos/total, "
              "tiempo transcurrido, restante y hora estimada de fin (se actualiza cada 10 s).",
    "Mostrar terminados": "Incluye los cálculos que ya terminaron.",
    "Actualizar avance": "Vuelve a leer los contadores ahora.",
}
