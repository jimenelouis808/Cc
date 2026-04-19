"""Pre-relaxation utilities.

Two entry points:

* :func:`relax_with_calculator` — generic ASE relaxation wrapper. The user
  provides any ASE calculator (Tersoff/AIREBO via LAMMPS-ASE, DFT codes,
  GAP/NEP, a neural-network potential ...) and this function runs the
  chosen :mod:`ase.optimize` algorithm to a force-norm tolerance.
* :func:`harmonic_pre_relax` — calculator-free, topology-based spring
  relaxation. Useful as a first cleanup for hand-built / stochastic
  structures (e.g. :func:`build_carbon_foam`) before handing them to a
  proper force field. It pulls every bonded pair toward the equilibrium
  C-C distance using a simple harmonic potential solved iteratively.

``harmonic_pre_relax`` is deliberately conservative: it preserves topology
(never breaks or creates bonds) and has no dependency beyond NumPy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms
from ase.optimize import BFGS, LBFGS, FIRE

from ..topology.graph import build_bond_graph
from ..utils.constants import CC_BOND


_OPTIMIZERS = {"bfgs": BFGS, "lbfgs": LBFGS, "fire": FIRE}


def relax_with_calculator(
    atoms: Atoms,
    calculator,
    algorithm: str = "lbfgs",
    fmax: float = 0.05,
    max_steps: int = 200,
    logfile: Optional[str] = None,
) -> Atoms:
    """Run an ASE optimizer until ``max(|F|) < fmax`` or ``max_steps`` hit.

    Parameters
    ----------
    atoms
        Input structure (modified in place and also returned).
    calculator
        Any object conforming to the ``ase.calculators`` interface
        (e.g. ``ase.calculators.lammpslib.LAMMPSlib`` for AIREBO/Tersoff,
        ``ase.calculators.espresso.Espresso`` for DFT).
    algorithm
        ``"lbfgs"`` (default), ``"bfgs"`` or ``"fire"``.
    fmax
        Force-norm convergence threshold in eV/Å.
    max_steps
        Hard cap on optimizer iterations.
    logfile
        Optional path for the ASE optimizer log.

    Returns
    -------
    ase.Atoms
        Relaxed structure.
    """
    key = algorithm.lower()
    if key not in _OPTIMIZERS:
        raise ValueError(f"Unknown algorithm {algorithm!r}. "
                         f"Allowed: {list(_OPTIMIZERS)}.")
    atoms.calc = calculator
    opt_cls = _OPTIMIZERS[key]
    opt = opt_cls(atoms, logfile=logfile)
    opt.run(fmax=fmax, steps=max_steps)
    atoms.info.setdefault("relax_history", []).append(
        {
            "algorithm": key,
            "fmax": fmax,
            "converged": bool(opt.converged()),
            "steps": int(opt.nsteps),
        }
    )
    return atoms


def harmonic_pre_relax(
    atoms: Atoms,
    equilibrium: float = CC_BOND,
    k: float = 20.0,
    steps: int = 200,
    step_size: float = 0.01,
    tol: float = 1e-4,
    max_displacement: float = 0.1,
) -> Atoms:
    """Calculator-free harmonic spring relaxation along existing bonds.

    Every pair connected in the bond graph contributes a force
    ``F = k * (|r| - r_eq) * r̂`` pulling the two atoms toward the
    equilibrium distance. Integrated with simple steepest descent. The
    topology is **frozen**: we guess bonds once from the input and keep
    that connectivity throughout.

    Parameters
    ----------
    atoms
        Input structure (modified in place and also returned).
    equilibrium
        Equilibrium bond length (Å). Defaults to 1.42.
    k
        Spring constant (arbitrary units — only the ratio to
        ``step_size`` matters here).
    steps
        Maximum number of iterations.
    step_size
        Damping factor applied to the displacement each step.
    tol
        Stop when the maximum per-atom displacement drops below ``tol``.

    Returns
    -------
    ase.Atoms
        Pre-relaxed structure. Stats recorded under
        ``atoms.info["harmonic_relax"]``.
    """
    g = build_bond_graph(atoms)
    edges = [(i, j) for i, j in g.edges]
    if not edges:
        atoms.info["harmonic_relax"] = {"edges": 0, "steps": 0}
        return atoms
    pairs = np.array(edges, dtype=int)
    positions = atoms.get_positions().copy()
    pbc = atoms.get_pbc()
    cell = np.array(atoms.cell)
    # Diagonal lengths for fast MIC on axis-aligned cells (our default).
    diag = np.diag(cell).copy()
    is_orthogonal = np.allclose(cell - np.diag(diag), 0.0)
    inv_cell = np.linalg.inv(cell) if not is_orthogonal else None

    def mic_delta(rij: np.ndarray) -> np.ndarray:
        """Minimum-image displacement respecting per-axis PBC flags."""
        if is_orthogonal:
            out = rij.copy()
            for ax in range(3):
                if pbc[ax] and diag[ax] > 0:
                    out[:, ax] -= diag[ax] * np.round(out[:, ax] / diag[ax])
            return out
        frac = rij @ inv_cell
        shift = np.round(frac) * np.array(pbc, dtype=float)
        return rij - shift @ cell

    final_step = 0
    max_move = 0.0
    for step in range(steps):
        final_step = step + 1
        ri = positions[pairs[:, 0]]
        rj = positions[pairs[:, 1]]
        rij = mic_delta(rj - ri)
        dist = np.linalg.norm(rij, axis=1)
        mask = dist > 1e-8
        if not np.any(mask):
            break
        unit = np.zeros_like(rij)
        unit[mask] = rij[mask] / dist[mask][:, None]
        forces = np.zeros_like(positions)
        f_mag = k * (dist - equilibrium)
        np.add.at(forces, pairs[:, 0], (f_mag[:, None] * unit))
        np.add.at(forces, pairs[:, 1], -(f_mag[:, None] * unit))
        disp = step_size * forces
        # Clip per-atom displacement for numerical stability.
        norms = np.linalg.norm(disp, axis=1)
        over = norms > max_displacement
        if np.any(over):
            disp[over] *= (max_displacement / norms[over])[:, None]
        max_move = float(np.linalg.norm(disp, axis=1).max())
        positions += disp
        if max_move < tol:
            break

    atoms.set_positions(positions)
    atoms.info["harmonic_relax"] = {
        "edges": len(edges),
        "steps": final_step,
        "max_move": max_move,
        "equilibrium": equilibrium,
    }
    return atoms
