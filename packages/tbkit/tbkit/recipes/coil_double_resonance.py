"""Double-resonance Raman (D, 2D) of the 204-atom 5-7 coil with xu_chn.

Run from packages/tbkit (resumable; files in out/coil_dr/)::

    python -m tbkit.recipes.coil_double_resonance fc [--part r/n]   # force constants
    python -m tbkit.recipes.coil_double_resonance check             # D(q=0) vs the Γ Hessian

Stage 2 of the double-resonance work: the phonons of the coil at any q along its
axis. The coil is periodic along z only, with a 15 Å period; force constants Φ(R)
between the cell and its neighbours R = −1, 0, +1 come from finite displacements of
the central cell's atoms in a 1×1×3 supercell (xu_chn with SCC, 1×1×2 k — the 1×1×4
of the primitive cell's Raman work, in the longer cell — kT 0.05 eV, ±0.01 Å). The
dynamical matrix D(q) is then their Fourier sum, exact at q = 0, ±1/3 (in 2π/c) and
smooth in between, as for graphene in ``tbkit.graphene``. ``check`` compares D(0)
with the Γ Hessian of ``recipes/doped_raman`` (1×1×4 k in the primitive cell).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read

WORK = Path("out/coil_dr")
#: xu_carbon (Xu's sp³ carbon, no SCC): a 612-atom supercell force call takes ~14 min
#: with xu_chn's self-consistent charges and ~1-2 with xu_carbon; for the undoped coil
#: xu_chn's C-C is Xu's, so the difference is the charge transfer between rings only.
MODEL = "xu_carbon"
SOURCE = Path("out/nanocoil/tb_xu_carbon/relaxed.extxyz")
GAMMA_MODES = Path("out/nanocoil/tb_xu_carbon/modes.npz")
REPEAT, KZ, KT, DELTA = 3, 2, 0.05, 0.01


def primitive():
    return read(SOURCE)


def supercell():
    return primitive().repeat((1, 1, REPEAT))


def _calculator():
    from ..calculator import TBCalculator
    from ..params import load_parameters

    return TBCalculator(load_parameters(MODEL), kpts=(1, 1, KZ), kT=KT)


def fc(part: tuple[int, int] = (0, 1)) -> None:
    """Forces of the supercell with each atom of the central cell displaced ±DELTA along
    x, y, z: one file per atom (``fc/a{atom}.npy``, shape (3, 2, n_super, 3))."""
    folder = WORK / "fc"
    folder.mkdir(parents=True, exist_ok=True)
    n = len(primitive())
    big = supercell()
    centre = n * (REPEAT // 2)                    # atoms of cell 1 (of 0, 1, 2)
    calc = _calculator()
    r, total = part
    for a in range(n):
        if a % total != r:
            continue
        path = folder / f"a{a:03d}.npy"
        if path.exists():
            continue
        forces = np.zeros((3, 2, len(big), 3))
        for alpha in range(3):
            for s, step in enumerate((DELTA, -DELTA)):
                moved = big.copy()
                moved.positions[centre + a, alpha] += step
                moved.calc = calc
                forces[alpha, s] = moved.get_forces()
                calc.reset()
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, forces)
        tmp.replace(path)


def force_constants() -> np.ndarray:
    """Φ[R, a, α, b, β] (eV/Å²), R = −1, 0, +1, from the central-cell displacements;
    symmetrised (Φ_ab(R) = Φ_ba(−R)) and with the acoustic sum rule imposed."""
    n = len(primitive())
    phi = np.zeros((3, n, 3, n, 3))
    for a in range(n):
        f = np.load(WORK / "fc" / f"a{a:03d}.npy")
        d = -(f[:, 0] - f[:, 1]) / (2 * DELTA)          # (α, n_super, 3) = Φ[a α, j β]
        for cell in range(REPEAT):
            phi[cell, a] = d[:, cell * n:(cell + 1) * n, :]
    # cells 0, 1, 2 of the supercell are R = −1, 0, +1 relative to the centre
    phi = 0.5 * (phi + np.transpose(phi[::-1], (0, 3, 4, 1, 2)))
    for a in range(n):
        for alpha in range(3):
            for beta in range(3):
                phi[1, a, alpha, a, beta] -= phi[:, a, alpha, :, beta].sum()
    return phi


def dynamical_matrix(phi: np.ndarray, masses: np.ndarray, q: float) -> np.ndarray:
    """D(q) for q in units of 2π/c (lattice convention: phase e^{2πi q R})."""
    n = len(masses)
    d = sum(phi[i] * np.exp(2j * np.pi * q * r) for i, r in enumerate((-1, 0, 1)))
    d = d.reshape(3 * n, 3 * n)
    m = np.repeat(masses, 3)
    return d / np.sqrt(np.outer(m, m))


def check() -> dict:
    from ase.units import invcm

    atoms = primitive()
    masses = atoms.get_masses()
    phi = force_constants()
    w2 = np.linalg.eigvalsh(0.5 * (dynamical_matrix(phi, masses, 0.0)
                                   + dynamical_matrix(phi, masses, 0.0).conj().T)).real
    from ase.units import _amu, _e, _hbar
    to_cm1 = np.sqrt(_e / _amu) * 1e10 * _hbar / _e / invcm
    ours = np.sign(w2) * np.sqrt(np.abs(w2)) * to_cm1
    ref = np.sort(np.load(GAMMA_MODES)["frequencies"]) if "frequencies" in \
        np.load(GAMMA_MODES).files else np.sort(np.load(GAMMA_MODES)["f"])
    diff = ours[6:] - ref[6:]
    out = {"modes": len(ours), "max_abs_diff_cm1": float(np.abs(diff).max()),
           "rms_diff_cm1": float(np.sqrt(np.mean(diff ** 2))),
           "highest_cm1": [float(ours[-1]), float(ref[-1])]}
    (WORK / "check.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("fc", "check"))
    parser.add_argument("--part", default="0/1")
    args = parser.parse_args(argv)
    if args.step == "fc":
        r, n = (int(x) for x in args.part.split("/"))
        fc((r, n))
    else:
        print(json.dumps(check(), indent=1))


if __name__ == "__main__":
    main()
