"""Quantum ESPRESSO (pw.x) input writer.

Given an :class:`ase.Atoms`, we write a complete ``pw.x`` input containing
``&CONTROL``, ``&SYSTEM``, ``&ELECTRONS`` (+ ``&IONS`` / ``&CELL`` when the
calculation is a relaxation) and the ``ATOMIC_SPECIES``, ``ATOMIC_POSITIONS``,
``CELL_PARAMETERS`` and ``K_POINTS`` cards.

Dimensionality, vacuum and k-point mesh are **auto-detected** from the
``pbc`` flags and the cell:

* 0 periodic axes → 1 × 1 × 1 k-mesh, ``assume_isolated='mp'`` (makov-payne).
* 1 periodic axis → nk × 1 × 1 along that axis.
* 2 periodic axes → nk × nk × 1, ``assume_isolated='2D'``.
* 3 periodic axes → full Monkhorst-Pack mesh.

The k-point density (points per Å⁻¹ along a reciprocal axis) is configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from ase import Atoms

from ..validation.checks import run_basic_checks


Calculation = Literal["scf", "relax", "vc-relax", "nscf", "bands"]


@dataclass
class QESettings:
    """Parameters inferred (or user-provided) for a QE pw.x run."""

    calculation: Calculation = "scf"
    prefix: str = "pwscf"
    pseudo_dir: str = "./pseudo"
    outdir: str = "./out"
    ecutwfc: float = 60.0
    ecutrho: float = 480.0
    kpoint_density: float = 0.20  # 1/Å, ~ 2π/kpoint_density Å-1 spacing
    occupations: str = "smearing"
    smearing: str = "mv"
    degauss: float = 0.01
    conv_thr: float = 1.0e-8
    mixing_beta: float = 0.4
    pseudopotentials: dict[str, str] | None = None
    assume_isolated: Optional[str] = None
    dimensionality: Optional[int] = None


_DEFAULT_PSEUDOS: dict[str, str] = {
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "N": "N.pbe-n-kjpaw_psl.1.0.0.UPF",
    "B": "B.pbe-n-kjpaw_psl.1.0.0.UPF",
    "S": "S.pbe-n-kjpaw_psl.1.0.0.UPF",
    "P": "P.pbe-n-kjpaw_psl.1.0.0.UPF",
    "H": "H.pbe-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
}


_ATOMIC_MASSES: dict[str, float] = {
    "C": 12.011,
    "N": 14.007,
    "B": 10.811,
    "S": 32.065,
    "P": 30.974,
    "H": 1.008,
    "O": 15.999,
}


def _kpoint_mesh(atoms: Atoms, density: float) -> tuple[int, int, int]:
    """Monkhorst-Pack mesh proportional to reciprocal-lattice length."""
    pbc = atoms.get_pbc()
    cell = np.array(atoms.cell)
    recip = 2.0 * np.pi * np.linalg.inv(cell).T  # columns are b_i
    mesh = [1, 1, 1]
    for ax in range(3):
        if not pbc[ax]:
            continue
        b_len = np.linalg.norm(recip[ax])
        mesh[ax] = max(1, int(np.ceil(b_len / density)))
    return tuple(mesh)  # type: ignore[return-value]


def infer_qe_settings(atoms: Atoms, base: Optional[QESettings] = None) -> QESettings:
    """Build a :class:`QESettings` by introspecting the structure.

    Parameters
    ----------
    atoms
        Structure to export.
    base
        Optional user-provided baseline; fields that are already set on it
        are preserved.

    Returns
    -------
    QESettings
        Ready-to-write settings. ``dimensionality`` and ``assume_isolated``
        are filled in based on ``atoms.pbc``.
    """
    s = QESettings() if base is None else base
    dim = int(sum(atoms.get_pbc()))
    s.dimensionality = dim
    if s.assume_isolated is None:
        if dim == 2:
            s.assume_isolated = "2D"
        elif dim == 0:
            s.assume_isolated = "mp"
    if s.pseudopotentials is None:
        s.pseudopotentials = {
            sym: _DEFAULT_PSEUDOS.get(sym, f"{sym}.UPF")
            for sym in set(atoms.get_chemical_symbols())
        }
    return s


def _fmt_namelist(name: str, fields: dict[str, object]) -> str:
    lines = [f"&{name}"]
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, bool):
            val = ".true." if v else ".false."
        elif isinstance(v, str):
            val = f"'{v}'"
        elif isinstance(v, float):
            val = f"{v:.8g}"
        else:
            val = str(v)
        lines.append(f"    {k} = {val}")
    lines.append("/")
    return "\n".join(lines)


def write_qe_input(
    atoms: Atoms,
    outdir: str | Path,
    settings: Optional[QESettings] = None,
    filename: str = "pw.in",
    force: bool = False,
) -> Path:
    """Write a complete pw.x input file.

    Parameters
    ----------
    atoms
        Structure to export.
    outdir
        Directory where the file is written. Created if missing.
    settings
        Optional user-supplied :class:`QESettings`. Missing fields are
        filled in by :func:`infer_qe_settings`.
    filename
        Output file name (default ``pw.in``).
    force
        If ``False`` (default) the export is aborted when validation reports
        any error. Pass ``True`` to bypass (e.g. for batch pre-relaxation).

    Returns
    -------
    pathlib.Path
        Path to the written file.

    Raises
    ------
    ValueError
        If validation fails and ``force`` is ``False``.
    """
    report = run_basic_checks(atoms)
    if not report.ok and not force:
        raise ValueError(
            "Structure failed validation, refusing to export:\n" + report.summary()
        )

    s = infer_qe_settings(atoms, base=settings)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    symbols = atoms.get_chemical_symbols()
    unique = sorted(set(symbols))
    mesh = _kpoint_mesh(atoms, s.kpoint_density)

    control = {
        "calculation": s.calculation,
        "prefix": s.prefix,
        "pseudo_dir": s.pseudo_dir,
        "outdir": s.outdir,
        "tprnfor": True,
        "tstress": True,
        "verbosity": "high",
    }
    system = {
        "ibrav": 0,
        "nat": len(atoms),
        "ntyp": len(unique),
        "ecutwfc": s.ecutwfc,
        "ecutrho": s.ecutrho,
        "occupations": s.occupations,
        "smearing": s.smearing,
        "degauss": s.degauss,
        "assume_isolated": s.assume_isolated,
    }
    electrons = {
        "conv_thr": s.conv_thr,
        "mixing_beta": s.mixing_beta,
    }

    parts = [
        _fmt_namelist("CONTROL", control),
        _fmt_namelist("SYSTEM", system),
        _fmt_namelist("ELECTRONS", electrons),
    ]
    if s.calculation in ("relax", "vc-relax"):
        parts.append(_fmt_namelist("IONS", {"ion_dynamics": "bfgs"}))
    if s.calculation == "vc-relax":
        parts.append(_fmt_namelist("CELL", {"cell_dynamics": "bfgs"}))

    # Cards ---------------------------------------------------------------
    species_lines = ["ATOMIC_SPECIES"]
    pseudos = s.pseudopotentials or {}
    for sym in unique:
        mass = _ATOMIC_MASSES.get(sym, 1.0)
        pp = pseudos.get(sym, f"{sym}.UPF")
        species_lines.append(f"  {sym}  {mass:.4f}  {pp}")
    parts.append("\n".join(species_lines))

    cell_lines = ["CELL_PARAMETERS angstrom"]
    for row in atoms.cell:
        cell_lines.append(f"  {row[0]:20.12f} {row[1]:20.12f} {row[2]:20.12f}")
    parts.append("\n".join(cell_lines))

    pos_lines = ["ATOMIC_POSITIONS angstrom"]
    for sym, p in zip(symbols, atoms.get_positions()):
        pos_lines.append(f"  {sym}  {p[0]:20.12f} {p[1]:20.12f} {p[2]:20.12f}")
    parts.append("\n".join(pos_lines))

    k_line = f"K_POINTS automatic\n  {mesh[0]} {mesh[1]} {mesh[2]} 0 0 0"
    parts.append(k_line)

    path = outdir / filename
    path.write_text("\n\n".join(parts) + "\n")
    return path
