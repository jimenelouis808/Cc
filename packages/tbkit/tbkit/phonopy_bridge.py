"""Phonons in the whole Brillouin zone with phonopy and the model's forces.

phonopy (optional: ``pip install tbkit[phonons]``) builds the supercell,
picks the symmetry-independent displacements and turns forces into force
constants; tbkit supplies the forces. From the force constants come the
dispersion along the cell's special-point path, the phonon DOS (total and per
element), the thermal properties (harmonic F, S, Cv) and the irreducible
representations at Γ, which name a mode by symmetry instead of by frequency
(the G mode of graphene is E2g; the check in the tests).

Run::

    result = phonopy_phonons(atoms, model, supercell=(4, 4, 1), cache_dir="out/ph")

Non-periodic axes keep a supercell of 1 and need vacuum (phonopy treats every
cell as periodic; the model does not, so nothing crosses the vacuum). The
k mesh of each supercell force call is the unit cell's divided by the supercell
size, so the electrons are sampled the same way in both. With ``cache_dir``
each displaced supercell's forces are a file and a rerun resumes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms

from .params import TBModel

THZ_TO_CM1 = 33.35641


def _require():
    try:
        import phonopy  # noqa: F401
    except ImportError as error:                         # pragma: no cover - optional
        raise ImportError("phonopy no está instalado: pip install 'tbkit[phonons]' "
                          "(o pip install phonopy).") from error


@dataclass
class PhonopyResult:
    phonopy: object                                       # phonopy.Phonopy with force constants
    supercell: tuple
    displacements: int
    warnings: list = field(default_factory=list)

    def gamma_frequencies(self) -> np.ndarray:
        self.phonopy.run_qpoints([[0, 0, 0]])
        return self.phonopy.get_qpoints_dict()["frequencies"][0] * THZ_TO_CM1


def _phonopy_atoms(atoms: Atoms):
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=np.asarray(atoms.get_cell()),
                        scaled_positions=atoms.get_scaled_positions(wrap=False),
                        masses=atoms.get_masses())


def _supercell_kmesh(atoms: Atoms, kmesh, supercell) -> tuple:
    counts = (kmesh,) * 3 if isinstance(kmesh, int) else tuple(kmesh)
    return tuple(max(1, int(np.ceil(c / s))) for c, s in zip(counts, supercell, strict=True))


def phonopy_phonons(atoms: Atoms, model: TBModel, supercell=(3, 3, 3), delta: float = 0.01,
                    kmesh=12, kT: float = 0.02, scc: Optional[bool] = None,
                    cache_dir: Optional[str | Path] = None, symmetrize: bool = True,
                    progress=None) -> PhonopyResult:
    """Force constants of ``atoms`` (relaxed!) with the model, through phonopy."""
    _require()
    from phonopy import Phonopy

    from .calculator import TBCalculator

    if model.repulsive is None:
        raise ValueError(f"«{model.name}» no tiene parte repulsiva: sin fuerzas no hay fonones.")
    pbc = atoms.get_pbc()
    if not pbc.any():
        raise ValueError("phonopy es para sistemas periódicos; para moléculas usa los modos "
                         "en Γ (tbkit.modes.vibrations).")
    supercell = tuple(int(s) if p else 1 for s, p in zip(supercell, pbc, strict=True))
    phonon = Phonopy(_phonopy_atoms(atoms), supercell_matrix=np.diag(supercell))
    phonon.generate_displacements(distance=delta)
    cells = phonon.supercells_with_displacements
    sc_kmesh = _supercell_kmesh(atoms, kmesh, supercell)
    folder = None
    if cache_dir is not None:
        folder = Path(cache_dir)
        folder.mkdir(parents=True, exist_ok=True)
        key = json.dumps({"model": model.name, "supercell": supercell, "delta": delta,
                          "kmesh": sc_kmesh, "kT": kT, "scc": scc,
                          "positions": np.round(atoms.get_positions(), 6).tolist(),
                          "cell": np.round(np.asarray(atoms.get_cell()), 6).tolist()})
        stamp = hashlib.sha256(key.encode()).hexdigest()[:16]
        tag = folder / "settings.json"
        if tag.exists() and json.loads(tag.read_text())["stamp"] != stamp:
            raise ValueError(f"{folder} tiene fuerzas de otro cálculo (otra estructura, modelo o "
                             "ajustes): usa otra carpeta.")
        tag.write_text(json.dumps({"stamp": stamp, "settings": json.loads(key)}, indent=1))
    forces = []
    for index, cell in enumerate(cells):
        path = folder / f"forces_{index:04d}.npy" if folder else None
        if path is not None and path.exists():
            forces.append(np.load(path))
            continue
        probe = Atoms(cell.symbols, cell=cell.cell, scaled_positions=cell.scaled_positions,
                      pbc=pbc)
        probe.calc = TBCalculator(model, kpts=sc_kmesh, kT=kT, scc=scc)
        f = probe.get_forces()
        f -= f.mean(axis=0)                     # drift: residual of the finite cell sum
        if path is not None:
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, f)
            tmp.replace(path)
        forces.append(f)
        if progress is not None:
            progress(index + 1, len(cells))
    phonon.forces = np.array(forces)
    phonon.produce_force_constants()
    if symmetrize:
        phonon.symmetrize_force_constants()
    warnings = []
    if any(s < 3 for s, p in zip(supercell, pbc, strict=True) if p):
        warnings.append("Superceldas de menos de 3 a lo largo de un eje periódico truncan las "
                        "constantes de fuerza: la dispersión lejos de Γ es aproximada.")
    return PhonopyResult(phonon, supercell, len(cells), warnings)


def dispersion(result: PhonopyResult, atoms: Atoms, npoints: int = 120) -> dict:
    """Frequencies (cm⁻¹) along the special-point path of the cell (periodic axes only)."""
    path = atoms.cell.bandpath(npoints=npoints, pbc=atoms.get_pbc())
    x, ticks, labels = path.get_linear_kpoint_axis()
    phonon = result.phonopy
    phonon.run_qpoints(path.kpts)
    frequencies = phonon.get_qpoints_dict()["frequencies"] * THZ_TO_CM1
    labels = [label.replace("G", "Γ") for label in labels]
    return {"x": np.asarray(x), "frequencies": frequencies, "ticks": np.asarray(ticks),
            "labels": labels, "path": path.path}


def phonon_dos(result: PhonopyResult, atoms: Atoms, mesh=20, sigma_cm1: float = 10.0) -> dict:
    """Total DOS and per element (states/cm⁻¹ per unit cell), Gaussian σ in cm⁻¹."""
    phonon = result.phonopy
    counts = [mesh if p else 1 for p in atoms.get_pbc()] if isinstance(mesh, int) else list(mesh)
    phonon.run_mesh(counts, with_eigenvectors=True, is_mesh_symmetry=False, is_gamma_center=True)
    phonon.run_projected_dos(sigma=sigma_cm1 / THZ_TO_CM1)
    data = phonon.get_projected_dos_dict()
    grid = data["frequency_points"] * THZ_TO_CM1
    projected = data["projected_dos"] / THZ_TO_CM1
    symbols = np.array(atoms.get_chemical_symbols())
    out = {"grid": grid, "total": projected.sum(axis=0)}
    for element in sorted(set(symbols)):
        out[element] = projected[symbols == element].sum(axis=0)
    return out


def thermal(result: PhonopyResult, atoms: Atoms, temperatures=(300.0,), mesh=20) -> list[dict]:
    """Harmonic free energy, entropy and heat capacity per unit cell at each T."""
    phonon = result.phonopy
    counts = [mesh if p else 1 for p in atoms.get_pbc()] if isinstance(mesh, int) else list(mesh)
    phonon.run_mesh(counts, is_gamma_center=True)
    temps = np.atleast_1d(np.asarray(temperatures, dtype=float))
    phonon.run_thermal_properties(temperatures=temps)
    data = phonon.get_thermal_properties_dict()
    # phonopy: kJ/mol (F) and J/K/mol (S, Cv) per unit cell
    return [{"T_K": float(t), "F_eV": float(f) / 96.48533212, "S_meV_per_K": float(s) / 96.48533212,
             "Cv_meV_per_K": float(c) / 96.48533212}
            for t, f, s, c in zip(data["temperatures"], data["free_energy"], data["entropy"],
                                  data["heat_capacity"], strict=True)]


def gamma_irreps(result: PhonopyResult, tolerance: float = 1e-4) -> list[dict]:
    """Γ modes grouped by degeneracy with their irreducible representation (point group)."""
    phonon = result.phonopy
    run = getattr(phonon, "run_irreps", None) or phonon.set_irreps
    run([0, 0, 0], degeneracy_tolerance=tolerance)
    irreps = phonon.irreps
    frequencies = np.asarray(irreps.frequencies) * THZ_TO_CM1
    labels = getattr(irreps, "_ir_labels", None)
    rows = []
    for k, indices in enumerate(irreps.band_indices):
        label = labels[k] if labels is not None and labels[k] is not None else None
        frequency = float(frequencies[indices[0]])
        if label is None:
            label = "acústicos" if abs(frequency) < 5.0 else "?"
        rows.append({"frequency_cm1": frequency, "degeneracy": len(indices),
                     "irrep": str(label), "bands": [int(i) for i in indices]})
    return rows


def point_group(result: PhonopyResult) -> str:
    return str(result.phonopy.symmetry.dataset.pointgroup)
