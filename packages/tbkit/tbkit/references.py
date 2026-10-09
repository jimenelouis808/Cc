"""Reference data from DFT for fitting: structures, levels, energies, forces.

A reference set is a list of :class:`ReferenceStructure`, each a geometry
with what a DFT code computed for it: the Kohn-Sham levels, the total
energy and the forces. It is saved as one JSON file together with the
settings of the calculation (code, version, functional, basis, grid...),
so a fit can be repeated -- and checked by the tests -- without the DFT
code installed.

:func:`generate_gpaw` produces such a set with GPAW: each molecule is
relaxed, then displaced at random and scaled uniformly, so the data cover
bond lengths around equilibrium (distance dependence of the hoppings) and
the forces off equilibrium (the repulsive part). GPAW is imported only
there; the rest of tbkit never needs it.

The zero of the Kohn-Sham levels of a finite system with zero boundary
conditions (GPAW's LCAO/FD default for ``pbc=False``) is the vacuum, the
same for every molecule; a fit may therefore use one common shift between
model and DFT levels for all of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
from ase import Atoms

#: GPAW settings used when none are given. LCAO with a double-zeta
#: polarised basis: its empty levels are basis-limited, which is why a fit
#: should weight only the lowest few of them.
GPAW_DEFAULTS = {"mode": "lcao", "basis": "dzp", "xc": "PBE", "h": 0.2, "vacuum": 5.0,
                 "convergence": {"density": 1e-6}, "extra_bands": 6, "fmax": 0.02}


@dataclass
class ReferenceStructure:
    """One geometry and its DFT results (eV, Å)."""

    label: str
    group: str                          # the molecule it derives from
    atoms: Atoms
    energy: float                       # total energy (code's own zero)
    forces: np.ndarray                  # (N, 3) eV/Å
    levels: np.ndarray                  # Kohn-Sham levels, eV, ascending
    n_occupied: int                     # doubly occupied levels (closed shell)
    role: str = "train"                 # "train" or "test"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {"label": self.label, "group": self.group, "role": self.role,
                "symbols": self.atoms.get_chemical_symbols(),
                "positions": np.round(self.atoms.get_positions(), 8).tolist(),
                "energy": self.energy, "forces": np.round(self.forces, 8).tolist(),
                "levels": np.round(self.levels, 6).tolist(), "n_occupied": self.n_occupied,
                "extra": self.extra}
        if self.atoms.pbc.any():
            # A crystal: without its cell the positions mean nothing.
            data["cell"] = np.round(self.atoms.cell.array, 8).tolist()
            data["pbc"] = [bool(p) for p in self.atoms.pbc]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ReferenceStructure":
        atoms = Atoms(data["symbols"], positions=data["positions"],
                      cell=data.get("cell"), pbc=data.get("pbc", False))
        return cls(data["label"], data["group"], atoms, float(data["energy"]),
                   np.array(data["forces"], dtype=float), np.array(data["levels"], dtype=float),
                   int(data["n_occupied"]), data.get("role", "train"), data.get("extra", {}))


def save_references(path: str | Path, structures: Iterable[ReferenceStructure],
                    settings: dict) -> Path:
    """Write a reference set (and the settings that produced it) as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"settings": settings, "structures": [s.to_dict() for s in structures]}
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def load_references(path: str | Path) -> tuple[list[ReferenceStructure], dict]:
    """``(structures, settings)`` from a file written by :func:`save_references`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ReferenceStructure.from_dict(s) for s in data["structures"]], data["settings"]


def distortions(atoms: Atoms, n_random: int = 4, sigma: float = 0.04,
                scales: tuple[float, ...] = (0.95, 1.05), seed: int = 0) -> list[tuple[str, Atoms]]:
    """Geometries around ``atoms``: random displacements and uniform scalings.

    Random displacements (Gaussian, ``sigma`` Å per component) probe forces
    and bond angles; a uniform scaling changes every bond length together
    and probes how levels depend on distance.
    """
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_random):
        moved = atoms.copy()
        moved.positions += rng.normal(0.0, sigma, size=moved.positions.shape)
        out.append((f"rnd{k}", moved))
    centre = atoms.get_positions().mean(axis=0)
    for scale in scales:
        scaled = atoms.copy()
        scaled.positions = centre + scale * (atoms.get_positions() - centre)
        out.append((f"x{scale:.2f}", scaled))
    return out


def _valence_electrons(calc) -> int:
    return int(round(calc.get_number_of_electrons()))


def gpaw_calculator(settings: dict, n_bands: int, txt=None):
    """A GPAW calculator from the settings (imports GPAW).

    Symmetry is off: displaced geometries (distortions, finite-difference
    vibrations) break the symmetry GPAW would detect in the first one.
    """
    from gpaw import GPAW

    return GPAW(mode=settings["mode"], basis=settings["basis"], xc=settings["xc"],
                h=settings["h"], convergence=settings["convergence"], nbands=n_bands,
                txt=txt, spinpol=False, symmetry="off")


def _n_bands(atoms: Atoms, settings: dict) -> int:
    valence = {"H": 1, "C": 4, "N": 5, "O": 6, "B": 3, "P": 5, "S": 6, "Se": 6}
    electrons = sum(valence[s] for s in atoms.get_chemical_symbols())
    return electrons // 2 + int(settings["extra_bands"])


def gpaw_single_point(atoms: Atoms, settings: dict) -> tuple[float, np.ndarray, np.ndarray, int]:
    """``(energy, forces, levels, n_occupied)`` of a closed-shell molecule with GPAW."""
    from .progress import watch_gpaw_scf

    atoms = atoms.copy()
    atoms.calc = gpaw_calculator(settings, _n_bands(atoms, settings))
    watch_gpaw_scf(atoms.calc, f"GPAW SCF ({atoms.get_chemical_formula()})")
    energy = float(atoms.get_potential_energy())
    forces = np.array(atoms.get_forces())
    levels = np.sort(np.array(atoms.calc.get_eigenvalues()))
    electrons = _valence_electrons(atoms.calc)
    if electrons % 2:
        raise ValueError("Referencia de capa abierta: solo moléculas de capa cerrada.")
    return energy, forces, levels, electrons // 2


def gpaw_relax(atoms: Atoms, settings: dict, log: Optional[str] = None) -> Atoms:
    """Relax a molecule with GPAW (BFGS to ``settings['fmax']``)."""
    from ase.optimize import BFGS

    from .progress import Progress, watch_optimizer

    atoms = atoms.copy()
    atoms.calc = gpaw_calculator(settings, _n_bands(atoms, settings))
    optimizer = BFGS(atoms, logfile=log)
    with Progress(None, f"GPAW relajación ({atoms.get_chemical_formula()})") as bar:
        watch_optimizer(optimizer, bar)
        optimizer.run(fmax=settings["fmax"], steps=200)
    relaxed = atoms.copy()
    relaxed.calc = None
    return relaxed


def generate_gpaw(molecules: dict[str, Atoms], settings: Optional[dict] = None,
                  role: str = "train", n_random: int = 4, sigma: float = 0.04,
                  scales: tuple[float, ...] = (0.95, 1.05), seed: int = 0,
                  progress: Optional[Callable[[str], None]] = None
                  ) -> list[ReferenceStructure]:
    """Relax each molecule with GPAW and compute it and its distortions.

    ``n_random=0`` and ``scales=()`` keep only the relaxed geometry (a test
    set). Molecules must be closed-shell and finite.
    """
    from .progress import Progress

    settings = {**GPAW_DEFAULTS, **(settings or {})}
    out = []
    bar = Progress(len(molecules) * (2 + n_random + len(scales)),
                   "referencias GPAW (relajación + geometrías por molécula)")
    for index, (name, atoms) in enumerate(molecules.items()):
        atoms = atoms.copy()
        atoms.pbc = False
        atoms.center(vacuum=settings["vacuum"])
        relaxed = gpaw_relax(atoms, settings)
        bar.step(note=f"{name} relajada")
        geometries = [("eq", relaxed)] + distortions(relaxed, n_random, sigma, scales,
                                                      seed + index)
        for tag, geometry in geometries:
            energy, forces, levels, n_occ = gpaw_single_point(geometry, settings)
            out.append(ReferenceStructure(f"{name}/{tag}", name, geometry, energy, forces,
                                          levels, n_occ, role))
            bar.step(note=f"{name}/{tag}")
            if progress:
                progress(f"{name}/{tag}: E = {energy:.4f} eV, |F|max = "
                         f"{np.abs(forces).max():.3f} eV/Å")
    return out


def gpaw_hessian(atoms: Atoms, settings: Optional[dict] = None,
                 delta: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """``(hessian, frequencies)`` with GPAW: the (3N, 3N) Hessian in eV/Å² (central
    differences of the forces, ``ase.vibrations``) and the harmonic frequencies
    (cm⁻¹, ascending, imaginary as negative, rigid-body modes included).

    ``atoms`` should be the GPAW-relaxed geometry. The Hessian is what a fit
    can use directly: it is linear in the pair-repulsion coefficients.
    """
    import os
    import tempfile

    from ase.units import invcm
    from ase.vibrations import Vibrations

    settings = {**GPAW_DEFAULTS, **(settings or {})}
    atoms = atoms.copy()
    atoms.pbc = False
    atoms.center(vacuum=settings["vacuum"])      # a translation: frequencies unchanged
    from .progress import Progress, count_calculations

    atoms.calc = gpaw_calculator(settings, _n_bands(atoms, settings))
    bar = Progress(6 * len(atoms) + 1, f"GPAW Hessiana ({atoms.get_chemical_formula()})")
    count_calculations(atoms.calc, bar)
    with tempfile.TemporaryDirectory() as directory:
        vibrations = Vibrations(atoms, name=os.path.join(directory, "vib"), delta=delta)
        vibrations.run()
        bar.close()
        data = vibrations.get_vibrations()
        hessian = data.get_hessian_2d()
        energies = data.get_energies()
    values = np.where(np.abs(energies.imag) > np.abs(energies.real), -np.abs(energies.imag),
                      np.abs(energies.real)) / invcm
    return hessian, np.sort(values)


def gpaw_frequencies(atoms: Atoms, settings: Optional[dict] = None,
                     delta: float = 0.01) -> np.ndarray:
    """Harmonic frequencies (cm⁻¹) with GPAW; see :func:`gpaw_hessian`."""
    return gpaw_hessian(atoms, settings, delta)[1]


#: GPAW settings for polarizabilities: a real-space grid (FD), which, unlike
#: an LCAO dzp basis, is not short of diffuse functions.
GPAW_ALPHA_DEFAULTS = {"mode": "fd", "xc": "PBE", "h": 0.18, "vacuum": 6.0,
                       "convergence": {"density": 1e-7, "eigenstates": 1e-10},
                       "field_v_per_angstrom": 0.01}


def gpaw_polarizability(atoms: Atoms, settings: Optional[dict] = None) -> np.ndarray:
    """Static α (Å³, 3x3) of a closed-shell molecule by ±E finite field with GPAW.

    ``α_ij = ∂μ_i/∂E_j`` by central differences (six SCF runs), converted
    from e·Å²/V with 1/(4πε0) = 14.3996 eV·Å/e².
    """
    from gpaw import GPAW
    from gpaw.external import ConstantElectricField

    settings = {**GPAW_ALPHA_DEFAULTS, **(settings or {})}
    atoms = atoms.copy()
    atoms.pbc = False
    atoms.center(vacuum=settings["vacuum"])
    field = float(settings["field_v_per_angstrom"])
    alpha = np.zeros((3, 3))
    for axis in range(3):
        dipoles = []
        for sign in (1.0, -1.0):
            direction = [0.0, 0.0, 0.0]
            direction[axis] = 1.0
            atoms.calc = GPAW(mode=settings["mode"], xc=settings["xc"], h=settings["h"],
                              convergence=settings["convergence"], spinpol=False,
                              symmetry="off", txt=None,
                              external=ConstantElectricField(sign * field, direction))
            atoms.get_potential_energy()
            dipoles.append(np.array(atoms.get_dipole_moment()))
        alpha[:, axis] = (dipoles[0] - dipoles[1]) / (2 * field) * 14.399645
    return 0.5 * (alpha + alpha.T)


def _valence_shells(symbol: str) -> tuple[int, Optional[int]]:
    """Principal quantum numbers of the valence s and p shells (p None for H, He)."""
    from gpaw.atom.configurations import configurations

    shells = configurations[symbol][1]
    n_s = max(n for n, ell, f, _ in shells if ell == 0 and f > 0)
    n_p = [n for n, ell, _, _ in shells if ell == 1 and n == n_s]
    return n_s, (n_p[0] if n_p else None)


def _gpaw_atom(symbol: str, xc: str = "PBE", extra: float = 0.0):
    """GPAW's all-electron atom, spin-paired, with ``extra`` electrons in the
    valence p shell (the s shell for H)."""
    from gpaw.atom.aeatom import AllElectronAtom

    atom = AllElectronAtom(symbol, xc=xc, spinpol=False, log=None)
    if extra:
        n_s, n_p = _valence_shells(symbol)
        atom.add(*((n_p, 1) if n_p else (n_s, 0)), extra)
    atom.run()
    return atom


def gpaw_atom_levels(symbol: str, xc: str = "PBE", extra: float = 0.0) -> dict[str, float]:
    """Valence levels ``{"s": ε_s, "p": ε_p}`` (eV) of the free atom, GPAW's atom."""
    from ase.units import Hartree

    atom = _gpaw_atom(symbol, xc, extra)
    n_s, n_p = _valence_shells(symbol)
    levels = {"s": float(atom.channels[0].e_n[n_s - 1] * Hartree)}
    if n_p:
        levels["p"] = float(atom.channels[1].e_n[n_p - 2] * Hartree)
    return levels


def gpaw_hubbard_u(symbol: str, xc: str = "PBE", delta: float = 0.05) -> float:
    """Hubbard U (eV) = dε/dn of the valence level (p; s for H), central
    differences of ±``delta`` electrons: the DFTB definition."""
    shell = "p" if _valence_shells(symbol)[1] else "s"
    plus = gpaw_atom_levels(symbol, xc, delta)[shell]
    minus = gpaw_atom_levels(symbol, xc, -delta)[shell]
    return (plus - minus) / (2 * delta)


def gpaw_onsite_dipole(symbol: str, xc: str = "PBE") -> float:
    """``d = (1/√3) ∫ R_ns R_np r³ dr`` of the free atom (Å), with GPAW's atom.

    Radial functions of the valence s and p shells (2s/2p, 3s/3p; spin-paired),
    each with its outer lobe positive; the magnitude used by :mod:`tbkit.dipoles`.
    """
    from ase.units import Bohr

    atom = _gpaw_atom(symbol, xc)
    r = atom.rgd.r_g
    n_s, n_p = _valence_shells(symbol)
    s = atom.channels[0].phi_ng[n_s - 1]
    p = atom.channels[1].phi_ng[n_p - 2]
    outer = np.searchsorted(r, 4.0)
    s, p = s * np.sign(s[outer]), p * np.sign(p[outer])
    return float(atom.rgd.integrate(s * p * r) / (4 * np.pi) / np.sqrt(3) * Bohr)


def gpaw_settings_record(settings: Optional[dict] = None) -> dict:
    """The settings plus the versions of GPAW and ASE, for the reference file."""
    import ase
    import gpaw

    return {"code": "GPAW", "gpaw_version": gpaw.__version__, "ase_version": ase.__version__,
            "spin": "closed shell, spin-paired", "boundary": "zero (pbc=False)",
            **GPAW_DEFAULTS, **(settings or {})}
