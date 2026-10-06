"""Hybrid workflows: geometry and phonons from another calculator, optics from TB.

The tight-binding sets give α and μ (Raman, IR) from one electronic structure,
but their phonons carry the repulsion's errors (a few % against GPAW, more
outside the training data). A machine-learned interatomic potential trained on
DFT (MACE-MP, PBE on the Materials Project) or DFT itself can supply the
geometry and the modes, and tbkit the polarizability along them: the
``phonons=(frequencies, L)`` route that Raman, resonant Raman and IR already
take for Quantum ESPRESSO modes.

Run::

    from tbkit.hybrid import mace_calculator, relax_with, external_vibrations
    atoms = relax_with(atoms, mace_calculator)
    vib = external_vibrations(atoms, mace_calculator)
    raman(atoms, model, phonons=(vib.frequencies, vib.modes))

MACE is optional and not a dependency of tbkit (it pulls PyTorch):
``pip install mace-torch``. Any ASE calculator works: pass a function that
returns a new one. What MACE-MP is worth here is measured against the stored
GPAW references by ``recipes/mace_validation.py`` (``docs/VALIDACION.md``);
it is a PBE surrogate, so it inherits PBE's own errors, and outside the
chemistry of its training set it is not checked.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable

import numpy as np
from ase import Atoms

from .modes import Vibrations

Factory = Callable[[], object]


def mace_calculator(model: str = "medium", dtype: str = "float64", device: str = "cpu"):
    """A MACE-MP foundation-model calculator (``pip install mace-torch``)."""
    try:
        from mace.calculators import mace_mp
    except ImportError as error:                     # pragma: no cover - optional
        raise ImportError("MACE no está instalado: pip install mace-torch") from error
    return mace_mp(model=model, device=device, default_dtype=dtype)


def relax_with(atoms: Atoms, make: Factory, fmax: float = 0.01, steps: int = 1000) -> Atoms:
    """Positions relaxed with ``make()`` (cell fixed); returns a copy without calculator."""
    from ase.optimize import BFGS

    moved = atoms.copy()
    moved.calc = make()
    BFGS(moved, logfile=None).run(fmax=fmax, steps=steps)
    out = moved.copy()
    out.calc = None
    return out


def external_vibrations(atoms: Atoms, make: Factory, delta: float = 0.005,
                        source: str = "") -> Vibrations:
    """Γ modes from ``make()``'s forces (central differences), as a tbkit Vibrations."""
    from ase.vibrations import Vibrations as AseVibrations

    probe = atoms.copy()
    probe.calc = make()
    residual = float(np.linalg.norm(probe.get_forces(), axis=1).max())
    with tempfile.TemporaryDirectory() as directory:
        ase_vib = AseVibrations(probe, name=os.path.join(directory, "vib"), delta=delta)
        ase_vib.run()
        hessian = ase_vib.get_vibrations().get_hessian_2d()
    result = Vibrations.from_hessian(atoms, hessian, source=source or type(probe.calc).__name__)
    if residual > 0.05:
        result.warnings.append(f"Fuerza residual {residual:.3f} eV/Å: geometría sin relajar "
                               "con esta calculadora.")
    return result
