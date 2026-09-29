"""Hand the current structure to another program's window (tbkit's), by file.

carbonforge never imports tbkit (the workspace rule): it writes the structure
to an extxyz file and starts ``tbkit-gui FILE`` as a separate process, the way
it hands inputs to QE or LAMMPS. No Tk here, so it is tested without a display.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from ase import Atoms


def tbkit_command() -> Optional[list[str]]:
    """How to start tbkit's window here, or None if tbkit is not installed.

    The ``tbkit-gui`` script when it is on the PATH; otherwise the module run
    with this interpreter, when the package can be found (located, not
    imported)."""
    script = shutil.which("tbkit-gui")
    if script:
        return [script]
    if importlib.util.find_spec("tbkit") is not None:
        return [sys.executable, "-m", "tbkit.gui"]
    return None


def write_for_tbkit(atoms: Atoms, directory: Optional[Path] = None) -> Path:
    """The structure as extxyz (cell and periodicity kept) in a fresh file."""
    from ase.io import write

    directory = Path(directory) if directory else Path(tempfile.mkdtemp(prefix="carbonforge-tb-"))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{atoms.get_chemical_formula()}.extxyz"
    write(path, atoms, format="extxyz")
    return path


def open_in_tbkit(atoms: Atoms, directory: Optional[Path] = None, launch=subprocess.Popen):
    """Write ``atoms`` and start tbkit's window on it; returns ``(path, command)``.

    Raises ``RuntimeError`` with the install hint when tbkit is missing."""
    command = tbkit_command()
    if command is None:
        raise RuntimeError("tbkit no está instalado en este entorno "
                           "(pip install \"tbkit[gui]\", o uv sync --extra gui).")
    path = write_for_tbkit(atoms, directory)
    launch(command + [str(path)])
    return path, command
