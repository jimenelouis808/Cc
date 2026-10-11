"""A snapshot of the window's session, to pick the work up later: ``Archivo → Guardar proyecto``.

One ``.tbkproj`` file (a zip) holds:

* ``proyecto.json``: format version, date, tbkit version, which geometry is active, the
  window's settings (model name, charge, SCC, kT, cores, RAM, repeat) and every page's
  controls (by attribute name), the recipe tables, the terminal's text;
* ``geometrias/NN.extxyz``: every geometry of the session, in order, with its name;
* ``modelo.json``: the full parameter set in use (``params.model_to_dict``), so a project
  made with a parameter file of one's own reopens on another machine;
* ``modos/modes.npz`` + ``modos/structure.extxyz``: the Γ modes, if computed (reopened by
  diagonalising the stored Hessian, without recomputing forces);
* ``registro.json``: the record of the last calculation (``tbkit.record``), if any.

Results that take seconds to redo (ground state, DOS, spectra) are not stored: the controls
that produced them are, and «Calcular» redoes them. Nothing here imports Qt.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ase import Atoms
from ase.io import read, write

SCHEMA = "tbkit-proyecto/1"
SUFFIX = ".tbkproj"


@dataclass
class Project:
    geometries: list[tuple[str, Atoms]] = field(default_factory=list)
    active: int = 0
    window: dict = field(default_factory=dict)       # model name, charge, scc, kT, …
    pages: dict = field(default_factory=dict)        # {page title: {attribute: value}}
    model: dict | None = None                        # params.model_to_dict of the model
    modes: dict | None = None                        # {"hessian", "masses", "atoms", "source"}
    record: dict | None = None
    console: str = ""
    saved: str = ""
    tbkit: str = ""


def save(path: str | Path, project: Project) -> Path:
    """Write ``project`` as one ``.tbkproj`` file (replaced atomically if it exists)."""
    import numpy as np

    from .. import __version__

    path = Path(path)
    if path.suffix != SUFFIX:
        path = path.with_suffix(SUFFIX)
    meta = {"schema": SCHEMA, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tbkit": __version__, "active": project.active, "window": project.window,
            "pages": project.pages, "console": project.console[-200000:],
            "geometries": []}
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        for k, (name, atoms) in enumerate(project.geometries):
            entry = f"geometrias/{k:02d}.extxyz"
            text = io.StringIO()
            write(text, atoms, format="extxyz")
            archive.writestr(entry, text.getvalue())
            meta["geometries"].append({"name": name, "file": entry})
        if project.model is not None:
            archive.writestr("modelo.json", json.dumps(project.model, ensure_ascii=False))
        if project.modes is not None:
            buffer = io.BytesIO()
            np.savez(buffer, hessian_ev_per_a2=project.modes["hessian"],
                     masses_amu=project.modes["masses"], source=project.modes.get("source", ""))
            archive.writestr("modos/modes.npz", buffer.getvalue())
            text = io.StringIO()
            write(text, project.modes["atoms"], format="extxyz")
            archive.writestr("modos/structure.extxyz", text.getvalue())
            meta["modes"] = "modos/modes.npz"
        if project.record is not None:
            archive.writestr("registro.json", json.dumps(project.record, ensure_ascii=False))
        archive.writestr("proyecto.json", json.dumps(meta, indent=1, ensure_ascii=False))
    tmp.replace(path)
    return path


def load(path: str | Path) -> Project:
    """Read a ``.tbkproj``; refuses a file of another format or a newer schema."""
    import numpy as np

    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "proyecto.json" not in names:
            raise ValueError(f"{path.name} no es un proyecto de tbkit (falta proyecto.json).")
        meta = json.loads(archive.read("proyecto.json"))
        if meta.get("schema") != SCHEMA:
            raise ValueError(f"{path.name}: formato {meta.get('schema')!r}; esta versión de "
                             f"tbkit lee {SCHEMA!r}.")
        geometries = []
        for item in meta["geometries"]:
            atoms = read(io.StringIO(archive.read(item["file"]).decode("utf-8")),
                         format="extxyz")
            geometries.append((item["name"], atoms))
        model = json.loads(archive.read("modelo.json")) if "modelo.json" in names else None
        modes = None
        if meta.get("modes") and meta["modes"] in names:
            data = np.load(io.BytesIO(archive.read(meta["modes"])))
            atoms = read(io.StringIO(archive.read("modos/structure.extxyz").decode("utf-8")),
                         format="extxyz")
            modes = {"hessian": data["hessian_ev_per_a2"], "masses": data["masses_amu"],
                     "atoms": atoms, "source": str(data["source"])}
        record = json.loads(archive.read("registro.json")) if "registro.json" in names \
            else None
    return Project(geometries, int(meta.get("active", 0)), meta.get("window", {}),
                   meta.get("pages", {}), model, modes, record, meta.get("console", ""),
                   meta.get("saved", ""), meta.get("tbkit", ""))
