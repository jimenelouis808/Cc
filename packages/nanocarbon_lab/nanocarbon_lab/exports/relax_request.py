"""Hand a structure to a real potential, through the exchange folder.

The valence force field in :mod:`nanocarbon_lab.relax.valence` is a
geometry cleaner: one rest length, one angle target per coordination, no
conjugation. It cannot say which of two structures is lower in energy,
cannot find a reconstruction, and cannot break or make a bond. When the
question is energy rather than geometry, the geometry has to go out to
something that computes one.

It goes out as a **folder**, never as an import. The root ``CLAUDE.md``
rule is that these packages do not import each other, and
``docs/INTERCAMBIO_DFT.md`` already defines the format they talk in:
``nanocarbon-dft/1``, a directory holding the request, the structure in
extxyz and — once somebody has run it — the result. carbonforge's
``carbonforge dft preparar`` reads such a folder and runs it; tbkit
writes them for its own reasons. This module is nanocarbon_lab's writer,
so a superstructure can reach an external relaxer without either package
knowing the other exists.

**Which code, for a carbon superstructure.** The format's ``codigo``
field takes ``gpaw``, ``qe``, ``dftb+`` and ``emt``. At the sizes this
package builds — a superfullerene comes out at eighteen thousand atoms —
plane-wave DFT is not a candidate and ``dftb+`` is: it is a
tight-binding method with analytic gradients that handles ten thousand
carbons, and the ``3ob`` or ``matsci`` Slater-Koster sets cover C, N, O
and H. That is the default here for that reason, not because it is the
most accurate thing in the list. ``emt`` exists in the format for tests
and is not a carbon potential at all; asking for it is allowed and says
so in the request's own ``motivo``.

**The bonds do not travel, and that is on purpose.** extxyz carries
positions, cell and periodicity; an external code derives its own
bonding from the geometry, which is what makes it an independent
opinion. Writing this package's bond list into the request would not
constrain the external calculation and would invite someone to compare
two topologies as though they were one.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import write as ase_write

#: The schema this folder speaks, from ``docs/INTERCAMBIO_DFT.md``.
SCHEMA = "nanocarbon-dft/1"

#: Codes the format defines. ``emt`` is ASE's toy potential, in the list
#: for testing the plumbing and not for carbon.
CODES = ("dftb+", "gpaw", "qe", "emt")

#: Default DFTB+ settings for a carbon framework: self-consistent
#: charges on, p shell on carbon. ``skf_dir`` is deliberately left for
#: the caller — a Slater-Koster set lives on whoever runs the job, and a
#: path written here would be a path from another machine.
DFTB_CARBON: dict[str, Any] = {
    "scc": True,
    "momento_angular": {"C": "p", "N": "p", "O": "p", "H": "s"},
}


def write_relaxation_request(
    atoms: Atoms,
    folder: str | Path,
    name: str = "",
    code: str = "dftb+",
    settings: dict[str, Any] | None = None,
    fmax: float = 0.05,
    steps: int = 300,
    reason: str = "",
    resources: dict[str, Any] | None = None,
) -> Path:
    """Write a ``nanocarbon-dft/1`` folder asking for a relaxation.

    Parameters
    ----------
    atoms
        The structure to relax. Written to ``estructura.extxyz`` with
        its cell and periodic axes; ``atoms.info`` is **not** written,
        because a bond list or a ring census in the comment line is both
        unread by the external code and large enough to break ASE's own
        extxyz reader on a structure of this size.
    folder
        Directory to create and fill. Must not already hold a request.
    name
        Label for the job. Defaults to the structure's
        ``network_kind`` or ``structure_type``, else the folder name.
    code
        One of :data:`CODES`. ``dftb+`` is the default and the reason is
        in this module's docstring.
    settings
        The ``ajustes`` block. For ``dftb+`` it is merged over
        :data:`DFTB_CARBON`, and ``skf_dir`` has to come from the
        caller: the Slater-Koster files live wherever the job runs.
    fmax, steps
        Force convergence (eV/Å) and the iteration cap for the
        ``relajar`` task.
    reason
        Free text for ``origen.motivo``. Say what the valence force
        field could not answer; the folder outlives the conversation.
    resources
        The ``recursos`` block (``mpi``, ``hilos``, ``memoria_gb``).

    Returns
    -------
    pathlib.Path
        The folder written.

    Raises
    ------
    ValueError
        If ``code`` is not one the format defines, or the folder already
        contains a request — overwriting one silently would throw away a
        job somebody may be part way through running.
    """
    if code not in CODES:
        raise ValueError(
            f"unknown code {code!r}; the exchange format defines {list(CODES)}."
        )
    folder = Path(folder)
    request_path = folder / "solicitud.json"
    if request_path.exists():
        raise ValueError(
            f"{request_path} already exists. A request folder carries its own "
            "partial results in tareas/ and progreso.json; write this one "
            "somewhere else rather than overwriting a job in flight."
        )
    folder.mkdir(parents=True, exist_ok=True)

    ajustes: dict[str, Any] = dict(DFTB_CARBON) if code == "dftb+" else {}
    ajustes.update(settings or {})

    label = (name or str(atoms.info.get("network_kind", ""))
             or str(atoms.info.get("structure_type", "")) or folder.name)

    # The structure goes out WITHOUT info. ASE writes info into the
    # extxyz comment line, and a superstructure's bond list is thousands
    # of pairs: the file it produces is one its own reader then refuses
    # ("inhomogeneous shape"), which is a trap worth not setting for
    # whoever opens this folder next.
    bare = Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.cell,
        pbc=atoms.get_pbc(),
    )
    ase_write(folder / "estructura.extxyz", bare, format="extxyz")

    request = {
        "esquema": SCHEMA,
        "nombre": label,
        "origen": {
            "programa": "nanocarbon_lab",
            "fecha": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
            "motivo": reason or (
                "geometría ya limpiada con el campo de valencia; hace falta "
                "una energía, que un campo de valencia no da"),
        },
        "estructura": "estructura.extxyz",
        "codigo": code,
        "ajustes": ajustes,
        "tareas": [
            {"id": "relax", "tipo": "relajar", "fmax": float(fmax),
             "pasos": int(steps)},
        ],
        "recursos": dict(resources or {"mpi": 4, "hilos": 1}),
        "atomos": len(atoms),
    }
    request_path.write_text(
        json.dumps(request, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return folder
