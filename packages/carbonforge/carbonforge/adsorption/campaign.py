"""An isotherm as a folder on disk, so a long run survives the window.

A single GCMC state point on the validation bundle takes **31 to 46
minutes** (measured: 1.6k to 2.6k molecules at 77 K, 1.5 M equilibration
and 1.5 M production moves each). Eight of them is most of a day. That
rules out the shape every other calculation in this package has — call a
function, wait, get a result — for three separate reasons, and each one
is a requirement here:

1. **The window must stay usable.** A minute of frozen Tk is a bug; an
   hour is unusable. So the work runs in a subprocess under
   :class:`carbonforge.jobs.queue.JobQueue`, the same queue that already
   drives Quantum ESPRESSO, SIESTA and LAMMPS directories, and for the
   same reason.
2. **A point that finished must never be recomputed.** The queue can be
   cancelled, a laptop can sleep, a run can be killed by the OOM killer
   at point seven of eight. Every finished point is appended to
   ``resultados.json`` the moment it is done, and a rerun skips what is
   already there. Resuming is the default, not a flag.
3. **Progress has to be readable from outside the process.** The queue
   reports a job's state by reading its directory, so the runner writes
   ``progreso.json`` as it goes rather than printing.

**The porosity is computed once, here, and handed to every point.** Not
an optimisation: :func:`~carbonforge.adsorption.gcmc.run_gcmc` needs the
**helium** void fraction to convert its absolute loading to an excess
one, and a per-point value computed from a different random seed would
make the excess column wobble for reasons that have nothing to do with
the adsorption. It also gives the run its cheap early exit — a framework
whose largest free sphere is under the probe cannot adsorb, and saying
so costs seconds against the day the sampling would take to agree.

**The structure is written without its ``info``.** A bond list or a ring
census in an extxyz comment line is unread here and large enough to
break ASE's own reader on a framework of this size, which is the same
fact :mod:`carbonforge.exchange` already works around.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from ase import Atoms
from ase.io import read, write

from .bulk import LEVESQUE_BULK, fit_virial
from .gcmc import CUTOFF, run_gcmc
from .porosity import analyse_porosity

#: The request, the running state and the results. Three files because
#: they have three lifetimes: the request is written once and never
#: touched, the progress is rewritten constantly, and the results only
#: ever grow.
REQUEST: str = "isoterma.json"
PROGRESS: str = "progreso.json"
RESULTS: str = "resultados.json"
STRUCTURE: str = "estructura.extxyz"

#: Format tag, so a folder written by a later version can be recognised
#: rather than half-read.
SCHEMA: str = "carbonforge-isoterma/1"


def write_isotherm_request(
    atoms: Atoms,
    folder: str | Path,
    temperature: float,
    pressures: Sequence[float],
    name: str = "",
    equilibration: int = 150_000,
    production: int = 300_000,
    seed: int = 0,
    cutoff: float = CUTOFF,
    probe_radius: float | None = None,
    porosity_samples: int = 200_000,
    reason: str = "",
) -> Path:
    """Write a folder the job queue can run as an isotherm.

    Parameters
    ----------
    atoms
        The framework. Must be orthorhombic and periodic; the check is
        the GCMC's own and is made **now** rather than an hour in.
    folder
        Directory to create and fill. Must not already hold a request.
    temperature
        In kelvin. A reservoir fitted to published densities exists only
        at 77, 150 and 293 K, so anything else is refused here instead
        of failing per point.
    pressures
        Reservoir pressures in MPa. Sorted and de-duplicated, because
        the results are keyed by pressure and two of the same would make
        resuming ambiguous.
    name
        Label for the run. Defaults to the structure's formula.
    equilibration, production
        Moves per point. Both are recorded in the result so a loading
        can say what produced it.
    seed, cutoff
        Passed through to each point.
    probe_radius
        Hard-sphere probe for the porosity, in Å. ``None`` takes
        :func:`~carbonforge.adsorption.porosity.analyse_porosity`'s own
        default, half H2's kinetic diameter.
    porosity_samples
        Monte Carlo points for the void fraction.
    reason
        Free text. Say what the run is for; the folder outlives the
        window that made it.

    Returns
    -------
    pathlib.Path
        The folder written.

    Raises
    ------
    ValueError
        If the folder already holds a request, the cell is not
        orthorhombic and periodic, no pressure was given, a pressure is
        not positive, or the temperature has no reservoir.
    """
    folder = Path(folder)
    if (folder / REQUEST).exists():
        raise ValueError(
            f"{folder / REQUEST} already exists. That folder carries its own "
            "partial results in " + RESULTS + "; write this run somewhere "
            "else rather than overwriting one that may be part way through.")

    wanted = sorted({float(p) for p in pressures})
    if not wanted:
        raise ValueError("no pressures given: an isotherm needs at least one.")
    if wanted[0] <= 0.0:
        raise ValueError(
            f"pressure {wanted[0]} MPa is not positive. The reservoir enters "
            "as a fugacity, which is undefined at zero pressure.")
    # Both of these would otherwise raise an hour into the run, inside a
    # subprocess, with the first point's work already thrown away.
    try:
        fit_virial(temperature)
    except KeyError as exc:
        # fit_virial raises a bare KeyError holding the number, which is
        # a fine message for a caller reading a traceback and a useless
        # one for somebody who typed a temperature into a form.
        raise ValueError(
            f"no reservoir at {temperature:g} K. The bulk densities this "
            "module fits a virial to are published at "
            f"{', '.join(f'{t:g}' for t in sorted(LEVESQUE_BULK))} K. For "
            "another temperature, fit one to your own measured densities "
            "and pass it to run_gcmc directly."
        ) from exc
    _check_cell(atoms, cutoff)

    folder.mkdir(parents=True, exist_ok=True)
    stripped = atoms.copy()
    stripped.info.clear()
    write(str(folder / STRUCTURE), stripped)

    request: dict[str, Any] = {
        "esquema": SCHEMA,
        "nombre": name or atoms.get_chemical_formula(),
        "origen": {
            "programa": "carbonforge",
            "fecha": time.strftime("%Y-%m-%d %H:%M"),
            "maquina": platform.node(),
            "motivo": reason,
        },
        "estructura": STRUCTURE,
        "atomos": len(atoms),
        "temperatura_k": float(temperature),
        "presiones_mpa": wanted,
        "muestreo": {
            "equilibrado": int(equilibration),
            "produccion": int(production),
            "semilla": int(seed),
            "corte_a": float(cutoff),
        },
        "porosidad": {
            "radio_sonda_a": probe_radius,
            "muestras": int(porosity_samples),
        },
    }
    (folder / REQUEST).write_text(
        json.dumps(request, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return folder


def _check_cell(atoms: Atoms, cutoff: float) -> None:
    """Refuse a cell the GCMC's neighbour search cannot handle.

    Two refusals, both of which :func:`~carbonforge.adsorption.gcmc.run_gcmc`
    makes itself. They are made here as well so that they happen when the
    folder is written and somebody is looking, rather than an hour later
    inside a subprocess whose output nobody is watching:

    * a triclinic cell, because every neighbour search here is a
      ``cKDTree`` with ``boxsize``, which is orthorhombic-only;
    * an edge shorter than twice the cutoff, because then the minimum
      image is not the nearest image.

    Both would otherwise return a plausible-looking wrong isotherm.
    """
    import numpy as np

    if not np.asarray(atoms.get_pbc()).all():
        raise ValueError(
            "the framework must be periodic in all three directions: a GCMC "
            "cell exchanges molecules with a reservoir through its faces.")
    cell = np.asarray(atoms.cell, dtype=float)
    if not np.allclose(cell - np.diag(np.diag(cell)), 0.0, atol=1e-8):
        raise ValueError(
            "the framework cell must be orthorhombic: the neighbour search "
            "is a cKDTree with boxsize, which cannot do a triclinic one, and "
            "the minimum image would be wrong across every face.")
    edges = np.abs(np.diag(cell))
    if edges.min() < 2.0 * cutoff:
        raise ValueError(
            f"the shortest cell edge is {edges.min():.2f} Å against a "
            f"{cutoff:.2f} Å pair cutoff, so the minimum image would not be "
            f"the nearest image. Use a supercell with every edge over "
            f"{2 * cutoff:.2f} Å, or ask for a smaller cutoff.")


def read_isotherm_request(folder: str | Path) -> dict[str, Any]:
    """Read and check a request folder's ``isoterma.json``."""
    folder = Path(folder)
    path = folder / REQUEST
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    request = json.loads(path.read_text(encoding="utf-8"))
    if request.get("esquema") != SCHEMA:
        raise ValueError(
            f"{path} is {request.get('esquema')!r}, not {SCHEMA!r}. "
            "A folder from another version is not half-read on purpose.")
    return request


def read_results(folder: str | Path) -> list[dict[str, Any]]:
    """Finished points, lowest pressure first. Empty when none are."""
    path = Path(folder) / RESULTS
    if not path.exists():
        return []
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A kill during the write leaves a truncated file. Losing the
        # points is bad; refusing to report the run at all is worse, and
        # the runner recomputes whatever is not readable.
        return []
    points = stored.get("puntos", [])
    return sorted(points, key=lambda row: row["presion_mpa"])


def status(folder: str | Path) -> dict[str, Any]:
    """What the job queue shows in the list, read from the folder alone.

    Returns a dict with ``state`` (``"sin empezar"``, ``"corriendo"``,
    ``"terminado"``, ``"sin salida"``), ``line`` (one line for the
    progress column), ``done`` and ``total``.
    """
    folder = Path(folder)
    try:
        request = read_isotherm_request(folder)
    except (FileNotFoundError, ValueError) as exc:
        return {"state": "sin salida", "line": str(exc), "done": 0, "total": 0}

    total = len(request["presiones_mpa"])
    done = len(read_results(folder))
    progress_path = folder / PROGRESS
    note = ""
    if progress_path.exists():
        try:
            note = json.loads(progress_path.read_text(encoding="utf-8")
                              ).get("linea", "")
        except json.JSONDecodeError:
            note = ""

    if done >= total and total:
        return {"state": "terminado",
                "line": f"{done}/{total} puntos", "done": done, "total": total}
    if done or note:
        return {"state": "corriendo",
                "line": f"{done}/{total} puntos" + (f" · {note}" if note else ""),
                "done": done, "total": total}
    return {"state": "sin empezar", "line": f"0/{total} puntos",
            "done": 0, "total": total}


def _write_progress(folder: Path, line: str) -> None:
    (folder / PROGRESS).write_text(
        json.dumps({"linea": line, "cuando": time.strftime("%H:%M:%S")},
                   ensure_ascii=False) + "\n", encoding="utf-8")


def _append_result(folder: Path, request: dict[str, Any],
                   point: dict[str, Any], porosity: dict[str, Any]) -> None:
    """Add one finished point, rewriting the whole file.

    Rewritten rather than appended to because the file is a JSON object
    and a valid one at every instant is worth more than the cost: these
    writes are seconds apart at best and half an hour apart in practice,
    against the 45 minutes of work each one protects.
    """
    points = [row for row in read_results(folder)
              if row["presion_mpa"] != point["presion_mpa"]]
    points.append(point)
    points.sort(key=lambda row: row["presion_mpa"])
    (folder / RESULTS).write_text(json.dumps({
        "esquema": SCHEMA,
        "nombre": request["nombre"],
        "temperatura_k": request["temperatura_k"],
        "muestreo": request["muestreo"],
        "porosidad": porosity,
        "puntos": points,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_isotherm(folder: str | Path,
                 pressures: Iterable[float] | None = None) -> list[dict[str, Any]]:
    """Run every point a folder still needs, writing each as it lands.

    Safe to call again on the same folder: points already in
    ``resultados.json`` are skipped, so a cancelled or killed run
    continues rather than restarting. That is the point of the format —
    at 45 minutes a point, recomputing seven to redo the eighth is a
    lost afternoon.

    Parameters
    ----------
    folder
        A directory written by :func:`write_isotherm_request`.
    pressures
        Run only these, of the ones the request names. For re-running a
        single point whose warnings said it had not equilibrated.

    Returns
    -------
    list of dict
        Every finished point in the folder, lowest pressure first —
        including ones earlier calls computed.
    """
    folder = Path(folder)
    request = read_isotherm_request(folder)
    atoms = read(str(folder / request["estructura"]))
    temperature = float(request["temperatura_k"])
    sampling = request["muestreo"]
    wanted = list(request["presiones_mpa"])
    if pressures is not None:
        chosen = {float(p) for p in pressures}
        unknown = chosen - set(wanted)
        if unknown:
            raise ValueError(
                f"{sorted(unknown)} are not pressures of this request "
                f"({wanted}). Write a new folder rather than growing one: "
                "its results would then not be the isotherm it describes.")
        wanted = [p for p in wanted if p in chosen]

    _write_progress(folder, "porosidad")
    porosity_options = request["porosidad"]
    kwargs: dict[str, Any] = {"samples": int(porosity_options["muestras"]),
                              "seed": int(sampling["semilla"])}
    if porosity_options.get("radio_sonda_a") is not None:
        kwargs["probe_radius"] = float(porosity_options["radio_sonda_a"])
    report = analyse_porosity(atoms, **kwargs)
    porosity = {k: v for k, v in asdict(report).items()
                if k not in ("pore_diameters", "pore_histogram")}
    porosity["warnings"] = list(report.warnings)

    # The cheap refusal the whole module is ordered around: a framework
    # with no channel the probe can cross adsorbs essentially nothing,
    # and the sampling would take a day to agree. Reported, not raised:
    # the number is the result.
    if report.largest_free_sphere < 2.0 * report.probe_radius:
        _write_progress(
            folder,
            f"sin canal: la esfera libre mayor es {report.largest_free_sphere:.2f} A "
            f"y la sonda mide {2 * report.probe_radius:.2f} A de diametro")

    already = {row["presion_mpa"] for row in read_results(folder)}
    for pressure in wanted:
        if pressure in already:
            continue
        _write_progress(folder, f"GCMC a {pressure:g} MPa")
        started = time.perf_counter()
        result = run_gcmc(
            atoms, temperature=temperature, pressure_mpa=pressure,
            equilibration=int(sampling["equilibrado"]),
            production=int(sampling["produccion"]),
            seed=int(sampling["semilla"]),
            cutoff=float(sampling["corte_a"]),
            void_fraction=report.helium_void_fraction,
        )
        point = {
            "presion_mpa": pressure,
            "n_medio": result.mean_count,
            "n_error": result.count_error,
            "densidad_absoluta_g_cm3": result.absolute_density_g_cm3,
            "densidad_exceso_g_cm3": result.excess_density_g_cm3,
            "densidad_gas_g_cm3": result.bulk_density_g_cm3,
            "wt_pct_total": result.weight_percent,
            "wt_pct_por_carbono": result.weight_percent_per_carbon,
            "aceptacion": result.acceptance,
            "segundos": time.perf_counter() - started,
            "avisos": [w for w in result.warnings
                       if "void fraction was given" not in w],
        }
        _append_result(folder, request, point, porosity)

    points = read_results(folder)
    _write_progress(folder, f"{len(points)}/{len(request['presiones_mpa'])} puntos")
    return points
