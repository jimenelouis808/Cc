"""Project files: a whole session in one document.

An analysis session is a set of measurements, the settings applied to
them, the figures made from them and the numbers that came out. Losing any
one of those means the work cannot be repeated, and by the time a referee
asks, the folder of ``.txt`` files has been reorganised twice.

An ``.rcproj`` is a ZIP holding a JSON manifest and one plain CSV per
measurement. Both decisions are deliberate:

**The data go inside.** A project that points at ``C:\\Users\\...\\datos``
opens on nobody else's computer, and that is exactly when a project file
is wanted — sending the analysis to a co-author. The original path is
recorded as provenance, not as the way back to the numbers.

**The parts stay readable.** ZIP, JSON and CSV can all be opened by hand
with no software of ours. A project file that only this program can read
is a way to lose data in five years, and the space a binary format would
save is not worth that.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

#: Bumped when the layout changes in a way an older reader cannot follow.
FORMAT_VERSION = 1

MANIFEST = "proyecto.json"
DATA_FOLDER = "datos"


class ProjectError(ValueError):
    """Raised when a project file cannot be read."""


@dataclass
class Dataset:
    """One measurement inside a project."""

    identifier: str
    kind: str
    """``raman``, ``xrd``, ``cv``, ``gcd`` or ``eis``."""
    name: str = ""
    columns: list[str] = field(default_factory=list)
    values: Optional[np.ndarray] = None
    source: str = ""
    """Where it came from. Provenance, not a dependency."""
    options: dict[str, Any] = field(default_factory=dict)
    """What the reader was told — laser, scan rate, wavelength. Without
    these the numbers can be reloaded and the analysis cannot."""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_object(self):
        """Rebuild the measurement object this dataset came from."""
        return _rebuild(self)


@dataclass
class Project:
    """Everything one session produced."""

    name: str = "proyecto"
    notes: str = ""
    datasets: list[Dataset] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    """:meth:`ramancarbon.plotting.engine.Plot.to_dict` payloads."""
    results: list[dict[str, Any]] = field(default_factory=list)
    """Tables of numbers, as ``{"titulo", "columnas", "filas", "notas"}``."""
    created: str = ""
    modified: str = ""
    application_version: str = ""

    # -- building ------------------------------------------------------
    def add(self, obj: Any, identifier: str = "", **options) -> Dataset:
        """Put a measurement in the project."""
        from .export import series_table

        table = series_table(obj)
        kind = _kind_of(obj)
        dataset = Dataset(
            identifier=identifier or _unique(self, getattr(obj, "name", kind)),
            kind=kind,
            name=getattr(obj, "name", ""),
            columns=list(table.columns),
            values=np.asarray(table.rows, dtype=float),
            source=str(getattr(obj, "metadata", {}).get("path", "")),
            options={**_options_of(obj), **options},
            metadata={k: v for k, v in getattr(obj, "metadata", {}).items()
                      if _jsonable(v)},
        )
        self.datasets.append(dataset)
        return dataset

    def add_figure(self, plot, include_data: bool = False) -> None:
        """Store a figure. Its series carry no data by default: they name
        the datasets already in the project instead of copying them."""
        self.figures.append(plot.to_dict(include_data=include_data))

    def add_table(self, table) -> None:
        """Store a results table (a :class:`~ramancarbon.dataio.export.Table`)."""
        self.results.append({
            "titulo": table.title,
            "columnas": list(table.columns),
            "unidades": table.units,
            "filas": [[_plain(v) for v in row] for row in table.rows],
            "notas": list(table.notes),
        })

    def dataset(self, identifier: str) -> Dataset:
        for item in self.datasets:
            if item.identifier == identifier:
                return item
        raise KeyError(
            f"no hay ningún conjunto «{identifier}»; hay: "
            + ", ".join(d.identifier for d in self.datasets)
        )

    def objects(self, kind: Optional[str] = None) -> list[Any]:
        """Every measurement, rebuilt."""
        return [d.to_object() for d in self.datasets
                if kind is None or d.kind == kind]

    # -- files ---------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the project. The suffix is forced to ``.rcproj``."""
        destination = Path(path)
        if destination.suffix.lower() != ".rcproj":
            destination = destination.with_suffix(".rcproj")
        destination.parent.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.created = self.created or now
        self.modified = now
        if not self.application_version:
            from .. import __version__

            self.application_version = __version__

        manifest = {
            "formato": FORMAT_VERSION,
            "aplicacion": "ramancarbon",
            "version_aplicacion": self.application_version,
            "nombre": self.name,
            "notas": self.notes,
            "creado": self.created,
            "modificado": self.modified,
            "conjuntos": [
                {
                    "id": d.identifier,
                    "tipo": d.kind,
                    "nombre": d.name,
                    "columnas": d.columns,
                    "archivo": f"{DATA_FOLDER}/{d.identifier}.csv",
                    "origen": d.source,
                    "opciones": {k: _plain(v) for k, v in d.options.items()},
                    "metadatos": {k: _plain(v) for k, v in d.metadata.items()},
                    "puntos": 0 if d.values is None else int(d.values.shape[0]),
                }
                for d in self.datasets
            ],
            "figuras": self.figures,
            "resultados": self.results,
        }

        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MANIFEST,
                             json.dumps(manifest, ensure_ascii=False, indent=2))
            for item in self.datasets:
                archive.writestr(f"{DATA_FOLDER}/{item.identifier}.csv",
                                 _csv(item))
            archive.writestr("LEEME.txt", _README)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        """Read a project file.

        A project written by a newer version opens with a warning in
        :attr:`notes` rather than an error: the parts it does not
        understand are ignored, and the measurements — the only
        irreplaceable thing in the file — always load.
        """
        source = Path(path)
        if not source.is_file():
            raise ProjectError(f"no existe: {source}")
        try:
            archive = zipfile.ZipFile(source)
        except zipfile.BadZipFile as error:
            raise ProjectError(
                f"{source.name} no es un proyecto de la suite (no es un ZIP)"
            ) from error
        with archive:
            try:
                manifest = json.loads(archive.read(MANIFEST).decode("utf-8"))
            except KeyError as error:
                raise ProjectError(
                    f"{source.name} es un ZIP pero no lleva {MANIFEST}"
                ) from error

            project = cls(
                name=manifest.get("nombre", source.stem),
                notes=manifest.get("notas", ""),
                created=manifest.get("creado", ""),
                modified=manifest.get("modificado", ""),
                application_version=manifest.get("version_aplicacion", ""),
                figures=list(manifest.get("figuras", [])),
                results=list(manifest.get("resultados", [])),
            )
            written_with = int(manifest.get("formato", 0))
            if written_with > FORMAT_VERSION:
                project.notes = (
                    f"[Aviso] este proyecto se guardó con el formato "
                    f"{written_with} y este programa entiende el "
                    f"{FORMAT_VERSION}: las medidas se han leído, lo que no "
                    f"se reconoce se ha ignorado.\n" + project.notes
                )

            for entry in manifest.get("conjuntos", []):
                name = entry.get("archivo", "")
                try:
                    text = archive.read(name).decode("utf-8")
                except KeyError:
                    raise ProjectError(
                        f"el manifiesto nombra {name} y el archivo no está en "
                        "el proyecto"
                    ) from None
                columns, values = _parse_csv(text)
                project.datasets.append(Dataset(
                    identifier=entry.get("id", name),
                    kind=entry.get("tipo", "desconocido"),
                    name=entry.get("nombre", ""),
                    columns=entry.get("columnas") or columns,
                    values=values,
                    source=entry.get("origen", ""),
                    options=entry.get("opciones", {}),
                    metadata=entry.get("metadatos", {}),
                ))
        return project


_README = """\
Proyecto de ramancarbon.

Es un ZIP corriente. Dentro:
  proyecto.json   el manifiesto: qué hay, de dónde vino y con qué ajustes
  datos/*.csv     las medidas, en texto plano

Se puede abrir con cualquier programa de compresión y leer con cualquier
editor. Eso es a propósito: unos datos que solo un programa sabe leer son
unos datos que se pierden.
"""


def _csv(dataset: Dataset) -> str:
    lines = [f"# {dataset.name or dataset.identifier} ({dataset.kind})"]
    if dataset.source:
        lines.append(f"# origen: {dataset.source}")
    for key, value in dataset.options.items():
        lines.append(f"# {key}: {value}")
    lines.append(",".join(dataset.columns))
    if dataset.values is not None:
        for row in dataset.values:
            lines.append(",".join(f"{v:.10g}" for v in row))
    return "\n".join(lines) + "\n"


def _parse_csv(text: str) -> tuple[list[str], np.ndarray]:
    columns: list[str] = []
    rows: list[list[float]] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            if not columns:
                columns = parts
    return columns, np.asarray(rows, dtype=float) if rows else np.zeros((0, 0))


def _kind_of(obj: Any) -> str:
    if hasattr(obj, "shift"):
        return "raman"
    if hasattr(obj, "two_theta"):
        return "xrd"
    if hasattr(obj, "scan_rate"):
        return "cv"
    if hasattr(obj, "frequency"):
        return "eis"
    if hasattr(obj, "time"):
        return "gcd"
    raise TypeError(f"no sé guardar un {type(obj).__name__} en un proyecto")


def _options_of(obj: Any) -> dict[str, Any]:
    """The arguments needed to rebuild this object, taken from it.

    These are the numbers that are *not* in the columns and without which
    the data cannot be analysed: the laser, the wavelength, the scan rate,
    the reference electrode.
    """
    options: dict[str, Any] = {}
    for attribute, key in (("laser_nm", "laser_nm"), ("wavelength", "wavelength"),
                           ("scan_rate", "scan_rate"), ("current", "current")):
        value = getattr(obj, attribute, None)
        if isinstance(value, (int, float)) and value is not None:
            options[key] = float(value)
    electrode = getattr(obj, "electrode", None)
    if electrode is not None:
        options["electrodo"] = {
            "referencia": getattr(electrode, "reference", ""),
            "masa_mg": getattr(electrode, "mass_mg", None),
            "area_cm2": getattr(electrode, "area_cm2", None),
            "volumen_cm3": getattr(electrode, "volume_cm3", None),
            "pH": getattr(electrode, "ph", None),
            "resistencia_ohm": getattr(electrode, "resistance_ohm", None),
            "fraccion_ir_compensada": getattr(
                electrode, "ir_compensated_fraction", 0.0),
            "etiqueta": getattr(electrode, "label", ""),
        }
    return options


def _rebuild(dataset: Dataset):
    """Turn a stored dataset back into the object it came from."""
    values = dataset.values
    if values is None or values.size == 0:
        raise ProjectError(f"«{dataset.identifier}» no tiene datos")
    x, y = values[:, 0], values[:, 1]
    options = dataset.options or {}

    if dataset.kind == "raman":
        from ..core.spectrum import Spectrum

        return Spectrum(shift=x, intensity=y, laser_nm=options.get("laser_nm"),
                        name=dataset.name or dataset.identifier,
                        metadata={"origen": dataset.source})
    if dataset.kind == "xrd":
        from ..xrd.pattern import Pattern

        return Pattern(two_theta=x, intensity=y,
                       wavelength=float(options.get("wavelength", 1.5406)),
                       name=dataset.name or dataset.identifier,
                       metadata={"origen": dataset.source})
    if dataset.kind == "cv":
        from ..echem.curve import Voltammogram

        return Voltammogram(potential=x, current=y,
                            scan_rate=float(options.get("scan_rate", 0.0)) or 0.05,
                            electrode=_electrode(options),
                            cycle=_column(dataset, "ciclo"),
                            name=dataset.name or dataset.identifier)
    if dataset.kind == "gcd":
        from ..echem.curve import ChargeDischarge

        if values.shape[1] < 3:
            raise ProjectError(
                f"«{dataset.identifier}» dice ser carga-descarga y no trae "
                "columna de corriente; sin su signo no se pueden separar las "
                "ramas"
            )
        return ChargeDischarge(time=x, potential=y, current=values[:, 2],
                               electrode=_electrode(options),
                               cycle=_column(dataset, "ciclo"),
                               name=dataset.name or dataset.identifier)
    if dataset.kind == "eis":
        from ..echem.curve import Impedance

        if values.shape[1] < 3:
            raise ProjectError(
                f"«{dataset.identifier}» dice ser impedancia y trae "
                f"{values.shape[1]} columnas; hacen falta tres"
            )
        return Impedance(frequency=x, z=values[:, 1] + 1j * values[:, 2],
                         electrode=_electrode(options),
                         name=dataset.name or dataset.identifier)
    raise ProjectError(f"tipo desconocido en el proyecto: {dataset.kind!r}")


def _column(dataset: Dataset, name: str):
    """One named column of a dataset, or ``None`` if it was not stored."""
    if dataset.values is None or name not in dataset.columns:
        return None
    index = dataset.columns.index(name)
    if index >= dataset.values.shape[1]:
        return None
    return dataset.values[:, index]


def _electrode(options: dict[str, Any]):
    from ..echem.curve import Electrode

    stored = options.get("electrodo") or {}
    return Electrode(
        mass_mg=stored.get("masa_mg"),
        area_cm2=stored.get("area_cm2"),
        volume_cm3=stored.get("volumen_cm3"),
        reference=stored.get("referencia") or "Ag/AgCl_3M",
        ph=stored.get("pH"),
        resistance_ohm=stored.get("resistencia_ohm"),
        ir_compensated_fraction=stored.get("fraccion_ir_compensada", 0.0) or 0.0,
        label=stored.get("etiqueta", "") or "",
    )


def _unique(project: Project, base: str) -> str:
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in (base or "dato"))
    taken = {d.identifier for d in project.datasets}
    if stem not in taken:
        return stem
    index = 2
    while f"{stem}_{index}" in taken:
        index += 1
    return f"{stem}_{index}"


def _jsonable(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict))


def _plain(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


__all__ = [
    "FORMAT_VERSION",
    "Dataset",
    "Project",
    "ProjectError",
]
