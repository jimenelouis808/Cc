"""What each simulation code accepts: a catalogue of input parameters.

carbonforge writes the settings it understands (spin, cutoffs, occupations,
...) and decides them from the structure. Everything else a code offers --
hundreds of keywords -- reaches the input file as **advanced parameters**,
and this module is what keeps that from being a blind text box:

* :class:`Parameter` describes one keyword: where it goes (a QE namelist, a
  SIESTA fdf key, a GPAW argument), its type, default, allowed values and a
  description.
* :class:`Catalog` holds a code's parameters, searches them, and checks a set
  of overrides before anything is written: unknown names, wrong types,
  values outside the allowed options, and keywords that override a decision
  carbonforge already made.

A catalogue starts from a small curated core (:mod:`carbonforge.codes.curated`)
and grows with the code's own documentation, imported from the user's
installation (:mod:`carbonforge.codes.importers`). Imported catalogues are
stored per user, so a manual is read once. The manuals themselves are never
bundled: they belong to their projects, under their licences.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..validation.checks import ValidationReport

#: Codes with a catalogue. The TB package will register its own when it exists.
CODES: dict[str, str] = {
    "qe": "Quantum ESPRESSO (pw.x)",
    "siesta": "SIESTA",
    "gpaw": "GPAW",
}

_TRUE = {"true", ".true.", "t", "yes", "y", "1", "sí", "si"}
_FALSE = {"false", ".false.", "f", "no", "n", "0"}


def _key(code: str, name: str, section: str = "") -> str:
    if code == "siesta":
        return re.sub(r"[._-]", "", name).lower()
    return f"{section}.{name}".lower() if section else name.lower()


@dataclass(frozen=True)
class Parameter:
    """One input keyword of one code.

    Attributes
    ----------
    code
        Key of :data:`CODES`.
    name
        The keyword as the code spells it (``ecutwfc``, ``Mesh.Cutoff``,
        ``convergence``).
    section
        Where it goes: a QE namelist (``system``), empty for SIESTA and GPAW.
    kind
        ``"integer"``, ``"real"``, ``"logical"``, ``"string"``,
        ``"quantity"`` (a number and a unit, SIESTA's ``300 Ry``), or
        ``"any"`` when the documentation does not say.
    default
        The code's default, as text (``"4 * ecutwfc"`` is a valid default).
    choices
        Allowed values, when the keyword is an enumeration.
    units
        Physical units, when documented (``Ry``, ``eV``, ``Ang``).
    description
        What it does.
    source
        ``"curado"`` or where it was imported from.
    managed
        True when carbonforge sets this keyword itself; overriding it is
        allowed, but reported.
    """

    code: str
    name: str
    section: str = ""
    kind: str = "any"
    default: Optional[str] = None
    choices: tuple[str, ...] = ()
    units: str = ""
    description: str = ""
    source: str = "curado"
    managed: bool = False

    @property
    def key(self) -> str:
        """``section.name`` for QE, ``name`` elsewhere; case-insensitive.

        SIESTA's fdf also ignores ``.``, ``_`` and ``-`` in names
        (``Mesh.Cutoff`` = ``MeshCutoff``), so its keys drop them.
        """
        return _key(self.code, self.name, self.section)

    @property
    def qualified(self) -> str:
        """How an override names it: ``system.nosym`` (QE) or ``Mesh.Cutoff``."""
        return f"{self.section}.{self.name}" if self.section else self.name

    def coerce(self, value: Any) -> Any:
        """Convert a raw value (usually text from the GUI) to this keyword's type.

        Raises
        ------
        ValueError
            With a message naming the keyword and what it expects.
        """
        text = str(value).strip()
        if self.kind == "integer":
            try:
                return int(text)
            except ValueError:
                raise ValueError(f"'{self.name}' espera un entero (recibido '{value}').") from None
        if self.kind == "real":
            try:
                return float(text.replace(",", ".").replace("d", "e").replace("D", "e"))
            except ValueError:
                raise ValueError(f"'{self.name}' espera un número (recibido '{value}').") from None
        if self.kind == "quantity":
            number, _, unit = text.partition(" ")
            try:
                float(number.replace(",", "."))
            except ValueError:
                raise ValueError(
                    f"'{self.name}' espera un número con unidad, p. ej. '300 Ry' "
                    f"(recibido '{value}')."
                ) from None
            return f"{number.replace(',', '.')} {unit.strip()}".strip()
        if self.kind == "logical":
            low = text.lower()
            if low in _TRUE:
                return True
            if low in _FALSE:
                return False
            raise ValueError(f"'{self.name}' espera verdadero/falso (recibido '{value}').")
        if self.choices:
            options = {c.strip("'\"").lower(): c.strip("'\"") for c in self.choices}
            low = text.strip("'\"").lower()
            if low not in options:
                raise ValueError(
                    f"'{self.name}': '{value}' no es una opción válida. "
                    f"Opciones: {', '.join(options.values())}."
                )
            return options[low]
        if self.kind == "any" and self.code == "gpaw":
            # GPAW takes Python values: {'name': 'fermi-dirac', 'width': 0.05}.
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return text.strip("'\"")
        return text.strip("'\"") if self.kind == "string" else text


@dataclass
class Catalog:
    """Every known parameter of one code."""

    code: str
    parameters: dict[str, Parameter] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    def add(self, parameter: Parameter, replace: bool = True) -> None:
        if replace or parameter.key not in self.parameters:
            self.parameters[parameter.key] = parameter

    def merge(self, other: "Catalog") -> "Catalog":
        """A new catalogue: ``other`` fills in and refines this one.

        Imported documentation wins on type, default, options and text; the
        curated ``managed`` flag is kept, since only carbonforge knows what
        it sets itself.
        """
        merged = Catalog(self.code, dict(self.parameters), [*self.sources, *other.sources])
        for key, parameter in other.parameters.items():
            ours = merged.parameters.get(key)
            if ours is not None and ours.managed:
                parameter = Parameter(**{**asdict(parameter), "managed": True})
            merged.parameters[key] = parameter
        return merged

    def get(self, name: str, section: str = "") -> Optional[Parameter]:
        found = self.parameters.get(_key(self.code, name, section))
        if found is None and not section:
            # QE names are unique across namelists; allow the bare name.
            matches = [p for p in self.parameters.values() if p.name.lower() == name.lower()]
            found = matches[0] if len(matches) == 1 else None
        return found

    def search(self, text: str = "", limit: int = 200) -> list[Parameter]:
        """Parameters whose name, section or description contain ``text``."""
        text = text.lower().strip()
        hits = [p for p in self.parameters.values()
                if not text or text in p.name.lower() or text in p.section.lower()
                or text in p.description.lower()]
        # Names that start with the query first, then alphabetical.
        hits.sort(key=lambda p: (not p.name.lower().startswith(text), p.section, p.name.lower()))
        return hits[:limit]

    @property
    def documented(self) -> bool:
        """Whether a manual was imported (not just the curated core)."""
        return any(source != "curado" for source in self.sources)

    def check(self, overrides: dict[str, Any]) -> tuple[dict[str, Any], ValidationReport]:
        """Validate advanced parameters; return them typed, and the report.

        ``overrides`` maps ``"section.name"`` (QE) or ``"name"`` to raw
        values. Unknown keywords are an **error** when the code's manual has
        been imported (the name is then really unknown) and a **warning**
        otherwise (the curated core is not exhaustive).
        """
        report = ValidationReport()
        typed: dict[str, Any] = {}
        for raw_key, value in overrides.items():
            section, _, name = raw_key.rpartition(".") if self.code == "qe" else ("", "", raw_key)
            parameter = self.get(name, section)
            if parameter is None:
                message = (f"'{raw_key}' no está en el catálogo de {CODES.get(self.code, self.code)}.")
                if self.documented:
                    report.errors.append(message + " Revisa el nombre (y la sección).")
                else:
                    report.warnings.append(
                        message + " Se escribirá tal cual; importa el manual del código "
                        "para validarlo."
                    )
                typed[raw_key] = value
                continue
            try:
                typed[parameter.qualified] = parameter.coerce(value)
            except ValueError as exc:
                report.errors.append(str(exc))
                continue
            if parameter.managed:
                report.warnings.append(
                    f"'{parameter.name}' lo decide carbonforge a partir de la estructura y "
                    "los controles; tu valor lo sustituye."
                )
        return typed, report

    # -- persistence --------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps({"code": self.code, "sources": self.sources,
                           "parameters": [asdict(p) for p in self.parameters.values()]},
                          ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Catalog":
        data = json.loads(text)
        catalog = cls(data["code"], sources=list(data.get("sources", [])))
        for item in data["parameters"]:
            item["choices"] = tuple(item.get("choices", ()))
            catalog.add(Parameter(**item))
        return catalog


def user_catalog_dir() -> Path:
    """Where imported catalogues are kept: ``$CARBONFORGE_HOME/catalogos``.

    ``CARBONFORGE_HOME`` defaults to ``~/.carbonforge``.
    """
    home = Path(os.environ.get("CARBONFORGE_HOME", Path.home() / ".carbonforge"))
    return home / "catalogos"


def save_imported(catalog: Catalog) -> Path:
    """Store an imported catalogue for this user; returns the file."""
    directory = user_catalog_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{catalog.code}.json"
    path.write_text(catalog.to_json(), encoding="utf-8")
    return path


def load_catalog(code: str) -> Catalog:
    """The curated catalogue of ``code``, refined by any imported manual."""
    from .curated import curated_catalog

    if code not in CODES:
        raise ValueError(f"Código desconocido: '{code}'. Opciones: {', '.join(CODES)}.")
    catalog = curated_catalog(code)
    stored = user_catalog_dir() / f"{code}.json"
    if stored.exists():
        catalog = catalog.merge(Catalog.from_json(stored.read_text(encoding="utf-8")))
    return catalog
