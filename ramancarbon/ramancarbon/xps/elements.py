"""Typed access to ``database/data/xps.json``.

The JSON keeps two kinds of line in two different fields, and the reason is
the single most common way of misreading a survey:

* A **photoemission** line sits at a fixed *binding* energy. Change the
  anode and it does not move.
* An **Auger** line sits at a fixed *kinetic* energy. Change the anode and
  its apparent binding energy moves by the difference in photon energy —
  233 eV between Mg Kα and Al Kα. The C KLL group is at 1219 eV apparent
  binding energy with Al and at 986 eV with Mg.

So an Auger line only has a binding energy once you say which source took
the spectrum, and :class:`AugerLine` refuses to pretend otherwise: it has
:meth:`AugerLine.binding_at`, not an ``energy_ev`` attribute.

The other thing encoded here is the **spin–orbit doublet**. Every level
with l > 0 is split into two lines whose area ratio is fixed by the
degeneracy 2j+1 — 1:2 for p, 2:3 for d, 3:4 for f — and whose separation is
a property of the element. :meth:`XPSLine.components` hands back both
components with their fractions of the total area so nothing downstream has
to rederive them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..database.loader import DATA_DIR, DatabaseError

#: Area ratio of the two spin–orbit components, by orbital letter, taken as
#: the ratio (2j+1) for the lower-j (higher binding energy) component over
#: the higher-j one. Used only to check the database against itself.
DEGENERACY_RATIO = {"p": 0.5, "d": 2.0 / 3.0, "f": 0.75}


@dataclass(frozen=True)
class Doublet:
    """The spin–orbit partner of a photoemission line.

    ``splitting_ev`` is positive and the partner always lies at *higher*
    binding energy: the lower-j component is the less tightly bound one in
    the j-j scheme used for core levels.
    """

    splitting_ev: float
    ratio: float
    """Area of the high-binding-energy component over the low one."""
    labels: tuple[str, str]

    @property
    def expected_ratio(self) -> Optional[float]:
        """The degeneracy ratio implied by the orbital letter in the labels."""
        for letter, value in DEGENERACY_RATIO.items():
            if letter in self.labels[0]:
                return value
        return None


@dataclass(frozen=True)
class XPSLine:
    """One photoemission line, at a binding energy that does not move."""

    element: str
    label: str
    orbital: str
    energy_ev: float
    """Binding energy in eV. For a doublet this is the *low* component
    (the higher-j one: 2p3/2, 3d5/2, 4f7/2)."""
    rsf: float
    doublet: Optional[Doublet]
    primary: bool
    """Whether this is the line quantification should use for the element."""
    confidence: str
    source: str = ""
    note: str = ""

    @property
    def is_doublet(self) -> bool:
        return self.doublet is not None

    def components(self) -> tuple[tuple[str, float, float], ...]:
        """``(label, binding energy, fraction of the line's total area)``.

        A singlet returns one entry with fraction 1. A doublet returns two,
        whose fractions sum to 1 and are fixed by the degeneracy — they are
        not free parameters of any fit.
        """
        if self.doublet is None:
            return ((self.label, self.energy_ev, 1.0),)
        r = float(self.doublet.ratio)
        low = 1.0 / (1.0 + r)
        return (
            (self.doublet.labels[0], self.energy_ev, low),
            (
                self.doublet.labels[1],
                self.energy_ev + self.doublet.splitting_ev,
                1.0 - low,
            ),
        )

    @property
    def span(self) -> tuple[float, float]:
        """Binding energies the whole line occupies, ignoring width."""
        energies = [e for _, e, _ in self.components()]
        return (min(energies), max(energies))


@dataclass(frozen=True)
class AugerLine:
    """One Auger group, at a fixed **kinetic** energy.

    Auger groups are broad (10–20 eV) and structured; ``width_ev`` is a
    rough total extent, not a FWHM of anything fittable.
    """

    element: str
    label: str
    kinetic_ev: float
    width_ev: float
    confidence: str
    source: str = ""
    note: str = ""

    def binding_at(self, photon_energy: float, work_function: float = 4.5) -> float:
        """Apparent binding energy with a given source, in eV.

        ``BE = hν − KE − φ``. The work function is the spectrometer's, not
        the sample's, and a few tenths of an eV either way does not matter
        for a group this broad.
        """
        return float(photon_energy) - float(self.kinetic_ev) - float(work_function)

    def window_at(
        self, photon_energy: float, work_function: float = 4.5
    ) -> tuple[float, float]:
        """Binding-energy range the group covers with a given source."""
        centre = self.binding_at(photon_energy, work_function)
        half = float(self.width_ev) / 2.0
        return (centre - half, centre + half)


@dataclass(frozen=True)
class Element:
    """One element's lines, as the database knows them."""

    symbol: str
    z: int
    name: str
    lines: tuple[XPSLine, ...]
    auger: tuple[AugerLine, ...]

    @property
    def primary_line(self) -> XPSLine:
        """The line used for quantification.

        Raises
        ------
        DatabaseError
            If no line in this element is marked ``primary``. That is a
            database error rather than a physical situation: quantification
            has to know which line carries the RSF.
        """
        for line in self.lines:
            if line.primary:
                return line
        raise DatabaseError(f"{self.symbol}: ninguna línea marcada como primaria")

    def line(self, label: str) -> XPSLine:
        """Look up one line by its label, e.g. ``"Fe 2p"``."""
        for candidate in self.lines:
            if candidate.label == label:
                return candidate
        known = ", ".join(c.label for c in self.lines)
        raise DatabaseError(f"{self.symbol}: línea desconocida {label!r}; hay: {known}")


@dataclass(frozen=True)
class ChemicalState:
    """One chemical state within a high-resolution region.

    ``window`` is the range a component's centre is allowed to take when it
    is assigned to this state — deliberately narrow, because the whole point
    of a literature-justified fit is that "N pyridinic" means 398.0–399.0 eV
    and not "whichever component came out lowest".
    """

    region: str
    key: str
    name: str
    energy_ev: float
    window: tuple[float, float]
    fwhm: tuple[float, float]
    asymmetric: bool
    satellite_ev: Optional[float]
    """Offset in eV of a shake-up satellite that accompanies this state."""
    satellite_of: Optional[str]
    """Set when this entry *is* the satellite of another state."""
    confidence: str
    source: str = ""
    note: str = ""

    @property
    def is_satellite(self) -> bool:
        return self.satellite_of is not None

    def contains(self, binding_energy: float, tolerance: float = 0.0) -> bool:
        """Whether a centre falls in this state's window, optionally padded."""
        return (
            self.window[0] - tolerance
            <= float(binding_energy)
            <= self.window[1] + tolerance
        )


@dataclass(frozen=True)
class CalibrationReference:
    """One charge-referencing standard."""

    key: str
    line: str
    energy_ev: float
    spread_ev: float
    conductor_required: bool
    confidence: str
    source: str = ""
    note: str = ""


@dataclass
class XPSDatabase:
    """The XPS half of the literature database, loaded and indexed."""

    elements: dict[str, Element]
    states: dict[str, tuple[ChemicalState, ...]]
    references: dict[str, CalibrationReference]
    sources: dict[str, dict]
    rsf_basis: dict

    # -- elements ------------------------------------------------------
    def element(self, symbol: str) -> Element:
        """Look up an element by symbol."""
        try:
            return self.elements[symbol]
        except KeyError:
            known = ", ".join(sorted(self.elements))
            raise DatabaseError(
                f"elemento {symbol!r} no está en la base XPS; hay: {known}"
            ) from None

    @property
    def symbols(self) -> tuple[str, ...]:
        """Every element symbol, in order of atomic number."""
        return tuple(sorted(self.elements, key=lambda s: self.elements[s].z))

    def all_lines(self, primary_only: bool = False) -> list[XPSLine]:
        """Every photoemission line in the database."""
        out: list[XPSLine] = []
        for element in self.elements.values():
            for line in element.lines:
                if primary_only and not line.primary:
                    continue
                out.append(line)
        return sorted(out, key=lambda ln: ln.energy_ev)

    def lines_in_range(
        self,
        low: float,
        high: float,
        elements: Optional[list[str]] = None,
        primary_only: bool = False,
    ) -> list[XPSLine]:
        """Photoemission lines with any component inside ``[low, high]`` eV."""
        out = []
        for line in self.all_lines(primary_only=primary_only):
            if elements is not None and line.element not in elements:
                continue
            span = line.span
            if span[1] >= low and span[0] <= high:
                out.append(line)
        return out

    def auger_in_range(
        self,
        low: float,
        high: float,
        photon_energy: float,
        work_function: float = 4.5,
        elements: Optional[list[str]] = None,
    ) -> list[AugerLine]:
        """Auger groups whose apparent binding energy falls in the range.

        Requires ``photon_energy``: without it the question has no answer.
        """
        out = []
        for element in self.elements.values():
            if elements is not None and element.symbol not in elements:
                continue
            for group in element.auger:
                lo, hi = group.window_at(photon_energy, work_function)
                if hi >= low and lo <= high:
                    out.append(group)
        return sorted(out, key=lambda g: g.binding_at(photon_energy, work_function))

    # -- chemical states -----------------------------------------------
    def region_names(self) -> tuple[str, ...]:
        """Every high-resolution region that has literature states."""
        return tuple(sorted(self.states))

    def states_for(self, region: str, include_satellites: bool = True) -> tuple[ChemicalState, ...]:
        """The literature states for one region, e.g. ``"N 1s"``.

        Returns an empty tuple for a region with no entries rather than
        raising: fitting a region the database does not cover is legitimate,
        it just means the components come back unnamed.
        """
        found = self.states.get(region, ())
        if include_satellites:
            return found
        return tuple(s for s in found if not s.is_satellite)

    def state(self, region: str, key: str) -> ChemicalState:
        """One named state."""
        for candidate in self.states.get(region, ()):
            if candidate.key == key:
                return candidate
        known = ", ".join(c.key for c in self.states.get(region, ()))
        raise DatabaseError(
            f"estado {key!r} desconocido en la región {region!r}; hay: {known or 'ninguno'}"
        )

    def region_for_line(self, label: str) -> Optional[str]:
        """The state-table key covering a line label, if there is one.

        ``"Fe 2p"`` maps to ``"Fe 2p3/2"`` because the chemical shifts are
        tabulated for the component that is actually fitted.
        """
        if label in self.states:
            return label
        for name in self.states:
            if name.startswith(label):
                return name
        return None

    # -- calibration ---------------------------------------------------
    def reference(self, key: str) -> CalibrationReference:
        """One charge-referencing standard."""
        try:
            return self.references[key]
        except KeyError:
            known = ", ".join(sorted(self.references))
            raise DatabaseError(
                f"referencia de carga {key!r} desconocida; hay: {known}"
            ) from None


def _line_from(element: str, entry: dict) -> XPSLine:
    doublet = entry.get("doublet")
    return XPSLine(
        element=element,
        label=str(entry["label"]),
        orbital=str(entry.get("orbital", "")),
        energy_ev=float(entry["energy_ev"]),
        rsf=float(entry.get("rsf", 0.0)),
        doublet=(
            Doublet(
                splitting_ev=float(doublet["splitting_ev"]),
                ratio=float(doublet["ratio"]),
                labels=(str(doublet["labels"][0]), str(doublet["labels"][1])),
            )
            if doublet
            else None
        ),
        primary=bool(entry.get("primary", False)),
        confidence=str(entry.get("confidence", "unknown")),
        source=str(entry.get("source", "")),
        note=str(entry.get("note", "")),
    )


def _build(path: Path) -> XPSDatabase:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DatabaseError(f"falta el archivo de base de datos XPS: {path}") from None
    except json.JSONDecodeError as exc:
        raise DatabaseError(f"{path.name} no es JSON válido: {exc}") from None

    elements: dict[str, Element] = {}
    for symbol, entry in raw.get("elements", {}).items():
        lines = tuple(_line_from(symbol, item) for item in entry.get("lines", ()))
        auger = tuple(
            AugerLine(
                element=symbol,
                label=str(item["label"]),
                kinetic_ev=float(item["kinetic_ev"]),
                width_ev=float(item.get("width_ev", 15.0)),
                confidence=str(item.get("confidence", "unknown")),
                source=str(item.get("source", "")),
                note=str(item.get("note", "")),
            )
            for item in entry.get("auger", ())
        )
        elements[symbol] = Element(
            symbol=symbol,
            z=int(entry.get("z", 0)),
            name=str(entry.get("name", symbol)),
            lines=lines,
            auger=auger,
        )

    states: dict[str, tuple[ChemicalState, ...]] = {}
    for region, entries in raw.get("chemical_states", {}).items():
        if region.startswith("_"):
            continue
        states[region] = tuple(
            ChemicalState(
                region=region,
                key=str(item["key"]),
                name=str(item["name"]),
                energy_ev=float(item["energy_ev"]),
                window=(float(item["window"][0]), float(item["window"][1])),
                fwhm=(float(item["fwhm"][0]), float(item["fwhm"][1])),
                asymmetric=bool(item.get("asymmetric", False)),
                satellite_ev=(
                    float(item["satellite_ev"]) if item.get("satellite_ev") is not None else None
                ),
                satellite_of=item.get("satellite_of"),
                confidence=str(item.get("confidence", "unknown")),
                source=str(item.get("source", "")),
                note=str(item.get("note", "")),
            )
            for item in entries
        )

    references = {
        str(item["key"]): CalibrationReference(
            key=str(item["key"]),
            line=str(item.get("line", "")),
            energy_ev=float(item["energy_ev"]),
            spread_ev=float(item.get("spread_ev", 0.2)),
            conductor_required=bool(item.get("conductor_required", False)),
            confidence=str(item.get("confidence", "unknown")),
            source=str(item.get("source", "")),
            note=str(item.get("note", "")),
        )
        for item in raw.get("calibration", {}).get("references", ())
    }

    return XPSDatabase(
        elements=elements,
        states=states,
        references=references,
        sources=dict(raw.get("sources", {})),
        rsf_basis=dict(raw.get("rsf_basis", {})),
    )


@lru_cache(maxsize=4)
def _load_cached(path: str) -> XPSDatabase:
    return _build(Path(path))


def load_xps_database(directory: Optional[str | Path] = None) -> XPSDatabase:
    """Load (and cache) the XPS line and state database.

    Parameters
    ----------
    directory:
        Where ``xps.json`` lives. Defaults to the package's own
        ``database/data``.
    """
    base = Path(directory) if directory else DATA_DIR
    return _load_cached(str(base / "xps.json"))


def clear_cache() -> None:
    """Forget the cached XPS database; call after editing ``xps.json``."""
    _load_cached.cache_clear()


__all__ = [
    "AugerLine",
    "CalibrationReference",
    "ChemicalState",
    "DEGENERACY_RATIO",
    "Doublet",
    "Element",
    "XPSDatabase",
    "XPSLine",
    "clear_cache",
    "load_xps_database",
]
