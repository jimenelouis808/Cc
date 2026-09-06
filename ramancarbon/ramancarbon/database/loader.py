"""Typed access to the JSON literature database.

The dataclasses here are thin: their job is to make the JSON's contract
explicit, to fail loudly when a hand-edited file is malformed, and to carry
the ``source``/``confidence`` metadata all the way through to the report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

#: Directory holding the JSON files. Overridable via :func:`load_database`.
DATA_DIR = Path(__file__).resolve().parent / "data"


class DatabaseError(ValueError):
    """Raised when a database file is missing or malformed."""


@dataclass(frozen=True)
class Band:
    """One Raman band as defined in ``bands.json``."""

    key: str
    name: str
    position: float
    """Centre in cm⁻¹ at the database's reference excitation."""
    window: tuple[float, float]
    """Search window in cm⁻¹, at the reference excitation."""
    dispersion: float
    """dω/dE_laser in cm⁻¹/eV."""
    typical_fwhm: tuple[float, float]
    order: str
    symmetry: Optional[str]
    default_profile: str
    origin: str
    occurs_in: tuple[str, ...]
    confidence: str
    source: str
    notes: str
    reference_laser_ev: float
    position_is_reference: bool = True
    """Whether ``position`` is a value a measurement can meaningfully be
    compared against. False for the RBM, whose frequency is set by the tube
    diameter: a "shift" from its nominal position carries no information."""
    multi_valued: bool = False
    """Whether a spectrum may legitimately contain several of this band (the
    RBM, one per resonant tube diameter)."""

    def position_at(self, laser_ev: Optional[float]) -> float:
        """Expected centre at a given excitation energy, in cm⁻¹.

        ``ω(E) = position + dispersion · (E − E_ref)``. With ``laser_ev``
        ``None`` the uncorrected reference position is returned, which is
        the honest fallback but means a D-band window centred 22 cm⁻¹ away
        from the truth at 633 nm — so the analysis modules ask for the
        laser rather than defaulting.
        """
        if laser_ev is None or self.dispersion == 0.0:
            return self.position
        return self.position + self.dispersion * (laser_ev - self.reference_laser_ev)

    def window_at(self, laser_ev: Optional[float], pad: float = 0.0) -> tuple[float, float]:
        """Search window at a given excitation, optionally padded, in cm⁻¹.

        The whole window translates with the dispersion; its width is not
        rescaled, since the width encodes sample-to-sample scatter rather
        than anything that depends on the laser.
        """
        if laser_ev is None or self.dispersion == 0.0:
            shift = 0.0
        else:
            shift = self.dispersion * (laser_ev - self.reference_laser_ev)
        return (self.window[0] + shift - pad, self.window[1] + shift + pad)

    @property
    def is_dispersive(self) -> bool:
        """Whether this band's position depends on the excitation energy."""
        return abs(self.dispersion) > 1e-9


@dataclass(frozen=True)
class Material:
    """One reference material from ``materials.json``."""

    key: str
    label: str
    family: str
    walls: int
    rbm: dict
    bands: dict
    ratios: dict
    intensity_basis: str
    features: dict
    confidence: str
    source: str
    notes: str

    def band_window(self, key: str) -> Optional[tuple[float, float]]:
        """Expected position range for one band in this material, cm⁻¹."""
        entry = self.bands.get(key)
        if not entry:
            return None
        low, high = entry["position"]
        return float(low), float(high)

    def ratio_range(self, key: str) -> Optional[tuple[float, float]]:
        """Expected range of a named intensity ratio, e.g. ``"ID_IG"``."""
        entry = self.ratios.get(key)
        if not entry:
            return None
        return float(entry[0]), float(entry[1])

    @property
    def expects_rbm(self) -> bool:
        """Whether an RBM must be present for this material."""
        return self.rbm.get("expected") == "required"

    @property
    def forbids_rbm(self) -> bool:
        """Whether an RBM should be absent for this material."""
        return self.rbm.get("expected") == "absent"


@dataclass(frozen=True)
class RBMParameterisation:
    """One ω_RBM ↔ diameter relation from ``rbm.json``."""

    key: str
    label: str
    A: float
    B: float
    environment: str
    diameter_range_nm: tuple[float, float]
    confidence: str
    source: str
    notes: str
    environment_correction: Optional[float] = None
    """``C_e`` for the multiplicative Araujo form; ``None`` for the linear form."""

    @property
    def is_multiplicative(self) -> bool:
        """Whether this entry uses ``ω = (A/d)·sqrt(1 + C_e d²)``."""
        return self.environment_correction is not None


@dataclass(frozen=True)
class PerturbationEffect:
    """One doping/strain/temperature signature from ``perturbations.json``."""

    key: str
    label: str
    data: dict

    @property
    def source(self) -> str:
        return str(self.data.get("source", ""))

    @property
    def notes(self) -> str:
        return str(self.data.get("notes", ""))

    @property
    def confidence(self) -> str:
        return str(self.data.get("confidence", "unknown"))


@dataclass(frozen=True)
class DopantSignature:
    """Expected band shifts for one dopant chemistry."""

    key: str
    label: str
    host: tuple[str, ...]
    carrier: str
    g_shift: tuple[float, float]
    d2_shift: tuple[float, float]
    id_ig: str
    confidence: str
    source: str
    notes: str

    def matches(self, delta_g: float, delta_2d: Optional[float], tolerance: float = 2.0) -> bool:
        """Whether a measured pair of shifts falls in this dopant's ranges.

        ``tolerance`` widens both ranges in cm⁻¹, to absorb the calibration
        uncertainty of a real spectrometer. A ``None`` 2D shift only tests
        the G band, and the caller should say so in its output — the sign of
        the 2D shift is the only thing that separates n from p doping.
        """
        g_ok = self.g_shift[0] - tolerance <= delta_g <= self.g_shift[1] + tolerance
        if delta_2d is None:
            return g_ok
        d_ok = self.d2_shift[0] - tolerance <= delta_2d <= self.d2_shift[1] + tolerance
        return g_ok and d_ok


@dataclass
class Database:
    """The whole literature database, loaded and indexed."""

    bands: dict[str, Band]
    materials: dict[str, Material]
    rbm: dict[str, RBMParameterisation]
    rbm_default: str
    rbm_raw: dict
    perturbations: dict[str, PerturbationEffect]
    dopants: dict[str, DopantSignature]
    perturbations_raw: dict
    reference_laser_ev: float

    # -- bands ---------------------------------------------------------
    def band(self, key: str) -> Band:
        """Look up a band, raising a helpful error if the key is unknown."""
        try:
            return self.bands[key]
        except KeyError:
            raise DatabaseError(
                f"unknown band {key!r}; known bands: {', '.join(sorted(self.bands))}"
            ) from None

    def bands_in_range(
        self, low: float, high: float, laser_ev: Optional[float] = None
    ) -> list[Band]:
        """Every band whose window overlaps ``[low, high]`` at this excitation."""
        out = []
        for band in self.bands.values():
            w_lo, w_hi = band.window_at(laser_ev)
            if w_hi >= low and w_lo <= high:
                out.append(band)
        return sorted(out, key=lambda b: b.position_at(laser_ev))

    # -- materials -----------------------------------------------------
    def material(self, key: str) -> Material:
        """Look up a reference material."""
        try:
            return self.materials[key]
        except KeyError:
            raise DatabaseError(
                f"unknown material {key!r}; known: {', '.join(sorted(self.materials))}"
            ) from None

    def materials_in_family(self, family: str) -> list[Material]:
        """All reference materials in one family (``"CNT"``, ``"graphene"``…)."""
        return [m for m in self.materials.values() if m.family == family]

    # -- RBM -----------------------------------------------------------
    def rbm_parameterisation(self, key: Optional[str] = None) -> RBMParameterisation:
        """Look up an RBM relation, defaulting to the database's own default."""
        chosen = key or self.rbm_default
        try:
            return self.rbm[chosen]
        except KeyError:
            raise DatabaseError(
                f"unknown RBM parameterisation {chosen!r}; known: "
                f"{', '.join(sorted(self.rbm))}"
            ) from None

    @property
    def wall_spacing_nm(self) -> tuple[float, float]:
        """Plausible wall-to-wall spacing range in a multi-walled tube, nm."""
        low, high = self.rbm_raw["wall_spacing_nm"]
        return float(low), float(high)

    @property
    def g_splitting(self) -> dict:
        """The ω_G⁻ = ω_G⁺ − C/d² constants and their provenance."""
        return dict(self.rbm_raw["g_band_splitting"])

    @property
    def chirality(self) -> dict:
        """Nanotube geometry constants."""
        return dict(self.rbm_raw["chirality"])

    # -- perturbations -------------------------------------------------
    def effect(self, key: str) -> PerturbationEffect:
        """Look up a doping/strain/temperature effect."""
        try:
            return self.perturbations[key]
        except KeyError:
            raise DatabaseError(
                f"unknown effect {key!r}; known: {', '.join(sorted(self.perturbations))}"
            ) from None

    @property
    def pristine(self) -> dict:
        """Undoped, unstrained reference band positions at the reference laser."""
        return dict(self.perturbations_raw["pristine_reference"])

    @property
    def strain_doping(self) -> dict:
        """Constants for the G–2D strain/doping decomposition."""
        return dict(self.perturbations_raw["strain_doping_separation"])

    @property
    def defect_type_ratios(self) -> dict:
        """I_D/I_D' values that identify the defect type."""
        return dict(self.perturbations_raw["defect_type_ID_IDprime"])

    def summary(self) -> str:
        """One line per table, for the CLI's ``db`` command."""
        return (
            f"{len(self.bands)} bandas, {len(self.materials)} materiales de "
            f"referencia, {len(self.rbm)} parametrizaciones RBM, "
            f"{len(self.dopants)} dopantes"
        )


def _read(directory: Path, filename: str) -> dict:
    path = directory / filename
    if not path.is_file():
        raise DatabaseError(f"missing database file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatabaseError(f"{path} is not valid JSON: {exc}") from exc


def _require(entry: dict, keys: Sequence[str], where: str) -> None:
    missing = [k for k in keys if k not in entry]
    if missing:
        raise DatabaseError(f"{where}: missing field(s) {', '.join(missing)}")


def _build(directory: Path) -> Database:
    bands_raw = _read(directory, "bands.json")
    materials_raw = _read(directory, "materials.json")
    rbm_raw = _read(directory, "rbm.json")
    perturbations_raw = _read(directory, "perturbations.json")

    reference_ev = float(bands_raw.get("reference_laser_ev", 2.33))

    bands: dict[str, Band] = {}
    for entry in bands_raw["bands"]:
        _require(entry, ("key", "position", "window", "dispersion_cm1_per_ev"), "bands.json")
        bands[entry["key"]] = Band(
            key=entry["key"],
            name=entry.get("name", entry["key"]),
            position=float(entry["position"]),
            window=(float(entry["window"][0]), float(entry["window"][1])),
            dispersion=float(entry["dispersion_cm1_per_ev"]),
            typical_fwhm=(
                float(entry.get("typical_fwhm", [5.0, 200.0])[0]),
                float(entry.get("typical_fwhm", [5.0, 200.0])[1]),
            ),
            order=entry.get("order", "first"),
            symmetry=entry.get("symmetry"),
            default_profile=entry.get("default_profile", "lorentzian"),
            origin=entry.get("origin", ""),
            occurs_in=tuple(entry.get("occurs_in", ())),
            confidence=entry.get("confidence", "unknown"),
            source=entry.get("source", ""),
            notes=entry.get("notes", ""),
            reference_laser_ev=reference_ev,
            position_is_reference=bool(entry.get("position_is_reference", True)),
            multi_valued=bool(entry.get("multi_valued", False)),
        )

    materials: dict[str, Material] = {}
    for entry in materials_raw["materials"]:
        _require(entry, ("key", "label", "family"), "materials.json")
        materials[entry["key"]] = Material(
            key=entry["key"],
            label=entry["label"],
            family=entry["family"],
            walls=int(entry.get("walls", 0)),
            rbm=dict(entry.get("rbm", {})),
            bands=dict(entry.get("bands", {})),
            ratios=dict(entry.get("ratios", {})),
            intensity_basis=entry.get("intensity_basis", "height"),
            features=dict(entry.get("features", {})),
            confidence=entry.get("confidence", "unknown"),
            source=entry.get("source", ""),
            notes=entry.get("notes", ""),
        )

    rbm: dict[str, RBMParameterisation] = {}
    for entry in rbm_raw["parameterisations"]:
        _require(entry, ("key", "A", "B"), "rbm.json")
        rbm[entry["key"]] = RBMParameterisation(
            key=entry["key"],
            label=entry.get("label", entry["key"]),
            A=float(entry["A"]),
            B=float(entry["B"]),
            environment=entry.get("environment", ""),
            diameter_range_nm=(
                float(entry.get("diameter_range_nm", [0.5, 3.0])[0]),
                float(entry.get("diameter_range_nm", [0.5, 3.0])[1]),
            ),
            confidence=entry.get("confidence", "unknown"),
            source=entry.get("source", ""),
            notes=entry.get("notes", ""),
            environment_correction=(
                float(entry["environment_correction"])
                if "environment_correction" in entry
                else None
            ),
        )

    perturbations = {
        entry["key"]: PerturbationEffect(
            key=entry["key"], label=entry.get("label", entry["key"]), data=dict(entry)
        )
        for entry in perturbations_raw["effects"]
    }

    dopants: dict[str, DopantSignature] = {}
    for entry in perturbations_raw["dopants"]:
        dopants[entry["key"]] = DopantSignature(
            key=entry["key"],
            label=entry.get("label", entry["key"]),
            host=tuple(entry.get("host", ())),
            carrier=entry.get("carrier", "unknown"),
            g_shift=(float(entry["G_shift_cm1"][0]), float(entry["G_shift_cm1"][1])),
            d2_shift=(float(entry["2D_shift_cm1"][0]), float(entry["2D_shift_cm1"][1])),
            id_ig=entry.get("ID_IG", ""),
            confidence=entry.get("confidence", "unknown"),
            source=entry.get("source", ""),
            notes=entry.get("notes", ""),
        )

    return Database(
        bands=bands,
        materials=materials,
        rbm=rbm,
        rbm_default=rbm_raw.get("default", next(iter(rbm))),
        rbm_raw=rbm_raw,
        perturbations=perturbations,
        dopants=dopants,
        perturbations_raw=perturbations_raw,
        reference_laser_ev=reference_ev,
    )


@lru_cache(maxsize=4)
def _load_cached(directory: str) -> Database:
    return _build(Path(directory))


def load_database(directory: Optional[str | Path] = None) -> Database:
    """Load (and cache) the literature database.

    Parameters
    ----------
    directory:
        Where the JSON files live. Defaults to the package's own
        ``database/data``. Point it at a copy to use a house-calibrated
        database without modifying the installed package.

    Returns
    -------
    Database

    Raises
    ------
    DatabaseError
        If a file is missing, is not valid JSON, or lacks a required field.
    """
    return _load_cached(str(Path(directory) if directory else DATA_DIR))


def clear_cache() -> None:
    """Forget cached databases; call after editing a JSON file in a session."""
    _load_cached.cache_clear()


__all__ = [
    "DATA_DIR",
    "Band",
    "Database",
    "DatabaseError",
    "DopantSignature",
    "Material",
    "PerturbationEffect",
    "RBMParameterisation",
    "clear_cache",
    "load_database",
]
