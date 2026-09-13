"""Session state and settings for the desktop application.

Deliberately free of Tkinter. Everything the GUI *decides* lives here —
which spectra are loaded, what the preprocessing settings are, how to build
a deconvolution model from the panel's controls, what the results table
should contain — and the Tk layer only renders it and forwards events. Two
reasons: the logic can be tested without a display (this repository's CI has
no X server), and a second front end would not have to reimplement it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from ..analysis.report import AnalysisResult, analyse
from ..core.io import COMMON_LASERS, read_spectrum
from ..core.history import Preferences
from ..core.preprocess import preprocess
from ..core.spectrum import Spectrum
from ..database import Database, load_database
from ..models.deconvolution import PRESET_LABELS, PRESETS, build_model
from ..models.fitting import FitModel, FitResult, PeakSpec, fit_model
from ..models.lineshapes import PROFILES

#: Baseline methods offered in the preprocessing panel, with labels.
BASELINE_METHODS = (
    ("asls", "Mínimos cuadrados asimétricos (recomendado)"),
    ("arpls", "arPLS (sin parámetro de asimetría)"),
    ("polynomial", "Polinómico iterativo"),
    ("rubberband", "Banda elástica (envolvente convexa)"),
    ("none", "Ninguna (los datos ya están corregidos)"),
)

#: Normalisation options.
NORMALISATIONS = (
    ("none", "Sin normalizar"),
    ("0-100", "A 0–100"),
    ("g", "A la banda G"),
    ("max", "Al máximo"),
    ("area", "Al área"),
    ("minmax", "A [0, 1]"),
)

#: Lineshape choices for the whole deconvolution.
DECONVOLUTION_PROFILES = (
    ("", "Según la base de datos"),
    ("pseudo_voigt", "Pseudo-Voigt (deja que el ajuste elija la forma)"),
    ("gaussian", "Gaussiana"),
    ("lorentzian", "Lorentziana"),
)

#: Intensity bases for the ratios.
BASES = (
    ("area", "Áreas integradas (recomendado)"),
    ("height", "Alturas de pico"),
)

#: Lineshape choices for the manual deconvolution editor.
PROFILE_CHOICES = tuple((key, PROFILES[key]["label"]) for key in PROFILES)


@dataclass
class PreprocessSettings:
    """What the preprocessing panel is asking for."""

    despike: bool = True
    resample: bool = False
    smooth_window: int = 0
    baseline_method: str = "asls"
    baseline_lam: float = 1e7
    baseline_p: float = 0.001
    baseline_order: int = 3
    normalise: str = "none"
    crop_low: Optional[float] = None
    crop_high: Optional[float] = None
    auto: bool = False
    """Let :func:`~ramancarbon.core.preprocess.auto_settings` choose the
    parameters. What the user has explicitly changed still wins."""
    auto_reasons: dict = field(default_factory=dict)
    """One sentence per automatic decision, filled in by :meth:`apply_auto`."""

    def apply_auto(self, spectrum: Spectrum) -> dict:
        """Overwrite the parameters with values chosen from the spectrum.

        Returns the explanations, so the panel can show what changed and
        why. This mutates the settings rather than bypassing them, so the
        user sees the chosen numbers in the same fields they would edit and
        can adjust any one of them afterwards — which is the whole point of
        having the automatic mode write into the controls instead of
        replacing them.
        """
        from ..core.preprocess import auto_settings

        chosen = auto_settings(spectrum, baseline_method=self.baseline_method)
        self.despike = chosen.despike
        self.smooth_window = chosen.smooth_window
        self.baseline_method = chosen.baseline_method
        self.baseline_lam = chosen.baseline_lam
        self.auto_reasons = dict(chosen.reasons)
        return self.auto_reasons

    def to_kwargs(self) -> dict:
        """Translate the panel's settings into ``preprocess`` arguments."""
        baseline_kwargs: dict[str, Any] = {}
        if self.baseline_method == "asls":
            baseline_kwargs = {"lam": self.baseline_lam, "p": self.baseline_p}
        elif self.baseline_method == "arpls":
            baseline_kwargs = {"lam": self.baseline_lam}
        elif self.baseline_method == "polynomial":
            baseline_kwargs = {"order": self.baseline_order}
        crop = None
        if self.crop_low is not None and self.crop_high is not None:
            crop = (self.crop_low, self.crop_high)
        return {
            "do_despike": self.despike,
            "do_resample": self.resample,
            "smooth_window": self.smooth_window,
            "baseline_method": None if self.baseline_method == "none" else self.baseline_method,
            "baseline_kwargs": baseline_kwargs,
            "normalise_method": None if self.normalise == "none" else self.normalise,
            "crop": crop,
        }

    def describe(self) -> str:
        """One line summarising the settings, for the status bar."""
        parts = []
        if self.despike:
            parts.append("sin picos cósmicos")
        if self.smooth_window > 2:
            parts.append(f"suavizado {self.smooth_window} pts")
        if self.baseline_method != "none":
            parts.append(f"línea base {self.baseline_method}")
        if self.normalise != "none":
            parts.append(f"normalizado {self.normalise}")
        return ", ".join(parts) if parts else "sin procesar"


@dataclass
class AnalysisSettings:
    """What the analysis panel is asking for."""

    basis: str = "area"
    presets: tuple[str, ...] = ("two_band", "three_band", "four_band", "five_band")
    metallic: Optional[bool] = None
    """``None`` lets the fit decide by comparing BWF against Lorentzian."""
    rbm_parameterisation: Optional[str] = None
    material_hint: Optional[str] = None
    profile: Optional[str] = None
    """Lineshape forced on every component; ``None`` uses the per-band
    defaults from the database."""
    check_interferences: bool = False
    """Look for non-carbon bands. Off by default; see
    :mod:`ramancarbon.analysis.interference`."""
    bootstrap_replicates: int = 0
    """Resampled refits for realistic uncertainties. 0 disables it — each
    replicate costs a full fit."""

    def to_kwargs(self) -> dict:
        return {
            "basis": self.basis,
            "presets": self.presets,
            "metallic": self.metallic,
            "rbm_parameterisation": self.rbm_parameterisation,
            "material_hint": self.material_hint,
            "profile": self.profile,
            "check_interferences": self.check_interferences,
        }


@dataclass
class LoadedSpectrum:
    """One spectrum in the session, with whatever has been computed for it."""

    raw: Spectrum
    processed: Optional[Spectrum] = None
    diagnostics: dict = field(default_factory=dict)
    result: Optional[AnalysisResult] = None
    tmd_result: Optional[Any] = None
    """Dichalcogenide analysis, when this spectrum was analysed as a TMD."""
    manual_fit: Optional[FitResult] = None
    is_control: bool = False
    error: str = ""

    @property
    def name(self) -> str:
        return self.raw.name

    @property
    def display(self) -> Spectrum:
        """Whichever version should be plotted: processed if there is one."""
        return self.processed or self.raw

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if self.result is not None:
            return "analizado"
        if self.processed is not None:
            return "procesado"
        return "cargado"


class Session:
    """Everything the application knows, independent of any widget toolkit."""

    def __init__(self, db: Optional[Database] = None) -> None:
        self.db: Database = db or load_database()
        self.spectra: list[LoadedSpectrum] = []
        self.current: int = -1
        self.preprocess_settings = PreprocessSettings()
        self.analysis_settings = AnalysisSettings()
        self.palette_name: str = "claro"
        self.preferences = Preferences.load()
        self.palette_name = str(self.preferences.get("palette", "claro"))
        self.plot_preset: str = str(self.preferences.get("plot_preset",
                                                         "predeterminado"))
        self.preprocess_settings.baseline_method = str(
            self.preferences.get("baseline_method", "asls"))
        self.analysis_settings.check_interferences = bool(
            self.preferences.get("check_interferences", False))
        self.messages: list[tuple[str, str]] = []
        if self.preferences.problem:
            self.messages.append(("warning", self.preferences.problem))
        """``(level, text)``; level is ``"info"``, ``"warning"`` or ``"error"``."""

    # -- preferences ---------------------------------------------------
    def remember(self) -> None:
        """Write the settings worth having back next time.

        Called when the window closes and after a successful load. Failing
        to write is not an error: a read-only home directory is real, and
        losing a preference must never lose the user's work.
        """
        self.preferences.set("palette", self.palette_name)
        self.preferences.set("plot_preset", self.plot_preset)
        self.preferences.set("baseline_method",
                             self.preprocess_settings.baseline_method)
        self.preferences.set("check_interferences",
                             self.analysis_settings.check_interferences)
        for item in self.spectra[-1:]:
            path = item.raw.metadata.get("path")
            if path:
                self.preferences.remember_file(path)
                if item.raw.laser_nm:
                    self.preferences.set("laser_nm", float(item.raw.laser_nm))
        self.preferences.save()

    def recent_files(self) -> list:
        """Files from previous sessions that are still there."""
        return self.preferences.existing_recent()

    # -- messages ------------------------------------------------------
    def log(self, level: str, text: str) -> None:
        """Record a message for the status area."""
        self.messages.append((level, text))
        if len(self.messages) > 300:
            del self.messages[:-300]

    # -- loading -------------------------------------------------------
    def load(self, paths: Iterable[str | Path], laser_nm: Optional[float] = None) -> int:
        """Read files into the session, keeping going past bad ones.

        Returns
        -------
        int
            How many spectra were added. Failures are recorded in
            :attr:`messages` rather than raised, so one unreadable file in a
            batch of two hundred does not abort the load.
        """
        added = 0
        for path in paths:
            p = Path(path)
            try:
                spectrum = read_spectrum(p, laser_nm=laser_nm)
            except (OSError, ValueError) as exc:
                self.log("error", f"{p.name}: {exc}")
                continue
            if spectrum.laser_nm is None:
                self.log(
                    "warning",
                    f"{spectrum.name}: no se ha encontrado la longitud de onda "
                    "del láser en la cabecera. Indícala antes de analizar; sin "
                    "ella no se corrigen las posiciones por dispersión ni se "
                    "puede calcular el tamaño de cristalito",
                )
            warning = spectrum.metadata.get("axis_warning")
            if warning:
                self.log("warning", f"{spectrum.name}: {warning}")
            self.spectra.append(LoadedSpectrum(raw=spectrum))
            added += 1
        if added and self.current < 0:
            self.current = 0
        return added

    def add(self, spectrum: Spectrum) -> None:
        """Add an already-constructed spectrum (used by the demo data)."""
        self.spectra.append(LoadedSpectrum(raw=spectrum))
        if self.current < 0:
            self.current = 0

    def remove(self, index: int) -> None:
        """Drop one spectrum from the session."""
        if not 0 <= index < len(self.spectra):
            return
        del self.spectra[index]
        self.current = min(self.current, len(self.spectra) - 1)

    def clear(self) -> None:
        """Empty the session."""
        self.spectra.clear()
        self.current = -1

    @property
    def active(self) -> Optional[LoadedSpectrum]:
        """The spectrum the panels are showing, or ``None``."""
        if 0 <= self.current < len(self.spectra):
            return self.spectra[self.current]
        return None

    def set_laser(self, wavelength_nm: Optional[float], all_spectra: bool = False) -> int:
        """Set the excitation wavelength on the active spectrum, or on all.

        Changing the laser invalidates every derived result, because band
        windows, dispersion corrections and the crystallite-size formula all
        depend on it. Those results are cleared rather than left stale.

        Returns
        -------
        int
            How many spectra were updated.
        """
        targets = self.spectra if all_spectra else ([self.active] if self.active else [])
        count = 0
        for item in targets:
            if item is None:
                continue
            item.raw.laser_nm = wavelength_nm
            if item.processed is not None:
                item.processed.laser_nm = wavelength_nm
            item.result = None
            item.manual_fit = None
            count += 1
        return count

    # -- processing ----------------------------------------------------
    def preprocess_active(self) -> Optional[LoadedSpectrum]:
        """Run preprocessing on the active spectrum."""
        item = self.active
        if item is None:
            return None
        try:
            item.processed, item.diagnostics = preprocess(
                item.raw, **self.preprocess_settings.to_kwargs()
            )
            item.error = ""
        except ValueError as exc:
            item.error = str(exc)
            self.log("error", f"{item.name}: {exc}")
        return item

    def analyse_active(self) -> Optional[LoadedSpectrum]:
        """Run the full analysis on the active spectrum."""
        item = self.active
        if item is None:
            return None
        control = next(
            (s.result for s in self.spectra if s.is_control and s.result and s is not item),
            None,
        )
        try:
            item.result = analyse(
                item.raw,
                control=control,
                preprocess_kwargs=self.preprocess_settings.to_kwargs(),
                db=self.db,
                **self.analysis_settings.to_kwargs(),
            )
            item.processed = item.result.processed
            item.diagnostics = item.result.diagnostics
            item.error = ""
            for warning in item.result.warnings:
                self.log("warning", f"{item.name}: {warning}")
        except (ValueError, RuntimeError) as exc:
            item.error = str(exc)
            self.log("error", f"{item.name}: {exc}")
        return item

    def bootstrap_active(self, replicates: Optional[int] = None):
        """Resampled uncertainties for the active spectrum's deconvolution.

        Returns ``None`` and logs why when there is no fit to resample.
        """
        from ..models.bootstrap import bootstrap_fit, position_of, ratio_of, width_of

        item = self.active
        if item is None or item.result is None or item.result.fit is None:
            self.log("error", "analiza el espectro antes de estimar incertidumbres")
            return None
        target = item.processed or item.raw
        model = self.build_manual_model(
            [
                PeakSpec(
                    name=p.name, profile=p.profile, centre=p.centre,
                    height=p.height, fwhm=p.fwhm, band=p.band,
                )
                for p in item.result.fit.peaks
            ],
            (float(item.result.fit.x[0]), float(item.result.fit.x[-1])),
            "linear",
        )
        quantities = {
            "I_D/I_G": ratio_of("D", "G"),
            "pos_G": position_of("G"),
            "Γ_G": width_of("G"),
            "pos_D": position_of("D"),
        }
        try:
            return bootstrap_fit(
                target, model, quantities,
                replicates=replicates or self.analysis_settings.bootstrap_replicates or 40,
            )
        except ValueError as exc:
            self.log("error", str(exc))
            return None

    def calibrate_active(self):
        """Correct the active spectrum's axis using the silicon line."""
        from ..analysis.quality import calibrate, silicon_offset

        item = self.active
        if item is None:
            self.log("error", "no hay ningún espectro seleccionado")
            return None
        offset = silicon_offset(item.raw)
        if offset is None:
            self.log(
                "error",
                "no se ha encontrado ninguna línea estrecha cerca de 520.7 cm⁻¹ "
                "que sirva de referencia",
            )
            return None
        item.raw = calibrate(item.raw, offset)
        item.result = None
        item.processed = None
        item.manual_fit = None
        self.log("info", f"eje corregido en {offset:+.2f} cm⁻¹")
        return offset

    # -- dichalcogenides ------------------------------------------------
    def analyse_tmd_active(self, material: Optional[str] = None):
        """Analyse the active spectrum as a transition-metal dichalcogenide.

        Kept separate from :meth:`analyse_active` rather than folded into
        it: a TMD spectrum and a carbon spectrum share almost no analysis,
        and a classifier that had to choose between "multi-walled nanotube"
        and "bilayer MoS₂" would be answering a question nobody asks — you
        know which sample you put under the objective.
        """
        from ..analysis.tmd import analyse_tmd

        item = self.active
        if item is None:
            self.log("error", "no hay ningún espectro seleccionado")
            return None
        try:
            item.tmd_result = analyse_tmd(
                item.raw,
                material=material,
                preprocess_kwargs=self.preprocess_settings.to_kwargs(),
            )
        except (ValueError, KeyError) as exc:
            self.log("error", str(exc))
            return None
        for warning in item.tmd_result.warnings:
            self.log("warning", f"{item.name}: {warning}")
        return item.tmd_result

    def tmd_table(self) -> tuple[list[str], list[list[str]]]:
        """Columns and rows for every spectrum analysed as a TMD."""
        rows_raw = []
        for item in self.spectra:
            if item.tmd_result is None:
                continue
            row = item.tmd_result.to_dict()
            row["nombre"] = item.name
            rows_raw.append(row)
        if not rows_raw:
            return [], []
        columns: list[str] = []
        for row in rows_raw:
            for key in row:
                if key not in columns:
                    columns.append(key)
        rows = [[_format_cell(row.get(c)) for c in columns] for row in rows_raw]
        return columns, rows

    def batch_summary(self):
        """Robust statistics over everything analysed in the session."""
        from ..analysis.batch import summarise

        results = [s.result for s in self.spectra if s.result is not None]
        return summarise(results) if results else None

    def analyse_all(self) -> tuple[int, int]:
        """Analyse every loaded spectrum.

        Returns
        -------
        (int, int)
            ``(succeeded, failed)``.
        """
        saved = self.current
        ok = bad = 0
        for index in range(len(self.spectra)):
            self.current = index
            item = self.analyse_active()
            if item and item.result is not None:
                ok += 1
            else:
                bad += 1
        self.current = saved
        return ok, bad

    # -- manual deconvolution -------------------------------------------
    def build_manual_model(
        self, specs: Sequence[PeakSpec], window: tuple[float, float], background: str
    ) -> FitModel:
        """Assemble a user-edited model from the deconvolution table."""
        return FitModel(
            peaks=list(specs), window=window, background=background, name="manual"
        )

    def fit_manual(
        self, specs: Sequence[PeakSpec], window: tuple[float, float], background: str
    ) -> Optional[FitResult]:
        """Fit a user-edited model to the active spectrum."""
        item = self.active
        if item is None:
            return None
        target = item.processed or item.raw
        try:
            model = self.build_manual_model(specs, window, background)
            item.manual_fit = fit_model(target, model)
        except ValueError as exc:
            item.error = str(exc)
            self.log("error", f"{item.name}: {exc}")
            return None
        for warning in item.manual_fit.warnings:
            self.log("warning", f"{item.name}: {warning}")
        return item.manual_fit

    def region_specs(self, n_d: int, n_g: int) -> list[PeakSpec]:
        """Starting components for a free-form ``n_d`` + ``n_g`` model."""
        from ..models.deconvolution import build_region_model

        item = self.active
        if item is None:
            raise ValueError("no hay ningún espectro seleccionado")
        target = item.processed or item.raw
        model = build_region_model(
            target,
            n_d=n_d,
            n_g=n_g,
            profile=self.analysis_settings.profile or "pseudo_voigt",
            metallic=bool(self.analysis_settings.metallic),
            db=self.db,
        )
        return list(model.peaks)

    def preset_specs(self, preset: str) -> list[PeakSpec]:
        """Starting components for a preset, for the editable table.

        Raises
        ------
        ValueError
            If the spectrum does not cover the preset's window, with a
            message naming both ranges.
        """
        item = self.active
        if item is None:
            raise ValueError("no hay ningún espectro seleccionado")
        target = item.processed or item.raw
        model = build_model(
            target,
            preset=preset,
            metallic=bool(self.analysis_settings.metallic),
            profile=self.analysis_settings.profile,
            db=self.db,
        )
        return list(model.peaks)

    # -- table for the batch view ---------------------------------------
    def results_table(self) -> tuple[list[str], list[list[Any]]]:
        """Column names and rows for the batch results view.

        Only spectra that have been analysed appear. The column set is the
        union over all rows, so a batch mixing nanotubes and graphene shows
        the diameter column for the ones that have it and blanks elsewhere.
        """
        rows_raw = [s.result.to_dict() for s in self.spectra if s.result is not None]
        if not rows_raw:
            return [], []
        columns: list[str] = []
        for row in rows_raw:
            for key in row:
                if key not in columns:
                    columns.append(key)
        rows = [[_format_cell(row.get(column)) for column in columns] for row in rows_raw]
        return columns, rows

    # -- multi-wavelength -----------------------------------------------
    def excitation_groups(self) -> dict[str, list[LoadedSpectrum]]:
        """Group analysed spectra that look like the same sample.

        Two spectra belong together when their names match once the laser
        wavelength is stripped out — ``muestra_532.txt`` and
        ``muestra_633.txt``. The wavelength removed is the one the spectrum
        actually carries, not a number guessed out of the filename, so a
        name like ``CNT_900C_532nm`` loses only the 532 and keeps the
        annealing temperature that would otherwise look just like a laser
        line.

        This is still a naming convention rather than a measurement, so the
        grouping is a suggestion for the user to confirm, not something
        applied silently.

        Returns
        -------
        dict[str, list[LoadedSpectrum]]
            Keyed by the common stem, only groups with two or more
            distinct excitations.
        """
        groups: dict[str, list[LoadedSpectrum]] = {}
        for item in self.spectra:
            if item.result is None or item.raw.laser_nm is None:
                continue
            stem = strip_laser_from_name(item.name, item.raw.laser_nm)
            groups.setdefault(stem, []).append(item)
        return {
            stem: items
            for stem, items in groups.items()
            if len({i.raw.laser_nm for i in items}) >= 2
        }

    def combine_excitations(self, items: Sequence[LoadedSpectrum]):
        """Run the multi-wavelength comparison over a group.

        Returns ``None`` and logs the reason when the group cannot be
        combined, rather than raising into the widget layer.
        """
        from ..analysis.multiwavelength import compare_excitations

        try:
            return compare_excitations([i.result for i in items if i.result])
        except ValueError as exc:
            self.log("error", str(exc))
            return None

    # -- export -----------------------------------------------------------
    def export_active(self, directory: str | Path, delimiter: str = ",") -> list[Path]:
        """Write every export for the active spectrum.

        Raises
        ------
        ValueError
            If nothing has been analysed.
        """
        from ..analysis.export import export_analysis

        item = self.active
        if item is None or item.result is None:
            raise ValueError("analiza el espectro antes de exportarlo")
        return export_analysis(item.result, directory, delimiter=delimiter)

    def export_fit(self, path: str | Path, delimiter: str = ",") -> list[Path]:
        """Write the components and curves of whichever fit is on screen."""
        from ..analysis.export import export_components, export_curves

        item = self.active
        if item is None:
            raise ValueError("no hay ningún espectro seleccionado")
        fit = item.manual_fit or (item.result.fit if item.result else None)
        if fit is None:
            raise ValueError("no hay ningún ajuste que exportar")
        base = Path(path)
        stem = base.with_suffix("")
        return [
            export_components(fit, stem.with_name(stem.name + "_componentes.csv"),
                              item.result, delimiter),
            export_curves(fit, stem.with_name(stem.name + "_curvas.csv"),
                          item.result, delimiter),
        ]

    def export_table(self, path: str | Path, delimiter: str = ",") -> Path:
        """Write the batch results table as CSV.

        Raises
        ------
        ValueError
            If nothing has been analysed yet.
        """
        columns, rows = self.results_table()
        if not columns:
            raise ValueError("no hay resultados que exportar: analiza algún espectro primero")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [delimiter.join(columns)]
        lines.extend(delimiter.join(str(cell) for cell in row) for row in rows)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target


def strip_laser_from_name(name: str, laser_nm: float) -> str:
    """Remove a spectrum's own laser wavelength from its name.

    Used to recognise the same sample measured at two excitations. Only the
    wavelength the spectrum reports is removed — matching any three-digit
    number would also eat annealing temperatures, sample numbers and dates.

    Parameters
    ----------
    name:
        The spectrum's name.
    laser_nm:
        Its excitation wavelength.

    Returns
    -------
    str
        The name with the wavelength and any adjoining separator removed,
        or the original name if that would leave nothing.
    """
    import re

    candidates = {f"{laser_nm:g}", f"{laser_nm:.0f}", f"{laser_nm:.1f}"}
    stem = name
    for text in sorted(candidates, key=len, reverse=True):
        stem = re.sub(
            rf"[_\-\s]*{re.escape(text)}\s*(?:nm)?", "", stem, flags=re.IGNORECASE
        )
    stem = stem.strip("_- .")
    return stem or name


def _format_cell(value: Any) -> str:
    """Render one table cell: short floats, empty string for missing."""
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if abs(value) >= 1e5 or (value != 0 and abs(value) < 1e-3):
            return f"{value:.3e}"
        return f"{value:.4g}"
    return str(value)


def laser_choices() -> list[str]:
    """Laser wavelengths offered in the combo box, as strings."""
    return [f"{value:g}" for value in COMMON_LASERS]


def preset_choices() -> list[tuple[str, str]]:
    """``(key, label)`` for every deconvolution preset."""
    return [(key, PRESET_LABELS[key]) for key in PRESETS]


__all__ = [
    "BASELINE_METHODS",
    "BASES",
    "DECONVOLUTION_PROFILES",
    "NORMALISATIONS",
    "PROFILE_CHOICES",
    "AnalysisSettings",
    "LoadedSpectrum",
    "PreprocessSettings",
    "Session",
    "laser_choices",
    "strip_laser_from_name",
    "preset_choices",
]
