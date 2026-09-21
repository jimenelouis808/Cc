"""XPS session state, with no Tkinter in it.

One session holds the spectra of a single sample — a survey and its
high-resolution regions — because that is the unit the analysis works on:
the charge reference is measured once and applied to all of them, the
survey decides which regions are worth fitting, and the composition needs
every region at once.

The one piece of state that is not obvious is the **per-region model**.
Choosing how many components a region has and which chemical states they
are is the whole of a high-resolution XPS analysis, and it is a decision
the user makes and then re-makes as they look at the residual. So the
session keeps the choice per region and rebuilds the model from it, rather
than keeping a fitted model that would have to be invalidated every time
anything changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..xps.calibrate import Calibration, calibrate, calibrate_to_state
from ..xps.elements import XPSDatabase, load_xps_database
from ..xps.fitting import XPSFitResult, fit_region, resolution_floor
from ..xps.io import read_xps, write_vamas
from ..xps.presets import compare_counts, count_model, free_model, state_model
from ..xps.quantify import Quantification, areas_from_fits, quantify
from ..xps.spectrum import XPSError, XPSSpectrum
from ..xps.survey import SurveyResult, add_region_evidence, identify
from ..xps.tables import read_components, write_fit

#: Background choices offered in the section, in the order they are usually
#: tried.
BACKGROUNDS: tuple[tuple[str, str], ...] = (
    ("shirley", "Shirley (el estándar)"),
    ("tougaard", "Tougaard (ventana ancha)"),
    ("lineal", "Lineal (solo regiones débiles)"),
    ("ninguno", "Ninguno"),
)

#: Line shapes offered per component, by profile key.
#:
#: Built from the engine's own table rather than written out again, so a
#: profile added there appears here: a list of shapes that has quietly
#: fallen behind the ones the fitter knows is a menu that lies.
def _profile_choices() -> tuple[tuple[str, str], ...]:
    from ..xps.lineshapes import XPS_PROFILES

    return tuple(
        (key, spec["label"] + (" — asimétrica" if spec["asymmetric"] else ""))
        for key, spec in XPS_PROFILES.items()
    )


PROFILES: tuple[tuple[str, str], ...] = _profile_choices()

#: Charge references offered, by database key.
REFERENCES: tuple[tuple[str, str], ...] = (
    ("C1s_adventitious", "C 1s adventicio 284.8 eV (universal y discutido)"),
    ("Au4f", "Au 4f7/2 83.96 eV"),
    ("Ag3d", "Ag 3d5/2 368.21 eV"),
    ("Cu2p", "Cu 2p3/2 932.62 eV"),
    ("Fermi", "Nivel de Fermi"),
    ("ninguna", "Ninguna: dejar el eje del instrumento"),
)


@dataclass
class RegionChoice:
    """What the user decided about one region."""

    label: str
    states: Optional[list[str]] = None
    """Chemical state keys, or ``None`` to let the count decide."""
    count: Optional[int] = None
    """How many components. ``None`` means "as many as the region shows",
    which is the honest default: a fixed number is a decision made about
    somebody else's sample."""
    background: str = "shirley"
    link_widths: bool = True
    satellites: bool = True
    free: bool = False
    """Fit ``count`` unnamed components instead of literature states, for a
    region the database does not describe."""
    window: Optional[tuple[float, float]] = None
    """The binding-energy window to fit, or ``None`` for the whole
    region. The ends of the window are a PARAMETER, not a detail: moving
    the high-binding-energy limit of a C 1s by one electronvolt moves the
    carbonyl area by several per cent. That is not a defect of the method
    — it is what "the area of a peak over a background" means — and it is
    why the ends go in the report."""
    extra: list[dict[str, Any]] = field(default_factory=list)
    """Components added by hand, each ``{name, centre, fwhm, profile}``.

    The operation the residual asks for: there is a shoulder at 285.5 eV
    that no tabulated state of this region explains, so put a component
    there and see whether it survives. It is also the easiest way to
    invent a chemical state, so a hand-added component carries no
    ``state`` and the report says where it came from — the composition
    counts its area, but nothing claims to know what it is."""
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-component changes, by component name: ``centre``, ``fwhm``,
    ``profile`` and ``fixed`` (a tuple of parameter names).

    Fixing is not free and not neutral, the same as everywhere else in
    this package: the parameter stops contributing a degree of freedom,
    so every other uncertainty comes out smaller, and a wrong fixed value
    moves into its neighbours instead of showing up as a bad fit. What it
    is FOR is a parameter the measurement cannot determine — the position
    of a satellite whose parent is clear, the width of a component buried
    under a stronger one — and it is a deception anywhere else."""


@dataclass
class XPSSession:
    """Everything the XPS section knows."""

    name: str = "muestra"
    spectra: list[XPSSpectrum] = field(default_factory=list)
    """As loaded, before any charge referencing."""
    shifted: list[XPSSpectrum] = field(default_factory=list)
    """After referencing. Equal to ``spectra`` when no reference is used."""
    reference: str = "C1s_adventitious"
    reference_state: Optional[tuple[str, str]] = None
    only_possible_states: bool = True
    """Offer only the states the sample's composition allows. Section 21
    of the user's specification: if no sulphur is detected, NiS is
    rejected as a candidate rather than fitted and then discussed."""
    transmission: str = "potencia"
    exponent: float = -0.65
    choices: dict[str, RegionChoice] = field(default_factory=dict)
    survey: Optional[SurveyResult] = None
    fits: dict[str, XPSFitResult] = field(default_factory=dict)
    composition: Optional[Quantification] = None
    calibration: Optional[Calibration] = None
    messages: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.database: XPSDatabase = load_xps_database()

    # -- messages ------------------------------------------------------
    def log(self, level: str, text: str) -> None:
        self.messages.append((level, text))
        if len(self.messages) > 300:
            del self.messages[:-300]

    # -- loading -------------------------------------------------------
    def load(self, path: str | Path, **options: Any) -> int:
        """Read a file, adding every spectrum it holds. Returns how many."""
        try:
            found = read_xps(path, **options)
        except (XPSError, OSError, ValueError) as error:
            # Some readers name the file themselves, because their message
            # is useful outside this session too. Saying it twice reads
            # like a bug in the message.
            name = Path(path).name
            text = str(error)
            self.log("error", text if text.startswith(name)
                     else f"{name}: {text}")
            return 0
        for item in found:
            self.spectra.append(item)
            note = item.looks_smoothed()
            if note:
                self.log("aviso", f"{item.name}: {note}")
            if item.pass_energy is None:
                self.log(
                    "aviso",
                    f"{item.name}: sin energía de paso no hay suelo de "
                    "resolución con el que juzgar las anchuras ajustadas",
                )
        self.shifted = list(self.spectra)
        self.log("info", f"{Path(path).name}: {len(found)} espectro(s)")
        return len(found)

    def clear(self) -> None:
        self.spectra.clear()
        self.shifted.clear()
        self.fits.clear()
        self.choices.clear()
        self.survey = None
        self.composition = None
        self.calibration = None

    @property
    def surveys(self) -> list[XPSSpectrum]:
        return [item for item in self.shifted if item.is_survey]

    @property
    def regions(self) -> list[XPSSpectrum]:
        return [item for item in self.shifted if not item.is_survey]

    def region_label(self, spectrum: XPSSpectrum) -> str:
        """The database region a spectrum belongs to, or its own label."""
        return self.database.region_for_line(spectrum.region) or spectrum.region

    def spectrum_rows(self) -> list[tuple[str, str, str]]:
        """``(name, what it is, the settings it carries)`` for the list."""
        rows = []
        for item in self.shifted:
            kind = "survey" if item.is_survey else self.region_label(item)
            floor = resolution_floor(item)
            detail = (f"{item.range[0]:.0f}–{item.range[1]:.0f} eV, paso "
                      f"{item.step:.2f}")
            if item.pass_energy:
                detail += f", E_paso {item.pass_energy:.0f} eV"
            if floor:
                detail += f" (resolución ≥ {floor:.2f} eV)"
            rows.append((item.name, kind, detail))
        return rows

    # -- calibration ---------------------------------------------------
    def apply_reference(self) -> Optional[Calibration]:
        """Measure the charge shift once and apply it to every spectrum."""
        self.shifted = list(self.spectra)
        self.calibration = None
        if not self.spectra:
            return None
        if self.reference_state:
            label, key = self.reference_state
            target = next((item for item in self.regions
                           if self.region_label(item) == label), None)
            if target is None:
                self.log("error", f"no hay ningún espectro de la región {label}")
                return None
            choice = self.choices.get(label)
            states = choice.states if choice else None
            try:
                _, calibration = calibrate_to_state(
                    target, label, key, states, database=self.database)
            except XPSError as error:
                self.log("error", f"calibración: {error}")
                return None
        elif self.reference and self.reference != "ninguna":
            entry = self.database.reference(self.reference)
            candidates = [item for item in self.spectra
                          if item.covers(entry.energy_ev - 5.0,
                                         entry.energy_ev + 5.0, 0.8)]
            if not candidates:
                self.log(
                    "aviso",
                    f"ningún espectro cubre {entry.line or self.reference}; el "
                    "eje se deja como venía",
                )
                return None
            best = min(candidates, key=lambda item: (item.is_survey, item.step))
            if self.reference == "C1s_adventitious" and any(
                self.region_label(item) == "C 1s" for item in self.regions
            ):
                self.log(
                    "aviso",
                    "hay una región C 1s de la propia muestra y la referencia "
                    "es el C 1s adventicio: en un material hecho de carbono "
                    "eso es circular. Escribe «C 1s:C-C sp2» en «…o "
                    "componente» para referenciar sobre una componente "
                    "ajustada",
                )
            try:
                _, calibration = calibrate(best, self.reference,
                                           database=self.database)
            except XPSError as error:
                self.log("error", f"calibración: {error}")
                return None
        else:
            return None

        self.calibration = calibration
        self.shifted = [item.shifted(calibration.shift_ev, calibration.reference)
                        for item in self.spectra]
        self.fits.clear()
        self.composition = None
        for text in calibration.warnings:
            self.log("aviso", text)
        self.log("info", calibration.describe())
        return calibration

    # -- survey --------------------------------------------------------
    def run_survey(self) -> Optional[SurveyResult]:
        surveys = self.surveys
        if not surveys:
            self.log("aviso", "no hay barrido ancho que identificar")
            return None
        try:
            found = identify(surveys[0], database=self.database)
            # The regions are evidence too, and better evidence than the
            # survey. See survey.add_region_evidence.
            self.survey = add_region_evidence(found, self.regions,
                                              database=self.database)
        except XPSError as error:
            self.log("error", f"survey: {error}")
            return None
        for text in self.survey.warnings:
            self.log("aviso", text)
        return self.survey

    def survey_rows(self) -> list[tuple[str, str, str]]:
        if not self.survey:
            return []
        rows = [(item.symbol, item.confidence,
                 "; ".join(str(match) for match in item.matched))
                for item in self.survey.elements]
        rows.extend((item.symbol, "sin corroborar",
                     "; ".join(str(match) for match in item.matched))
                    for item in self.survey.uncorroborated)
        # Tentative last, and labelled: the window around the line carries
        # intensity but there is no peak. "Not found" was hiding this.
        rows.extend((item.symbol, "tentativo",
                     item.notes[0] if item.notes else "")
                    for item in self.survey.tentative)
        return rows

    # -- fitting -------------------------------------------------------
    def choice_for(self, label: str) -> RegionChoice:
        return self.choices.setdefault(label, RegionChoice(label=label))

    def present_elements(self) -> Optional[list[str]]:
        """The elements this sample is known to contain, or ``None``.

        Only elements with real evidence count: what the survey
        identified, plus what a loaded region actually shows a peak for.
        A region that was SCANNED is not evidence on its own -- the
        sulphur region of a sample with no sulphur is flat, and it was
        still measured.

        ``None`` when nothing has been established yet, which keeps every
        state on offer: a filter that fires on no information is a filter
        that hides things.
        """
        found: set[str] = set()
        if self.survey is not None:
            found.update(self.survey.symbols())
        if self.regions:
            from ..xps.survey import SurveyResult, add_region_evidence

            blank = SurveyResult(peaks=[], elements=[], uncorroborated=[],
                                 unexplained=[], overlaps=[])
            found.update(item.symbol for item in
                         add_region_evidence(blank, self.regions,
                                             database=self.database).elements)
        return sorted(found) or None

    def available_states(self, label: str) -> list[tuple[str, str]]:
        """``(key, name)`` of every state the database has for a region.

        Filtered to what the sample can chemically contain unless the
        user has turned that off. Binding energy is not a functional
        group: «O de red (óxido metálico)» at 530 eV and a quinone C=O at
        531 eV are one electronvolt apart, and which one a peak is
        depends on whether the sample has a metal in it -- which the
        region cannot see and the survey can.
        """
        present = self.present_elements() if self.only_possible_states else None
        return [(item.key, item.name)
                for item in self.database.states_for(label,
                                                     include_satellites=False,
                                                     present=present)]

    def impossible_states(self, label: str) -> list[tuple[str, str]]:
        """``(name, why)`` for the states this sample rules out."""
        present = self.present_elements()
        if present is None:
            return []
        out = []
        for item in self.database.states_for(label, include_satellites=False):
            if item.possible_in(present):
                continue
            out.append((item.name,
                        "hace falta " + " o ".join(item.requires_any)
                        + f", y en la muestra hay {', '.join(present)}"))
        return out

    def fit(self, label: str) -> Optional[XPSFitResult]:
        """Fit one region with whatever the user chose for it."""
        spectrum = next((item for item in self.regions
                         if self.region_label(item) == label), None)
        if spectrum is None:
            self.log("error", f"no hay espectro de la región {label}")
            return None
        choice = self.choice_for(label)
        try:
            window = choice.window or spectrum.range
            if choice.free:
                model = free_model(
                    spectrum, window, choice.count or 2,
                    background=choice.background, region=label,
                    link_widths=choice.link_widths,
                )
            elif choice.states:
                model = state_model(
                    spectrum, label, choice.states, window=choice.window,
                    background=choice.background,
                    link_widths=choice.link_widths,
                    include_satellites=choice.satellites, database=self.database,
                )
            else:
                model, notes = count_model(
                    spectrum, label, choice.count, window=choice.window,
                    background=choice.background, link_widths=choice.link_widths,
                    include_satellites=choice.satellites, database=self.database,
                    present=(self.present_elements()
                             if self.only_possible_states else None),
                )
                for note in notes:
                    self.log("info", f"{label}: {note}")
            self._add_extra(choice, model, spectrum)
            self._apply_overrides(choice, model)
            result = fit_region(spectrum, model, database=self.database)
        except (XPSError, ValueError) as error:
            self.log("error", f"{label}: {error}")
            return None
        self.fits[label] = result
        for text in result.warnings:
            self.log("aviso", f"{label}: {text}")
        self.composition = None
        return result

    def _add_extra(self, choice: RegionChoice, model, spectrum) -> None:
        """Append the user's hand-added components to a built model."""
        from ..xps.fitting import XPSComponent
        from ..xps.lineshapes import fwhm_for_total

        if not choice.extra:
            return
        low, high = model.window
        for entry in choice.extra:
            centre = float(entry["centre"])
            if not low - 2.0 <= centre <= high + 2.0:
                self.log(
                    "error",
                    f"la componente en {centre:g} eV cae fuera de la ventana "
                    f"{low:g}–{high:g} eV; ensánchala o mueve la componente",
                )
                continue
            name = str(entry.get("name") or f"manual_{centre:.1f}")
            name = name.replace(" ", "_")
            if any(c.name == name for c in model.components):
                name = f"{name}_2"
            index = int(np.argmin(np.abs(spectrum.binding_energy - centre)))
            model.components.append(XPSComponent(
                name=name,
                label=str(entry.get("label") or name),
                centre=centre,
                height=max(float(spectrum.counts[index]), 1.0),
                fwhm=fwhm_for_total(str(entry.get("profile", "gl")),
                                    float(entry.get("fwhm", 1.4))),
                profile=str(entry.get("profile", "gl")),
                justification="añadida a mano; no corresponde a ningún "
                              "estado tabulado de esta región",
            ))

    def add_component(self, label: str, centre: float, fwhm: float = 1.4,
                      name: str = "", profile: str = "gl") -> bool:
        """Put a component where the residual says there is one."""
        if fwhm <= 0:
            self.log("error", "la anchura tiene que ser positiva")
            return False
        self.choice_for(label).extra.append({
            "name": name or f"manual_{centre:.1f}",
            "label": name or f"manual {centre:.1f} eV",
            "centre": float(centre),
            "fwhm": float(fwhm),
            "profile": profile,
        })
        return True

    def remove_component(self, label: str, name: str) -> bool:
        """Drop a hand-added component. The tabulated ones are chosen by
        picking states, not by deleting them one at a time."""
        choice = self.choice_for(label)
        for entry in list(choice.extra):
            if name in (entry.get("name"), entry.get("label")):
                choice.extra.remove(entry)
                return True
        self.log(
            "error",
            f"«{name}» no la añadiste tú: las componentes de la base de "
            "datos se eligen en la lista de estados químicos",
        )
        return False

    def _apply_overrides(self, choice: RegionChoice, model) -> None:
        """Put the user's per-component edits onto a freshly built model.

        Applied AFTER the model is built rather than instead of building
        it, so the literature windows, the doublets and the width links
        are still the ones the database says. What the user is editing is
        a starting point and a set of holds, not the physics.
        """
        from ..xps.lineshapes import fwhm_for_total, resolve_xps_profile

        # The width the user typed is the total width of the component
        # they were LOOKING at, and for an asymmetric profile the total
        # depends on the extra parameters as much as on the width one.
        # Converting against the freshly built model's defaults instead
        # of against the fitted shape makes the round trip miss: measured,
        # 1.22 eV typed back came out as 1.19.
        fitted = {c.name: tuple(c.extra)
                  for c in getattr(self.fits.get(choice.label), "components",
                                   ())}
        for component in model.components:
            edit = choice.overrides.get(component.name)
            if not edit:
                continue
            if "fwhm" in edit and component.name in fitted:
                component.extra = fitted[component.name]
            if "centre" in edit:
                component.centre = float(edit["centre"])
                # The bounds came from the state's published window, and
                # a starting value the user moved outside it would be
                # clipped straight back. Their number wins; it is their
                # sample.
                if component.centre_bounds is not None:
                    low, high = component.centre_bounds
                    if not low <= component.centre <= high:
                        component.centre_bounds = (
                            min(low, component.centre - 0.5),
                            max(high, component.centre + 0.5),
                        )
                        self.log(
                            "aviso",
                            f"{component.label}: {component.centre:.2f} eV "
                            "queda fuera de la ventana publicada de ese "
                            "estado; se ha ampliado la ventana para "
                            "admitirlo, pero comprueba que sigue siendo ese "
                            "estado",
                        )
            if edit.get("profile"):
                component.profile = resolve_xps_profile(str(edit["profile"]))
                component.extra = None
                component.extra_bounds = None
                component.__post_init__()
            if "fwhm" in edit:
                # The user typed the TOTAL width, because that is what
                # the table shows and what the literature publishes. For
                # every profile but the plain Gaussian and Lorentzian it
                # is not the width PARAMETER -- a GL product is 4 % narrower
                # than its parameter and a Doniach-Sunjic 13 % wider -- so
                # storing the typed number directly would mean typing the
                # displayed value back changed the peak.
                component.fwhm = fwhm_for_total(
                    component.profile, float(edit["fwhm"]), component.extra)
                if component.fwhm_bounds is not None:
                    low, high = component.fwhm_bounds
                    component.fwhm_bounds = (min(low, component.fwhm),
                                             max(high, component.fwhm))
            if "fixed" in edit:
                held = tuple(edit["fixed"])
                # Holding the width means holding the width you SEE. For a
                # Doniach-Šunjić the total is the Lorentzian parameter
                # convolved with a Gaussian, so fixing only the parameter
                # leaves the Gaussian free and the total drifts anyway --
                # measured, a width typed back as 1.24 eV came out at 1.27.
                # Whatever else carries the width gets held with it.
                if "fwhm" in held:
                    from ..xps.lineshapes import XPS_PROFILES

                    spec = XPS_PROFILES[resolve_xps_profile(component.profile)]
                    held = held + tuple(
                        name for name in spec.get("extra", ())
                        if "fwhm" in name and name not in held)
                component.fixed = held

    def component_rows(self, label: str) -> list[tuple[str, ...]]:
        """``(name, E, FWHM, area, %, profile, held)`` for a fitted region.

        The held column is not decoration: holding a parameter removes a
        degree of freedom, so every uncertainty in the row next to it
        comes out smaller, and a reader who cannot see which were held
        cannot read the table.
        """
        result = self.fits.get(label)
        if result is None:
            return []
        rows = []
        for component in result.components:
            rows.append((
                component.label,
                f"{component.centre:.2f}",
                # The TOTAL width. In a Doniach-Šunjić the ``fwhm``
                # parameter is only the Lorentzian part, and it is the
                # total that the literature publishes and that has to be
                # compared against the resolution floor.
                f"{component.true_fwhm:.2f}",
                f"{component.area:.4g}",
                f"{100 * component.area_fraction:.1f}",
                component.profile,
                ", ".join(component.fixed) if component.fixed else "",
            ))
        return rows

    def set_component(self, label: str, name: str, **edits: Any) -> None:
        """Override one component's starting values, profile or holds.

        ``name`` is the component's *label* as the table shows it; it is
        resolved back to the model name, because the constraint syntax and
        the report identify components by name and the table shows the
        readable one.
        """
        result = self.fits.get(label)
        internal = name
        if result is not None:
            internal = next((c.name for c in result.components
                             if c.label == name or c.name == name), name)
        choice = self.choice_for(label)
        entry = choice.overrides.setdefault(internal, {})
        for key, value in edits.items():
            if value is None:
                entry.pop(key, None)
            else:
                entry[key] = value
        if not entry:
            choice.overrides.pop(internal, None)

    def reset_components(self, label: str) -> None:
        """Back to what the database and the data suggest."""
        choice = self.choice_for(label)
        choice.overrides.clear()
        choice.extra.clear()

    def set_window(self, label: str,
                   window: Optional[tuple[float, float]]) -> bool:
        """Set the region's fit window, or ``None`` for the whole region."""
        if window is None:
            self.choice_for(label).window = None
            return True
        low, high = sorted(float(v) for v in window)
        spectrum = next((item for item in self.regions
                         if self.region_label(item) == label), None)
        if spectrum is not None:
            available = spectrum.range
            if high <= available[0] or low >= available[1]:
                self.log(
                    "error",
                    f"{low:g}–{high:g} eV no solapa con el espectro, que va "
                    f"de {available[0]:.1f} a {available[1]:.1f} eV",
                )
                return False
        if high - low < 1.0:
            self.log("error", "una ventana de menos de 1 eV no da para ajustar")
            return False
        self.choice_for(label).window = (low, high)
        return True

    def window_for(self, label: str) -> tuple[float, float]:
        """The window that would be fitted: the user's, or the region's."""
        choice = self.choice_for(label)
        if choice.window:
            return choice.window
        spectrum = next((item for item in self.regions
                         if self.region_label(item) == label), None)
        return spectrum.range if spectrum is not None else (0.0, 0.0)

    def fit_all(self) -> int:
        """Fit every region the database describes. Returns how many worked."""
        done = 0
        for spectrum in self.regions:
            label = self.region_label(spectrum)
            if not self.database.states_for(label) and not self.choice_for(label).free:
                self.log(
                    "aviso",
                    f"la región «{spectrum.region}» no está en la base de "
                    "datos; ajústala como libre y nombra sus componentes tú",
                )
                continue
            if self.fit(label) is not None:
                done += 1
        return done

    def compare(self, label: str, counts: tuple[int, ...] = (2, 3, 4, 5)):
        """Fit several component counts and report what each one buys."""
        spectrum = next((item for item in self.regions
                         if self.region_label(item) == label), None)
        if spectrum is None:
            return [], "no hay espectro de esa región"
        choice = self.choice_for(label)
        try:
            return compare_counts(spectrum, label, counts,
                                  database=self.database,
                                  background=choice.background)
        except XPSError as error:
            return [], str(error)

    # -- quantification ------------------------------------------------
    def quantify(self) -> Optional[Quantification]:
        if not self.fits:
            self.log("aviso", "no hay ninguna región ajustada que cuantificar")
            return None
        photon = next((item.photon_energy for item in self.shifted
                       if item.photon_energy), None)
        if photon is None:
            self.log(
                "error",
                "sin energía del fotón no hay escala cinética, así que no hay "
                "corrección de transmisión ni cuantificación. Declara el ánodo",
            )
            return None
        try:
            self.composition = quantify(
                areas_from_fits(list(self.fits.values())), photon_energy=photon,
                transmission=self.transmission, exponent=self.exponent,
                database=self.database,
            )
        except XPSError as error:
            self.log("error", f"cuantificación: {error}")
            return None
        for text in self.composition.warnings:
            self.log("aviso", text)
        return self.composition

    # -- output --------------------------------------------------------
    def export_tables(self, directory: str | Path) -> list[Path]:
        """Write every fitted region as a pair of tables."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for label, result in self.fits.items():
            stem = label.replace(" ", "").replace("/", "")
            written.extend(write_fit(result, directory / f"{stem}.csv"))
        return written

    def export_vamas(self, path: str | Path) -> Path:
        """Write the referenced spectra in the format other programs read."""
        return write_vamas(path, self.shifted or self.spectra)

    def import_model(self, path: str | Path) -> Optional[str]:
        """Load a components table as the model for its region."""
        try:
            components, label = read_components(path, database=self.database)
        except (XPSError, OSError) as error:
            self.log("error", f"{Path(path).name}: {error}")
            return None
        if not label:
            self.log("error",
                     f"{Path(path).name}: la tabla no dice de qué región es")
            return None
        choice = self.choice_for(label)
        choice.states = [item.state for item in components
                         if item.state and not item.satellite]
        choice.count = len(components)
        self.log(
            "info",
            f"modelo importado para {label}: "
            + ", ".join(item.label for item in components)
            + ". Las posiciones y anchuras son las de la otra muestra y "
              "entran como punto de partida, no como valores fijos",
        )
        return label

    # -- cross-region validation ---------------------------------------
    def cross_checks(self) -> list[tuple[str, str]]:
        """``(veredicto, texto)`` comparing what each region claims.

        Section 25 of the specification, and the one check a single
        region can never do on itself. A C 1s fit will happily put a C–N
        component at 285.9 eV whether or not the sample has nitrogen in
        it: the shoulder is there, C–O sits on top of C–N, and least
        squares has no opinion about chemistry. What decides is the other
        region -- and by the time the N 1s is on screen, nobody is
        looking at the C 1s any more.

        ``veredicto`` is ``"ok"``, ``"aviso"`` or ``"incoherente"``.
        Nothing here changes a fit; an inconsistency is a result.
        """
        from ..xps.checks import bound_checks

        present = self.present_elements() or []
        checks: list[tuple[str, str]] = []
        # A parameter that stopped at its bound is not a measurement.
        for label, result in self.fits.items():
            checks.extend(bound_checks(result, label, self.database))

        # Which states each fitted region actually used.
        used: dict[str, set[str]] = {}
        for label, result in self.fits.items():
            used[label] = {c.state for c in result.components if c.state}

        def area_of(label: str, keys: set[str]) -> float:
            result = self.fits.get(label)
            if result is None:
                return 0.0
            return sum(float(c.area) for c in result.components
                       if c.state in keys)

        # C–N against the nitrogen region.
        if "C-N" in used.get("C 1s", set()):
            share = area_of("C 1s", {"C-N"})
            total = area_of("C 1s", used.get("C 1s", set())) or 1.0
            fraction = 100.0 * share / total
            if "N" not in present:
                checks.append(("incoherente", (
                    f"el C 1s asigna un {fraction:.0f} % a C–N y no hay "
                    "nitrógeno en la muestra. A 285.9 eV el C–N y el C–O se "
                    "solapan y el ajuste no distingue: lo decide el N 1s")))
            elif "N 1s" not in self.fits:
                checks.append(("aviso", (
                    f"el C 1s asigna un {fraction:.0f} % a C–N; ajusta el "
                    "N 1s para contrastarlo")))
            else:
                checks.append(("ok", (f"C–N ({fraction:.0f} % del C 1s) con "
                                      "N 1s ajustado: coherente")))

        # Oxygenated carbon against the oxygen region.
        oxygenated = {"C-O", "C=O", "O-C=O", "carbonate"} & used.get("C 1s", set())
        if oxygenated and "O" not in present:
            checks.append((
                "incoherente",
                "el C 1s asigna carbono oxigenado ("
                + ", ".join(sorted(oxygenated))
                + ") y no hay oxígeno en la muestra"))

        # A sulphide needs a metal to be a sulphide of.
        metals = {"Fe", "Co", "Ni", "Mo", "W", "Se", "Ti", "Zn", "Cu", "Mn"}
        sulfide = {"sulfide", "metal-sulfide"} & used.get("S 2p3/2", set())
        if sulfide and not (metals & set(present)):
            checks.append(("incoherente", (
                "el S 2p asigna sulfuro metálico y no hay ningún metal "
                "detectado. Sin el 2p del metal eso no se sostiene")))

        # A state the composition rules out, wherever it was used.
        for label, keys in used.items():
            for name, why in self.impossible_states(label):
                state = next((s for s in
                              self.database.states_for(label,
                                                       include_satellites=True)
                              if s.name == name and s.key in keys), None)
                if state is not None:
                    checks.append((
                        "incoherente",
                        f"{label}: «{name}» está ajustado y {why}"))
        return checks

    def report(self) -> str:
        """The written report, built from what is currently in the session."""
        from ..xps.report import XPSAnalysis, build_report

        analysis = XPSAnalysis(
            name=self.name, survey=self.survey,
            regions=list(self.fits.values()), composition=self.composition,
            calibration=self.calibration,
            warnings=[text for level, text in self.messages if level == "aviso"],
        )
        text = build_report(analysis)
        checks = self.cross_checks()
        if not checks:
            return text
        marks = {"ok": "✓", "aviso": "⚠", "incoherente": "✗"}
        lines = ["", "Validación cruzada entre regiones",
                 "-" * 34]
        lines.extend(f"{marks.get(verdict, '·')} {body}"
                     for verdict, body in checks)
        lines.append(
            "Una región no puede comprobarse a sí misma: el ajuste de un "
            "C 1s pone un C–N donde hay hombro, haya nitrógeno o no.")
        return text + "\n".join(lines) + "\n"


__all__ = ["BACKGROUNDS", "PROFILES", "REFERENCES", "RegionChoice",
           "XPSSession"]
