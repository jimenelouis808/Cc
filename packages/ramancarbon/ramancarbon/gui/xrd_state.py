"""Diffraction session state, with no Tkinter in it.

Same rule as :mod:`ramancarbon.gui.state`: everything the diffraction tab
knows and decides lives here, so it can be tested on a machine with no
display. The Tk layer in :mod:`ramancarbon.gui.xrd_app` only reads and
writes these objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..xrd.io import read_pattern
from ..xrd.pattern import Pattern
from ..xrd.reference import LibraryEntry, load_library
from ..xrd.report import XRDResult, analyse_pattern
from ..xrd.rietveld import Parameter, PhaseModel, RietveldResult, auto_refine, build_parameters, free_kinds, refine
from ..xrd.scattering import ANODES
from ..xrd.structure import Crystal

#: Parameter groups the interface offers, and what each one is for.
PARAMETER_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("scale", "Escala", "Una por fase. De aquí salen las fracciones en peso."),
    ("background", "Fondo", "Coeficientes de Chebyshev."),
    ("zero", "Cero", "Desplazamiento constante del eje 2θ."),
    ("displacement", "Desplazamiento de muestra",
     "Término en cos θ. Es OTRA función del ángulo, no un cero disfrazado."),
    ("lattice", "Celda", "a, b, c, con las ligaduras de la simetría."),
    ("profile_w", "Anchura (W)", "El término constante de Caglioti."),
    ("profile_uv", "Perfil (U, V)", "Dependencia de la anchura con el ángulo."),
    ("eta", "Forma (η)", "Mezcla gaussiana/lorentziana."),
    ("preferred", "Orientación preferente",
     "March-Dollase. Necesita que declares un eje de textura."),
    ("u_iso", "U_iso", "Desplazamiento atómico. El que se traga los errores de los demás."),
)


@dataclass
class LoadedPattern:
    """One diffractogram in the session, with whatever has been done to it."""

    pattern: Pattern
    result: Optional[XRDResult] = None
    refinement: Optional[RietveldResult] = None
    parameters: Optional[list[Parameter]] = None
    models: list[PhaseModel] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.pattern.name

    @property
    def analysed(self) -> bool:
        return self.result is not None


class XRDSession:
    """Everything the diffraction section knows."""

    def __init__(self) -> None:
        self.patterns: list[LoadedPattern] = []
        self.current: int = -1
        # Restored from the preferences file, not empty: a folder the user
        # added once and lost on the next start is the same as no folder.
        # Directories that have gone (an unmounted share, a deleted
        # download) are dropped quietly rather than reported every start.
        from ..core.history import Preferences

        self._preferences = Preferences.load()
        self.cif_directories: list[str] = [
            folder for folder in self._preferences.get("cif_directories", [])
            if Path(folder).is_dir()
        ]
        self.selected_phases: list[str] = []
        """Names of reference phases the user pinned. Empty means "search
        the whole library"."""
        self.overlay_phases: list[str] = []
        """Names of phases to DRAW over the measured pattern, with no
        verdict attached. A different thing from `selected_phases`: that
        one narrows what the search will consider, this one narrows
        nothing and asserts nothing. It exists for the case the search
        is worst at -- it came back with nothing, and the question is
        which candidate to go and look up."""
        self.anode: str = "Cu"
        self.wavelength: Optional[float] = None
        self.kalpha2_ratio: float = 0.5
        self.counts: bool = True
        self.texture_axis: Optional[tuple[int, int, int]] = None
        self.instrument_fwhm: float = 0.06
        self.background_order: int = 6
        self.max_phases: int = 4
        self.smooth_window: int = 0
        """Savitzky-Golay window for the PEAK SEARCH, in points. 0 is off.
        Never applied to the refinement."""
        self.background_lambda: Optional[float] = None
        """Stiffness of the background removed before peak finding. Raise
        it for nanocrystalline patterns, whose reflections are wide enough
        that the default treats them as background."""
        self.min_significance: Optional[float] = None
        self.messages: list[tuple[str, str]] = []

    # -- messages ------------------------------------------------------
    def log(self, level: str, text: str) -> None:
        self.messages.append((level, text))
        if len(self.messages) > 300:
            del self.messages[:-300]

    # -- data ----------------------------------------------------------
    def load(self, paths: Iterable[str | Path]) -> int:
        """Read diffractograms, keeping going past unreadable files."""
        added = 0
        for path in paths:
            location = Path(path)
            try:
                pattern = read_pattern(
                    location,
                    wavelength=self.wavelength,
                    anode=self.anode,
                    counts=self.counts,
                    kalpha2_ratio=self.kalpha2_ratio,
                )
            except (OSError, ValueError) as exc:
                self.log("error", f"{location.name}: {exc}")
                continue
            self.patterns.append(LoadedPattern(pattern=pattern))
            self._report_header(pattern)
            added += 1
        if added and self.current < 0:
            self.current = 0
        return added

    def _report_header(self, pattern: Pattern) -> None:
        """Say what the file's own header supplied, and what it did not.

        A .ras records the wavelength the tube actually ran at, the step,
        the scan speed and the counting unit. Until the reader had a
        branch for the format these were all discarded -- the file opened
        and the numbers were right, so nothing looked wrong, while every
        d-spacing sat on a default 1.540598 Å instead of the recorded
        1.540593. That is a part in three thousand, a tenth of a degree
        at 80° 2θ, and wider than the peak-matching window.

        So the header is now reported rather than absorbed silently: the
        user can see that the file was read as a .ras and not as three
        anonymous columns.
        """
        meta = getattr(pattern, "metadata", None) or {}
        recorded = meta.get("wavelength_recorded")
        parts: list[str] = []
        if recorded is not None:
            parts.append(f"λ(Kα₁) = {float(recorded):.6f} Å del archivo")
        if meta.get("wavelength_alpha2") is not None:
            parts.append(f"Kα₂ = {float(meta['wavelength_alpha2']):.6f} Å")
        if meta.get("step") is not None:
            parts.append(f"paso {float(meta['step']):g}°")
        if meta.get("dwell") is not None:
            parts.append(f"{float(meta['dwell']):.3f} s/punto")
        if meta.get("unit"):
            parts.append(f"unidad «{meta['unit']}»")
        if meta.get("sample"):
            parts.append(f"muestra «{meta['sample']}»")
        if not parts:
            fmt = meta.get("format")
            if fmt in ("ras", "asc"):                    # pragma: no cover
                self.log("warn", f"{pattern.name}: el archivo .{fmt} no trajo "
                                 "cabecera legible; se usan los valores del "
                                 "panel Equipo.")
            return
        self.log("info", f"{pattern.name}: " + ", ".join(parts) + ".")
        if meta.get("unit") == "cps" and self.counts:
            self.log(
                "warn",
                f"{pattern.name}: la cabecera dice «cps», así que las "
                "intensidades no son cuentas y σ = √N no es su "
                "incertidumbre. Desmarca «Los datos son cuentas» o "
                "multiplica por el tiempo por punto.")

    def add_pattern(self, pattern: Pattern) -> None:
        self.patterns.append(LoadedPattern(pattern=pattern))
        if self.current < 0:
            self.current = 0

    @property
    def item(self) -> Optional[LoadedPattern]:
        if 0 <= self.current < len(self.patterns):
            return self.patterns[self.current]
        return None

    def remove_current(self) -> None:
        if self.item is None:
            return
        del self.patterns[self.current]
        self.current = min(self.current, len(self.patterns) - 1)

    # -- library -------------------------------------------------------
    def remember_cif_directories(self) -> None:
        """Persist the added folders. Never raises: a preferences file that
        cannot be written is a nuisance, not a reason to lose the session."""
        try:
            self._preferences.set("cif_directories",
                                  list(self.cif_directories))
            self._preferences.save()
        except OSError:
            pass

    def library(self) -> list[LibraryEntry]:
        return load_library(self.cif_directories)

    def overlay_crystals(self) -> list[Crystal]:
        """The structures the user asked to see drawn on the pattern.

        Separate from `selected_phases`, which narrows the SEARCH. This
        narrows nothing and claims nothing: it answers "would this one
        line up?", which is the question left when the identification
        came back empty, and which the search cannot answer because its
        job is to refuse.
        """
        out: list[Crystal] = []
        by_name = {entry.crystal.name: entry for entry in self.library()}
        for name in self.overlay_phases:
            entry = by_name.get(name)
            if entry is None:
                self.log("aviso", f"«{name}» ya no está en la biblioteca")
                continue
            out.append(entry.crystal)
        return out

    def add_overlay(self, name: str) -> bool:
        """Draw one more library phase over the measured pattern."""
        if not name:
            return False
        if name in self.overlay_phases:
            return True
        if name not in {entry.crystal.name for entry in self.library()}:
            self.log("error", f"«{name}» no está en la biblioteca")
            return False
        self.overlay_phases.append(name)
        return True

    def remove_overlay(self, name: str) -> bool:
        if name not in self.overlay_phases:
            return False
        self.overlay_phases.remove(name)
        return True

    def clear_overlays(self) -> None:
        self.overlay_phases.clear()

    def candidates(self) -> Optional[list[Crystal]]:
        """The structures to test, or ``None`` for the whole library."""
        entries = self.library()
        if not self.selected_phases:
            return [entry.crystal for entry in entries]
        wanted = set(self.selected_phases)
        chosen = [e.crystal for e in entries if e.crystal.name in wanted]
        return chosen or [e.crystal for e in entries]

    # -- analysis ------------------------------------------------------
    def analyse_current(self, refine_after: bool = True) -> Optional[XRDResult]:
        """Identify phases and optionally refine, on the selected pattern."""
        item = self.item
        if item is None:
            return None
        result = analyse_pattern(
            item.pattern,
            candidates=self.candidates(),
            refine=refine_after,
            max_phases=self.max_phases,
            preferred_axis=self.texture_axis,
            instrument_fwhm=self.instrument_fwhm,
            smooth_window=self.smooth_window,
            background_lambda=self.background_lambda,
            min_significance=self.min_significance,
        )
        item.result = result
        item.refinement = result.refinement
        if result.refinement is not None:
            item.models = result.refinement.phases
            item.parameters = result.refinement.parameters
        for warning in result.warnings:
            self.log("warning", warning)
        return result

    def prepare_manual(self) -> Optional[list[Parameter]]:
        """Build a parameter list for hand refinement of the current phases.

        The phases come from the identification when there is one and from
        the pinned selection otherwise, because Rietveld fits a model — it
        does not discover one, and offering a manual refinement with no
        phases would just be a way of fitting the background.
        """
        item = self.item
        if item is None:
            return None
        if not item.models:
            crystals: list[Crystal] = []
            if item.result and item.result.search.accepted:
                crystals = [m.crystal for m in item.result.search.accepted]
            elif self.selected_phases:
                crystals = self.candidates() or []
            if not crystals:
                self.log(
                    "error",
                    "no hay fases que refinar. Identifícalas primero, o fija "
                    "las que sepas que hay en la lista de la biblioteca",
                )
                return None
            item.models = [PhaseModel(crystal=c) for c in crystals]
        item.parameters = build_parameters(item.models, self.background_order)
        free_kinds(item.parameters, ("scale", "background"))
        return item.parameters

    def model_phase_names(self) -> list[str]:
        """Names of the phases currently IN the refinement model."""
        item = self.item
        return [m.crystal.name for m in item.models] if item and item.models else []

    def add_phase_to_model(self, name: str) -> bool:
        """Put one library phase into the refinement model.

        The case this exists for is the one every refinement runs into: a
        systematic bump left in the difference curve that no amount of
        refining the phases already in the model will remove, because it
        belongs to a phase that is not in it. Until now the only way to
        add one was to re-run the identification and hope it found it.
        """
        item = self.item
        if item is None:
            return False
        if name in self.model_phase_names():
            self.log("info", f"{name} ya está en el modelo")
            return False
        crystal = next((e.crystal for e in self.library() if e.crystal.name == name),
                       None)
        if crystal is None:
            self.log("error", f"{name} no está en la biblioteca de referencia")
            return False
        item.models = list(item.models) + [PhaseModel(crystal=crystal)]
        # The parameter list is built from the model, so it has to be
        # rebuilt; keeping the old one would refine a phase that is no
        # longer the model's shape.
        item.parameters = build_parameters(item.models, self.background_order)
        free_kinds(item.parameters, ("scale", "background"))
        self.log("info",
                 f"{name} añadida al modelo. Los parámetros se han vuelto a "
                 "preparar: solo escala y fondo están libres")
        return True

    def remove_phase_from_model(self, name: str) -> bool:
        """Take one phase out of the refinement model."""
        item = self.item
        if item is None or not item.models:
            return False
        kept = [m for m in item.models if m.crystal.name != name]
        if len(kept) == len(item.models):
            return False
        if not kept:
            self.log("error",
                     "no se puede quitar la última fase: un refinamiento sin "
                     "fases solo ajusta el fondo")
            return False
        item.models = kept
        item.parameters = build_parameters(item.models, self.background_order)
        free_kinds(item.parameters, ("scale", "background"))
        self.log("info", f"{name} quitada del modelo")
        return True

    def refinement_metrics(self) -> list[tuple[str, str]]:
        """The numbers a refinement is judged by, as label/value rows.

        χ² reduced next to the goodness of fit because they are the same
        statement and different communities read different ones, and the
        evaluation count next to both because a refinement that returns
        instantly is either converged or never started, and the two look
        identical from outside.
        """
        item = self.item
        outcome = item.refinement if item else None
        if outcome is None:
            return []
        rows = [
            ("Rp", f"{100 * outcome.r_p:.2f} %"),
            ("Rwp", f"{100 * outcome.r_wp:.2f} %"),
            ("Rexp", f"{100 * outcome.r_expected:.2f} %"),
            ("GOF", f"{outcome.gof:.3f}"),
            ("χ² reducida", f"{outcome.chi_squared:.3f}"),
            ("evaluaciones", str(outcome.n_evaluations)),
            ("parámetros libres", f"{outcome.free_parameters} / {outcome.pattern.n} pts"),
            ("convergido", "sí" if outcome.converged else "NO"),
        ]
        fractions = outcome.weight_fractions()
        sizes = outcome.crystallite_sizes(self.instrument_fwhm)
        for name, fraction in fractions.items():
            size = sizes.get(name)
            text = "no disponible" if fraction is None else f"{100 * fraction:.1f} % peso"
            if size:
                text += f", {size:.0f} nm"
            rows.append((name, text))
        return rows

    def refine_current(
        self,
        progress: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Optional[RietveldResult]:
        """Run one refinement with whatever parameters are marked free.

        ``progress`` and ``should_stop`` are the same pair
        :meth:`auto_refine_current` takes, and they were missing here --
        so the manual button ran a fit of arbitrary length with no
        counter and no way out, which from the outside is exactly what a
        hang looks like. With five phases the budget is thousands of
        evaluations, and a refinement that is working looks identical to
        one that is stuck.
        """
        item = self.item
        if item is None or item.parameters is None or not item.models:
            self.log("error", "prepara primero los parámetros del refinamiento")
            return None
        try:
            outcome = refine(
                item.pattern,
                item.models,
                parameters=item.parameters,
                background_order=self.background_order,
                instrument_fwhm=self.instrument_fwhm,
                callback=progress,
                should_stop=should_stop,
            )
        except ValueError as exc:
            self.log("error", str(exc))
            return None
        item.refinement = outcome
        for warning in outcome.warnings:
            self.log("warning", warning)
        return outcome

    def auto_refine_current(
        self,
        progress: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Optional[RietveldResult]:
        """Run the staged automatic protocol.

        ``progress`` is called with one line per stage and every tenth
        residual evaluation, so a refinement can be watched instead of
        guessed at. A staged refinement with a good starting point really
        does converge in a few seconds, and without a counter that is
        indistinguishable from one that never ran.
        """
        item = self.item
        if item is None:
            return None
        if not item.models:
            if self.prepare_manual() is None:
                return None
        outcome = auto_refine(
            item.pattern,
            item.models,
            background_order=self.background_order,
            preferred_axis=self.texture_axis,
            instrument_fwhm=self.instrument_fwhm,
            callback=progress,
            should_stop=should_stop,
        )
        item.refinement = outcome
        item.parameters = outcome.parameters
        for warning in outcome.warnings:
            self.log("warning", warning)
        return outcome

    def background_sensitivity(
        self,
        progress: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        """Refine the same phases against several background orders.

        The check a weight fraction cannot perform on itself. A broad
        reflection from a nanocrystalline phase and a flexible polynomial
        describe the same shape, and no single refinement can tell which
        of the two it just fitted: it converges either way, with plausible
        R factors and a difference curve that looks fine.

        Measured on a real CVD pattern of carbon on FeSe, the
        turbostratic carbon came back at 25.9 % of the sample by weight
        with a second-order background, 36.6 % at fourth, 52.4 % at sixth,
        33.1 % at eighth and 33.1 % at tenth. The package default is
        sixth, and sixth was the outlier.

        Costs one full refinement per order, which is why it is a
        separate button.
        """
        from ..xrd.rietveld import background_order_sensitivity

        item = self.item
        if item is None:
            self.log("error", "carga un difractograma primero")
            return None
        if not item.models and self.prepare_manual() is None:
            return None
        return background_order_sensitivity(
            item.pattern,
            item.models,
            preferred_axis=self.texture_axis,
            instrument_fwhm=self.instrument_fwhm,
            callback=progress,
            should_stop=should_stop,
        )

    # -- tables --------------------------------------------------------
    def phase_rows(self) -> list[tuple[str, ...]]:
        """Rows for the identified-phase table."""
        item = self.item
        if item is None or item.result is None:
            return []
        rows: list[tuple[str, ...]] = []
        fractions = (
            item.refinement.weight_fractions() if item.refinement is not None else {}
        )
        sizes = (
            item.refinement.crystallite_sizes(self.instrument_fwhm)
            if item.refinement is not None
            else {}
        )
        for match in item.result.search.accepted:
            name = match.crystal.name
            fraction = fractions.get(name)
            size = sizes.get(name)
            rows.append(
                (
                    name,
                    match.crystal.formula,
                    match.verdict,
                    f"{match.score:.2f}",
                    f"{100 * fraction:.1f}" if fraction is not None else "—",
                    f"{size:.1f}" if size is not None else "—",
                )
            )
        return rows

    def peak_rows(self) -> list[tuple[str, ...]]:
        item = self.item
        if item is None or item.result is None:
            return []
        explained = {
            id(peak)
            for match in item.result.search.accepted
            for _, peak in match.matched
        }
        satellites = {id(p) for p in item.result.search.kalpha2_residuals}
        rows = []
        for peak in item.result.peaks:
            if id(peak) in explained:
                label = "asignado"
            elif id(peak) in satellites:
                label = "cola Kα₂"
            else:
                label = "SIN EXPLICAR"
            rows.append(
                (
                    f"{peak.two_theta:.3f}",
                    f"{peak.d:.4f}",
                    f"{peak.height:.0f}",
                    f"{peak.fwhm:.3f}" if peak.fwhm else "—",
                    f"{peak.significance:.0f}",
                    label,
                )
            )
        return rows

    def refinement_rows(self) -> list[tuple[str, str]]:
        """The agreement factors, as a refinement program reports them.

        The engine has carried all of these since the counter was added;
        the section was showing two of them in a plot title. FullProf
        prints Rp, Rwp, Rexp and chi-squared together because they answer
        different questions — Rwp against Rexp is whether the fit is as
        good as the counting statistics allow, and neither is worth
        anything until the difference curve has no structure in it.
        """
        item = self.item
        result = item.refinement if item is not None else None
        if result is None:
            return []
        rows = [
            ("Estado", "convergido" if result.converged else "NO convergido"),
            ("Mensaje", result.message or "—"),
            ("Rp", f"{100 * result.r_p:.2f} %"),
            ("Rwp", f"{100 * result.r_wp:.2f} %"),
            ("Rexp", f"{100 * result.r_expected:.2f} %"),
            ("GOF (Rwp/Rexp)", f"{result.gof:.3f}"),
            ("χ² reducida", f"{result.chi_squared:.3f}"),
            ("Evaluaciones del residuo", f"{result.n_evaluations}"),
            ("Parámetros libres",
             f"{result.free_parameters} sobre {result.pattern.n} puntos"),
            ("Cero", f"{result.zero:+.4f}°"),
            ("Desplazamiento de muestra", f"{result.displacement:+.4f}°"),
        ]
        if result.stages:
            rows.append(("Etapas", " → ".join(result.stages)))
        return rows

    def fraction_rows(self) -> list[tuple[str, str, str, str]]:
        """``(phase, formula, weight %, crystallite size)``.

        The fractions are of the CRYSTALLINE, MODELLED part and they sum
        to 100 % always: a phase nobody modelled does not lower the
        others, its intensity is shared out among them, and the amorphous
        content does not appear at all. That warning travels with the
        numbers rather than being left in the report.
        """
        item = self.item
        result = item.refinement if item is not None else None
        if result is None:
            return []
        fractions = result.weight_fractions()
        sizes = result.crystallite_sizes(self.instrument_fwhm)
        rows = []
        for phase in result.phases:
            crystal = phase.current_crystal()
            fraction = fractions.get(crystal.name)
            size = sizes.get(crystal.name)
            rows.append((
                crystal.name,
                crystal.formula,
                f"{100 * fraction:.1f} %" if fraction is not None else "—",
                f"{size:.1f} nm" if size else "—",
            ))
        return rows

    def refinement_output(self) -> str:
        """The full refinement report, for reading or saving."""
        item = self.item
        result = item.refinement if item is not None else None
        if result is None:
            return "No hay ningún refinamiento todavía."
        return result.summary(self.instrument_fwhm)

    def parameter_rows(self) -> list[tuple[str, ...]]:
        item = self.item
        if item is None or item.parameters is None:
            return []
        rows = []
        for parameter in item.parameters:
            error = (
                f"{parameter.error:.3g}" if parameter.error is not None else "—"
            )
            rows.append(
                (
                    parameter.name,
                    "sí" if parameter.free else "no",
                    f"{parameter.value:.6g}",
                    error,
                    parameter.kind,
                )
            )
        return rows

    def set_free(self, kinds: Iterable[str]) -> None:
        """Free exactly the named parameter groups."""
        item = self.item
        if item is None or item.parameters is None:
            return
        free_kinds(item.parameters, list(kinds))

    def toggle_parameter(self, name: str) -> None:
        item = self.item
        if item is None or item.parameters is None:
            return
        for parameter in item.parameters:
            if parameter.name == name:
                parameter.free = not parameter.free
                return

    def set_parameter_value(self, name: str, value: float) -> bool:
        item = self.item
        if item is None or item.parameters is None:
            return False
        for parameter in item.parameters:
            if parameter.name == name:
                if not parameter.lower <= value <= parameter.upper:
                    self.log(
                        "error",
                        f"{name}: {value:g} está fuera de sus límites "
                        f"[{parameter.lower:g}, {parameter.upper:g}]",
                    )
                    return False
                parameter.value = float(value)
                return True
        return False

    # -- misc ----------------------------------------------------------
    def report(self) -> str:
        item = self.item
        if item is None:
            return "Carga un difractograma (.xy, .xye, .dat, .txt, .xrdml…)."
        if item.result is None:
            return item.pattern.describe() + "\n\nSin analizar todavía."
        return item.result.report()

    def anode_choices(self) -> list[str]:
        return sorted(ANODES)


__all__ = ["PARAMETER_GROUPS", "LoadedPattern", "XRDSession"]
