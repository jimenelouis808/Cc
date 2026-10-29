"""Electrochemistry session state, with no Tkinter in it.

One session holds the several measurements of a single electrode — a
voltammogram, a series of them at different scan rates, a charge–discharge
curve, an impedance spectrum, a polarisation curve — because that is the
unit the analysis works on: the mechanism classification is stronger the
more of them there are, and the cross-checks between them are the most
useful thing the section produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

from ..echem.curve import (ChargeDischarge, CurveError, Electrode, Impedance,
                           Voltammogram)
from ..echem.eis import (
    AUTO_CIRCUIT,
    CircuitError,
    CircuitTemplate,
    analyse_eis,
    brug_capacitance,
    circuit_for,
    circuit_templates,
    compare_circuits,
    delete_user_circuit,
    draw_circuit,
    hsu_mansfeld_capacitance,
    parse_circuit,
    save_user_circuit,
)
from ..echem.io import EchemIOError, read_cv, read_eis, read_gcd
from .figures_echem import (PRESET_LABELS, CompositeFigure,
                            apply_preset, available_panels,
                            default_panels, draw_composite, every_axes,
                            save_composite)
from ..echem.report import EchemResult, analyse_sample

#: The kinds of measurement a session holds, and what each one is for.
MEASUREMENT_KINDS: tuple[tuple[str, str], ...] = (
    ("cv", "Voltamperometría cíclica"),
    ("rates", "Serie de velocidades"),
    ("gcd", "Carga-descarga"),
    ("gcd_series", "Serie de carga-descarga (Ragone)"),
    ("eis", "Impedancia"),
    ("eis_series", "Serie de impedancias (comparar muestras)"),
    ("ms_series", "Serie de potenciales (Mott-Schottky)"),
    ("lsv", "Curva de polarización (HER/OER)"),
)

#: Header of :meth:`EchemSession.mott_schottky_rows`.
#:
#: One row per FREQUENCY, because that is the axis along which a
#: Mott-Schottky is checked: the answer depends on the frequency chosen
#: and a single row is not a measurement.
MOTT_SCHOTTKY_COLUMNS: tuple[str, ...] = (
    "f_hz", "tipo", "E_fb_V", "E_fb_error_V", "N_cm3", "N_error_cm3",
    "pendiente_F-2_V-1", "r2", "ventana_V", "puntos",
)

#: Header of :meth:`EchemSession.eis_comparison_rows`.
#:
#: The circuit is a column and not a caption because every spectrum in a
#: series gets the one its own shape asks for, and which one that was
#: decides what the rest of the row means.
EIS_COMPARISON_COLUMNS: tuple[str, ...] = (
    "muestra", "R_s_ohm", "R_ion_ohm", "codo_hz", "fase_f_min_grados",
    "C_entregable_mF", "C_elemento_mF", "circuito", "residuo_pct",
)


#: Unit choices offered for the columns of a text export, as
#: ``(label, multiplier to SI)``.
#:
#: Explicit, not guessed. A column of numbers around 0.001 is equally
#: plausibly amps or milliamps, and the consequence of getting it wrong
#: is not subtle: a current density a thousand times too large drags the
#: iR correction with it, so an overpotential of 170 mV comes back as
#: 400 V and the axis runs to 2·10⁵ mA/cm². A file with a unit header
#: still wins — this is for the ones without one.
POTENTIAL_UNITS: tuple[tuple[str, float], ...] = (
    ("V / cabecera", 1.0),
    ("mV", 1e-3),
)

CURRENT_UNITS: tuple[tuple[str, float], ...] = (
    ("A / cabecera", 1.0),
    ("mA", 1e-3),
    ("µA", 1e-6),
)


def _typical_current(curve: ChargeDischarge) -> float:
    """The magnitude of the current a charge-discharge curve was run at.

    The median of |i| over the whole curve, which needs only that the
    current sat at its set value for more than half the record -- which
    is what "galvanostatic" means. Nothing is filtered out first: an
    earlier version dropped points below a fraction of the MAXIMUM, and
    then a single overshoot at a reversal threw away every real point and
    the "typical current" came back as the spike. A maximum is the wrong
    statistic here for the same reason it is everywhere else in this
    package.
    """
    import numpy as np

    magnitude = np.abs(np.asarray(curve.current, dtype=float))
    return float(np.median(magnitude)) if magnitude.size else 0.0


#: Diameter in mm of the disc electrode most measurements here are made
#: on. It is a default, not an assumption: the field is editable and the
#: area it produces is shown.
DEFAULT_DISC_MM = 3.0

#: Electrolyte the electrode card starts on, as a key of the table in
#: ``echem.json``.
DEFAULT_ELECTROLYTE = "KOH_0.5M"

#: Reference the electrode card starts on. Saturated KCl rather than
#: 3 M: they are 13 mV apart and the saturated one is what is usually on
#: the bench.
DEFAULT_REFERENCE = "Ag/AgCl_sat"


#: Columns of the tables :meth:`save_tables` writes, in order. Kept
#: here rather than built inline so the CSV header and the values
#: cannot drift apart.
TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "ciclos": ("archivo", "mv_s", "ciclo", "completo", "puntos",
               "duracion_s", "ventana_v", "recorrido_v", "q_ox_c",
               "q_red_c", "eficiencia", "c_lazo_f", "c_lazo_f_g",
               "c_rama_f", "c_red_f", "c_red_f_g", "c_red_mf_cm2"),
    "velocidades": ("mv_s", "c_lazo_f", "c_lazo_f_g", "c_red_f",
                    "c_red_f_g", "eficiencia", "duracion_ciclo_s",
                    "q_ox_c", "q_red_c", "delta_q_c"),
    "leyes": ("e_v", "e_eje_v", "b", "r2_potencia", "r2_dunn", "r2_fuga",
              "k_cap_f", "i_par_a", "empate"),
    "dunn": ("e_v", "e_eje_v", "k1_a_por_vs", "k2_a_por_sqrtvs", "r2",
             "i_cap_anodica_a", "i_cap_catodica_a", "no_definida"),
}


def _default_electrode() -> Electrode:
    """A glassy carbon disc in the electrolyte most of these runs use.

    Defaults exist so that the common case needs no typing, and every one
    of these is editable in the sidebar. The mass is deliberately NOT
    defaulted: there is no sensible value for it, a wrong one is
    invisible, and a blank one makes the F/g disappear, which is the
    failure that can be seen.
    """
    electrode = Electrode(reference=DEFAULT_REFERENCE)
    electrode.use_disc(DEFAULT_DISC_MM)
    electrode.use_electrolyte(DEFAULT_ELECTROLYTE)
    return electrode


@dataclass
class EchemSession:
    """Everything the electrochemistry section knows."""

    name: str = "muestra"
    electrode: Electrode = field(default_factory=lambda: _default_electrode())
    cv: Optional[Voltammogram] = None
    rate_series: list[Voltammogram] = field(default_factory=list)
    gcd: Optional[ChargeDischarge] = None
    gcd_series: list[ChargeDischarge] = field(default_factory=list)
    """Charge-discharge curves at several currents. One Ragone point per
    curve, and a Ragone plot with one point is not a Ragone plot: the
    whole content of the figure is how the energy falls as the power
    rises, which needs the curves at the other currents."""
    eis: Optional[Impedance] = None
    eis_series: list[Impedance] = field(default_factory=list)
    """Several spectra to compare — one per sample, per temperature, per
    potential. Comparing them is the whole point of measuring more than
    one, and the section could hold exactly one: the user's three
    carbons pyrolysed at 800, 900 and 1020 °C had to be opened one at a
    time and written down by hand. Each spectrum is analysed on its own
    and the comparison table is built across them."""
    ms_series: list[Impedance] = field(default_factory=list)
    """Spectra of the SAME electrode at different DC biases, for a
    Mott-Schottky plot. Separate from :attr:`eis_series`, which holds
    one spectrum per SAMPLE: the two are different experiments and
    mixing them would put a sample's name on a potential axis."""
    ms_material: str = ""
    """Key of a semiconductor in ``echem.json``, for its ε."""
    ms_epsilon: Optional[float] = None
    """ε typed by hand, which wins over the material's. ``None`` reads
    the database."""
    ms_frequencies: tuple[float, ...] = (1000.0, 500.0, 100.0)
    """Where to read the capacitance. THREE by default, not one: the
    answer of a Mott-Schottky depends on the frequency, and the only way
    to find out how much is to do it at several."""
    ms_temperature_k: float = 298.15
    lsv: Optional[Voltammogram] = None
    reaction: str = "OER"
    circuit: str = AUTO_CIRCUIT
    """The circuit NAME, or a circuit string typed by hand, or
    ``"auto"``. All three work: the name is resolved against the bundled
    and user libraries, anything unrecognised is parsed as a circuit,
    and ``"auto"`` reads the shape of the spectrum and settles a
    shortlist by comparison.

    ``"auto"`` is the default because the previous default was
    ``randles_cpe``, and a default circuit is a hypothesis nobody
    stated. On a porous carbon that one has no way to describe a
    capacitive tail except by sending its charge-transfer resistance to
    its 1e9 bound, which it did on all three of the user's spectra."""
    circuit_initial: dict[str, float] = field(default_factory=dict)
    """Starting values by label, for the ones the user has overridden.
    Empty means "read them off the spectrum", which is what should
    normally happen."""
    circuit_fixed: set[str] = field(default_factory=set)
    """Labels held rather than refined."""
    potential_scale: float = 1.0
    """Multiplier from the file's potential column to VOLTS."""
    current_scale: float = 1.0
    """Multiplier from the file's current column to AMPS. A thousand here
    is a thousand in every capacitance, every current density and every
    overpotential."""
    dunn_sweep: str = "media"
    """Which branch of the cycle the Dunn separation reads. The anodic
    sweep alone is the usual published choice, and it is a choice: on a
    material whose oxidation and reduction are not mirror images the two
    halves give different coefficients, and that difference is
    information."""
    dunn_rate: Optional[float] = None
    """Which scan rate the Dunn figure is drawn at, in V/s. ``None``
    takes the SLOWEST, because that is where the diffusive contribution
    is largest and therefore where the separation has something to
    show — the fastest sweep is where every electrode looks surface
    controlled."""
    drt_regularisation: Optional[float] = None
    """λ for the distribution of relaxation times. ``None`` chooses it
    by the L-curve. Whichever it is, it is reported: a DRT is an
    ill-posed inversion and λ is a choice about how much structure to
    believe, so a DRT without its λ is not a measurement."""
    cycle_choice: Optional[int] = None
    """Which cycle of every loaded voltammogram to report, 1-based.

    ``None`` is the last COMPLETE one, which is the right default and is
    still a default: it is not the cycle to quote when the later ones
    drifted, nor when several files have to be compared at the same cycle
    number. The choice is pushed onto the curves themselves, so it
    survives into every capacitance the package computes and into a saved
    project."""
    potential_scale_choice: str = "medido"
    """Which scale the potential axes are drawn and exported on. The
    MEASURED one by default: a conversion is a claim and the measured
    numbers are not."""
    dunn_nonnegative: bool = True
    """Hold the Dunn coefficients at or above zero, which is what the
    method means. Off only to see the unconstrained fit."""
    dunn_points: int = 60
    """How many potentials the Dunn separation is run at."""
    dunn_window: Optional[tuple[float, float]] = None
    """Potential range for Dunn. ``None`` is the range common to every
    curve, which is the only honest default."""
    capacitance_window: Optional[tuple[float, float]] = None
    """Window the CV capacitance is divided by. ``None`` is the measured
    one. Setting it narrower than the sweep does NOT re-integrate over
    the narrower window — it changes the divisor — so it is for quoting a
    capacitance over a stated window, not for excluding a region."""
    subtract_parasitic: bool = False
    """Remove the constant anodic current the charge balance measures
    from the rate series before b and Dunn. Off by default because it is
    a correction, and a correction that is on by default stops being
    visible."""
    non_faradaic: bool = False
    """Whether the rate study's window was chosen free of faradaic current.
    It is off by default because most windows are not, and claiming
    otherwise turns a C_dl that includes reaction current into an ECSA."""
    composite: CompositeFigure = field(default_factory=CompositeFigure)
    """Which panels the multi-panel figure holds, and how they are laid
    out. A figure is a VALUE here, the same rule the plotting engine
    follows, so the choice survives a saved project and comes back
    identical."""
    result: Optional[EchemResult] = None
    messages: list[tuple[str, str]] = field(default_factory=list)
    _eis_analyses: Optional[list] = field(default=None, repr=False)
    _ms_result: Optional[object] = field(default=None, repr=False)
    """Cache for :meth:`eis_analyses`. Every impedance panel needs the
    same per-spectrum analysis and each one costs a circuit comparison,
    so a six-panel figure would run it six times per spectrum."""

    # -- messages ------------------------------------------------------
    def log(self, level: str, text: str) -> None:
        self.messages.append((level, text))
        if len(self.messages) > 300:
            del self.messages[:-300]

    # -- loading -------------------------------------------------------
    def _apply_electrode(self, curve) -> None:
        curve.electrode = self.electrode

    def load_cv(
        self, path: str | Path, scan_rate: Optional[float] = None,
        into_series: bool = False,
    ) -> bool:
        try:
            curve = read_cv(path, scan_rate=scan_rate, electrode=self.electrode,
                            potential_scale=self.potential_scale,
                            current_scale=self.current_scale)
        except (OSError, ValueError, EchemIOError) as exc:
            self.log("error", f"{Path(path).name}: {exc}")
            return False
        if into_series:
            self.rate_series.append(curve)
            self.rate_series.sort(key=lambda c: c.scan_rate)
        else:
            self.cv = curve
        return True

    def load_gcd(
        self, path: str | Path, current: Optional[float] = None
    ) -> bool:
        try:
            self.gcd = read_gcd(path, electrode=self.electrode, current=current,
                                potential_scale=self.potential_scale,
                                current_scale=self.current_scale)
        except (OSError, ValueError, EchemIOError) as exc:
            self.log("error", f"{Path(path).name}: {exc}")
            return False
        return True

    def add_gcd_to_series(
        self, path: str | Path, current: Optional[float] = None
    ) -> bool:
        """Another charge-discharge curve, at a different current."""
        try:
            curve = read_gcd(path, electrode=self.electrode, current=current,
                             potential_scale=self.potential_scale,
                             current_scale=self.current_scale)
        except (OSError, ValueError, EchemIOError) as exc:
            self.log("error", f"{Path(path).name}: {exc}")
            return False
        self.gcd_series.append(curve)
        self.gcd_series.sort(key=_typical_current)
        return True

    def load_eis(self, path: str | Path) -> bool:
        try:
            self.eis = read_eis(path, electrode=self.electrode)
        except (OSError, ValueError, EchemIOError) as exc:
            self.log("error", f"{Path(path).name}: {exc}")
            return False
        return True

    def add_eis_to_series(self, path: str | Path,
                          label: str = "") -> bool:
        """Load one more spectrum into the comparison series.

        ``label`` is what the legend says — the sample, the temperature,
        the potential. Taken from the file name when empty, which is
        almost never what anyone wants on a figure, so it is asked for.
        """
        try:
            spectrum = read_eis(path, electrode=self.electrode)
        except (OSError, ValueError, EchemIOError) as exc:
            self.log("error", f"{Path(path).name}: {exc}")
            return False
        spectrum.name = label.strip() or Path(path).stem
        if any(s.name == spectrum.name for s in self.eis_series):
            self.log("warning",
                     f"ya hay un espectro llamado «{spectrum.name}» en la "
                     "serie; dos curvas con el mismo nombre no se "
                     "distinguen en la leyenda")
        self.eis_series.append(spectrum)
        self._eis_analyses = None
        if self.eis is None:
            # So the circuit editor, the DRT and C(ω) have something to
            # work on without a second load.
            self.eis = spectrum
        self.result = None
        return True

    def add_to_mott_schottky(self, path: str | Path) -> bool:
        """Load one more spectrum into the Mott-Schottky series.

        No label is asked for: these are the SAME electrode at different
        potentials, and the potential is in the file. The EC-Lab reader
        puts it in the metadata from the file's own ``<Ewe>/V`` column,
        so a folder of PEIS files measured along a sweep already is the
        experiment.
        """
        try:
            spectrum = read_eis(path, electrode=self.electrode)
        except (OSError, ValueError, EchemIOError) as exc:
            self.log("error", f"{Path(path).name}: {exc}")
            return False
        bias = spectrum.metadata.get("potencial_cd_v")
        if bias is None:
            self.log("error",
                     f"{Path(path).name} no declara el potencial de "
                     "polarización, así que no puede ser un punto de "
                     "Mott-Schottky")
            return False
        spectrum.name = f"{bias:+.3f} V"
        self.ms_series.append(spectrum)
        self.ms_series.sort(
            key=lambda s: s.metadata.get("potencial_cd_v", 0.0))
        self._ms_result = None
        return True

    def mott_schottky_frequencies(self) -> list[float]:
        """The chosen frequencies that every loaded spectrum covers.

        One that only some spectra reach would build the plot out of a
        different set of potentials at each frequency, and then the
        spread between frequencies — which is what the method is
        checked by — would partly be a spread between experiments.
        """
        if not self.ms_series:
            return []
        low = max(float(s.frequency.min()) for s in self.ms_series)
        high = min(float(s.frequency.max()) for s in self.ms_series)
        return [f for f in self.ms_frequencies if low <= f <= high]

    def mott_schottky(self):
        """The Mott-Schottky analysis of the loaded series, cached."""
        from ..echem.mott_schottky import analyse_mott_schottky

        if self._ms_result is not None:
            return self._ms_result
        if len(self.ms_series) < 2:
            self.log("error",
                     "hacen falta espectros a varios potenciales: un "
                     "Mott-Schottky es la capacitancia CONTRA el potencial")
            return None
        frequencies = self.mott_schottky_frequencies()
        if not frequencies:
            self.log("error",
                     "ninguna de las frecuencias elegidas cae dentro del "
                     "barrido de todos los espectros")
            return None
        result = analyse_mott_schottky(
            self.ms_series, frequencies,
            epsilon=self.ms_epsilon, material=self.ms_material,
            area_cm2=self.electrode.area_cm2,
            temperature_k=self.ms_temperature_k)
        for warning in result.warnings:
            self.log("warning", warning)
        self._ms_result = result
        return result

    def mott_schottky_rows(self) -> list[tuple[str, ...]]:
        """One row per frequency, plus the header."""
        rows: list[tuple[str, ...]] = [MOTT_SCHOTTKY_COLUMNS]
        result = self.mott_schottky()
        if result is None or not result.available:
            return rows
        for fit in result.fits:
            rows.append((
                f"{fit.frequency_hz:.4g}",
                fit.carrier_type,
                f"{fit.flat_band_v:+.4f}",
                f"{fit.flat_band_error_v:.4f}",
                f"{fit.carrier_density_cm3:.3g}",
                f"{fit.carrier_density_error_cm3:.3g}",
                f"{fit.slope:+.4g}",
                f"{fit.r_squared:.4f}",
                f"{fit.window[0]:+.3f} … {fit.window[1]:+.3f}",
                str(fit.points),
            ))
        return rows

    def mott_schottky_materials(self) -> list[str]:
        from ..echem.mott_schottky import semiconductors

        return [""] + sorted(semiconductors())

    def mott_schottky_report(self) -> str:
        result = self.mott_schottky()
        if result is None:
            return ""
        return result.summary()

    def eis_analyses(self) -> list:
        """One :class:`EISResult` per loaded spectrum, cached.

        Each is analysed with ``circuit="auto"``, so each gets the
        circuit its own shape asks for. Comparing several samples through
        one forced circuit hides exactly the differences worth having —
        on the user's three carbons, only the 900 °C spectrum needs the
        wiring inductance, and forcing the other two to carry an L leaves
        it undetermined in both.
        """
        if self._eis_analyses is not None:
            return self._eis_analyses
        spectra = (list(self.eis_series) if self.eis_series
                   else ([self.eis] if self.eis is not None else []))
        out = []
        for spectrum in spectra:
            try:
                out.append(analyse_eis(spectrum, circuit=AUTO_CIRCUIT))
            except (CircuitError, ValueError) as exc:
                self.log("error", f"{spectrum.name}: {exc}")
                out.append(None)
        self._eis_analyses = out
        return out

    def eis_series_choices(self) -> list[str]:
        return [s.name for s in self.eis_series]

    def use_eis_from_series(self, name: str) -> bool:
        """Make one of the series the spectrum the circuit editor acts on."""
        for spectrum in self.eis_series:
            if spectrum.name == name:
                self.eis = spectrum
                self.result = None
                return True
        self.log("error", f"no hay ningún espectro llamado «{name}»")
        return False

    def eis_comparison_rows(self) -> list[tuple[str, ...]]:
        """One row per spectrum in the series, analysed the same way.

        The header row comes first so the table can be written straight
        to a file or into a tree widget. Every spectrum is analysed on
        its own with ``circuit="auto"``, so each gets the circuit its own
        shape asks for, and the column says which — comparing three
        samples through one forced circuit hides exactly the differences
        worth having.
        """
        rows: list[tuple[str, ...]] = [EIS_COMPARISON_COLUMNS]
        for spectrum in self.eis_series:
            try:
                analysis = analyse_eis(spectrum, circuit=AUTO_CIRCUIT)
            except (CircuitError, ValueError) as exc:
                rows.append((spectrum.name, f"falló: {exc}", "", "", "", "",
                             "", "", ""))
                continue
            shape = analysis.shape
            capacitance = analysis.capacitance_f
            rows.append((
                spectrum.name,
                f"{analysis.series_resistance:.4g}"
                if analysis.series_resistance is not None else "—",
                f"{analysis.ionic_resistance_ohm:.4g}"
                if analysis.ionic_resistance_ohm is not None else "—",
                f"{analysis.knee_frequency:.4g}"
                if analysis.knee_frequency is not None else "—",
                f"{shape.low_phase_deg:.1f}" if shape else "—",
                f"{1e3 * capacitance:.4g}" if capacitance else "—",
                f"{1e3 * analysis.storage_capacitance.farads:.4g}"
                if analysis.storage_capacitance is not None else "—",
                analysis.fit.circuit if analysis.fit else "—",
                f"{analysis.fit.residual_pct:.3f}" if analysis.fit else "—",
            ))
        return rows

    def load_eis_pair(self, nyquist: str | Path, bode: str | Path) -> bool:
        """One spectrum from EC-Lab's two separate exports.

        A Nyquist export has no frequency column and a Bode export has no
        magnitude, so neither opens on its own — which is correct, and was
        also a dead end for anyone who had exported exactly those two
        files. Together they are a complete spectrum, and the pairing is
        checked against the phase both of them describe.
        """
        from ..echem.io import read_eis_pair

        try:
            self.eis = read_eis_pair(nyquist, bode, electrode=self.electrode)
        except (OSError, ValueError, EchemIOError) as exc:
            self.log("error", str(exc))
            return False
        agreement = self.eis.metadata.get("acuerdo_de_fase_grados")
        if agreement is not None:
            self.log("info",
                     f"Nyquist y Bode emparejados: la fase coincide dentro "
                     f"de {agreement:.3f}°")
        return True

    def load_lsv(self, path: str | Path, scan_rate: float = 0.005) -> bool:
        try:
            self.lsv = read_cv(path, scan_rate=scan_rate,
                               electrode=self.electrode,
                               potential_scale=self.potential_scale,
                               current_scale=self.current_scale)
        except (OSError, ValueError, EchemIOError) as exc:
            self.log("error", f"{Path(path).name}: {exc}")
            return False
        return True

    def add_curves(self, **curves) -> None:
        """Attach already-built curves, as the demo loader does."""
        for key, value in curves.items():
            if value is None:
                continue
            if key in ("rate_series", "gcd_series"):
                setattr(self, key, list(value))
            else:
                setattr(self, key, value)
        self.retag_electrode()

    def retag_electrode(self) -> None:
        """Push the current electrode and cycle choice onto every curve.

        Editing the mass after loading is the normal order of events, and
        without this the curves keep the electrode they were read with and
        the specific capacitances silently stay wrong. The same goes for
        the cycle: it is read by :meth:`Voltammogram.reporting_cycle`,
        which every capacitance goes through.
        """
        for curve in self.all_curves():
            curve.electrode = self.electrode
            if hasattr(curve, "preferred_cycle"):
                curve.preferred_cycle = self.cycle_choice

    def cycle_choices(self) -> list[tuple[Optional[int], str]]:
        """``(número, etiqueta)`` for the cycle selector.

        Built from the voltammogram that has the most cycles, so a rate
        series whose files differ in length still offers every number
        that exists somewhere. Each label says whether that cycle is
        complete, because an incomplete one integrated over part of its
        own path gives a capacitance low by that fraction and nothing
        downstream says so.
        """
        curves = [c for c in ([self.cv] + list(self.rate_series)) if c is not None]
        if not curves:
            return [(None, "último completo")]
        reference = max(curves, key=lambda c: len(c.cycles()))
        every = reference.cycles()
        complete = {cycle.name for cycle in reference.complete_cycles()}
        out: list[tuple[Optional[int], str]] = [
            (None, f"último completo (de {len(every)})")
        ]
        for index, cycle in enumerate(every, start=1):
            mark = "" if cycle.name in complete else "  — INCOMPLETO"
            out.append((index, f"ciclo {index}{mark}"))
        return out

    def reporting_cycle_note(self) -> str:
        """What the analysis will actually integrate, in one line."""
        curve = self.cv or (self.rate_series[0] if self.rate_series else None)
        if curve is None:
            return ""
        try:
            chosen, why = curve.reporting_cycle()
        except CurveError as exc:
            return str(exc)
        return why or f"{curve.name}: un solo ciclo, {chosen.n} puntos"

    def all_curves(self) -> list:
        out = [c for c in (self.cv, self.gcd, self.eis, self.lsv) if c is not None]
        out.extend(self.rate_series)
        out.extend(self.gcd_series)
        out.extend(s for s in self.eis_series
                   if not any(s is c for c in out))
        out.extend(s for s in self.ms_series
                   if not any(s is c for c in out))
        return out

    def clear(self, kind: str) -> None:
        if kind == "rates":
            self.rate_series = []
        elif kind == "gcd_series":
            self.gcd_series = []
        elif kind == "ms_series":
            self.ms_series = []
            self._ms_result = None
        elif kind == "eis_series":
            # Identity, not equality: these are dataclasses with arrays
            # inside, so `in` calls the generated __eq__ and then fails
            # evaluating an array's truth value.
            if any(self.eis is s for s in self.eis_series):
                self.eis = None
            self.eis_series = []
            self._eis_analyses = None
        elif kind in ("cv", "gcd", "eis", "lsv"):
            setattr(self, kind, None)
        self.result = None

    @property
    def loaded(self) -> dict[str, str]:
        """Kind → a one-line description of what is loaded, for the sidebar."""
        out: dict[str, str] = {}
        if self.cv is not None:
            out["cv"] = self.cv.describe()
        if self.rate_series:
            rates = ", ".join(f"{1e3 * c.scan_rate:g}" for c in self.rate_series)
            out["rates"] = f"{len(self.rate_series)} curvas: {rates} mV/s"
        if self.gcd is not None:
            out["gcd"] = self.gcd.describe()
        if self.gcd_series:
            currents = ", ".join(f"{1e3 * _typical_current(c):g}"
                                 for c in self.gcd_series)
            out["gcd_series"] = f"{len(self.gcd_series)} curvas: {currents} mA"
        if self.eis is not None:
            out["eis"] = self.eis.describe()
        if self.eis_series:
            names = ", ".join(s.name for s in self.eis_series)
            out["eis_series"] = f"{len(self.eis_series)} espectros: {names}"
        if self.ms_series:
            biases = [s.metadata.get("potencial_cd_v", 0.0)
                      for s in self.ms_series]
            out["ms_series"] = (
                f"{len(self.ms_series)} potenciales: "
                f"{min(biases):+.3f} … {max(biases):+.3f} V")
        if self.lsv is not None:
            out["lsv"] = f"{self.reaction}: {self.lsv.describe()}"
        return out

    @property
    def has_data(self) -> bool:
        return bool(self.all_curves())

    # -- analysis ------------------------------------------------------
    def analyse(self) -> Optional[EchemResult]:
        if not self.has_data:
            self.log("error", "no hay ninguna medida cargada")
            return None
        self.retag_electrode()
        self.result = analyse_sample(
            name=self.name,
            cv=self.cv,
            rate_series=self.rate_series or None,
            gcd=self.gcd,
            eis=self.eis,
            catalysis_curve=self.lsv,
            reaction=self.reaction,
            circuit=self.circuit,
            circuit_initial=dict(self.circuit_initial) or None,
            circuit_fixed=sorted(self.circuit_fixed) or None,
            drt_regularisation=self.drt_regularisation,
            non_faradaic=self.non_faradaic,
            dunn_sweep=self.dunn_sweep,
            dunn_nonnegative=self.dunn_nonnegative,
            dunn_points=self.dunn_points,
            dunn_window=self.dunn_window,
            capacitance_window=self.capacitance_window,
            subtract_parasitic=self.subtract_parasitic,
        )
        for warning in self.result.warnings:
            self.log("warning", warning)
        return self.result

    # -- tables --------------------------------------------------------
    def summary_rows(self) -> list[tuple[str, str, str]]:
        """``(quantity, value, note)`` for the headline table."""
        result = self.result
        if result is None:
            return []
        rows: list[tuple[str, str, str]] = []

        if result.storage:
            rows.append(
                ("Mecanismo", result.storage.label,
                 f"confianza {result.storage.confidence}")
            )
            rows.append(("Magnitud correcta", result.storage.report_as, ""))
        # The side-by-side comparison supersedes the single-method rows when
        # there is more than one method: the same number twice in one table
        # reads as two measurements.
        compared = bool(result.capacitance and len(result.capacitance.entries) > 1)
        if result.cv and result.cv.capacitance_loop and not compared:
            entry = result.cv.capacitance_loop
            rows.append(
                ("C (CV, lazo)", f"{1e3 * entry.farads:.4g} mF",
                 f"{entry.specific_f_per_g:.4g} F/g"
                 if entry.specific_f_per_g is not None else "sin masa")
            )
        if compared:
            for entry in result.capacitance.entries:
                rows.append(
                    (f"C ({entry.method})", f"{1e3 * entry.farads:.4g} mF",
                     f"{entry.condition}"
                     + (f" — {entry.per_gram:.4g} F/g"
                        if entry.per_gram is not None else ""))
                )
            if result.capacitance.spread is not None:
                rows.append(
                    ("Dispersión entre métodos",
                     f"{100 * result.capacitance.spread:.0f} %",
                     "lo que entrega el dispositivo es la de GCD")
                )
        if result.dunn and result.dunn.fractions:
            fastest = max(result.dunn.fractions)
            rows.append(
                ("Capacitivo (Dunn)",
                 f"{100 * result.dunn.fractions[fastest]:.0f} %",
                 f"a {1e3 * fastest:g} mV/s")
            )
        if result.gcd:
            if result.gcd.capacitance_f is not None and not compared:
                rows.append(
                    ("C (GCD, descarga)", f"{1e3 * result.gcd.capacitance_f:.4g} mF",
                     f"{result.gcd.specific_f_per_g:.4g} F/g"
                     if result.gcd.specific_f_per_g is not None else "sin masa")
                )
            if result.gcd.capacity_mah_per_g is not None:
                rows.append(
                    ("Capacidad", f"{result.gcd.capacity_mah_per_g:.4g} mAh/g",
                     f"{result.gcd.capacity_c_per_g:.4g} C/g")
                )
            if result.gcd.coulombic_efficiency is not None:
                rows.append(
                    ("Eficiencia culómbica",
                     f"{100 * result.gcd.coulombic_efficiency:.2f} %", "")
                )
            if result.gcd.energy_wh_per_kg is not None:
                rows.append(
                    ("Energía / potencia",
                     f"{result.gcd.energy_wh_per_kg:.4g} Wh/kg",
                     f"{result.gcd.power_w_per_kg:.4g} W/kg (un solo electrodo)")
                )
            if result.gcd.resistance_ohm is not None:
                rows.append(
                    ("R interna (caída IR)", f"{result.gcd.resistance_ohm:.4g} Ω", "")
                )
        if result.rates:
            if result.rates.b_values:
                low = min(v.b for v in result.rates.b_values)
                rows.append(("b mínima", f"{low:.3f}", "en el pico"))
            if result.rates.ecsa_cm2 is not None:
                low, high = result.rates.ecsa_range_cm2
                rows.append(
                    ("ECSA", f"{result.rates.ecsa_cm2:.4g} cm²",
                     f"entre {low:.3g} y {high:.3g} según la Cs")
                )
        if result.eis:
            if result.eis.series_resistance is not None:
                rows.append(("R serie", f"{result.eis.series_resistance:.4g} Ω", ""))
            if result.eis.charge_transfer_resistance is not None:
                rows.append(
                    ("R transferencia de carga",
                     f"{result.eis.charge_transfer_resistance:.4g} Ω", "")
                )
            if result.eis.kk:
                rows.append(
                    ("Kramers-Kronig",
                     "consistente" if result.eis.kk.passes else "NO CONSISTENTE",
                     f"residuo RMS {100 * result.eis.kk.rms_residual:.2f} %")
                )
        if result.catalysis:
            catalysis = result.catalysis
            if catalysis.overpotential_at_benchmark is not None:
                rows.append(
                    (f"η a {catalysis.benchmark_ma_cm2:g} mA/cm²",
                     f"{1e3 * catalysis.overpotential_at_benchmark:.0f} mV", "")
                )
            if catalysis.tafel and catalysis.tafel.valid:
                rows.append(
                    ("Pendiente de Tafel",
                     f"{catalysis.tafel.slope_mv_per_decade:.1f} mV/dec",
                     f"{catalysis.tafel.decades:.2f} décadas")
                )
        return rows

    def circuit_rows(self) -> list[tuple[str, str, str, str]]:
        """``(label, value, uncertainty, fixed?)`` for the fitted circuit.

        The fixed column is not decoration. Holding a parameter removes a
        degree of freedom, so every other uncertainty in the table comes
        out smaller; a reader who cannot see which were held cannot read
        the column next to it.
        """
        result = self.result
        if result is None or result.eis is None or result.eis.fit is None:
            return []
        fit = result.eis.fit
        rows = []
        for label, value in fit.values().items():
            error = fit.errors.get(label)
            held = label in fit.fixed
            rows.append(
                (
                    label,
                    f"{value:.6g}",
                    "fijado" if held
                    else f"± {100 * error / abs(value):.1f} %"
                    if error is not None and value
                    else "—",
                    "sí" if held else "",
                )
            )
        return rows

    def drt_rows(self) -> list[tuple[str, str, str, str]]:
        """``(τ, R, C = τ/R, share of the total)`` per resolved process.

        The capacitance is the useful column: two processes with the same
        resistance and time constants a decade apart are different things,
        and τ/R is what says which.
        """
        result = self.result
        if result is None or result.drt is None:
            return []
        total = sum(result.drt.peak_resistances) or 1.0
        rows = []
        for tau, resistance in zip(result.drt.peaks_s,
                                   result.drt.peak_resistances):
            rows.append((
                f"{tau:.4g}",
                f"{resistance:.4g}",
                f"{1e3 * tau / resistance:.4g}" if resistance > 0 else "—",
                f"{100 * resistance / total:.0f} %",
            ))
        return rows

    def gcd_capacitance_rows(self) -> list[tuple[str, str, str]]:
        """``(convenio, mF, F/g)`` for the chosen discharge.

        Three, not one. They agree to about a per cent on a straight
        discharge and by tens of per cent on a plateau, and the
        disagreement is the diagnostic: it measures how far the curve is
        from the straight line the ΔV convention assumes.
        """
        result = self.result
        if result is None or result.gcd is None or not result.gcd.discharges:
            return []
        branch = result.gcd.discharges[-1]
        per_gram = result.gcd.specific_by_method
        rows = []
        for name, value in branch.capacitances().items():
            gram = per_gram.get(name)
            rows.append((
                name,
                f"{1e3 * value:.4g}" if value else "—",
                f"{gram:.4g}" if gram else "—",
            ))
        spread = branch.capacitance_spread
        if spread is not None:
            rows.append(("dispersión entre convenios",
                         f"{100 * spread:.0f} %",
                         f"R² recta = {branch.linearity:.3f}"))
        return rows

    def capacitance_vs_current(self) -> list[tuple[float, float, str]]:
        """``(current density or mA, specific capacitance, method)``.

        One point per charge–discharge curve, for the rate-capability
        plot. Built from the ENERGY convention, because that is the one
        that stays honest when the discharge bends, and a rate plot whose
        points are each over-reported by a different amount is worse than
        no plot.
        """
        from ..echem.gcd import analyse_gcd

        curves = list(self.gcd_series)
        if self.gcd is not None and not any(c is self.gcd for c in curves):
            curves.insert(0, self.gcd)
        points: list[tuple[float, float, str]] = []
        for curve in curves:
            try:
                analysis = analyse_gcd(curve)
            except (ValueError, ZeroDivisionError):
                continue
            if not analysis.discharges:
                continue
            branch = analysis.discharges[-1]
            value = (analysis.specific_by_method.get("energía (2E/ΔV²)")
                     or analysis.specific_by_method.get("ΔV (I·Δt/ΔV)"))
            if not value:
                continue
            points.append((1e3 * _typical_current(curve), float(value),
                           f"{1e3 * abs(branch.current_a):g} mA"))
        points.sort(key=lambda item: item[0])
        return points

    def ragone_points(self) -> list[tuple[float, float, str]]:
        """``(Wh/kg, W/kg, label)``, one per charge-discharge curve.

        Both per KILOGRAM, and the thousand is the point: ``specific``
        normalises by the mass in grams, so it returns J/g. A Ragone plot
        whose power axis is three decades out still looks like a Ragone
        plot.
        """
        from ..echem.gcd import analyse_gcd

        points: list[tuple[float, float, str]] = []
        curves = list(self.gcd_series)
        # Identity, not equality: these dataclasses hold arrays, so `in`
        # calls the generated __eq__, which compares arrays element-wise
        # and raises on the truth value of the result.
        if self.gcd is not None and not any(c is self.gcd for c in curves):
            curves.insert(0, self.gcd)
        for curve in curves:
            try:
                analysis = analyse_gcd(curve)
            except (ValueError, ZeroDivisionError) as exc:
                self.log("warning", f"Ragone: {exc}")
                continue
            energy = analysis.energy_wh_per_kg
            power = analysis.power_w_per_kg
            if not energy or not power:
                continue
            points.append((energy, power,
                           f"{1e3 * _typical_current(curve):g} mA"))
        return points

    def capacitance_rows(self) -> list[tuple[str, str, str, str]]:
        """``(method, condition, mF, F/g)``.

        The three are not the same measurement and the table says so by
        carrying the condition: what a device delivers is the GCD value,
        while the EIS one is measured with 10 mV about a fixed point
        where nothing is rate-limited and is an upper bound the device
        never sees.
        """
        result = self.result
        if result is None or result.capacitance is None:
            return []
        return [
            (entry.method, entry.condition, f"{1e3 * entry.farads:.4g}",
             f"{entry.per_gram:.4g}" if entry.per_gram is not None else "—")
            for entry in result.capacitance.entries
        ]

    # -- the circuit ---------------------------------------------------
    def dunn_rate_choices(self) -> list[float]:
        """The scan rates the separation can be drawn at, slowest first."""
        result = self.result
        if result is None or result.dunn is None:
            return sorted({c.scan_rate for c in self.rate_series})
        return sorted(result.dunn.rates)

    def dunn_rate_for_plot(self) -> Optional[float]:
        """The rate the figure should use: the user's, or the slowest."""
        rates = self.dunn_rate_choices()
        if not rates:
            return None
        if self.dunn_rate is not None:
            nearest = min(rates, key=lambda r: abs(r - self.dunn_rate))
            if abs(nearest - self.dunn_rate) < 1e-9:
                return nearest
        return rates[0]

    def circuit_choices(self) -> list[str]:
        """``"auto"`` first, then the catalogue in the order to try it in."""
        return [AUTO_CIRCUIT] + [t.name for t in circuit_templates()]

    def circuit_catalogue(self) -> list[CircuitTemplate]:
        """Every circuit on offer, bundled and user, in the order to try
        them in."""
        return circuit_templates()

    def circuit_text(self) -> str:
        """The circuit string behind the current choice, for the entry
        box: a preset the user can then edit is more useful than one they
        can only accept."""
        return circuit_for(self.circuit)

    def circuit_note(self) -> tuple[str, str]:
        """``(what it is for, what it gets wrong)`` for the current
        choice, or empty strings for a hand-typed one."""
        if self.circuit == AUTO_CIRCUIT:
            return (
                "Lee la forma del espectro —R en serie, inductancia "
                "arriba, arco cerrado o no, codo, décadas a 45°, cola "
                "capacitiva— propone los circuitos que esa forma pide y "
                "los compara por BIC y por parámetros determinados.",
                "Es un cribado, no un veredicto: ninguna regla "
                "estructural puede decir si el circuito es el apropiado "
                "para el MATERIAL. Mira la tabla de comparación y la "
                "lista de candidatos antes de citar nada.",
            )
        for template in circuit_templates():
            if template.name == self.circuit:
                return template.use, template.caution
        return "", ""

    def set_circuit(self, text: str) -> bool:
        """Choose a circuit by name or by string, checking it parses.

        Checked here rather than at fit time so a typo is a message next
        to the box that produced it, not a traceback ten seconds into an
        analysis. Changing the circuit clears the per-parameter overrides,
        because ``R1.R`` in one circuit is not ``R1.R`` in another.
        """
        candidate = text.strip()
        if not candidate:
            self.log("error", "el circuito no puede estar vacío")
            return False
        if candidate == AUTO_CIRCUIT:
            # Not a circuit, so there is nothing to parse and no
            # per-parameter override that could survive: which circuit
            # gets fitted is not known until the spectrum is read.
            self.circuit_initial.clear()
            self.circuit_fixed.clear()
            self.circuit = candidate
            return True
        try:
            parse_circuit(circuit_for(candidate))
        except CircuitError as exc:
            self.log("error", f"circuito no válido: {exc}")
            return False
        if candidate != self.circuit:
            self.circuit_initial.clear()
            self.circuit_fixed.clear()
        self.circuit = candidate
        return True

    def circuit_parameters(self) -> list[tuple[str, float, bool]]:
        """``(label, starting value, fixed)`` for every parameter of the
        current circuit, before any fit.

        Read from the circuit itself rather than from the last result, so
        the table is there to set up the fit rather than only to look at
        afterwards.
        """
        text = circuit_for(self.circuit)
        if text == AUTO_CIRCUIT:
            # Before the fit there is no circuit to list. After it there
            # is: the one the comparison chose, with the values it found,
            # which is what the user needs in order to take it over by
            # hand.
            fit = (self.result.eis.fit
                   if self.result is not None and self.result.eis else None)
            if fit is None:
                return []
            return [(label, value, label in self.circuit_fixed)
                    for label, value in fit.values().items()]
        try:
            elements = parse_circuit(text).elements()
        except CircuitError:
            return []
        rows = []
        for element in elements:
            for label, value in zip(element.labels, element.values):
                rows.append((label, self.circuit_initial.get(label, value),
                             label in self.circuit_fixed))
        return rows

    def set_circuit_parameter(self, label: str,
                              value: Optional[float] = None,
                              fixed: Optional[bool] = None) -> None:
        """Override a starting value, hold a parameter, or both."""
        if value is not None:
            self.circuit_initial[label] = float(value)
        if fixed is not None:
            if fixed:
                self.circuit_fixed.add(label)
                # Fixing a parameter at whatever the library default
                # happens to be is a way to get a confident wrong answer.
                # If no value was given, take the one on the table.
                self.circuit_initial.setdefault(
                    label,
                    next((v for lab, v, _ in self.circuit_parameters()
                          if lab == label), 0.0),
                )
            else:
                self.circuit_fixed.discard(label)

    def reset_circuit_parameters(self) -> None:
        """Back to values read off the spectrum, nothing held."""
        self.circuit_initial.clear()
        self.circuit_fixed.clear()

    def refit_circuit(self) -> bool:
        """Fit the CURRENT circuit against the loaded spectrum, and only that.

        The circuit editor could change the circuit and not act on it:
        applying a new one updated the tables and then nothing happened
        until the user went back to the sidebar and re-ran the whole
        analysis — CV, GCD, Dunn, catalysis and all — to see what the
        change did. That is seconds of work to answer a question that
        costs one fit, and it is the loop a circuit is chosen in: write
        one, look at the residual, write another.

        Only the impedance part of the result is replaced. Everything
        else in `self.result` stays as it was, because nothing else
        depends on the circuit.
        """
        if self.eis is None:
            self.log("error", "no hay ningún espectro de impedancia cargado")
            return False
        if self.result is None:
            return self.analyse() is not None
        try:
            self.result.eis = analyse_eis(
                self.eis,
                circuit=circuit_for(self.circuit),
                initial=dict(self.circuit_initial) or None,
                fixed=sorted(self.circuit_fixed) or None,
            )
        except (CircuitError, ValueError) as exc:
            self.log("error", f"el ajuste falló: {exc}")
            return False
        for warning in self.result.eis.warnings:
            self.log("warning", warning)
        return True

    def adopt_chosen_circuit(self) -> bool:
        """Replace ``"auto"`` with the circuit the comparison picked.

        The point of automatic choice is to get to a circuit, not to stay
        on "automatic": once there is one, the user wants to edit it, hold
        a parameter, see it drawn. This is the handover, and it is an
        explicit action rather than something that happens behind the
        analysis — ``"auto"`` means "re-decide from the data", and a
        session that silently stopped doing that would answer a different
        question the next time the spectrum changed.
        """
        if self.circuit != AUTO_CIRCUIT:
            return False
        fit = (self.result.eis.fit
               if self.result is not None and self.result.eis else None)
        if fit is None:
            self.log("error",
                     "todavía no hay ningún circuito elegido: analiza el "
                     "espectro primero")
            return False
        self.circuit = fit.circuit
        self.circuit_initial = dict(fit.values())
        self.circuit_fixed.clear()
        self.log("info", f"circuito adoptado: {fit.circuit}, con los "
                         "valores del ajuste como punto de partida")
        return True

    def circuit_drawing(self) -> str:
        """The current circuit as a schematic, with fitted values if any.

        Answers the question the string cannot: a circuit is written left
        to right and read as a shape, and the two stop agreeing as soon
        as there is nesting. The bundled adsorption-loop template shipped
        with its branch unparenthesised for exactly that reason.
        """
        values = None
        eis = getattr(self.result, "eis", None) if self.result else None
        if eis is not None and eis.fit is not None:
            values = eis.fit.values()
        text = circuit_for(self.circuit)
        if text == AUTO_CIRCUIT:
            # Draw what was CHOSEN. "auto" has no shape of its own, and
            # returning a parse error here reads as the drawing being
            # broken rather than as there being nothing to draw yet.
            if eis is None or eis.fit is None:
                return ("Elige automáticamente: analiza el espectro y aquí\n"
                        "se dibuja el circuito que la comparación eligió.")
            text = eis.fit.circuit
        try:
            return draw_circuit(text, values)
        except CircuitError as exc:
            return f"no se puede dibujar: {exc}"

    def fit_curve_rows(self) -> list[tuple[str, ...]]:
        """The fitted curve as columns, for export.

        A figure and a report answer "what happened"; neither lets the
        fit be redrawn on a journal's axes or beside somebody else's
        data. These are the columns that do, and the residual is in them
        because it is the part that says where the model fails — the
        same argument as the Rietveld difference curve.
        """
        eis = getattr(self.result, "eis", None) if self.result else None
        if eis is None or eis.fit is None or eis.fit.fitted is None:
            return []
        spectrum = eis.spectrum
        model = np.asarray(eis.fit.fitted)
        rows = []
        for index in range(spectrum.n):
            observed = spectrum.z[index]
            calculated = model[index]
            rows.append((
                f"{spectrum.frequency[index]:.6g}",
                f"{observed.real:.6g}",
                f"{observed.imag:.6g}",
                f"{calculated.real:.6g}",
                f"{calculated.imag:.6g}",
                f"{observed.real - calculated.real:.6g}",
                f"{observed.imag - calculated.imag:.6g}",
                f"{abs(observed):.6g}",
                f"{np.degrees(np.angle(observed)):.6g}",
            ))
        return rows

    #: Column names for :meth:`fit_curve_rows`. Z″ carries its PHYSICAL
    #: sign here, as everywhere in this package; the −Z″ of a Nyquist plot
    #: is a drawing convention and exporting it negated is how a sign
    #: error travels into somebody else's plotting script.
    FIT_CURVE_COLUMNS = (
        "f (Hz)", "Zr_obs (ohm)", "Zi_obs (ohm)",
        "Zr_calc (ohm)", "Zi_calc (ohm)",
        "res_r (ohm)", "res_i (ohm)", "|Z| (ohm)", "fase (grados)",
    )

    def save_fit_curve(self, path: str | Path) -> bool:
        """Write the fitted curve as a CSV next to the figure."""
        rows = self.fit_curve_rows()
        if not rows:
            self.log("error", "no hay ninguna curva ajustada que exportar")
            return False
        eis = self.result.eis  # type: ignore[union-attr]
        destination = Path(path)
        header = [
            f"# {self.name}",
            f"# circuito: {circuit_for(self.circuit)}",
        ]
        if eis.fit is not None:
            header.append(
                f"# chi2 = {eis.fit.chi_squared:.6g}  residuo medio "
                f"{eis.fit.residual_pct:.4f} %")
            header.append(
                "# " + "  ".join(
                    f"{label}={value:.6g}"
                    for label, value in eis.fit.values().items()))
        header.append("# Z'' va con su signo FÍSICO (negativo si capacitivo)")
        header.append(",".join(self.FIT_CURVE_COLUMNS))
        body = [",".join(row) for row in rows]
        destination.write_text("\n".join(header + body) + "\n",
                               encoding="utf-8")
        self.log("info", f"curva ajustada escrita en {destination}")
        return True

    def compare_circuits(self) -> str:
        """Fit a ladder of candidate circuits and return the comparison."""
        if self.eis is None:
            self.log("error", "no hay ningún espectro de impedancia cargado")
            return ""
        comparison = compare_circuits(self.eis)
        return comparison.describe()

    def effective_capacitance_report(self) -> str:
        """The CPE's capacitance by both relations, with the arithmetic.

        A ZARC needs none of it — its capacitance is ``τ/R`` by
        construction — and saying so is more useful than running a
        conversion that would return the same number by a longer route.
        """
        eis = getattr(self.result, "eis", None) if self.result else None
        if eis is None or eis.fit is None:
            return ""
        fit = eis.fit
        values = fit.values()
        blocks: list[str] = []

        for element in fit.elements:
            if element.symbol == "Za":
                r, tau, gamma = element.values
                blocks.append(
                    f"{element.name} (ZARC) — sin conversión\n"
                    + "─" * 46 + "\n"
                    f"  R     = {r:.6g} Ω\n"
                    f"  τ     = {tau:.6g} s\n"
                    f"  γ     = {gamma:.4f}\n\n"
                    f"  C = τ/R = {tau / r:.6g} F\n\n"
                    "  Exacta por construcción: un ZARC está escrito en "
                    "(R, τ, γ),\n  así que su capacitancia efectiva no "
                    "necesita ni Brug ni\n  Hsu-Mansfeld. Es la razón de "
                    "escribir el arco así."
                )

        resistances = [v for label, v in values.items() if label.endswith(".R")]
        series = resistances[0] if resistances else None
        charge_transfer = eis.charge_transfer_resistance
        converted = False
        for element in fit.elements:
            if element.symbol != "Q":
                continue
            converted = True
            t, p_exponent = element.values
            if charge_transfer:
                blocks.append(
                    hsu_mansfeld_capacitance(
                        t, p_exponent, charge_transfer).describe())
            if series:
                blocks.append(
                    brug_capacitance(t, p_exponent, series,
                                     charge_transfer).describe())
        if not blocks:
            return ""
        # Only when there is actually a conversion to compare. With a
        # circuit of ZARCs alone there is no Brug and no Hsu-Mansfeld,
        # and a note explaining the difference between two relations
        # that were not used reads as though one of them had been.
        if not converted:
            return "\n\n".join(blocks)
        blocks.append(
            "LAS DOS NO SON LA MISMA MEDIDA\n" + "─" * 46 + "\n"
            "  Hsu-Mansfeld supone una distribución de constantes de "
            "tiempo\n  NORMAL a la superficie (a través del espesor de "
            "una película)\n  y usa la R en paralelo con el CPE.\n\n"
            "  Brug supone una distribución A LO LARGO de la superficie\n"
            "  —rugosidad, corriente no uniforme— y usa la resistencia de "
            "la\n  disolución, que es la que acopla los parches entre "
            "sí.\n\n"
            "  Como R_s ∥ R_ct < R_ct, Brug devuelve SIEMPRE menos. La "
            "literatura\n  las aplica indistintamente; cuál te toca "
            "depende de dónde esté\n  la dispersión, y eso no sale del "
            "espectro."
        )
        return "\n\n".join(blocks)

    def save_circuit(self, name: str, circuit: Optional[str] = None) -> bool:
        """Store the current circuit under a name of the user's own."""
        try:
            save_user_circuit(name, circuit or circuit_for(self.circuit))
        except CircuitError as exc:
            self.log("error", str(exc))
            return False
        self.log("info", f"circuito guardado como «{name.strip()}»")
        return True

    def delete_circuit(self, name: str) -> bool:
        """Forget a user circuit. The bundled ones cannot be deleted."""
        if delete_user_circuit(name):
            self.log("info", f"circuito «{name}» borrado")
            if self.circuit == name:
                self.set_circuit("randles_cpe")
            return True
        self.log(
            "error",
            f"«{name}» no es un circuito tuyo. Los del programa no se "
            "borran: guárdate uno con el mismo nombre para sustituirlo",
        )
        return False

    def potential_axis(self, curve) -> tuple[np.ndarray, str, str]:
        """``(valores, etiqueta, explicación)`` on the chosen scale."""
        return curve.electrode.on_scale(curve.potential,
                                        self.potential_scale_choice)

    def table_rows(self, which: str) -> list[list]:
        """One of :attr:`TABLE_COLUMNS` as rows, or ``[]``.

        Everything the voltammetry tabs show, as numbers rather than as
        formatted strings: a table you can read on screen and cannot take
        to a plotting program is half a result.
        """
        from ..echem.cv import (capacitance, compare_current_laws,
                                coulombic_efficiency, elapsed_time,
                                integrate_charge, reversible_capacitance)

        curves = [c for c in ([self.cv] + list(self.rate_series))
                  if c is not None]
        if not curves:
            return []
        shift = 0.0
        if curves:
            values, _, _ = self.potential_axis(curves[0])
            shift = float(values[0] - curves[0].potential[0])

        rows: list[list] = []
        if which == "ciclos":
            for curve in curves:
                complete = {c.name for c in curve.complete_cycles()}
                for index, cycle in enumerate(curve.cycles(), start=1):
                    try:
                        time, _ = elapsed_time(cycle)
                        oxidation, absolute, span = integrate_charge(cycle)
                        loop, branch = capacitance(cycle)
                        reduction = reversible_capacitance(cycle)
                    except CurveError:
                        continue
                    rows.append([
                        curve.name, 1e3 * curve.scan_rate, index,
                        cycle.name in complete, cycle.n, float(time[-1]),
                        cycle.span, cycle.traversed, oxidation,
                        absolute - oxidation,
                        (absolute - oxidation) / oxidation if oxidation else None,
                        loop.farads, loop.specific_f_per_g, branch.farads,
                        reduction.farads, reduction.specific_f_per_g,
                        None if reduction.specific_f_per_cm2 is None
                        else 1e3 * reduction.specific_f_per_cm2,
                    ])
        elif which == "velocidades":
            for curve in sorted(curves, key=lambda c: c.scan_rate):
                try:
                    cycle, _ = curve.reporting_cycle()
                    time, _ = elapsed_time(cycle)
                    oxidation, absolute, _ = integrate_charge(cycle)
                    loop, _ = capacitance(curve)
                    reduction = reversible_capacitance(curve)
                except CurveError:
                    continue
                rows.append([
                    1e3 * curve.scan_rate, loop.farads, loop.specific_f_per_g,
                    reduction.farads, reduction.specific_f_per_g,
                    coulombic_efficiency(curve), float(time[-1]), oxidation,
                    absolute - oxidation, 2.0 * oxidation - absolute,
                ])
        elif which == "leyes":
            for law in compare_current_laws(self.rate_series or curves):
                rows.append([
                    law.potential, law.potential + shift, law.b,
                    law.r_squared.get("potencia"), law.r_squared.get("dunn"),
                    law.r_squared.get("fuga"), law.capacitance_f,
                    law.leakage_a, not law.decisive,
                ])
        elif which == "dunn":
            analysis = self.result.dunn if self.result else None
            if analysis is None:
                return []
            fastest = max(self.rate_series or curves, key=lambda c: c.scan_rate)
            try:
                cycle, _ = fastest.reporting_cycle()
                anodic, cathodic, exceeded = analysis.capacitive_inside(cycle)
            except CurveError:
                return []
            for index, potential in enumerate(analysis.potentials):
                rows.append([
                    float(potential), float(potential) + shift,
                    float(analysis.k1[index]), float(analysis.k2[index]),
                    float(analysis.r_squared[index]), float(anodic[index]),
                    float(cathodic[index]), bool(exceeded[index]),
                ])
        return rows

    def save_tables(self, folder: str | Path) -> list[str]:
        """Write every voltammetry table as CSV, plus the report.

        Returns the names written. One call rather than four buttons
        because the four tables are one result: a capacitance without its
        charge balance is the number this package spent a week showing is
        not a capacitance.
        """
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for which, columns in TABLE_COLUMNS.items():
            rows = self.table_rows(which)
            if not rows:
                continue
            path = target / f"{which}.csv"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(",".join(columns) + "\n")
                for row in rows:
                    handle.write(",".join(
                        "" if value is None else
                        (f"{value:.6g}" if isinstance(value, float) else str(value))
                        for value in row) + "\n")
            written.append(path.name)
        text = self.report()
        if text:
            (target / "informe.txt").write_text(text, encoding="utf-8")
            written.append("informe.txt")
        if not written:
            self.log("error", "no hay nada que exportar: carga una medida")
        else:
            self.log("info", f"escritos en {target}: " + ", ".join(written))
        return written

    # -- the multi-panel figure ----------------------------------------
    def figure_panels(self) -> list:
        """The panels whose data are loaded, as :class:`PanelSpec`."""
        return available_panels(self)

    def figure_presets(self) -> tuple[tuple[str, str], ...]:
        """``(key, label)`` for every layout preset on offer."""
        return PRESET_LABELS

    def use_figure_preset(self, name: str) -> bool:
        """Set width, height, columns, font size and dpi from a preset.

        The panels chosen are left alone: a preset is about where the
        figure is GOING — one journal column, two, a slide — and that is
        independent of which panels it holds.
        """
        if not apply_preset(self.composite, name):
            self.log("error", f"no hay ningún preajuste llamado «{name}»")
            return False
        self.log("info", f"preajuste «{name}»: "
                         f"{self.composite.width_in:g}″ de ancho, letra "
                         f"{self.composite.font_size:g}, "
                         f"{self.composite.dpi} ppp")
        return True

    def standard_figure(self) -> list[str]:
        """Choose the panels this project publishes, for what is loaded.

        One action instead of a column of checkboxes. The friction of
        picking panels by hand every time is the kind that stops a
        figure being made at all, and the defaults are the figures the
        field actually prints — not the first entries of the list.
        """
        self.composite.panels = default_panels(self)
        if not self.composite.panels:
            self.log("error",
                     "no hay nada cargado con lo que hacer una figura")
        return list(self.composite.panels)

    def set_figure_panels(self, keys) -> None:
        """Choose the panels, keeping the order they are offered in.

        The order is not the order they were ticked: a figure's letters
        run (a), (b), (c) down the page, and a user who ticks the
        residual before the Nyquist does not mean the residual to be
        panel (a)."""
        offered = [panel.key for panel in self.figure_panels()]
        chosen = set(keys)
        self.composite.panels = [key for key in offered if key in chosen]

    def draw_figure(self, figure, palette) -> list[str]:
        """Draw the composite onto a matplotlib figure, reporting
        whatever panel failed instead of raising."""
        problems = draw_composite(figure, self.composite, self, palette)
        for problem in problems:
            self.log("error", problem)
        return problems

    def save_figure(self, path, palette) -> bool:
        """Write the composite at ITS own width and resolution.

        Not a screenshot of the canvas: the canvas is whatever size the
        pane happens to be, and a figure drawn at one width and saved at
        another arrives with its labels at the wrong size — which is the
        single most common way a figure comes out unreadable.
        """
        try:
            target, problems = save_composite(
                self.composite, self, palette, path)
        except (OSError, ValueError) as exc:
            self.log("error", f"no se ha podido guardar la figura: {exc}")
            return False
        for problem in problems:
            self.log("error", problem)
        self.log("info", f"figura escrita en {target}")
        return True

    def figure_data_rows(self) -> dict[str, list[tuple]]:
        """The numbers each chosen panel is DRAWING, panel by panel.

        What goes out is what is drawn, with every transformation the
        figure applied already in it — the same rule the canvases follow.
        A figure that can only be looked at has to be retyped to be used
        anywhere else, and retyped numbers are wrong numbers.
        """
        import matplotlib
        from matplotlib.figure import Figure

        from .theme import PALETTES, matplotlib_style

        palette = PALETTES["claro"]
        out: dict[str, list[tuple]] = {}
        with matplotlib.rc_context(matplotlib_style(palette)):
            for key in self.composite.panels:
                # One panel per figure, deliberately. Zipping the chosen
                # panels against `figure.axes` of the composite looks
                # equivalent and is not: a panel with a twin axis adds a
                # second entry, so every panel after the first Bode plot
                # was reading another panel's lines and two of them came
                # back with nothing but their header.
                single = replace(self.composite, panels=[key], columns=1)
                figure = Figure(figsize=(single.width_in,
                                         single.panel_height_in))
                draw_composite(figure, single, self, palette)
                rows: list[tuple] = [("serie", "x", "y")]
                for axis in every_axes(figure):
                    for line in axis.get_lines():
                        name = line.get_label()
                        if name.startswith("_"):
                            continue
                        for x, y in zip(*line.get_data()):
                            rows.append((name, float(x), float(y)))
                out[key] = rows
        return out

    def save_figure_data(self, folder) -> list[str]:
        """Write one CSV per panel of the composite figure.

        One file per panel rather than one wide table: the panels do not
        share an x axis — frequency, potential, scan rate and time
        constant all appear in the same figure — and columns side by side
        under different abscissae is a table nobody can read back.
        """
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for key, rows in self.figure_data_rows().items():
            if len(rows) <= 1:
                continue
            path = target / f"figura_{key}.csv"
            with open(path, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(",".join(
                        f"{value:.6g}" if isinstance(value, float)
                        else str(value) for value in row) + "\n")
            written.append(path.name)
        if not written:
            self.log("error",
                     "la figura no tiene ningún panel con datos que exportar")
        else:
            self.log("info", f"escritos en {target}: " + ", ".join(written))
        return written

    def report(self) -> str:
        if self.result is None:
            return (
                "Carga al menos una medida (CV, GCD, EIS o curva de "
                "polarización) y pulsa Analizar."
            )
        return self.result.report()


__all__ = ["CURRENT_UNITS", "MEASUREMENT_KINDS", "POTENTIAL_UNITS",
           "EchemSession"]
