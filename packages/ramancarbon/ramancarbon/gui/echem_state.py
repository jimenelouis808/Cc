"""Electrochemistry session state, with no Tkinter in it.

One session holds the several measurements of a single electrode — a
voltammogram, a series of them at different scan rates, a charge–discharge
curve, an impedance spectrum, a polarisation curve — because that is the
unit the analysis works on: the mechanism classification is stronger the
more of them there are, and the cross-checks between them are the most
useful thing the section produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..echem.curve import ChargeDischarge, Electrode, Impedance, Voltammogram
from ..echem.eis import (
    CircuitError,
    CircuitTemplate,
    circuit_for,
    circuit_templates,
    delete_user_circuit,
    parse_circuit,
    save_user_circuit,
)
from ..echem.io import EchemIOError, read_cv, read_eis, read_gcd
from ..echem.report import EchemResult, analyse_sample

#: The kinds of measurement a session holds, and what each one is for.
MEASUREMENT_KINDS: tuple[tuple[str, str], ...] = (
    ("cv", "Voltamperometría cíclica"),
    ("rates", "Serie de velocidades"),
    ("gcd", "Carga-descarga"),
    ("eis", "Impedancia"),
    ("lsv", "Curva de polarización (HER/OER)"),
)


@dataclass
class EchemSession:
    """Everything the electrochemistry section knows."""

    name: str = "muestra"
    electrode: Electrode = field(default_factory=Electrode)
    cv: Optional[Voltammogram] = None
    rate_series: list[Voltammogram] = field(default_factory=list)
    gcd: Optional[ChargeDischarge] = None
    eis: Optional[Impedance] = None
    lsv: Optional[Voltammogram] = None
    reaction: str = "OER"
    circuit: str = "randles_cpe"
    """The circuit NAME, or a circuit string typed by hand. Both work:
    the name is resolved against the bundled and user libraries and
    anything unrecognised is parsed as a circuit."""
    circuit_initial: dict[str, float] = field(default_factory=dict)
    """Starting values by label, for the ones the user has overridden.
    Empty means "read them off the spectrum", which is what should
    normally happen."""
    circuit_fixed: set[str] = field(default_factory=set)
    """Labels held rather than refined."""
    non_faradaic: bool = False
    """Whether the rate study's window was chosen free of faradaic current.
    It is off by default because most windows are not, and claiming
    otherwise turns a C_dl that includes reaction current into an ECSA."""
    result: Optional[EchemResult] = None
    messages: list[tuple[str, str]] = field(default_factory=list)

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
            curve = read_cv(path, scan_rate=scan_rate, electrode=self.electrode)
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
            self.gcd = read_gcd(path, electrode=self.electrode, current=current)
        except (OSError, ValueError, EchemIOError) as exc:
            self.log("error", f"{Path(path).name}: {exc}")
            return False
        return True

    def load_eis(self, path: str | Path) -> bool:
        try:
            self.eis = read_eis(path, electrode=self.electrode)
        except (OSError, ValueError, EchemIOError) as exc:
            self.log("error", f"{Path(path).name}: {exc}")
            return False
        return True

    def load_lsv(self, path: str | Path, scan_rate: float = 0.005) -> bool:
        try:
            self.lsv = read_cv(path, scan_rate=scan_rate, electrode=self.electrode)
        except (OSError, ValueError, EchemIOError) as exc:
            self.log("error", f"{Path(path).name}: {exc}")
            return False
        return True

    def add_curves(self, **curves) -> None:
        """Attach already-built curves, as the demo loader does."""
        for key, value in curves.items():
            if value is None:
                continue
            if key == "rate_series":
                self.rate_series = list(value)
            else:
                setattr(self, key, value)
        self.retag_electrode()

    def retag_electrode(self) -> None:
        """Push the current electrode onto every loaded curve.

        Editing the mass after loading is the normal order of events, and
        without this the curves keep the electrode they were read with and
        the specific capacitances silently stay wrong.
        """
        for curve in self.all_curves():
            curve.electrode = self.electrode

    def all_curves(self) -> list:
        out = [c for c in (self.cv, self.gcd, self.eis, self.lsv) if c is not None]
        out.extend(self.rate_series)
        return out

    def clear(self, kind: str) -> None:
        if kind == "rates":
            self.rate_series = []
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
        if self.eis is not None:
            out["eis"] = self.eis.describe()
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
            non_faradaic=self.non_faradaic,
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

    # -- the circuit ---------------------------------------------------
    def circuit_choices(self) -> list[str]:
        return [t.name for t in circuit_templates()]

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
        try:
            elements = parse_circuit(circuit_for(self.circuit)).elements()
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

    def report(self) -> str:
        if self.result is None:
            return (
                "Carga al menos una medida (CV, GCD, EIS o curva de "
                "polarización) y pulsa Analizar."
            )
        return self.result.report()


__all__ = ["MEASUREMENT_KINDS", "EchemSession"]
