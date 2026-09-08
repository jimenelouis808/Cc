"""Measuring the suite, so that "optimised" is a number and not a claim.

Run it::

    python -m ramancarbon.benchmarks
    python -m ramancarbon.benchmarks --json tiempos.json

Every entry runs on synthetic data of a realistic size and reports the
**best** of several repeats rather than the mean. The best is the honest
figure for a benchmark on a shared machine: the mean measures whatever
else the machine was doing, and only the best is a property of the code.

The point of keeping this in the package rather than in a scratch script
is that a change that makes something four times slower should be
noticeable without anybody remembering to check.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

#: What each measurement is allowed to take before it is skipped in the
#: quick run. Not a limit on the operation, a limit on the benchmark.
QUICK_BUDGET_S = 1.5


@dataclass
class Measurement:
    """One timed operation."""

    name: str
    group: str
    seconds: float
    repeats: int
    detail: str = ""

    @property
    def milliseconds(self) -> float:
        return self.seconds * 1000.0

    def line(self) -> str:
        if self.seconds >= 1.0:
            timing = f"{self.seconds:8.2f} s "
        else:
            timing = f"{self.milliseconds:8.1f} ms"
        return f"  {self.name:<34s} {timing}   {self.detail}"


@dataclass
class Report:
    """Everything that was measured."""

    measurements: list[Measurement] = field(default_factory=list)
    machine: dict[str, str] = field(default_factory=dict)

    def add(self, measurement: Measurement) -> None:
        self.measurements.append(measurement)

    def to_dict(self) -> dict:
        return {
            "maquina": self.machine,
            "medidas": [
                {"nombre": m.name, "grupo": m.group, "segundos": m.seconds,
                 "repeticiones": m.repeats, "detalle": m.detail}
                for m in self.measurements
            ],
        }

    def text(self) -> str:
        lines = ["Tiempos de ramancarbon", "=" * 62]
        for key, value in self.machine.items():
            lines.append(f"{key}: {value}")
        current = None
        for measurement in self.measurements:
            if measurement.group != current:
                current = measurement.group
                lines += ["", current]
            lines.append(measurement.line())
        total = sum(m.seconds for m in self.measurements)
        lines += ["", f"Suma de los mejores tiempos: {total:.2f} s"]
        return "\n".join(lines)


def measure(function: Callable[[], object], repeats: int = 3) -> tuple[float, object]:
    """Best of ``repeats`` runs, and whatever the last one returned."""
    best = float("inf")
    result: object = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def _machine() -> dict[str, str]:
    import platform

    import numpy
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sistema": f"{platform.system()} {platform.machine()}",
    }


def run(quick: bool = False, repeats: int = 3) -> Report:
    """Time the operations a user actually waits for."""
    from .analysis.report import analyse
    from .core.baseline import arpls_baseline, asls_baseline, snip_baseline
    from .echem.eis import drt, fit_circuit
    from .echem.kinetics import differential_capacity
    from .examples.demo_data import (
        make_demo,
        make_eis_demo,
        make_map_demo,
        make_plateau_gcd_demo,
        make_xrd_demo,
    )
    from .mapping import despike_map, kmeans, mcr_als, pca
    from .models.deconvolution import build_model
    from .xrd.lebail import pawley
    from .xrd.reference import find_phase
    from .xrd.rietveld import PhaseModel, auto_refine
    from .xrd.search import find_peaks, identify_phases

    report = Report(machine=_machine())
    spectrum = make_demo("MWCNT")
    pattern = make_xrd_demo("CNT_FeSe")
    mos2_pattern = make_xrd_demo("MoS2_texturado")
    mos2 = find_phase("MoS2_2H")
    impedance = make_eis_demo()

    def add(name, group, function, count=repeats, detail=""):
        seconds, value = measure(function, count)
        report.add(Measurement(name, group, seconds, count,
                               detail(value) if callable(detail) else detail))
        return value

    # -- Raman ---------------------------------------------------------
    add("línea base asLS", "Raman (3200 puntos)",
        lambda: asls_baseline(spectrum.intensity), 5)
    add("línea base arPLS", "Raman (3200 puntos)",
        lambda: arpls_baseline(spectrum.intensity), 5)
    add("línea base SNIP", "Raman (3200 puntos)",
        lambda: snip_baseline(spectrum.intensity), 5)
    from .models.fitting import fit_model

    five_band = build_model(spectrum, preset="five_band")
    add("deconvolución D–G (5 bandas)", "Raman (3200 puntos)",
        lambda: fit_model(spectrum, five_band), repeats,
        detail=lambda r: f"R² = {r.r_squared:.5f}")
    add("analyse() completo", "Raman (3200 puntos)",
        lambda: analyse(spectrum, auto_preprocess=True), repeats,
        detail=lambda r: r.classification.label)

    # -- maps ----------------------------------------------------------
    cube = make_map_demo("dos_fases", seed=1)
    clean = add("quitar rayos cósmicos", "Mapa Raman (20×24×900)",
                lambda: despike_map(cube)[0], 2)
    add("PCA (5 componentes)", "Mapa Raman (20×24×900)",
        lambda: pca(clean, 5), 2)
    add("k-medias (k=3)", "Mapa Raman (20×24×900)",
        lambda: kmeans(clean, 3), 2)
    add("MCR-ALS (2 componentes)", "Mapa Raman (20×24×900)",
        lambda: mcr_als(clean, 2), 2)

    # -- diffraction ---------------------------------------------------
    add("buscar picos", "Difracción (3500 puntos)",
        lambda: find_peaks(pattern), 2, detail=lambda p: f"{len(p)} picos")
    add("identificar fases", "Difracción (3500 puntos)",
        lambda: identify_phases(pattern), 2)
    if not quick:
        add("Pawley (1 fase, 15 ciclos)", "Difracción (3500 puntos)",
            lambda: pawley(mos2_pattern, mos2, cycles=15), 1,
            detail=lambda r: f"R_wp = {100 * r.r_wp:.2f} %")
        add("Rietveld automático (1 fase)", "Difracción (3500 puntos)",
            lambda: auto_refine(mos2_pattern, [PhaseModel(crystal=mos2)]), 1,
            detail=lambda r: f"R_wp = {100 * r.r_wp:.2f} %")

    # -- electrochemistry ----------------------------------------------
    add("circuito equivalente", "Electroquímica",
        lambda: fit_circuit(impedance, "R0-(R1|Q1)"), 3)
    add("DRT (Tikhonov + curva L)", "Electroquímica",
        lambda: drt(impedance), 3, detail=lambda r: f"{len(r.peaks_s)} procesos")
    gcd = make_plateau_gcd_demo()
    add("dQ/dV", "Electroquímica", lambda: differential_capacity(gcd), 3)

    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ramancarbon.benchmarks",
        description="Mide cuánto tarda cada operación de la suite.")
    parser.add_argument("--json", metavar="ARCHIVO",
                        help="guardar los tiempos como JSON")
    parser.add_argument("--rapido", action="store_true",
                        help="saltarse los refinamientos, que son los lentos")
    parser.add_argument("--repeticiones", type=int, default=3,
                        help="repeticiones de cada medida (se informa la mejor)")
    arguments = parser.parse_args(argv)

    report = run(quick=arguments.rapido, repeats=arguments.repeticiones)
    print(report.text())
    if arguments.json:
        from pathlib import Path

        Path(arguments.json).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\nGuardado en {arguments.json}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
