"""Computed IR spectrum against an experimental FTIR: the analysis half of vibspec.

Everything the comparison needs, and nothing that draws:

* :func:`computed_curve` -- broaden the computed lines (Lorentzian or
  Gaussian, width given as **FWHM**, as spectroscopists quote it), after
  applying the frequency scale factor. The stored frequencies are never
  scaled in place.
* :func:`read_ftir` -- a two-column CSV/TXT export from the spectrometer,
  whatever the delimiter or decimal separator, in absorbance or
  transmittance.
* :func:`to_absorbance`, :func:`rubberband_baseline`, :func:`normalise` --
  put the experiment on the same footing as the calculation. IR intensity is
  proportional to absorbance, not to transmittance: comparing a computed
  spectrum with a %T trace compares the wrong quantity.
* :func:`find_bands`, :func:`match_bands`, :func:`search_scale_factor` --
  suggest which computed mode goes with which experimental band, and derive
  the scale factor that best maps one onto the other. The search matters:
  unscaled DFT frequencies are often further from experiment than any sane
  matching tolerance (water in PBE: 3698 against 3756 cm^-1), so pairing
  first and fitting afterwards finds nothing to fit.

What this cannot do, and says: a nearest-band match within a tolerance is a
*suggestion*. Two computed modes can compete for one band, a band can be an
overtone or a combination the harmonic calculation does not have, and a
finite ribbon is not a nanotube. The table is where assignment starts, not
where it ends.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

from ...results.spectra import VibrationalSpectrum, broaden

Profile = Literal["lorentzian", "gaussian"]
Quantity = Literal["absorbance", "transmittance"]

#: Default plotting window, cm^-1: the mid-IR a typical FTIR covers.
MID_IR: tuple[float, float] = (400.0, 4000.0)


# --------------------------------------------------------------------------
# Computed side
# --------------------------------------------------------------------------

def computed_curve(
    spectrum: VibrationalSpectrum,
    grid: np.ndarray,
    fwhm_cm1: float = 10.0,
    profile: Profile = "lorentzian",
    scale_factor: float = 1.0,
) -> np.ndarray:
    """Broadened computed IR spectrum on ``grid``.

    Parameters
    ----------
    spectrum
        From :func:`carbonforge.vibspec.core.workflow.collect`.
    grid
        Wavenumbers, cm^-1.
    fwhm_cm1
        Full width at half maximum of every line. 8-20 cm^-1 resembles a
        solid-state FTIR band; the calculation itself has no width.
    profile
        ``"lorentzian"`` or ``"gaussian"``.
    scale_factor
        Multiplies every frequency before broadening.

    Returns
    -------
    numpy.ndarray
        Intensity in the calculation's units, (D/Å)²/amu per cm^-1.
    """
    if fwhm_cm1 <= 0:
        raise ValueError("fwhm_cm1 debe ser positivo.")
    if scale_factor <= 0:
        raise ValueError("scale_factor debe ser positivo.")
    _, intensity = broaden(
        spectrum.frequencies * scale_factor, spectrum.activities("ir"),
        width_cm1=fwhm_cm1 / 2.0, grid=np.asarray(grid, dtype=float), profile=profile,
    )
    return intensity


def sticks(
    spectrum: VibrationalSpectrum, scale_factor: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled line positions and their intensities relative to the strongest."""
    frequencies = spectrum.frequencies * scale_factor
    intensities = spectrum.activities("ir")
    top = intensities.max() if intensities.size and intensities.max() > 0 else 1.0
    return frequencies, intensities / top


# --------------------------------------------------------------------------
# Experimental side
# --------------------------------------------------------------------------

@dataclass
class ExperimentalSpectrum:
    """An FTIR trace, always stored with ascending wavenumber.

    ``quantity_source`` says how :attr:`quantity` was decided -- ``"given"``,
    ``"header"`` or ``"values"`` -- so a guess is never mistaken for a fact.
    """

    wavenumber: np.ndarray
    values: np.ndarray
    quantity: Quantity
    percent: bool = False
    name: str = "FTIR"
    source: Optional[str] = None
    quantity_source: str = "given"
    header: list[str] = field(default_factory=list)


_NUMBER = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")


def _split(line: str) -> list[str]:
    for delimiter in ("\t", ";", ","):
        if delimiter in line:
            parts = [p.strip() for p in line.split(delimiter)]
            # A comma is a delimiter only if it yields numeric columns; in
            # "1234,5 0,12" it is a decimal separator.
            if delimiter != "," or all(_NUMBER.match(p) for p in parts if p):
                return [p for p in parts if p]
    return line.split()


def _to_float(token: str) -> Optional[float]:
    token = token.strip()
    if not _NUMBER.match(token):
        return None
    return float(token.replace(",", "."))


def read_ftir(
    path: str | Path,
    quantity: Optional[Quantity] = None,
    name: Optional[str] = None,
) -> ExperimentalSpectrum:
    """Read a two-column FTIR export (wavenumber, absorbance or transmittance).

    Tab, semicolon, comma and whitespace delimiters are recognised, and so is
    a decimal comma (``1234,5;0,87``). Header lines before the data are kept.
    Extra columns are ignored.

    Parameters
    ----------
    path
        CSV or TXT file.
    quantity
        ``"absorbance"`` or ``"transmittance"``. When omitted it is read from
        the header (``abs``, ``transm``, ``%T``) or, failing that, guessed
        from the values: a trace whose median exceeds 1.5 is taken as %T,
        anything else as absorbance. The guess is recorded in
        :attr:`ExperimentalSpectrum.quantity_source`; pass ``quantity`` when
        it matters.
    name
        Label for plots; defaults to the file name.
    """
    path = Path(path)
    header: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        values = [_to_float(p) for p in _split(line)]
        if len(values) >= 2 and values[0] is not None and values[1] is not None:
            xs.append(values[0])
            ys.append(values[1])
        elif not xs:
            header.append(line.strip())
    if len(xs) < 2:
        raise ValueError(
            f"{path}: no se encontraron dos columnas numéricas (número de onda, "
            "intensidad). Exporta el espectro como texto desde el software del equipo."
        )

    x, y = np.asarray(xs), np.asarray(ys)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) == 0):
        raise ValueError(f"{path}: hay números de onda repetidos.")

    source = "given"
    if quantity is None:
        text = " ".join(header).lower()
        if "transm" in text or "%t" in text:
            quantity, source = "transmittance", "header"
        elif "abs" in text:
            quantity, source = "absorbance", "header"
        else:
            quantity = "transmittance" if float(np.median(y)) > 1.5 else "absorbance"
            source = "values"
    if quantity not in ("absorbance", "transmittance"):
        raise ValueError(f"quantity debe ser 'absorbance' o 'transmittance', no {quantity!r}.")
    percent = quantity == "transmittance" and float(np.max(y)) > 1.5

    return ExperimentalSpectrum(
        wavenumber=x, values=y, quantity=quantity, percent=percent,
        name=name or path.stem, source=str(path), quantity_source=source, header=header,
    )


def to_absorbance(experiment: ExperimentalSpectrum) -> ExperimentalSpectrum:
    """Return the spectrum as absorbance, ``A = -log10(T)``.

    Transmittance at or below zero (noise at the bottom of a saturated band)
    is clipped to 1e-4, i.e. A = 4, rather than producing infinities.
    """
    if experiment.quantity == "absorbance":
        return experiment
    transmittance = experiment.values / (100.0 if experiment.percent else 1.0)
    absorbance = -np.log10(np.clip(transmittance, 1e-4, None))
    return ExperimentalSpectrum(
        wavenumber=experiment.wavenumber, values=absorbance, quantity="absorbance",
        name=experiment.name, source=experiment.source,
        quantity_source=experiment.quantity_source, header=experiment.header,
    )


def rubberband_baseline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Lower convex hull of the trace, interpolated: a parameter-free baseline.

    Good for the broad, smoothly curving background of a KBr pellet or an
    ATR spectrum of a carbon powder. It assumes the true baseline touches the
    spectrum from below and is convex; where scattering makes it concave,
    subtract nothing rather than something wrong.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    hull: list[int] = []
    for i in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.interp(x, x[hull], y[hull])


def normalise(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """Scale ``y`` so its maximum inside ``window`` (default: everywhere) is 1."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if window is None:
        mask = np.ones_like(x, dtype=bool)
    else:
        mask = (x >= min(window)) & (x <= max(window))
    if not mask.any():
        raise ValueError(f"La ventana {window} no contiene datos.")
    top = float(np.max(y[mask]))
    if top <= 0:
        raise ValueError("El máximo en la ventana no es positivo; no se puede normalizar.")
    return y / top


def prepare_experiment(
    experiment: ExperimentalSpectrum,
    baseline: bool = True,
    window: tuple[float, float] = MID_IR,
) -> tuple[np.ndarray, np.ndarray]:
    """Absorbance, baseline-corrected, normalised to 1 inside ``window``."""
    absorbance = to_absorbance(experiment)
    x, y = absorbance.wavenumber, absorbance.values
    mask = (x >= min(window)) & (x <= max(window))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        raise ValueError(f"El espectro experimental no tiene datos en {window} cm⁻¹.")
    if baseline:
        y = y - rubberband_baseline(x, y)
    return x, normalise(x, y)


# --------------------------------------------------------------------------
# Bands and assignment
# --------------------------------------------------------------------------

def find_bands(
    x: np.ndarray,
    y: np.ndarray,
    prominence: float = 0.05,
    min_separation_cm1: float = 15.0,
) -> np.ndarray:
    """Positions of the experimental bands, cm^-1.

    ``prominence`` is relative to the strongest band (the trace is expected
    normalised). Bands closer than ``min_separation_cm1`` keep the stronger.
    """
    from scipy.signal import find_peaks

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    step = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    distance = max(1, int(round(min_separation_cm1 / step)))
    peaks, _ = find_peaks(y, prominence=prominence * float(np.max(y)), distance=distance)
    return x[peaks]


@dataclass
class BandMatch:
    """One computed IR-active mode and the experimental band it lands near."""

    mode_index: int
    computed_cm1: float          # after scaling
    relative_intensity: float    # to the strongest computed mode
    experimental_cm1: Optional[float]

    @property
    def delta_cm1(self) -> Optional[float]:
        if self.experimental_cm1 is None:
            return None
        return self.computed_cm1 - self.experimental_cm1


def match_bands(
    spectrum: VibrationalSpectrum,
    bands_cm1: Sequence[float],
    scale_factor: float = 1.0,
    tolerance_cm1: float = 30.0,
    min_relative_intensity: float = 0.05,
) -> list[BandMatch]:
    """Pair each IR-active computed mode with the nearest experimental band.

    Only modes with at least ``min_relative_intensity`` of the strongest are
    considered: the rest would not be visible in the experiment anyway. A
    mode with no band within ``tolerance_cm1`` gets ``experimental_cm1=None``.
    Several modes may land on the same band -- which is real (overlapping
    modes) as often as it is a mismatch, and is left for you to judge.
    """
    frequencies, relative = sticks(spectrum, scale_factor)
    bands = np.asarray(sorted(bands_cm1), dtype=float)
    matches = []
    for mode, frequency, strength in zip(
        spectrum.modes, frequencies, relative, strict=True
    ):
        if strength < min_relative_intensity or frequency <= 0:
            continue
        nearest = None
        if bands.size:
            j = int(np.argmin(np.abs(bands - frequency)))
            if abs(bands[j] - frequency) <= tolerance_cm1:
                nearest = float(bands[j])
        matches.append(BandMatch(mode.index, float(frequency), float(strength), nearest))
    return sorted(matches, key=lambda m: m.computed_cm1)


def fit_scale_factor(matches: Sequence[BandMatch], unscaled_by: float = 1.0) -> float:
    """Least-squares scale factor mapping computed onto experimental bands.

    Minimises ``sum((λ ν_calc - ν_exp)²)`` over matched pairs, giving
    ``λ = sum(ν_calc ν_exp) / sum(ν_calc²)``, the usual definition (Scott &
    Radom, 1996). ``unscaled_by`` is the factor the matches were made with,
    so the result is relative to the raw frequencies.

    Raises
    ------
    ValueError
        With fewer than three matched pairs: a scale factor fitted to one or
        two bands mostly fits their assignment errors.
    """
    pairs = [(m.computed_cm1 / unscaled_by, m.experimental_cm1)
             for m in matches if m.experimental_cm1 is not None]
    if len(pairs) < 3:
        raise ValueError(
            f"Solo {len(pairs)} banda(s) emparejada(s): hacen falta al menos 3 para "
            "ajustar un factor de escala con sentido."
        )
    computed = np.array([p[0] for p in pairs])
    experimental = np.array([p[1] for p in pairs])
    return float(computed @ experimental / (computed @ computed))


def search_scale_factor(
    spectrum: VibrationalSpectrum,
    bands_cm1: Sequence[float],
    tolerance_cm1: float = 30.0,
    min_relative_intensity: float = 0.05,
    bounds: tuple[float, float] = (0.90, 1.05),
    step: float = 0.001,
) -> tuple[float, list[BandMatch]]:
    """Find the scale factor that brings the most computed intensity onto bands.

    Scans ``bounds`` (the range of published factors) in steps of ``step``,
    scoring each factor by the summed relative intensity of the modes it
    matches, with the RMS deviation as tie-breaker; then refines the winner
    by least squares on its pairs (:func:`fit_scale_factor`).

    Returns
    -------
    (scale_factor, matches)
        The refined factor, relative to the raw frequencies, and the matches
        it gives.

    Raises
    ------
    ValueError
        When no factor in ``bounds`` matches at least three modes.
    """
    best: Optional[tuple[tuple[float, float], float, list[BandMatch]]] = None
    for factor in np.arange(min(bounds), max(bounds) + step / 2, step):
        matches = match_bands(spectrum, bands_cm1, float(factor), tolerance_cm1,
                              min_relative_intensity)
        paired = [m for m in matches if m.delta_cm1 is not None]
        if not paired:
            continue
        weight = sum(m.relative_intensity for m in paired)
        rms = float(np.sqrt(np.mean([m.delta_cm1 ** 2 for m in paired])))
        score = (weight, -rms)
        if best is None or score > best[0]:
            best = (score, float(factor), matches)
    if best is None:
        raise ValueError(
            f"Ningún factor entre {min(bounds)} y {max(bounds)} acerca un modo calculado "
            f"a menos de {tolerance_cm1} cm⁻¹ de una banda experimental."
        )
    _, factor, matches = best
    refined = fit_scale_factor(matches, unscaled_by=factor)
    return refined, match_bands(spectrum, bands_cm1, refined, tolerance_cm1,
                                min_relative_intensity)


def match_table(matches: Sequence[BandMatch]) -> str:
    """The matches as a plain-text table."""
    lines = [f"{'modo':>5} {'calc (cm⁻¹)':>12} {'I rel':>6} {'exp (cm⁻¹)':>11} {'Δ':>7}",
             "-" * 46]
    for m in matches:
        exp = f"{m.experimental_cm1:11.1f}" if m.experimental_cm1 is not None else f"{'—':>11}"
        delta = f"{m.delta_cm1:+7.1f}" if m.delta_cm1 is not None else f"{'':>7}"
        lines.append(f"{m.mode_index:5d} {m.computed_cm1:12.1f} {m.relative_intensity:6.2f} "
                     f"{exp} {delta}")
    lines.append(
        "\nEmparejado por cercanía: es una propuesta de asignación, no una prueba. "
        "Revisa el modo (modes.npz) antes de dar una banda por asignada."
    )
    return "\n".join(lines)


def export_csv(
    stem: str | Path,
    grid: np.ndarray,
    computed: np.ndarray,
    stick_positions: np.ndarray,
    stick_heights: np.ndarray,
    experiment: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> list[Path]:
    """Write one two-column CSV per curve, the format ramancarbon reads.

    ``<stem>_calculado.csv``, ``<stem>_barras.csv`` and, with an experiment,
    ``<stem>_experimental.csv``. Separate files because the curves have
    different abscissae; each opens directly in ramancarbon's plot engine.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []

    def write(path: Path, header: tuple[str, str], x, y) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for a, b in zip(x, y, strict=True):
                writer.writerow([f"{a:.4f}", f"{b:.6g}"])
        written.append(path)

    write(stem.with_name(stem.name + "_calculado.csv"),
          ("numero_de_onda_cm-1", "ir_calculado_norm"), grid, computed)
    write(stem.with_name(stem.name + "_barras.csv"),
          ("numero_de_onda_cm-1", "intensidad_relativa"), stick_positions, stick_heights)
    if experiment is not None:
        write(stem.with_name(stem.name + "_experimental.csv"),
              ("numero_de_onda_cm-1", "absorbancia_norm"), *experiment)
    return written
