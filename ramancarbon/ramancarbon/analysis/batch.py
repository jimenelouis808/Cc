"""Many spectra at once: in parallel, and summarised properly.

Two things a single-spectrum tool gets wrong about a batch.

**Speed.** Analysing one spectrum costs about a fifth of a second, almost
all of it in the least-squares fits. That is nothing for one spectrum and
several minutes for a map. The work is embarrassingly parallel — no
spectrum depends on any other — so :func:`analyse_many` spreads it over
processes.

The first attempt at that made it **eight times slower**, and the reason is
worth stating because it is invisible and it bites every scientific Python
program that mixes NumPy with multiprocessing. NumPy's linear algebra runs
on a threaded BLAS, which by default opens one thread per core. Four worker
processes each opening four BLAS threads puts sixteen threads on four cores,
and they spend their time fighting rather than working. Measured on four
cores, sixteen spectra: 47 seconds with the default threading, 1.1 seconds
with BLAS pinned to one thread per worker.

The pinning is done here, so callers get it for free. What this module
*cannot* do is pin the calling process's own BLAS, because NumPy is already
imported by the time any of this code runs. That matters more than it
sounds: in the same measurement, pinning BLAS in the parent too made even
the **serial** path 1.6x faster, because the matrices in a peak fit are
tiny — a few hundred rows by twenty columns — and threading them costs more
in synchronisation than it saves. If you process large batches often, set

    export OMP_NUM_THREADS=1

before starting Python and everything here gets faster, parallel or not.

Realistic figures from this package's own benchmark, 24 spectra on 4 cores,
without that environment variable set: 6.2 s serial, 3.0 s on four workers.
The speed-up is about two rather than four because the parent still runs a
threaded BLAS and each spawned worker costs roughly a second to start.

**Statistics.** A map of a real sample is heterogeneous, and the honest
result is a distribution, not a number. Reporting the mean alone hides that;
reporting the mean of a distribution with outliers in it is worse. So
:func:`summarise` gives the median and a robust spread alongside the mean,
flags outliers by a criterion that does not itself depend on the outliers,
and says how many points went into each figure.

The robust statistics matter more here than they look. One spectrum taken
on a lump of catalyst, or on a hole in the sample, can shift a mean I_D/I_G
by tens of percent while barely moving the median.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from ..core.spectrum import Spectrum
from .report import AnalysisResult, analyse

#: Environment variables that cap the thread count of the various BLAS
#: implementations NumPy may be linked against.
BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@contextmanager
def single_threaded_blas():
    """Temporarily pin BLAS to one thread per process.

    Applies to processes started *inside* the block, which is why
    :func:`analyse_many` uses the ``spawn`` start method: a spawned worker
    boots a fresh interpreter and imports NumPy after inheriting this
    environment, so the setting takes effect. A forked worker would
    inherit the parent's already-initialised BLAS and ignore it.

    The parent's own NumPy is already imported and is unaffected, so this
    is safe to wrap around anything.

    Any value the user had set is restored on exit.
    """
    previous = {var: os.environ.get(var) for var in BLAS_THREAD_VARS}
    try:
        for var in BLAS_THREAD_VARS:
            os.environ[var] = "1"
        yield
    finally:
        for var, value in previous.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


#: Modified z-score above which a point is called an outlier.
#:
#: Based on the median absolute deviation rather than the standard
#: deviation, because the standard deviation is itself inflated by the
#: outliers you are trying to find — with two bad points in twenty, an
#: ordinary 3σ test finds neither.
OUTLIER_Z = 3.5


@dataclass
class Statistic:
    """One measured quantity summarised over a batch."""

    key: str
    n: int
    mean: float
    std: float
    median: float
    mad_std: float
    """Robust spread: the median absolute deviation scaled to be comparable
    with a standard deviation for normally distributed data."""
    minimum: float
    maximum: float
    outliers: list[int] = field(default_factory=list)
    """Indices, into the list of results, of the points flagged as outliers."""

    @property
    def relative_spread(self) -> float:
        """Robust spread as a fraction of the median."""
        return self.mad_std / abs(self.median) if self.median else float("nan")

    def __str__(self) -> str:
        text = (
            f"{self.key:<12s} n={self.n:<4d} mediana={self.median:9.4g} "
            f"± {self.mad_std:.3g} (robusto)   media={self.mean:9.4g} "
            f"± {self.std:.3g}"
        )
        if self.outliers:
            text += f"   [{len(self.outliers)} atípico(s)]"
        return text


@dataclass
class BatchSummary:
    """Statistics over a set of analyses."""

    statistics: dict[str, Statistic] = field(default_factory=dict)
    n_results: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    materials: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Lote: {self.n_results} espectros analizados"]
        if self.failures:
            lines.append(f"  {len(self.failures)} con error")
        if self.materials:
            lines.append("")
            lines.append("Identificación:")
            for name, count in sorted(
                self.materials.items(), key=lambda kv: -kv[1]
            ):
                share = 100.0 * count / max(self.n_results, 1)
                lines.append(f"  {count:4d} ({share:5.1f} %)  {name}")
        if self.statistics:
            lines.append("")
            lines.append("Estadística:")
            lines.extend("  " + str(s) for s in self.statistics.values())
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


def _analyse_one(payload: tuple[Spectrum, dict]) -> tuple[Optional[AnalysisResult], str]:
    """Worker entry point. Returns the result or the failure message.

    Failures are returned rather than raised so that one unreadable
    spectrum in three hundred does not take the batch down with it.
    """
    spectrum, kwargs = payload
    try:
        return analyse(spectrum, **kwargs), ""
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def analyse_many(
    spectra: Sequence[Spectrum],
    workers: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    **kwargs: Any,
) -> tuple[list[AnalysisResult], list[tuple[str, str]]]:
    """Analyse many spectra, in parallel where that helps.

    Parameters
    ----------
    spectra:
        The spectra to analyse.
    workers:
        Number of processes. ``None`` uses the CPU count, capped at the
        number of spectra. Pass ``1`` to run in the current process, which
        is what to do when debugging — a traceback from a worker process is
        much harder to read.
    progress:
        Called as ``progress(done, total)`` after each spectrum. Runs on
        the calling thread, so a GUI can use it directly.
    **kwargs:
        Passed to :func:`~ramancarbon.analysis.report.analyse`.

    Returns
    -------
    (list[AnalysisResult], list[(name, message)])
        The analyses that succeeded, in input order, and the failures.

    Notes
    -----
    Workers are started with the ``spawn`` method and BLAS pinned to one
    thread each; see :func:`single_threaded_blas` for why that is not
    optional. A spawned worker pays about a second of interpreter start-up,
    so for a handful of spectra the serial path wins and is chosen
    automatically.

    **Calling this from a script requires the standard guard**::

        if __name__ == "__main__":
            results, failures = analyse_many(spectra)

    Without it each worker re-imports the script, reaches this call again,
    and tries to start its own pool. That is a property of ``spawn``, not
    of this package. It is detected and handled — the batch falls back to
    running serially with a warning explaining the fix — but the guard is
    worth having, because the serial path is several times slower.
    """
    items = list(spectra)
    if not items:
        return [], []

    count = workers if workers is not None else min(len(items), os.cpu_count() or 1)
    # Below this, process start-up costs more than it saves.
    if count <= 1 or len(items) < 4:
        results: list[AnalysisResult] = []
        failures: list[tuple[str, str]] = []
        for index, spectrum in enumerate(items, start=1):
            result, error = _analyse_one((spectrum, kwargs))
            if result is not None:
                results.append(result)
            else:
                failures.append((spectrum.name, error))
            if progress:
                progress(index, len(items))
        return results, failures

    try:
        return _run_parallel(items, count, kwargs, progress)
    except (RuntimeError, BrokenProcessPool, OSError) as exc:
        # The usual cause is a caller without an ``if __name__ ==
        # "__main__":`` guard, which the spawn start method requires: the
        # worker re-imports the calling script, reaches the same
        # analyse_many call, and tries to start its own pool. Falling back
        # is much friendlier than propagating multiprocessing's error,
        # which does not mention this package at all.
        import warnings

        warnings.warn(
            f"no se ha podido paralelizar ({type(exc).__name__}); se continúa "
            "en serie. Si llamas a analyse_many desde un script, protégelo "
            'con  if __name__ == "__main__":  — el método spawn reimporta el '
            "script en cada proceso hijo y sin esa guarda el hijo vuelve a "
            "lanzar el lote.",
            RuntimeWarning,
            stacklevel=2,
        )
        return analyse_many(items, workers=1, progress=progress, **kwargs)


def _run_parallel(
    items: Sequence[Spectrum],
    count: int,
    kwargs: dict,
    progress: Optional[Callable[[int, int], None]],
) -> tuple[list[AnalysisResult], list[tuple[str, str]]]:
    """The process-pool path. Raises if the pool cannot be built or used."""
    ordered: list[Optional[AnalysisResult]] = [None] * len(items)
    failures: list[tuple[str, str]] = []
    with single_threaded_blas():
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=count, mp_context=context) as pool:
            futures = {
                pool.submit(_analyse_one, (spectrum, kwargs)): index
                for index, spectrum in enumerate(items)
            }
            for done, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                result, error = future.result()
                if result is not None:
                    ordered[index] = result
                else:
                    failures.append((items[index].name, error))
                if progress:
                    progress(done, len(items))
    return [r for r in ordered if r is not None], failures


#: Quantities summarised by :func:`summarise`, in report order.
TRACKED = (
    "ID_IG",
    "I2D_IG",
    "ID_IDp",
    "gamma_G",
    "pos_G",
    "pos_D",
    "fwhm_D",
    "La_nm",
    "R2",
)


def summarise(
    results: Sequence[AnalysisResult],
    keys: Iterable[str] = TRACKED,
    failures: Sequence[tuple[str, str]] = (),
) -> BatchSummary:
    """Summarise a batch, robustly.

    Parameters
    ----------
    results:
        The analyses to summarise.
    keys:
        Which columns of :meth:`AnalysisResult.to_dict` to summarise.
    failures:
        Failures from :func:`analyse_many`, carried into the summary.

    Returns
    -------
    BatchSummary
    """
    summary = BatchSummary(n_results=len(results), failures=list(failures))
    if not results:
        return summary

    rows = [r.to_dict() for r in results]
    for row in rows:
        label = str(row.get("material", "?"))
        summary.materials[label] = summary.materials.get(label, 0) + 1

    for key in keys:
        values = np.array(
            [row.get(key) for row in rows if isinstance(row.get(key), (int, float))],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        # 1.4826 makes the MAD an estimator of sigma for normal data.
        mad_std = 1.4826 * mad
        outliers: list[int] = []
        if mad > 0:
            z = 0.6745 * (values - median) / mad
            outliers = [int(i) for i in np.flatnonzero(np.abs(z) > OUTLIER_Z)]
        summary.statistics[key] = Statistic(
            key=key,
            n=int(values.size),
            mean=float(np.mean(values)),
            std=float(np.std(values, ddof=1)),
            median=median,
            mad_std=mad_std,
            minimum=float(np.min(values)),
            maximum=float(np.max(values)),
            outliers=outliers,
        )

    _add_warnings(summary)
    return summary


def _add_warnings(summary: BatchSummary) -> None:
    """Say what the numbers imply about the sample, not just what they are."""
    ratio = summary.statistics.get("ID_IG")
    if ratio is not None:
        # Two tests, because they catch different shapes. The robust spread
        # assumes something near-normal and is the right measure when a few
        # points sit apart from a tight core. It badly understates a broad,
        # flat distribution — a set running uniformly from 1.05 to 2.41 gives
        # a relative spread of only 0.21 — so the full range is checked too.
        broad = ratio.relative_spread > 0.25
        wide = (
            (ratio.maximum - ratio.minimum) / abs(ratio.median) > 0.5
            if ratio.median
            else False
        )
        if broad or wide:
            summary.warnings.append(
                f"I_D/I_G va de {ratio.minimum:.2f} a {ratio.maximum:.2f} "
                f"(mediana {ratio.median:.2f}, dispersión robusta "
                f"{ratio.relative_spread * 100:.0f} %). La muestra es "
                "heterogénea a la escala del haz: un valor único no la "
                "representa. Da la mediana con su dispersión y di sobre "
                "cuántos puntos"
            )
    if len(summary.materials) > 1:
        names = ", ".join(sorted(summary.materials))
        summary.warnings.append(
            f"el lote no se clasifica de forma homogénea ({names}). O la "
            "muestra tiene varias fases, o algunos espectros son de mala "
            "calidad — comprueba los que caen aparte antes de promediar nada"
        )
    for statistic in summary.statistics.values():
        if statistic.outliers and len(statistic.outliers) <= 3:
            summary.warnings.append(
                f"{statistic.key}: {len(statistic.outliers)} punto(s) atípico(s) "
                f"en las posiciones {statistic.outliers}. Míralos antes de "
                "descartarlos: pueden ser el residuo de catalizador, un agujero "
                "en la muestra, o la parte interesante"
            )


__all__ = [
    "BLAS_THREAD_VARS",
    "OUTLIER_Z",
    "TRACKED",
    "BatchSummary",
    "Statistic",
    "analyse_many",
    "single_threaded_blas",
    "summarise",
]
