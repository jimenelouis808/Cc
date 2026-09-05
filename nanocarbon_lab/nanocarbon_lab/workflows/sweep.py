"""Parameter sweeps over any structure mode, not just nanotubes.

:func:`~nanocarbon_lab.workflows.batch.batch_cnt_sweep` sweeps chirality
and length, which was the whole program when it was written. There are
nineteen modes now -- cages, junctions, schwarzites, nanotube networks,
dichalcogenides, twisted bilayers -- and none of them could be swept.

The fix is not another `batch_X_sweep` per mode. :mod:`nanocarbon_lab.jobs`
already maps "what to build" onto builder arguments for every mode, and
already knows how to estimate the size and cost of one. A sweep is then
just a Cartesian product over that mapping, and adding a mode to
``jobs.py`` gets it a sweep for free.

Two things this owes the user, because a sweep is the one place where a
mistake is expensive:

* **The cost, before it runs.** Four values of three parameters is 64
  structures, and if each is a two-minute schwarzite that is two hours.
  :func:`describe_sweep` reports the count and the predicted atoms so
  the arithmetic happens before the wait, not during it.
* **A name that says what it is.** Each structure is named from the
  parameters that vary, so ``capped_tube__freq3__rings8`` is legible in
  a directory listing months later -- and the parameters that did *not*
  vary are left out, since repeating them in every name says nothing.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

from ase import Atoms

from ..jobs import Job, build, estimate_atoms, family_of, parameter_names
from .batch import BatchJob, ExportFormat


def _format_value(value: Any) -> str:
    """A short, filename-safe rendering of one parameter value."""
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return "".join(ch if (ch.isalnum() or ch in "-.") else "_" for ch in text)


def expand(mode: str,
           base: Mapping[str, Any] | None = None,
           vary: Mapping[str, Sequence[Any]] | None = None) -> list[Job]:
    """Every combination of ``vary``, on top of ``base``, as jobs.

    Parameters
    ----------
    mode
        Any mode in :data:`nanocarbon_lab.jobs.MODES`.
    base
        Parameters held fixed across the sweep.
    vary
        Parameter name to the list of values it takes. The product is
        taken in the order the keys are given, so the first key varies
        slowest -- which is what makes a sorted directory listing read
        as a table.

    Returns
    -------
    list[Job]
        One job per combination; a sweep with nothing to vary is a single
        job, which is a useful degenerate case rather than an error.

    Raises
    ------
    ValueError
        If ``mode`` is unknown, or a varied parameter has no values --
        an empty list would silently collapse the whole product to
        nothing, which looks like a sweep that ran and found no work.
    """
    base = dict(base or {})
    vary = dict(vary or {})
    for name, values in vary.items():
        if not list(values):
            raise ValueError(
                f"Parameter {name!r} has no values to sweep. An empty list "
                "would silently collapse the whole product to zero jobs."
            )

    # Names are checked against the builder's real signature, before
    # anything is built. Without this a typo is accepted in silence: the
    # estimate quietly falls back to the default and the mistake surfaces
    # as a TypeError on the first build -- which on a long sweep is hours
    # in, with a directory of half-written output to clean up.
    allowed = set(parameter_names(mode))
    unknown = sorted((set(base) | set(vary)) - allowed)
    if unknown:
        raise ValueError(
            f"{mode!r} has no parameter(s) {', '.join(repr(u) for u in unknown)}. "
            f"It accepts: {', '.join(sorted(allowed))}."
        )

    names = list(vary)
    combinations = list(itertools.product(*(list(vary[n]) for n in names)))
    jobs: list[Job] = []
    for combination in combinations:
        params = {**base, **dict(zip(names, combination, strict=True))}
        # Job validates the mode, so an unknown one fails here rather
        # than on the hundredth build.
        jobs.append(Job(mode=mode, params=params))
    return jobs


def sweep_name(mode: str, job: Job, vary: Mapping[str, Sequence[Any]]) -> str:
    """A directory-listing-legible name for one job of a sweep.

    Only the varying parameters appear: repeating the fixed ones in every
    name adds length without adding information.
    """
    stem = mode.replace(" ", "_").replace("(", "").replace(")", "")
    parts = [f"{name}{_format_value(job.params[name])}"
             for name in vary if name in job.params]
    return "__".join([stem, *parts]) if parts else stem


def sweep_jobs(
    mode: str,
    base: Mapping[str, Any] | None = None,
    vary: Mapping[str, Sequence[Any]] | None = None,
    dopant: str | None = None,
    dopant_conc: float = 0.0,
    dopant_site: str = "random",
    seed: int = 0,
    export: ExportFormat = "qe",
    post: Sequence[Callable[[Atoms], Atoms]] = (),
) -> list[BatchJob]:
    """Turn a sweep into :class:`~.batch.BatchJob` objects.

    The builder of each is ``jobs.build`` bound to that job, so anything
    the GUI or the CLI can build, a sweep can build too -- including the
    doping and the MX2 chemistry, which go through the same single
    policy rather than being reimplemented here.

    Each job gets its **own derived seed** (``seed + index``). A shared
    seed would put the identical defect pattern in every structure of the
    sweep, which is the one thing a dataset must not have: the model
    would learn the pattern rather than the physics.
    """
    vary = dict(vary or {})
    built = expand(mode, base, vary)
    out: list[BatchJob] = []
    for index, job in enumerate(built):
        seeded = Job(
            mode=job.mode,
            params=job.params,
            dopant=dopant,
            dopant_conc=dopant_conc,
            dopant_site=dopant_site,
            seed=seed + index,
        )
        out.append(BatchJob(
            name=sweep_name(mode, job, vary),
            builder=partial(build, seeded),
            post=list(post),
            export=export,
        ))
    return out


def describe_sweep(mode: str,
                   base: Mapping[str, Any] | None = None,
                   vary: Mapping[str, Sequence[Any]] | None = None) -> dict:
    """Count and cost a sweep without building anything.

    A sweep is where a small mistake becomes an expensive one: three
    parameters at four values each is 64 structures, and if each is a
    two-minute schwarzite that is two hours. This is the arithmetic done
    up front.
    """
    jobs = expand(mode, base, vary)
    per_job = [estimate_atoms(job) for job in jobs]
    return {
        "mode": mode,
        "family": family_of(mode),
        "n_structures": len(jobs),
        "atoms_min": min(per_job) if per_job else 0,
        "atoms_max": max(per_job) if per_job else 0,
        "atoms_total": sum(per_job),
        "names": [sweep_name(mode, job, dict(vary or {})) for job in jobs],
    }


__all__ = ["describe_sweep", "expand", "sweep_jobs", "sweep_name"]
