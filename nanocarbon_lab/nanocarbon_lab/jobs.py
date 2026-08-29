"""One description of "what to build", shared by the GUI and the CLI.

The GUI used to carry its own ninety-line ``if mode == ...`` chain mapping
widget values onto builder arguments, duplicating what the CLI already
knew. Three things fall out of writing that mapping down once:

* the GUI can hand a job to a **separate process**, because a
  :class:`Job` is a plain dataclass of built-in types and pickles
  cleanly, whereas a bound method closing over Tk variables does not;
* it can tell you **how big and how slow** a build will be before you
  commit to it, which matters when the same button produces either a
  60-atom cage in a tenth of a second or a 3000-atom coil in six
  minutes;
* it can show you the **equivalent command line**, so anything found by
  dragging sliders can be reproduced, scripted and put in a paper's
  methods section.

Nothing here imports tkinter, matplotlib or anything else the GUI needs.
That is deliberate: the worker subprocess imports this module, and it
should pay for numpy and the builders, not for a windowing toolkit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

# Canonical mode names. The GUI shows these verbatim in its dropdown.
MODES = (
    "capped tube",
    "coil (relaxed)",
    "fullerene",
    "nano-onion",
    "junction",
    "schwarzite",
    "multi-wall",
    "bundle",
)

# Modes that go through marching cubes + isotropic remeshing rather than a
# seed polyhedron. They cost orders of magnitude more time per atom, which
# is the single most useful thing to know before pressing Build.
IMPLICIT_MODES = frozenset({"coil (relaxed)", "junction", "schwarzite"})

# Area of one period of each triply periodic minimal surface, in units of
# the cell length squared. Standard values for the trigonometric
# approximations; used only to estimate atom counts.
TPMS_AREA = {"primitive": 2.345, "diamond": 3.838, "gyroid": 3.091}

# Area a single graphitic ring covers: a regular hexagon of side 1.42 Å.
RING_AREA = 1.5 * math.sqrt(3.0) * 1.42**2

# Atoms per ring in a closed honeycomb (Euler: F = 2V - 4).
ATOMS_PER_RING = 2.0


@dataclass(frozen=True)
class Job:
    """A structure to build: the mode, the builder's arguments, chemistry.

    Attributes
    ----------
    mode
        One of :data:`MODES`.
    params
        Keyword arguments for that mode's builder, already in the
        builder's own units and spelling.
    dopant
        ``None`` or an element symbol substituted after the build.
    dopant_conc
        Substitution fraction, ignored when ``dopant`` is ``None``.
    seed
        RNG seed, threaded into both the builder and the doping.
    """

    mode: str
    params: dict[str, Any] = field(default_factory=dict)
    dopant: str | None = None
    dopant_conc: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(
                f"Unknown mode {self.mode!r}; expected one of {list(MODES)}."
            )

    def with_params(self, **changes: Any) -> Job:
        """A copy with some builder arguments replaced."""
        return replace(self, params={**self.params, **changes})


def build(job: Job):
    """Run a job and return the resulting :class:`ase.Atoms`.

    Imports the builders lazily so that merely describing or estimating a
    job -- which the GUI does on every keystroke -- costs nothing.
    """
    from .builders import (
        build_bundle,
        build_capped_cnt,
        build_coil,
        build_fullerene,
        build_junction,
        build_multiwall_cnt,
        build_nano_onion,
        build_schwarzite,
    )

    builders = {
        "capped tube": build_capped_cnt,
        "coil (relaxed)": build_coil,
        "fullerene": build_fullerene,
        "nano-onion": build_nano_onion,
        "junction": build_junction,
        "schwarzite": build_schwarzite,
        "multi-wall": build_multiwall_cnt,
        "bundle": build_bundle,
    }
    atoms = builders[job.mode](**job.params, seed=job.seed)
    if job.dopant and job.dopant_conc > 0:
        from .dopants import dope_random

        atoms = dope_random(atoms, job.dopant, job.dopant_conc, seed=job.seed)
    return atoms


def estimate_atoms(job: Job) -> int:
    """Predict the atom count without building anything.

    Exact for the seed-polyhedron modes, where the count is a closed form:
    subdividing a seed with ``F`` faces by frequency ``f`` gives ``F*f**2``
    triangles, and the dual puts one atom on each. Approximate for the
    implicit modes, where it is the meshed surface area divided by the
    area one ring covers -- good to a few per cent against the real
    builds (gyroid at 36 Å: 1529 predicted, 1502 built).
    """
    p = job.params
    mode = job.mode

    if mode == "capped tube":
        # Seed capsule has 10*n_rings faces; subdivision multiplies by f^2.
        return int(10 * int(p.get("n_body_rings", 8)) * int(p.get("freq", 3)) ** 2)

    if mode == "fullerene":
        base = 60 if p.get("family", "C60") == "C60" else 20
        return int(base * int(p.get("freq", 1)) ** 2)

    if mode == "nano-onion":
        base = 60 if p.get("family", "C60") == "C60" else 20
        inner = int(p.get("inner_freq", 1))
        step = int(p.get("freq_step", 1))
        shells = int(p.get("n_shells", 3))
        return int(sum(base * (inner + k * step) ** 2 for k in range(shells)))

    if mode == "multi-wall":
        rings = int(p.get("n_body_rings", 10))
        inner = int(p.get("inner_freq", 3))
        step = int(p.get("freq_step", 2))
        shells = int(p.get("n_shells", 2))
        return int(sum(10 * rings * (inner + k * step) ** 2 for k in range(shells)))

    if mode == "bundle":
        across = int(p.get("n_rings_across", 1))
        n_tubes = 3 * across * across + 3 * across + 1
        per_tube = 10 * int(p.get("n_body_rings", 10)) * int(p.get("freq", 3)) ** 2
        return int(n_tubes * per_tube)

    if mode == "schwarzite":
        cell = float(p.get("cell", 36.0))
        area = TPMS_AREA.get(p.get("kind", "primitive"), 2.345) * cell**2
        return int(ATOMS_PER_RING * area / RING_AREA)

    if mode == "junction":
        radius = float(p.get("tube_radius", 6.0))
        arm = float(p.get("arm_length", 22.0))
        n_arms = {"L": 2, "T": 3, "Y": 3, "X": 4, "cross3d": 6}.get(p.get("kind", "Y"), 3)
        # Arms as cylinders plus one hemispherical tip each; the shared
        # neck is ignored, which is why this reads slightly high.
        area = n_arms * (2 * math.pi * radius * arm + 2 * math.pi * radius**2)
        return int(ATOMS_PER_RING * area / RING_AREA)

    if mode == "coil (relaxed)":
        radius = float(p.get("tube_radius", 6.0))
        coil_radius = float(p.get("coil_radius", 40.0))
        pitch = float(p.get("pitch", 25.0))
        turns = float(p.get("turns", 2.0))
        taper = float(p.get("taper", 1.0))
        mean_radius = coil_radius * (1.0 + taper) / 2.0
        arc = turns * math.hypot(2 * math.pi * mean_radius, pitch)
        area = 2 * math.pi * radius * arc + 4 * math.pi * radius**2
        return int(ATOMS_PER_RING * area / RING_AREA)

    return 0


def estimate_cost(job: Job) -> tuple[str, str]:
    """``(severity, human text)`` for how long a job will take.

    Severity is ``"fast"``, ``"slow"`` or ``"very slow"``, meant for
    colouring a label. The split is not about atom count alone: the
    implicit modes mesh and remesh a surface before they relax it, so a
    2000-atom coil takes minutes where a 2000-atom cage takes under a
    second.
    """
    n = estimate_atoms(job)
    if job.mode in IMPLICIT_MODES:
        if n < 900:
            return "slow", f"~{n} atoms, tens of seconds"
        if n < 2500:
            return "slow", f"~{n} atoms, a minute or two"
        return "very slow", f"~{n} atoms, several minutes — consider smaller"
    if n > 20000:
        return "very slow", f"~{n} atoms, slow to relax and to draw"
    if n > 4000:
        return "slow", f"~{n} atoms, a few seconds"
    return "fast", f"~{n} atoms, near-instant"


# Mode -> (sub-command, {builder argument: command-line flag}). Arguments
# not listed have no flag and are dropped from the generated command.
_CLI_MAP: dict[str, tuple[str, dict[str, str]]] = {
    "capped tube": ("cnt-cap", {
        "n_body_rings": "--rings", "freq": "--freq", "bond": "--bond",
        "bend_angle": "--bend-angle", "shape": "--shape",
        "waviness": "--waviness", "max_strain": "--max-strain",
        "shape_points": "--shape-points", "helix_turns": "--helix-turns",
        "helix_radius": "--helix-radius", "helix_pitch": "--helix-pitch",
        "helix_taper": "--helix-taper", "roughness": "--roughness",
    }),
    "coil (relaxed)": ("coil", {
        "coil_radius": "--coil-radius", "pitch": "--pitch", "turns": "--turns",
        "tube_radius": "--tube-radius", "taper": "--taper", "bond": "--bond",
        "anneal_sweeps": "--anneal-sweeps", "roughness": "--roughness",
    }),
    "fullerene": ("fullerene", {
        "freq": "--freq", "family": "--family", "bond": "--bond",
        "roughness": "--roughness",
    }),
    "nano-onion": ("onion", {
        "n_shells": "--shells", "inner_freq": "--inner-freq",
        "freq_step": "--freq-step", "family": "--family", "bond": "--bond",
        "roughness": "--roughness",
    }),
    "junction": ("junction", {
        "kind": "--kind", "tube_radius": "--tube-radius",
        "arm_length": "--arm-length", "blend": "--blend", "bond": "--bond",
        "anneal_sweeps": "--anneal-sweeps", "roughness": "--roughness",
    }),
    "schwarzite": ("schwarzite", {
        "kind": "--kind", "cell": "--cell", "thickness": "--thickness",
        "bond": "--bond", "anneal_sweeps": "--anneal-sweeps",
        "roughness": "--roughness",
    }),
    "multi-wall": ("mwcnt", {
        "n_shells": "--shells", "inner_freq": "--inner-freq",
        "freq_step": "--freq-step", "n_body_rings": "--rings",
        "bond": "--bond", "roughness": "--roughness",
    }),
    "bundle": ("bundle", {
        "n_rings_across": "--shells", "freq": "--freq",
        "n_body_rings": "--rings", "gap": "--gap", "bond": "--bond",
        "roughness": "--roughness",
    }),
}


def to_cli(job: Job, out: str = "out/structure") -> str:
    """The ``nanocarbon`` command line equivalent to this job.

    Lets anything found by dragging sliders be reproduced exactly, put in
    a script, or pasted into a methods section. Flags whose value equals
    the builder default are still emitted: the point is a command that
    states the whole structure, not the shortest one.

    Handedness is a special case -- the builders take ``+1``/``-1`` while
    the CLI takes ``right``/``left`` -- so it is translated rather than
    printed as a number that the CLI would reject.
    """
    subcommand, flags = _CLI_MAP[job.mode]
    parts = ["nanocarbon", subcommand]

    for name, value in sorted(job.params.items()):
        if value is None:
            continue
        if name in ("helix_handedness", "handedness"):
            parts += ["--helix-handedness" if name.startswith("helix") else
                      "--handedness", "right" if int(value) >= 0 else "left"]
            continue
        if name == "pin_ends":
            if value:
                parts.append("--pin-ends")
            continue
        flag = flags.get(name)
        if flag is None:
            continue
        if isinstance(value, bool):
            if value:
                parts.append(flag)
        elif isinstance(value, float):
            parts += [flag, f"{value:g}"]
        else:
            parts += [flag, str(value)]

    if job.dopant and job.dopant_conc > 0:
        parts += ["--dopant", job.dopant, "--dopant-conc", f"{job.dopant_conc:g}"]
    parts += ["--seed", str(job.seed), "--out", out]
    return " ".join(parts)


__all__ = [
    "IMPLICIT_MODES",
    "MODES",
    "Job",
    "build",
    "estimate_atoms",
    "estimate_cost",
    "to_cli",
]
