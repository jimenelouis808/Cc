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

# Canonical mode names, grouped by material family. The GUI shows the
# family in one dropdown and the modes of that family in the next, so a
# carbon control never appears next to a dichalcogenide one.
CARBON_MODES = (
    "capped tube",
    "coil (relaxed)",
    "fullerene",
    "nano-onion",
    "junction",
    "schwarzite",
    "multi-wall",
    "bundle",
)

#: MX2 dichalcogenides. Flat and rolled structures only for now -- the
#: curved topologies (Y junction, schwarzite) need even-membered rings to
#: keep the M/X alternation, which is a different remeshing problem; see
#: the README.
TMD_MODES = (
    "TMD layers",
    "TMD bulk",
    "TMD ribbon",
    "TMD nanotube",
    "TMD coil",
)

FAMILIES = {"carbon": CARBON_MODES, "dichalcogenide": TMD_MODES}

MODES = CARBON_MODES + TMD_MODES


def family_of(mode: str) -> str:
    """Which material family a mode belongs to."""
    for family, modes in FAMILIES.items():
        if mode in modes:
            return family
    raise ValueError(f"Unknown mode {mode!r}.")

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
    from .tmd import (
        build_tmd_bulk,
        build_tmd_coil,
        build_tmd_layers,
        build_tmd_nanotube,
        build_tmd_ribbon,
    )

    builders = {
        "TMD layers": build_tmd_layers,
        "TMD bulk": build_tmd_bulk,
        "TMD ribbon": build_tmd_ribbon,
        "TMD nanotube": build_tmd_nanotube,
        "TMD coil": build_tmd_coil,
        "capped tube": build_capped_cnt,
        "coil (relaxed)": build_coil,
        "fullerene": build_fullerene,
        "nano-onion": build_nano_onion,
        "junction": build_junction,
        "schwarzite": build_schwarzite,
        "multi-wall": build_multiwall_cnt,
        "bundle": build_bundle,
    }
    if job.mode in TMD_MODES:
        # The TMD builders are deterministic -- exact crystallography, no
        # random defect placement -- so they take no seed, and passing one
        # would be a TypeError rather than a no-op.
        return builders[job.mode](**job.params)

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

    if mode in TMD_MODES:
        return _estimate_tmd_atoms(mode, p)

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


def _estimate_tmd_atoms(mode: str, p: dict) -> int:
    """Atom counts for the dichalcogenides -- exact, all of them.

    Every TMD structure here is placed on ideal lattice sites, so the
    count is combinatorics rather than a surface-area guess: three atoms
    per formula unit, times the cells.
    """
    if mode == "TMD layers":
        return 3 * int(p.get("n_layers", 1)) * int(p.get("nx", 1)) * int(
            p.get("ny", 1)) * (2 if p.get("phase") == "1T'" else 1)
    if mode == "TMD bulk":
        repeat = {"2H": 2, "3R": 3, "AA": 1}.get(p.get("stacking", "2H"), 2)
        return 3 * repeat * int(p.get("nx", 1)) * int(p.get("ny", 1))
    if mode == "TMD ribbon":
        # `width` rows of one formula unit each, times the length repeats.
        return 3 * int(p.get("width", 6)) * int(p.get("length", 1))
    if mode in ("TMD nanotube", "TMD coil"):
        n, m = int(p.get("n", 20)), int(p.get("m", 0))
        divisor = math.gcd(2 * n + m, 2 * m + n)
        # Lattice points in one translational cell of an (n, m) tube.
        cells = 2 * (n * n + n * m + m * m) // divisor
        per_period = 3 * cells
        if mode == "TMD nanotube":
            return per_period * int(p.get("length", 1))
        return per_period * _coil_periods(p)
    return 0


def _coil_periods(p: dict) -> int:
    """How many tube periods a coil's helix arc length calls for.

    Mirrors the builder, which tiles one period along the axis: the
    count is what makes a coil expensive, and it grows with the coil
    radius rather than with anything the user thinks of as "size".
    """
    from .tmd.materials import get_material

    material = get_material(p.get("material", "MoS2"))
    n, m = int(p.get("n", 30)), int(p.get("m", 0))
    divisor = math.gcd(2 * n + m, 2 * m + n)
    # |axial vector| for the (n, m) cell, in the 60-degree basis.
    ca = ((2 * m + n) // divisor)
    cb = ((2 * n + m) // divisor)
    period = material.a * math.sqrt(ca * ca - ca * cb + cb * cb)
    radius = float(p.get("coil_radius", 220.0))
    pitch = float(p.get("pitch", 90.0))
    turns = float(p.get("turns", 0.5))
    arc = math.hypot(2.0 * math.pi * radius * turns, pitch * turns)
    return max(1, int(round(arc / period))) if period > 0 else 1


def estimate_cost(job: Job) -> tuple[str, str]:
    """``(severity, human text)`` for how long a job will take.

    Severity is ``"fast"``, ``"slow"`` or ``"very slow"``, meant for
    colouring a label. The split is not about atom count alone: the
    implicit modes mesh and remesh a surface before they relax it, so a
    2000-atom coil takes minutes where a 2000-atom cage takes under a
    second.
    """
    n = estimate_atoms(job)
    if job.mode == "TMD coil":
        # Still no meshing or relaxation, but a coil is not "one cell":
        # its length is the helix arc, so the atom count follows the coil
        # radius rather than anything that reads as a size, and it is the
        # one TMD mode that can reach six figures by accident.
        return ("very slow" if n > 60000 else "slow" if n > 20000 else "fast",
                f"~{n} atoms — a swept tube, so the count grows with the "
                "coil radius")
    if job.mode in TMD_MODES:
        # Exact lattice placement, no meshing and no relaxation, so these
        # are instant regardless of size; only the drawing is a cost.
        return ("slow" if n > 20000 else "fast",
                f"~{n} atoms, placed directly on lattice sites")
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
    "TMD layers": ("tmd", {
        "material": "--material", "n_layers": "--layers", "phase": "--phase",
        "stacking": "--stacking", "nx": "--nx", "ny": "--ny",
        "vacuum": "--vacuum",
    }),
    "TMD bulk": ("tmd-bulk", {
        "material": "--material", "phase": "--phase", "stacking": "--stacking",
        "nx": "--nx", "ny": "--ny",
    }),
    "TMD ribbon": ("tmd-ribbon", {
        "material": "--material", "width": "--width", "length": "--length",
        "edge": "--edge", "termination": "--termination", "phase": "--phase",
    }),
    "TMD nanotube": ("tmd-tube", {
        "material": "--material", "n": "--n", "m": "--m",
        "length": "--length", "phase": "--phase",
    }),
    "TMD coil": ("tmd-coil", {
        "material": "--material", "n": "--n", "m": "--m",
        "coil_radius": "--coil-radius", "pitch": "--pitch",
        "turns": "--turns", "phase": "--phase",
        "handedness": "--handedness",
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

    if job.mode in TMD_MODES:
        # No random placement to seed and no substitutional doping yet, so
        # emitting either would be a flag the parser does not have.
        parts += ["--out", out]
        return " ".join(parts)

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
