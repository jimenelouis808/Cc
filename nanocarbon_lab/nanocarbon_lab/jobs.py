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

#: MX2 dichalcogenides, flat through curved. The curved two need even
#: rings to keep the M/X alternation, which is why they carry a `parity`
#: choice the carbon ones do not; see `tmd/curved.py`. Their defaults
#: differ on purpose: a junction is genus 0, where even rings are
#: sufficient and `split` always reaches a perfectly alternating net; a
#: schwarzite is not, so `flip` trades alternation for geometry.
TMD_MODES = (
    "TMD layers",
    "TMD bulk",
    "TMD ribbon",
    "TMD nanotube",
    "TMD coil",
    "TMD schwarzite",
    "TMD junction",
)

#: Stacks of two or more 2D layers, twisted or aligned. A third family
#: because a stack is neither carbon nor dichalcogenide -- it may be
#: either, or one of each, and its controls (twist angle, layer order)
#: belong to none of the other panels.
HETERO_MODES = (
    "twisted bilayer",
    "vdW stack",
)

FAMILIES = {
    "carbon": CARBON_MODES,
    "dichalcogenide": TMD_MODES,
    "heterostructure": HETERO_MODES,
}

MODES = CARBON_MODES + TMD_MODES + HETERO_MODES


def family_of(mode: str) -> str:
    """Which material family a mode belongs to."""
    for family, modes in FAMILIES.items():
        if mode in modes:
            return family
    raise ValueError(f"Unknown mode {mode!r}.")

# Modes that go through marching cubes + isotropic remeshing rather than a
# seed polyhedron. They cost orders of magnitude more time per atom, which
# is the single most useful thing to know before pressing Build.
IMPLICIT_MODES = frozenset({"coil (relaxed)", "junction", "schwarzite",
                            "TMD schwarzite", "TMD junction"})

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
        ``None`` or an element symbol substituted after the build. The
        host is always carbon; ``None`` is the default and means a pure
        carbon structure.
    dopant_conc
        Substitution fraction, ignored when ``dopant`` is ``None``.
    dopant_site
        Where the substitutions go: ``"random"`` anywhere, ``"pentagon"``
        on the five-membered rings that carry a curved structure's
        curvature and its reactivity, ``"edge"`` on under-coordinated
        atoms, ``"bulk"`` on fully sp2 ones. For ``"pentagon"`` the
        fraction is of the pentagon sites, not of the whole structure --
        those differ by a large factor on a long tube.
    tmd_edit
        Post-build chemistry for a dichalcogenide: ``None``, ``"janus"``,
        ``"alloy"``, ``"vacancies"`` or ``"antisites"``. The carbon
        `dopant` fields do not apply to an MX2 -- substituting a
        heteroatom for a carbon means nothing there -- so the two
        families get separate fields rather than one overloaded pair.
    tmd_edit_element
        Which species the edit introduces: the new chalcogen for a Janus
        layer, or the substituent for an alloy. Ignored by the two defect
        edits.
    tmd_edit_amount
        Fraction for an alloy, count for the two defect edits, side
        (+1 outer / -1 inner) for a Janus layer.
    seed
        RNG seed, threaded into the builder, the doping and the MX2
        chemistry.
    """

    mode: str
    params: dict[str, Any] = field(default_factory=dict)
    dopant: str | None = None
    dopant_conc: float = 0.0
    dopant_site: str = "random"
    tmd_edit: str | None = None
    tmd_edit_element: str = "Se"
    tmd_edit_amount: float = 1.0
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
    from .hetero import build_twisted_bilayer, build_vdw_stack
    from .tmd import (
        build_tmd_bulk,
        build_tmd_coil,
        build_tmd_junction,
        build_tmd_layers,
        build_tmd_nanotube,
        build_tmd_ribbon,
        build_tmd_schwarzite,
    )

    builders = {
        "twisted bilayer": build_twisted_bilayer,
        "vdW stack": build_vdw_stack,
        "TMD layers": build_tmd_layers,
        "TMD bulk": build_tmd_bulk,
        "TMD ribbon": build_tmd_ribbon,
        "TMD nanotube": build_tmd_nanotube,
        "TMD coil": build_tmd_coil,
        "TMD schwarzite": build_tmd_schwarzite,
        "TMD junction": build_tmd_junction,
        "capped tube": build_capped_cnt,
        "coil (relaxed)": build_coil,
        "fullerene": build_fullerene,
        "nano-onion": build_nano_onion,
        "junction": build_junction,
        "schwarzite": build_schwarzite,
        "multi-wall": build_multiwall_cnt,
        "bundle": build_bundle,
    }
    if job.mode in HETERO_MODES:
        # Commensurate stacking is exact crystallography, no randomness.
        return builders[job.mode](**job.params)

    if job.mode in TMD_MODES:
        # The TMD builders are deterministic -- exact crystallography, no
        # random defect placement -- so they take no seed, and passing one
        # would be a TypeError rather than a no-op. The chemistry that
        # *is* random is applied afterwards, like carbon's doping.
        atoms = builders[job.mode](**job.params)
        if job.tmd_edit:
            atoms = apply_tmd_chemistry(atoms, job)
        return atoms

    atoms = builders[job.mode](**job.params, seed=job.seed)
    if job.dopant and job.dopant_conc > 0:
        atoms = apply_doping(atoms, job)
    return atoms


#: Where a substitution may be placed. "pentagon" is the one that needs
#: ring metadata, which every mesh-based builder records and a plain
#: sheet does not.
DOPANT_SITES = ("random", "pentagon", "edge", "bulk")


def apply_doping(atoms, job: Job):
    """Substitute ``job.dopant`` into a freshly built structure.

    Split out of :func:`build` so the CLI and the GUI go through one
    placement policy rather than three. ``"pentagon"`` counts its
    fraction against the pentagon sites; the others against all carbons,
    or against the eligible pool for edge and bulk.
    """
    from .dopants import dope_directed, dope_random, dope_rings

    site = job.dopant_site
    if site not in DOPANT_SITES:
        raise ValueError(
            f"Unknown dopant_site {site!r}; expected one of {list(DOPANT_SITES)}."
        )
    if site == "random":
        return dope_random(atoms, job.dopant, job.dopant_conc, seed=job.seed)
    if site == "pentagon":
        return dope_rings(atoms, job.dopant, ring_size=5,
                          concentration=job.dopant_conc, seed=job.seed)
    # Edge and bulk take a count rather than a fraction, so turn the
    # fraction into one against that pool -- against the whole structure
    # it would mean something different for each mode.
    from .dopants.substitutional import _bulk_indices, _edge_indices

    pool = _edge_indices(atoms) if site == "edge" else _bulk_indices(atoms)
    count = max(1, int(round(job.dopant_conc * len(pool))))
    # dope_directed spells them "edges" and "bulk", which do not pluralise
    # the same way -- hence a map rather than an f-string.
    where = {"edge": "edges", "bulk": "bulk"}[site]
    return dope_directed(atoms, job.dopant, where=where,
                         count=count, seed=job.seed)


#: Post-build edits for a dichalcogenide. These are the chemistry an MX2
#: actually undergoes -- there is no such thing as substituting a
#: heteroatom for a "carbon" here -- so they are a separate axis from the
#: carbon dopants rather than more entries in the same list.
TMD_EDITS = ("janus", "alloy", "vacancies", "antisites")


def apply_tmd_chemistry(atoms, job: Job):
    """Apply one post-build edit to a dichalcogenide.

    The counterpart of :func:`apply_doping`, and split out for the same
    reason: the GUI and the CLI must not each carry their own version of
    what "40% alloy" means.

    ``tmd_edit_amount`` is deliberately one field doing three jobs -- a
    fraction for an alloy, a count for the two defect edits, a side for
    a Janus layer. The alternative is four fields of which three are
    always ignored, and the GUI would then have to show or hide them per
    edit anyway.
    """
    from .tmd.modify import alloy, antisites, chalcogen_vacancies, make_janus

    edit = job.tmd_edit
    if edit not in TMD_EDITS:
        raise ValueError(
            f"Unknown tmd_edit {edit!r}; expected one of {list(TMD_EDITS)}."
        )
    if edit == "janus":
        side = 1 if job.tmd_edit_amount >= 0 else -1
        return make_janus(atoms, chalcogen=job.tmd_edit_element, side=side)
    if edit == "alloy":
        # Which sublattice follows from the element: a chalcogen replaces
        # chalcogens and anything else replaces the metal. Asking the user
        # to say so as well would only let the two disagree.
        site = "chalcogen" if job.tmd_edit_element in ("S", "Se", "Te") else "metal"
        return alloy(atoms, job.tmd_edit_element,
                     fraction=job.tmd_edit_amount, seed=job.seed, site=site)
    if edit == "vacancies":
        return chalcogen_vacancies(atoms, n_defects=int(job.tmd_edit_amount),
                                   seed=job.seed)
    return antisites(atoms, n_defects=int(job.tmd_edit_amount), seed=job.seed)


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

    if mode in HETERO_MODES:
        return _estimate_hetero_atoms(mode, p)

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


def _estimate_hetero_atoms(mode: str, p: dict) -> int:
    """Atom counts for a stack -- exact, from the commensurate cell.

    A twist is not a free parameter: the cell holds ``m^2 + mn + n^2``
    primitive cells per layer, and that jumps from 7 at 21.8 deg to 2791
    at the magic angle, so the count is worth knowing before building.
    """
    from .hetero.moire import get_layer, nearest_commensurate

    if mode == "twisted bilayer":
        bottom = get_layer(p.get("layer", "graphene"))
        top = get_layer(p.get("top_layer") or p.get("layer", "graphene"))
        *_, cells = nearest_commensurate(float(p.get("target_angle", 5.0)),
                                         int(p.get("max_index", 40)))
        return cells * (bottom.n_sites + top.n_sites)

    names = p.get("layers", ["graphene", "graphene"])
    sites = sum(get_layer(name).n_sites for name in names)
    return sites * int(p.get("nx", 1)) * int(p.get("ny", 1))


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
    if mode == "TMD junction":
        # Same surface-area argument as the schwarzite, on a junction's
        # arms and central sphere instead of a periodic minimal surface.
        from .tmd.materials import get_material

        a = get_material(p.get("material", "MoS2")).a
        radius = float(p.get("tube_radius", 12.0))
        arm = float(p.get("arm_length", 26.0))
        n_arms = {"L": 2, "T": 3, "Y": 3, "X": 4}.get(p.get("kind", "Y"), 3)
        area = n_arms * (2 * math.pi * radius * arm
                         + 2 * math.pi * radius**2)
        # Arms bury each other where they meet, and summing them counts
        # the junction body once per arm. The shortfall grows with the
        # arm count -- measured 0.96 for an L, 0.85 for a Y, 0.73 for an
        # X -- so correct for it linearly rather than pretending the
        # arms are disjoint.
        overlap = min(1.0, max(0.5, 1.19 - 0.115 * n_arms))
        triangles = area * overlap / (math.sqrt(3.0) / 4.0 * a * a)
        crowding = 1.22 if p.get("parity", "split") == "split" else 1.0
        return int(round(1.5 * triangles * crowding))
    if mode == "TMD schwarzite":
        # Meshed, not enumerated: surface area over the area one
        # equilateral triangle of side `a` covers, then 1.5 atoms per
        # triangle (one metal or two chalcogens, half the sites each).
        from .tmd.materials import get_material

        a = get_material(p.get("material", "MoS2")).a
        cell = float(p.get("cell", 36.0))
        area = TPMS_AREA.get(p.get("kind", "primitive"), 2.345) * cell**2
        triangles = area / (math.sqrt(3.0) / 4.0 * a * a)
        # Splitting adds a vertex per repair step. How many depends on how
        # many odd vertices the remesh happened to leave, so this is the
        # one genuinely approximate factor here: measured 1.22 on Schwarz P.
        crowding = 1.22 if p.get("parity") == "split" else 1.0
        return int(round(1.5 * triangles * crowding))

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
    if job.mode in HETERO_MODES:
        # Exact lattice placement, so only the drawing costs anything --
        # but a small twist angle makes a very large cell, and that is
        # the surprise worth warning about.
        return ("very slow" if n > 30000 else "slow" if n > 8000 else "fast",
                f"~{n} atoms — a smaller twist angle means a bigger cell")
    if job.mode in ("TMD schwarzite", "TMD junction"):
        # The one TMD mode that meshes and relaxes -- twice, in fact (the
        # site net, then the atoms) -- so it is costed like the carbon
        # implicit modes rather than like its own family.
        return ("very slow" if n > 2500 else "slow",
                f"~{n} atoms, meshed and relaxed — a minute or more")
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
    "twisted bilayer": ("twist", {
        "layer": "--layer", "top_layer": "--top-layer",
        "target_angle": "--angle", "max_index": "--max-index",
        "gap": "--gap",
    }),
    "vdW stack": ("stack", {
        "layers": "--layers", "gap": "--gap", "nx": "--nx", "ny": "--ny",
    }),
    "TMD junction": ("tmd-junction", {
        "material": "--material", "kind": "--kind",
        "tube_radius": "--tube-radius", "arm_length": "--arm-length",
        "blend": "--blend", "parity": "--parity", "phase": "--phase",
    }),
    "TMD schwarzite": ("tmd-schwarzite", {
        "material": "--material", "kind": "--kind", "cell": "--cell",
        "parity": "--parity", "phase": "--phase",
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


def _tmd_edit_flags(job: Job) -> list[str]:
    """The command-line form of one dichalcogenide edit.

    Each edit gets its own flag rather than a generic
    ``--tmd-edit NAME --amount N`` pair, because the amount means a
    different thing for each and a shared flag would have to be
    documented four ways.
    """
    edit = job.tmd_edit
    if edit == "janus":
        flags = ["--janus", job.tmd_edit_element]
        if job.tmd_edit_amount < 0:
            flags += ["--janus-side", "inner"]
        return flags
    if edit == "alloy":
        return ["--alloy", job.tmd_edit_element,
                "--alloy-fraction", f"{job.tmd_edit_amount:g}",
                "--seed", str(job.seed)]
    if edit == "vacancies":
        return ["--chalcogen-vacancies", str(int(job.tmd_edit_amount)),
                "--seed", str(job.seed)]
    return ["--antisites", str(int(job.tmd_edit_amount)),
            "--seed", str(job.seed)]


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
        elif isinstance(value, (list, tuple)):
            # An nargs="+" flag such as `--layers graphene hBN graphene`.
            parts += [flag, *(str(item) for item in value)]
        else:
            parts += [flag, str(value)]

    if job.mode in TMD_MODES or job.mode in HETERO_MODES:
        # Exact crystallography in both families, so no carbon dopant and
        # nothing to seed -- emitting either would be a flag the parser
        # does not have. The MX2 chemistry *is* random, though, so it
        # carries its seed with it.
        if job.mode in TMD_MODES and job.tmd_edit:
            parts += _tmd_edit_flags(job)
        parts += ["--out", out]
        return " ".join(parts)

    if job.dopant and job.dopant_conc > 0:
        parts += ["--dopant", job.dopant, "--dopant-conc", f"{job.dopant_conc:g}"]
        if job.dopant_site != "random":
            parts += ["--dopant-site", job.dopant_site]
    parts += ["--seed", str(job.seed), "--out", out]
    return " ".join(parts)


__all__ = [
    "DOPANT_SITES",
    "TMD_EDITS",
    "IMPLICIT_MODES",
    "MODES",
    "Job",
    "apply_doping",
    "apply_tmd_chemistry",
    "build",
    "estimate_atoms",
    "estimate_cost",
    "to_cli",
]
