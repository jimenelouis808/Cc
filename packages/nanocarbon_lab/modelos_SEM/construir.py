"""Structural models for the SEM morphologies of FeSe@C at 900 and 1020 C.

Usage::

    uv run python modelos_SEM/construir.py

What this builds, and why each one is shaped the way it is:

* **A hexagonal plate of hexagonal (NiAs-type) FeSe.** The thin hexagonal
  platelets in the 900 C micrograph. A hexagonal habit is what a P6_3/mmc
  crystal gives for free, which is the user's own reason for suspecting
  this phase -- it is a hypothesis the model makes explicit, not a result.
* **A rod of tetragonal (PbO-type) FeSe.** The needles in the same
  micrograph. Built as a capsule rather than a cylinder: a cylinder's
  sharp rim puts shell carbons 1.6 A from the core and the coating
  refuses it, which is also what a real sharp edge does -- it reconstructs.
* **A cementite particle**, for the 1020 C sample, wrapped in graphite.
* **A capped nanotube**, likewise.
* **Two compositions** putting the last two together: the particle at the
  tube's tip, and the anemone. These are ILLUSTRATIONS and are labelled
  as such -- `scene.compose` places finished structures beside each other
  and does not bond them, so the tube and the particle in them are two
  objects in one file, not one relaxed structure.

Every carbon object comes in three versions, which is the comparison the
user asked for:

``_C``
    Pure carbon. The control.
``_NSe``
    Shell co-doped with N and Se, substitutional, in one pass so both
    fractions are of the same denominator.
``_NSeO``
    The same doping plus O/N/Se surface groups, grafted on the OUTER face
    only -- including a selenone (C=Se), reached by swapping S for Se in
    the shipped thiocarbonyl, so its bond length is rebuilt rather than
    copied.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from scipy.spatial import cKDTree

from nanocarbon_lab import scene
from nanocarbon_lab.builders.capped_cnt import build_capped_cnt
from nanocarbon_lab.dopants.codoping import codope
from nanocarbon_lab.exports.xyz import write_render_bundle, write_xyz
from nanocarbon_lab.functionalize.attach import functionalize
from nanocarbon_lab.functionalize.groups import GROUPS, substitute
from nanocarbon_lab.nanoparticle import shapes as shape_module
from nanocarbon_lab.nanoparticle.carve import carve_nanoparticle
from nanocarbon_lab.nanoparticle.coat import (
    GRAPHITIC_GAP,
    graphitic_coat,
    stacked_bonds,
)
from nanocarbon_lab.nanoparticle.crystal import read_crystal
from nanocarbon_lab.nanoparticle.quality import particle_quality
from nanocarbon_lab.viz.plot import save_structure_png

HERE = Path(__file__).resolve().parent
CIF = HERE / "cif"
XYZ = HERE / "xyz"
IMG = HERE / "img"
BUNDLE = HERE / "bundle"

#: Everything the build can be told, with the defaults these models
#: settled on. Exposed as command-line options rather than constants
#: because the whole point of this folder is sweeping them: a diameter,
#: a length or a dopant fraction should not need the source edited.
DEFAULTS: dict[str, Any] = {
    # -- the hexagonal plate (900 C) --
    "placa_radio": 30.0,       # Å, circumradius of the hexagon
    "placa_semigrosor": 4.5,   # Å; 3.0 is the crystal's floor, see README
    # -- the tetragonal rod (900 C) --
    "barra_radio": 6.0,        # Å
    "barra_semilargo": 40.0,   # Å
    # -- the big carbide particle (1020 C) --
    "fe3c_radio": 12.0,        # Å, SPHERE now, not an ellipsoid
    "fe3c_capas": 2, "capas": 1,
    # -- the one that goes in the tube --
    "fe3c_interior_radio": 6.0,
    # -- the iron that decorates an arm --
    # The decoration: rings of beads along each arm, so many per ring.
    # A count alone gave a random scatter; a lattice is what reads as
    # decoration.
    "fe_radio": 4.0, "fe_particulas": 34, "fe_contacto": 2.8,
    "barras": 16,              # how many rods in the field
    "barras_separacion": 0.72, # x the rod's own length between them
    "barras_holgura": 2.6,     # A a rod must clear its neighbours by
    "barras_intentos": 60,     # draws per rod before the best is kept
    "anemonas_separacion": 1.9,  # x the arm's span between centres
    # -- the arm --
    "tubo_freq": 5,            # radius = 1.96 * freq Å, quantised
    "tubo_anillos": 12,
    "tubo_curva": 0.5,         # `waviness` of the arc centreline
    # -- the anemone --
    "brazos": 8,          # 8 x 3726 = 29 808 atomos, bajo el techo de 32 k
    "brazos_separacion": 2.0,  # tube radii out from the centre
    # -- chemistry --
    "n_frac": 0.06, "se_frac": 0.025,
    # -- the coating --
    "hueco": 3.45,             # Å
    # 3.0, and the trade is deliberate. The anchor holds the wall on
    # its own envelope, which is what makes it look like graphene
    # rather than a lumpy sack; it costs bond length. Measured on the
    # rod, roughness RMS against the envelope and the worst bond:
    #     anchor 0.0 -> RMS 0.580, bond 1.523 (sp2 OK)
    #     anchor 1.0 -> RMS 0.381, bond 1.592
    #     anchor 3.0 -> RMS 0.335, bond 1.656
    #     anchor 8.0 -> the builder refuses it outright
    # A 42 % smoother wall for bonds outside the sp2 range. These are
    # pictures, and that is the choice the pictures asked for -- the
    # run says so at the end, every time.
    "ancla": 3.0,
    # Annealing is OFF and stays off: measured on the rod at anchor 3,
    # 40 sweeps move the roughness from 0.335 to 0.327 and the
    # non-hexagon share from 8.2 % to 8.6 %. It buys two per cent and
    # costs a pass.
    "recocido": 0,
    "rejilla": 128, "remallado": 60, "relajacion": 8000,
    "semilla": 20251031,
}

#: Substitutional co-doping of the carbon, as the user's material is:
#: N and Se together, each fraction against the ORIGINAL carbon count
#: and each under its own ceiling in `dopants.chemistry` (N 0.20,
#: Se 0.03 -- Se is the binding one).

#: "avoid", so no two heteroatoms end up bonded. N and Se are not a
#: donor-acceptor pair like B-N: there is no reason for them to sit
#: together, and letting them would concentrate the strain Se already
#: brings (it is 48 % larger than carbon and puckers its site).
AFFINITY = "avoid"

#: Surface groups for the third version, in the order they are grafted.
#: Coverage is deliberately low and falls along the list: each pass sees
#: the sites the previous ones left, and a real oxidised carbon carries
#: a few per cent of groups, not a monolayer.
GRAFTS = [
    ("hydroxyl", 0.05),
    ("carboxyl", 0.03),
    ("carbonyl", 0.03),
    ("amine", 0.03),
    ("selenol", 0.02),
    ("selenona", 0.02),
]

#: The two groups that are not in the catalogue, built by element swap
#: from ones that are. `substitute` keeps the angles and REBUILDS the
#: bond lengths, so the C-Se of the selenone comes out at selenium's
#: length and not sulfur's.
EXTRA_GROUPS = {
    "selenol": ("hydroxyl", {"O": "Se"}),
    "selenona": ("thiocarbonyl", {"S": "Se"}),
}

VACUUM = 10.0

#: Above this many atoms `record` writes no matplotlib view.
#:
#: The 3D viewer costs seconds per thousand atoms and it is the SCATTER
#: that costs them, not the bonds -- a 3055-atom arm takes ~18 s even
#: with every bond in one collection, and a 55 000-atom composition did
#: not finish in fifteen minutes. It is a convenience for looking at
#: one small structure, and for anything this size the contact sheets
#: (`contacto.py`, a flat orthographic projection, seconds for all
#: nineteen) and the camera previews (`camara.py`) are both faster and
#: more useful. So the big ones simply do not get one, and the run says
#: so rather than appearing to hang.
MAX_VIEW_ATOMS = 4000

#: Filled by :func:`main` from the command line.
OPT: dict[str, Any] = dict(DEFAULTS)


def group_for(name: str):
    """A shipped group, or one built from a shipped group by swap."""
    if name in EXTRA_GROUPS:
        base, swaps = EXTRA_GROUPS[name]
        return substitute(GROUPS[base], swaps, name=name)
    return GROUPS[name]


def core_from(cif: Path, shape: str, params: dict, centre: str = "cell",
              round_by: float = 0.0):
    """Carve one inorganic core and return it with its report.

    ``round_by`` rounds the cut's edges by that radius, by shrinking the
    shape and offsetting it back out -- `offset` on a polyhedron rounds
    rather than grows, which is what the shapes module is for. It is not
    cosmetic: a sharp rim leaves the graphitic envelope nowhere to go,
    and on the thin plate it put shell carbons **0.91 Å** from the core,
    which is not a distance any pair of atoms has. A real platelet edge
    reconstructs; this model cannot, so the honest move is to stop
    asking the coating to wrap a knife edge.
    """
    crystal, crystal_report = read_crystal(cif)
    cut = shape_module.make_shape(shape, **params)
    if round_by > 0.0:
        shrunk = {key: (value - round_by if isinstance(value, (int, float))
                        and key != "sides" else value)
                  for key, value in params.items()}
        cut = shape_module.offset(shape_module.make_shape(shape, **shrunk),
                                  round_by)
    core, carve_report = carve_nanoparticle(
        crystal, cut, centre=centre, trim_loose=True,
        stoichiometric=True, vacuum=VACUUM)
    verdict, why = particle_quality(carve_report)
    return core, crystal_report, carve_report, (verdict, why)


def dress(carbon: Atoms, version: str) -> Atoms:
    """Apply one version's chemistry to a finished carbon object.

    The three versions differ only here, which is the point of having
    them: anything that changes between two of them is the doping or the
    groups and never the geometry underneath.
    """
    if version == "C":
        return carbon
    out = codope(carbon, [("N", OPT["n_frac"]), ("Se", OPT["se_frac"])],
                 affinity=AFFINITY, seed=OPT["semilla"])
    if version == "NSe":
        return out
    for index, (name, coverage) in enumerate(GRAFTS):
        try:
            out = functionalize(out, group_for(name), coverage=coverage,
                                face="outer", seed=OPT["semilla"] + index)
        except ValueError as exc:
            print(f"      {name}: no se injertó ({exc})")
    return out


def record(atoms: Atoms, label: str, notes: dict) -> dict:
    """Write the xyz and a PNG, and collect what the structure is."""
    XYZ.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    BUNDLE.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    comment = f"{label} | " + " ".join(f"{k}{v}" for k, v in sorted(counts.items()))
    write_xyz(atoms, XYZ / f"{label}.xyz", comment=comment)
    # And the render bundle: the same atoms plus the bonds and the ring
    # sizes, which is what the Blender pipeline reads. Written for every
    # model rather than for the few that get rendered, because which
    # ones those are is a decision taken later and a sidecar is cheap.
    write_render_bundle(atoms, BUNDLE / label, cif=False)
    # Two views, because one does not show a habit. The oblique one is
    # how a particle is usually drawn; the one down the c axis is where
    # a hexagonal plate looks hexagonal and a rod looks round, which is
    # the whole question these models were built to put side by side.
    if len(atoms) <= MAX_VIEW_ATOMS:
        for suffix, view in (("", (22.0, -60.0)), ("_eje", (90.0, -90.0))):
            save_structure_png(atoms, IMG / f"{label}{suffix}.png",
                               figsize=(5.5, 5.5), dpi=130, view=view)
    else:
        print(f"    (sin vista 3D: {len(atoms)} átomos pasan de "
              f"{MAX_VIEW_ATOMS}; mira contacto.py y camara.py)")
    row = {"modelo": label, "atomos": len(atoms),
           "formula": atoms.get_chemical_formula(), **notes}
    print(f"  {label:26s} n={len(atoms):5d}  {atoms.get_chemical_formula()}")
    return row


def coated_particle(label: str, cif: Path, shape: str, params: dict,
                    shells: int, gap: float, rows: list,
                    round_by: float = 0.0) -> dict:
    """One inorganic core in its graphitic shell, in the three versions.

    The core is carved once and the shells are built once; only the
    chemistry on the carbon changes between versions. Doing it the other
    way -- three independent builds -- would let the meshing differ
    between them and make the comparison meaningless.
    """
    print(f"\n{label}")
    t0 = time.time()
    core, crystal_report, carve_report, (verdict, why) = core_from(
        cif, shape, params, round_by=round_by)
    print(f"  núcleo: {core.get_chemical_formula()} "
          f"({len(core)} átomos, {time.time() - t0:.1f} s) -> {verdict}")
    print(f"    {why}")
    for warning in carve_report.warnings:
        print(f"    aviso: {warning}")

    # A stiff anchor is what makes the wall flat, and past some
    # stiffness the builder refuses the result outright -- it raises
    # rather than returning, so an unguarded call loses the whole run
    # over one family. Step down until one holds, and say which.
    shell_list = coat_report = None
    for anchor in sorted({OPT["ancla"], 1.0, 0.0}, reverse=True):
        if anchor > OPT["ancla"]:
            continue
        try:
            shell_list, coat_report = graphitic_coat(
                core, gap=gap, shells=shells, vacuum=VACUUM,
                seed=OPT["semilla"], wall_anchor=anchor, roughness=0.0,
                grid_resolution=OPT["rejilla"],
                remesh_iterations=OPT["remallado"],
                relax_iterations=OPT["relajacion"],
                anneal_sweeps=OPT["recocido"])
        except RuntimeError as exc:
            print(f"    ancla {anchor} rechazada ({str(exc)[:60]}…)")
            continue
        if anchor != OPT["ancla"]:
            print(f"    *** ancla bajada a {anchor}: la pedida "
                  f"({OPT['ancla']}) no daba una pared construible, "
                  "así que esta coraza es más rugosa que las demás")
        break
    if shell_list is None:
        raise RuntimeError(f"{label}: ninguna rigidez de ancla construye "
                           "una pared aquí")
    print(f"  coraza: {sum(len(s) for s in shell_list)} C en {shells} capa(s), "
          f"{time.time() - t0:.1f} s")
    for warning in coat_report.warnings:
        print(f"    aviso: {warning}")
    for verdict in coat_report.verdict:
        if "broken" in verdict:
            BROKEN.append((label, verdict))
            print(f"    *** CORAZA ROTA: {verdict}")

    # The bare core as well. Once the shell is on, the habit the whole
    # model is a hypothesis about is invisible -- the coating's envelope
    # is an offset of the solid and rounds every corner by the gap -- so
    # the picture that can be compared with the micrograph is this one.
    core.info["solid_indices"] = list(range(len(core)))
    rows.append(record(core, f"{label}_nucleo_desnudo", {
        "familia": label, "version": "núcleo sin recubrir",
        "nucleo": core.get_chemical_formula(), "capas_grafiticas": 0,
        "veredicto_nucleo": f"{verdict}: {why}",
        "veredicto_coraza": "—",
    }))

    out: dict = {}
    for version in ("C", "NSe", "NSeO"):
        dressed = [dress(shell, version) for shell in shell_list]
        whole = core.copy()
        for shell in dressed:
            whole += shell
        whole.set_cell(core.get_cell())
        whole.set_pbc(False)
        # The shells' own bonds, shifted to where each shell starts.
        # Without this the combined particle carries no bond list and
        # the renderer draws it as a cloud of atoms with no graphitic
        # mesh -- which is exactly how the plate and the rod came out
        # beside a nanotube that drew correctly.
        whole.info["bonds"] = stacked_bonds(dressed, len(core))
        # Which atoms are the SOLID. A region and not an element: the
        # same selenium is in the FeSe core, where it is part of a
        # solid, and substituted into the shell, where it is one atom
        # in a lattice. The core is written first, so it is the first
        # `len(core)` indices.
        whole.info["solid_indices"] = list(range(len(core)))
        rows.append(record(whole, f"{label}_{version}", {
            "familia": label, "version": version,
            "nucleo": core.get_chemical_formula(),
            "capas_grafiticas": shells,
            "veredicto_nucleo": f"{verdict}: {why}",
            "veredicto_coraza": "; ".join(coat_report.verdict),
        }))
        out[version] = whole
    return out


def coated_tube(label: str, rows: list,
                record_rows: bool = True) -> dict:
    """A capped nanotube, in the three versions.

    No core: this is the tube itself. The particle it grows from is a
    separate object and the two only meet in a composition, because they
    are two structures and pretending otherwise is what a scene is for.
    """
    print(f"\n{label}")
    t0 = time.time()
    tube = build_capped_cnt(n_body_rings=OPT["tubo_anillos"],
                            freq=OPT["tubo_freq"], shape="arc",
                            waviness=OPT["tubo_curva"],
                            vacuum=VACUUM, seed=OPT["semilla"])
    geometry = tube.info.get("geometry", {})
    print(f"  tubo: {len(tube)} C, radio {tube.info.get('radius', 0):.2f} Å, "
          f"{time.time() - t0:.1f} s")
    print(f"    enlaces {geometry.get('bond_min', 0):.3f}–"
          f"{geometry.get('bond_max', 0):.3f} Å")
    out: dict = {}
    for version in ("C", "NSe", "NSeO"):
        whole = dress(tube, version)
        out[version] = whole
        if not record_rows:
            continue
        rows.append(record(whole, f"{label}_{version}", {
            "familia": label, "version": version, "nucleo": "—",
            "capas_grafiticas": 0,
            "veredicto_nucleo": "—",
            "veredicto_coraza": tube.info.get("quality_verdict", "—"),
        }))
    return out


def _tip_frame(tube: Atoms,
               outward: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Where the tube's far cap is, and which way it points there.

    Measured from the atoms rather than assumed to be +z, because the
    arm is an arc: its tip has moved sideways and its axis has turned.
    The cap centre is the centroid of the atoms past the 95th percentile
    along the tube's own long axis, and the direction is that centre
    minus the centroid of the slab just behind it.
    """
    positions = np.asarray(tube.positions, dtype=float)
    centred = positions - positions.mean(axis=0)
    # The long axis is the first principal direction, not z.
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    along = centred @ axis
    if along.max() < -along.min():      # point it at the longer end
        axis, along = -axis, -along
    # The arm is placed radially outward from the anemone's centre
    # along +z of its own frame, so its OUTWARD tip is the one at
    # positive z. The principal axis alone does not know which that is.
    if outward and (axis @ np.array([0.0, 0.0, 1.0])) < 0.0:
        axis, along = -axis, -along
    cap = positions[along >= np.percentile(along, 95)].mean(axis=0)
    behind = positions[(along >= np.percentile(along, 70))
                       & (along < np.percentile(along, 90))].mean(axis=0)
    direction = cap - behind
    return cap, direction / np.linalg.norm(direction)


def arm_with_particle_inside(tube: Atoms, particle: Atoms,
                             outward: bool = True) -> Atoms:
    """One arm: the carbide sitting INSIDE the tube's tip.

    This is the other reading of the micrograph, and it is the one the
    user asked for. It is still an assembly and not a relaxed interface
    -- the particle is placed, not bonded -- but unlike the two scenes
    it is placed against a MEASURED clearance rather than by eye: the
    particle is pulled back along the tube's own axis until its far end
    clears the cap by the graphitic gap, and the build refuses outright
    if it does not fit across the tube.
    """
    cap, direction = _tip_frame(tube, outward=outward)
    local = np.asarray(particle.positions, dtype=float)
    local = local - local.mean(axis=0)
    half_length = float(np.abs(local @ direction).max())
    # How far the particle reaches ACROSS the axis, which is what has to
    # clear the wall.
    across = local - np.outer(local @ direction, direction)
    reach = float(np.linalg.norm(across, axis=1).max())
    clearance = float(tube.info.get("radius", 0.0)) - GRAPHITIC_GAP
    if reach > clearance:
        raise ValueError(
            f"la partícula llega a {reach:.2f} Å del eje y el tubo sólo "
            f"deja {clearance:.2f} Å: no cabe dentro. Sube "
            f"`--tubo-freq` o baja `--fe3c-interior-radio`.")
    centre = cap - direction * (half_length + GRAPHITIC_GAP)
    placed = particle.copy()
    placed.positions = local + centre
    arm = tube.copy()
    arm += placed
    arm.info = dict(tube.info)
    arm.info["particle_clearance"] = clearance - reach
    arm.info["solid_indices"] = list(range(len(tube), len(arm)))
    print(f"    partícula dentro: llega a {reach:.2f} Å del eje, el tubo "
          f"deja {clearance:.2f} Å ({clearance - reach:.2f} Å de holgura)")
    return arm


def compositions(tube: Atoms, particle: Atoms, pieces: dict,
                 rows: list) -> None:
    """The cover scenes. ILLUSTRATIONS, every one of them.

    `scene.compose` places finished structures beside each other and
    does not bond them, so each of these is several objects in one
    coordinate system: right for a figure, and not something to relax
    or hand to a plane-wave code. The arm below is the exception only
    in that its placement is MEASURED rather than done by eye.
    """
    print("\ncomposiciones (ilustraciones, no estructuras)")
    rng = np.random.default_rng(OPT["semilla"])
    arm = arm_with_particle_inside(tube, particle)
    radius = float(tube.info.get("radius", 8.0))
    rows.append(record(arm, "brazo_particula_dentro", {
        "familia": "escena", "version": "ilustración", "nucleo": "Fe3C",
        "capas_grafiticas": 0, "veredicto_nucleo": "—",
        "veredicto_coraza": "ensamblaje: la partícula está colocada, no "
                            "enlazada; holgura medida"}))

    def finish(items, label, note):
        composed, report = scene.compose(items, vacuum=VACUUM, seams="drop")
        rows.append(record(composed, label, {
            "familia": "escena", "version": "ilustración",
            "nucleo": "—", "capas_grafiticas": 0,
            "veredicto_nucleo": "—", "veredicto_coraza": note}))
        # Which item each atom belongs to, into the sidecar. Without
        # it the whole composition is one object in Blender and no
        # plate, rod or arm can be moved on its own -- which is what
        # `render_cnt --split-items` reads to take them apart.
        sidecar = BUNDLE / f"{label}.json"
        carried = json.loads(sidecar.read_text())
        carried["scene"] = composed.info.get("scene", {})
        carried["item_per_atom"] = [int(t) for t in composed.get_tags()]
        carried["is_illustration"] = True
        sidecar.write_text(json.dumps(carried, indent=2))
        for warning in report.warnings:
            if "ILLUSTRATION" not in warning:
                print(f"    aviso: {warning}")

    # 1. One anemone.
    finish(anemone_items(arm, OPT["brazos"], radius), "escena_anemona",
           f"composición: {OPT['brazos']} brazos")

    # 2. A FIELD OF RODS, which is what the 900 C micrograph mostly is:
    # needles lying every which way. Orientations are uniform on the
    # sphere and positions jittered on a grid, so they read as a heap
    # rather than as a pattern -- the grid alone looks like a crystal
    # of rods, and pure random placement makes them pile up.
    rod = pieces.get("barra_T")
    if rod is not None:
        # Below one rod length the rods interleave rather than sit in
        # cells, which is what the micrograph's heap looks like. The
        # composition measures every pair and reports anything that
        # touches, so this can be tightened by eye and checked.
        span = OPT["barras_separacion"] * float(
            max(np.ptp(rod.positions[:, k]) for k in range(3)))
        items = []
        side = int(np.ceil(np.sqrt(OPT["barras"])))
        # Each rod is DRAWN AND MEASURED before it is kept. Tuning the
        # span and the jitter by hand against the composition's pair
        # report got the field from two interpenetrating pairs (0.26 and
        # 0.98 A) down to one (0.38 A) and no further: with random
        # orientations at this density some draw always lands on a
        # neighbour, and widening the field is the one thing asked
        # against. So the draw is rejected instead, which keeps the
        # spacing and removes the collisions outright.
        placed: list[np.ndarray] = []
        for index in range(OPT["barras"]):
            row, column = divmod(index, side)
            best: tuple[float, scene.Item] | None = None
            for _ in range(OPT["barras_intentos"]):
                jitter = rng.uniform(-0.12, 0.12, size=3) * span
                candidate = scene.Item(
                    rod, label=f"barra{index}",
                    rotation=tuple(rng.uniform(0.0, 360.0, size=3)),
                    translation=(float((column - 0.5 * (side - 1)) * span
                                       + jitter[0]),
                                 float((row - 0.5 * (side - 1)) * span
                                       + jitter[1]),
                                 float(0.35 * jitter[2])))
                here = candidate.placed().positions
                gap = (float("inf") if not placed else
                       min(float(cKDTree(other).query(here)[0].min())
                           for other in placed))
                if best is None or gap > best[0]:
                    best = (gap, candidate)
                if gap >= OPT["barras_holgura"]:
                    break
            assert best is not None
            if best[0] < OPT["barras_holgura"]:
                print(f"    barra{index}: la mejor de "
                      f"{OPT['barras_intentos']} tiradas deja "
                      f"{best[0]:.2f} Å")
            items.append(best[1])
            placed.append(best[1].placed().positions)
        finish(items, "escena_barras", f"composición: {OPT['barras']} barras")

    # 3. A POSTER LINE-UP: each piece standing, seen side on, one
    # behind the other down the viewing axis. The first version turned
    # each one to a different three-quarter angle, which reads as three
    # objects scattered in space; a line-up reads as three *characters*,
    # which is what a cover wants, and it is also the view in which a
    # plate, a rod and a tube are told apart at a glance.
    plate = pieces.get("placa_H")
    if plate is not None and rod is not None:
        reach = 0.5 * float(max(np.ptp(plate.positions[:, k])
                                for k in range(3)))
        # Depth between neighbours. They stand thin side on, so this is
        # about the silhouette overlapping by the right amount rather
        # than about clearance: the composition measures every pair and
        # the build refuses to ship one that interpenetrates.
        step = 1.90 * reach
        # The sideways stagger grows with depth, so each one appears
        # past the shoulder of the one in front instead of behind it.
        finish([
            scene.Item(plate, label="placa", rotation=standing_angles(plate),
                       translation=(0.0, -step, 0.0)),
            scene.Item(rod, label="barra", rotation=standing_angles(rod),
                       translation=(0.46 * reach, 0.0, 0.10 * reach)),
            scene.Item(arm, label="tubo", rotation=standing_angles(arm),
                       translation=(0.80 * reach, step, 0.02 * reach)),
        ], "escena_profundidad", "composición: placa, barra y tubo de pie, "
           "de perfil, uno detrás de otro")

    # 4. Several anemones.
    items = []
    arms_each = max(4, OPT["brazos"] - 2)
    cluster = OPT["anemonas_separacion"] * float(
        max(np.ptp(arm.positions[:, k]) for k in range(3)))
    for which, centre in enumerate(((0.0, 0.0, 0.0),
                                    (cluster, 0.25 * cluster, 0.4 * cluster),
                                    (0.45 * cluster, -cluster,
                                     -0.3 * cluster))):
        items += anemone_items(arm, arms_each, radius, centre=centre,
                               label=f"a{which}_")
    finish(items, "escena_anemonas", f"composición: 3 anémonas de "
           f"{arms_each} brazos")

    # 5. One anemone with iron beads on its arms -- the bright specks
    # along the tubes in the micrograph.
    bead = pieces.get("fe_decoracion")
    if bead is not None:
        dressed = decorate(arm, bead, OPT["fe_particulas"],
                           OPT["semilla"], OPT["fe_contacto"])
        rows.append(record(dressed, "brazo_decorado", {
            "familia": "escena", "version": "ilustración", "nucleo": "Fe",
            "capas_grafiticas": 0, "veredicto_nucleo": "—",
            "veredicto_coraza": f"{OPT['fe_particulas']} partículas "
                                "de Fe repartidas sobre la pared"}))
        finish(anemone_items(dressed, OPT["brazos"], radius),
               "escena_anemona_decorada",
               f"composición: {OPT['brazos']} brazos con Fe encima")


def standing_angles(atoms: Atoms) -> tuple[float, float, float]:
    """Euler angles that stand a piece up and turn it face on.

    The item's own principal axes are measured and mapped onto the
    scene's: the LONGEST onto +z, so it stands, and the SHORTEST onto
    -y, so its broad side faces a camera placed on the -y side. For a
    plate that shows the hexagon; for a rod or a tube it shows the
    length. Both are the silhouette the piece is recognised by.

    Measured rather than typed in, because each piece leaves its
    builder in whatever frame that builder uses, and three hand-tuned
    triples drift the moment one of them is rebuilt.

    The decomposition matches :func:`scene.euler_matrix`'s stated
    convention, ``R = Rz @ Ry @ Rx``, and nothing else here may assume
    a different one.
    """
    centred = (np.asarray(atoms.positions, dtype=float)
               - np.asarray(atoms.positions, dtype=float).mean(axis=0))
    axes = np.linalg.svd(centred, full_matrices=False)[2]
    long_axis, thin = axes[0], axes[2]
    thin = thin - np.dot(thin, long_axis) * long_axis
    thin /= np.linalg.norm(thin)
    source = np.column_stack([long_axis, np.cross(thin, long_axis), thin])
    if np.linalg.det(source) < 0:
        source[:, 1] *= -1.0
    target = np.column_stack([[0.0, 0.0, 1.0],      # length stands up
                              [-1.0, 0.0, 0.0],
                              [0.0, -1.0, 0.0]])    # broad side to camera
    turn = target @ source.T
    beta = np.arcsin(float(np.clip(-turn[2, 0], -1.0, 1.0)))
    alpha = np.arctan2(turn[2, 1], turn[2, 2])
    gamma = np.arctan2(turn[1, 0], turn[0, 0])
    return tuple(float(np.degrees(a)) for a in (alpha, beta, gamma))


def anemone_items(arm: Atoms, arms: int, radius: float,
                  centre=(0.0, 0.0, 0.0), label: str = "brazo",
                  spread: float = 1.0) -> list:
    """One anemone's worth of placed arms, as scene items.

    Split out of :func:`compositions` so a scene can hold several. The
    directions are Fibonacci points on the sphere, which spreads them
    evenly without the clumping a random draw gives at this count -- an
    anemone is not a flower and should not be flat.
    """
    reach = 0.5 * float(max(np.ptp(arm.positions[:, axis])
                            for axis in range(3)))
    start = OPT["brazos_separacion"] * radius + GRAPHITIC_GAP
    items = []
    for index in range(arms):
        height = 1.0 - 2.0 * (index + 0.5) / arms
        radial = float(np.sqrt(max(0.0, 1.0 - height * height)))
        angle = index * np.pi * (3.0 - np.sqrt(5.0))
        direction = np.array([radial * np.cos(angle),
                              radial * np.sin(angle), height])
        tilt = float(np.degrees(np.arccos(np.clip(direction[2], -1.0, 1.0))))
        azimuth = float(np.degrees(np.arctan2(direction[1], direction[0])))
        items.append(scene.Item(
            arm, label=f"{label}{index}", rotation=(0.0, tilt, azimuth),
            translation=tuple(np.asarray(centre, dtype=float)
                              + direction * (start + reach) * spread)))
    return items


def decorate(arm: Atoms, bead: Atoms, count: int, seed: int,
             contact: float) -> Atoms:
    """Seat metal beads over an arm's wall, spread EVENLY but not in rows.

    The bright specks along the tubes in the micrograph. They are
    PLACED, not grown: each sits on the wall at a van der Waals contact
    along the local outward normal, and nothing here computes whether
    iron would wet graphite that way.

    Three layouts were tried and the first two are visible AS layouts,
    which is the fault:

    * **A uniform random draw** clumps. At this count the gaps are as
      big as the particles and the eye finds the clumps first, so it
      reads as dirt rather than as decoration.
    * **Hoops along the barrel**, staggered half a step, fixes the
      clumping and introduces a worse artefact: the beads line up into
      visible *rings* round the tube, which the micrograph does not
      show. A lattice is even, and even is not the same as uniform.

    What the micrograph shows is **blue noise**: no clumps, no rows, no
    preferred angle. So the seats are chosen by best-candidate
    sampling -- for each new seat, ``DECORATION_CANDIDATES`` wall atoms
    are drawn at random and the one furthest from every seat already
    taken is kept.

    **The seats are atoms of the wall, not points on an analytic
    cylinder**, and that is the part that was wrong first. An arm is
    BENT: parameterising it by height and angle about its own principal
    axis is a cylinder's parameterisation, and on a curved tube it
    stretches on the outside of the bend and folds on the inside. The
    measured gaps were honest and the picture still showed clumps,
    because the gaps were being measured in the wrong space. Distances
    between atoms do not care how the tube is shaped.

    Random, so it takes a ``seed``; deterministic given one.

    The outward normal is measured per seat, for the same reason: the
    local surface normal is the smallest-variance direction of the
    atom's own neighbourhood, and its sign is fixed against the vector
    from the middle of a short length of tube, which points outward
    whatever the tube is doing globally.

    **Each bead is turned at random**, because a crystal carved out of
    a lattice has flat faces and an orientation, and thirty-four copies
    of one orientation read as a repeated stamp rather than as a
    scatter of particles. The turn is a uniform random rotation
    (Shoemake's quaternion, not three Euler angles, which bunch toward
    the poles), from the same seeded generator the seats come from.

    ``contact`` is how close a bead's nearest atom may come to the
    wall. The graphitic van der Waals contact is 3.4 A and this
    defaults to **2.8**, which is closer than that: a particle resting
    at the true contact, drawn with the metal at van der Waals radius
    and the carbon as thin ball-and-stick, reads as floating -- its
    centre ends about 6 A above a wall whose drawn surface is a thin
    mesh. This is an ILLUSTRATION and the number is a drawing decision,
    so it is a parameter with its default stated rather than a constant
    quietly below the physics.

    Beads go on the barrel only: one on a cap reads as the tip particle
    and the two should not be confused. Anything the arm already
    declared solid -- the particle in the tip -- is not wall.
    """
    positions = np.asarray(arm.positions, dtype=float)
    solid = set(int(i) for i in arm.info.get("solid_indices", []))
    wall = np.array([i for i in range(len(arm)) if i not in solid], dtype=int)
    if len(wall) < 8 * count:
        return arm

    # Caps off. The principal axis is a poor parameterisation of a bent
    # tube and a perfectly good way to find its two ends.
    centred = positions - positions[wall].mean(axis=0)
    axis = np.linalg.svd(centred[wall], full_matrices=False)[2][0]
    along = centred @ axis
    low, high = along[wall].min(), along[wall].max()
    margin = 0.10 * (high - low)
    free = wall[(along[wall] > low + margin) & (along[wall] < high - margin)]
    if len(free) < 8 * count:
        free = wall

    rng = np.random.default_rng(seed)
    seats: list[int] = []
    for _ in range(count):
        draw = rng.choice(free, size=DECORATION_CANDIDATES, replace=False)
        if not seats:
            pick = int(draw[0])
        else:
            taken = positions[np.asarray(seats)]
            far = np.linalg.norm(positions[draw][:, None, :] - taken[None],
                                 axis=2).min(axis=1)
            pick = int(draw[int(np.argmax(far))])
        seats.append(pick)

    shape = np.asarray(bead.positions, dtype=float)
    shape = shape - shape.mean(axis=0)
    tree = cKDTree(positions[wall])
    out = arm.copy()
    for index in seats:
        local = shape @ _random_rotation(rng).T
        # The patch that gives the normal, and the longer one that
        # gives its sign. Both are counts rather than radii, so neither
        # needs the tube's diameter to be known or constant.
        patch = positions[wall][tree.query(positions[index], k=30)[1]]
        normal = np.linalg.svd(patch - patch.mean(axis=0),
                               full_matrices=False)[2][2]
        segment = positions[wall][tree.query(positions[index], k=400)[1]]
        outward = positions[index] - segment.mean(axis=0)
        if float(np.dot(normal, outward)) < 0.0:
            normal = -normal
        placed = bead.copy()
        placed.positions = local + positions[index] + normal * _seating(
            local, positions[index], normal, tree, contact)
        out += placed
    out.info = dict(arm.info)
    out.info["solid_indices"] = (list(arm.info.get("solid_indices", []))
                                 + list(range(len(arm), len(out))))
    beads = np.asarray(out.positions[len(arm):], dtype=float)
    print(f"    decoración: {count} partículas giradas al azar, separación "
          f"mínima entre centros "
          f"{_closest_seat(positions[np.asarray(seats)]):.1f} Å, "
          f"la más hundida a {tree.query(beads)[0].min():.2f} Å de la pared "
          f"(contacto pedido {contact:.1f} Å)")
    return out


def _seating(local: np.ndarray, seat: np.ndarray, normal: np.ndarray,
             wall: cKDTree, contact: float, ceiling: float = 12.0) -> float:
    """How far along the normal a bead has to go to rest on the wall.

    The smallest push for which NO atom of the bead comes closer to the
    wall than ``contact``, found by bisection against the wall's own
    tree. Bisection is valid because the distance from the
    bead to the wall rises monotonically once the bead is outside.

    Two cheaper rules were tried first and both leave beads floating,
    which the render showed before any number did:

    * **Half the bead's longest extent, plus the gap.** A bead whose
      nearest-to-wall atom is a corner ends up to 3 Å too far out.
    * **The lowest atom ALONG THE NORMAL, plus the gap.** Better, and
      still wrong by the sideways part: that atom is at ``gap`` along
      the normal, so its real distance is ``hypot(gap, sideways)`` --
      measured, up to 5.65 Å against a 3.4 Å target.

    Measuring against the wall rather than against the seat atom also
    takes the curvature in: on the inside of the bend the wall comes up
    to meet the bead, and a push computed from one atom does not know.
    """
    low, high = 0.0, ceiling
    for _ in range(40):
        middle = 0.5 * (low + high)
        here = local + seat + normal * middle
        if float(wall.query(here)[0].min()) < contact:
            low = middle
        else:
            high = middle
    return high


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniform random rotation matrix.

    Shoemake's quaternion method, which samples SO(3) EVENLY. Three
    random Euler angles do not: they bunch toward the poles, so a field
    of particles turned that way still shares a preferred axis, which
    is the artefact the random turn exists to remove.
    """
    u1, u2, u3 = rng.random(3)
    root, other = np.sqrt(1.0 - u1), np.sqrt(u1)
    x = root * np.sin(2.0 * np.pi * u2)
    y = root * np.cos(2.0 * np.pi * u2)
    z = other * np.sin(2.0 * np.pi * u3)
    w = other * np.cos(2.0 * np.pi * u3)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _closest_seat(seats: np.ndarray) -> float:
    """The smallest distance between two seats, in Å.

    Reported rather than assumed: blue noise is a claim about this
    number, and a sampler that quietly degenerates to a random draw
    would look fine in the log without it.
    """
    if len(seats) < 2:
        return float("inf")
    gap = np.linalg.norm(seats[:, None, :] - seats[None, :, :], axis=2)
    np.fill_diagonal(gap, np.inf)
    return float(gap.min())


#: Draws per seat in the decoration's best-candidate sampler. More is
#: more even and the return falls off fast; 24 is where the measured
#: minimum gap stops improving by more than a few per cent.
DECORATION_CANDIDATES = 24

#: Shells whose carbon-carbon bonds left the sp2 range, collected so
#: the run ENDS by saying so.
#:
#: This exists because it already went wrong once and silently: the
#: per-model lines scroll past, the verdict was only in the JSON, and
#: nine models were written and handed over with walls carrying bonds
#: of 1.59-1.72 A against the ceiling of 1.55. A quality gate that
#: nobody reads is not a gate.
BROKEN: list[tuple[str, str]] = []


def wanted(name: str) -> bool:
    """Is this family in the run? ``--solo`` narrows it to one.

    Worth having because the families cost very different amounts: the
    coated particles are a minute each and the tube and the scenes are
    seconds, so iterating on the anemone should not rebuild the plate.
    """
    return not OPT["solo"] or name.startswith(OPT["solo"])


def main() -> None:
    rows: list = []
    # The finished objects the scenes are built from, kept as they are
    # made so a scene does not rebuild them.
    pieces: dict[str, Atoms] = {}

    if wanted("placa_H"):
        # The THICKNESS is set by the coating, not by the crystal: at
        # 6.0 A -- one P6_3/mmc cell, the thinnest stoichiometric slab
        # there is -- the shell comes back broken at every gap and
        # anchor tried, because a 6 A rim is finer than the wall that
        # would wrap it. In a real sample that edge is where the
        # coating is defective or simply open.
        # --capas stays 1 here, and that is measured rather than
        # chosen. Two shells is what makes the Fe3C sphere read as
        # coated -- 0 % of its core atoms poke through against 1-2 %
        # for these two -- so it was the obvious thing to try. On the
        # rod it TEARS: both shells came back broken, with 1.66 and
        # 1.69 Å bonds against the 1.55 ceiling, because a second
        # shell sits 3.4 Å further out over a rim that is already the
        # tightest curvature in the model. A compact sphere takes a
        # second shell; an 85 x 9.4 Å capsule does not. Widening the
        # rod would allow it and is a different model.
        pieces["placa_H"] = coated_particle(
            "placa_H", CIF / "FeSe_hexagonal.cif", "prism",
                        {"radius": OPT["placa_radio"],
                         "half_height": OPT["placa_semigrosor"],
             "sides": 6},
            shells=OPT["capas"], gap=OPT["hueco"], rows=rows)["C"]

        # And the thinnest plate the CRYSTAL allows, bare. Half a cell
        # terminates Se-Se and comes out Fe110Se200, which is not FeSe
        # any more, so 6.0 A is the floor. This is the one to compare
        # against the micrograph.
        thin, _, _, (verdict, why) = core_from(
            CIF / "FeSe_hexagonal.cif", "prism",
            {"radius": OPT["placa_radio"], "half_height": 3.0, "sides": 6})
        thin.info["solid_indices"] = list(range(len(thin)))
        print(f"\nplaca delgada desnuda: {thin.get_chemical_formula()} "
              f"({len(thin)} átomos, "
              f"{np.ptp(thin.positions[:, 2]):.1f} Å de grosor) -> {verdict}")
        rows.append(record(thin, "placa_H_delgada_desnuda", {
            "familia": "placa_H",
            "version": "la más delgada posible, sin cubrir",
            "nucleo": thin.get_chemical_formula(), "capas_grafiticas": 0,
            "veredicto_nucleo": f"{verdict}: {why}",
            "veredicto_coraza": "—"}))

    if wanted("barra_T"):
        # A capsule and not a cylinder: a cylinder's sharp rim leaves
        # the envelope nowhere to go and the coating refuses it, which
        # is also what a real sharp edge does -- it reconstructs.
        pieces["barra_T"] = coated_particle(
            "barra_T", CIF / "FeSe_tetragonal.cif", "rod",
            {"radius": OPT["barra_radio"],
             "half_length": OPT["barra_semilargo"]},
            shells=OPT["capas"], gap=OPT["hueco"], rows=rows)["C"]

    if wanted("particula_Fe3C"):
        coated_particle("particula_Fe3C", CIF / "Fe3C_cementita.cif",
                        "sphere", {"radius": OPT["fe3c_radio"]},
                        shells=OPT["fe3c_capas"], gap=OPT["hueco"], rows=rows)

    # The small one, bare, for putting inside a tube. No graphitic shell
    # on it: inside the tube the TUBE is the graphite, and a shell would
    # be a second wall 3.35 A from the first with nothing between them.
    inside, _, _, (verdict, why) = core_from(
        CIF / "Fe3C_cementita.cif", "sphere",
        {"radius": OPT["fe3c_interior_radio"]})
    inside.info["solid_indices"] = list(range(len(inside)))
    if wanted("particula_Fe3C"):
        print(f"\npartícula interior: {inside.get_chemical_formula()} "
              f"({len(inside)} átomos) -> {verdict}")
        rows.append(record(inside, "particula_Fe3C_interior", {
            "familia": "particula_Fe3C",
            "version": "para ir dentro del tubo",
            "nucleo": inside.get_chemical_formula(),
            "capas_grafiticas": 0,
            "veredicto_nucleo": f"{verdict}: {why}",
            "veredicto_coraza": "—"}))

    # The iron that decorates an arm. alpha-Fe, carved as a bead; it is
    # PLACED on the wall, not grown there, and the scene says so.
    bead, _, _, (bead_verdict, _) = core_from(
        CIF / "Fe_alfa.cif", "sphere", {"radius": OPT["fe_radio"]})
    bead.info["solid_indices"] = list(range(len(bead)))
    pieces["fe_decoracion"] = bead
    print(f"\npartícula de Fe decorativa: {len(bead)} átomos "
          f"({np.ptp(bead.positions, axis=0).max():.1f} Å) -> {bead_verdict}")

    tubes = coated_tube("nanotubo", rows, record_rows=wanted("nanotubo"))
    if wanted("escena"):
        # A scene needs pieces this run may not have built. Read them
        # back from disk rather than rebuilding: `--solo escena` is for
        # iterating on a composition, and a minute per coated particle
        # is the whole reason that flag exists. The `solid` list comes
        # back from the sidecar, so the recovered piece still knows
        # which of its atoms are the core.
        for name in ("placa_H", "barra_T"):
            if name in pieces:
                continue
            path = XYZ / f"{name}_C.xyz"
            if not path.exists():
                print(f"    {name}: sin construir y sin archivo, "
                      "las escenas que lo usan se saltan")
                continue
            from ase.io import read as ase_read
            piece = ase_read(path)
            sidecar = BUNDLE / f"{name}_C.json"
            if sidecar.exists():
                piece.info["solid_indices"] = json.loads(
                    sidecar.read_text()).get("solid", [])
            pieces[name] = piece
            print(f"    {name}: releído de {path.name} ({len(piece)} átomos)")
        compositions(tubes["C"], inside, pieces, rows)

    out = HERE / "modelos.json"
    if OPT["solo"] and out.exists():
        # A partial run must not wipe the families it did not build.
        keep = [row for row in json.loads(out.read_text(encoding="utf-8"))
                if not any(row["modelo"] == new["modelo"] for new in rows)]
        # ...and a row it keeps has to still describe the file on disk.
        # It happened: the Fe3C family's models were rebuilt to 2838
        # atoms while its row went on saying 2770, because a run that
        # did not build that family carried the old row forward. The
        # structures were right the whole time and the TABLE lied,
        # which is worse than either being wrong on its own.
        for row in keep:
            path = XYZ / f"{row['modelo']}.xyz"
            if not path.exists():
                continue
            on_disk = int(path.read_text(encoding="utf-8").split("\n", 1)[0])
            if on_disk != int(row["atomos"]):
                print(f"    *** {row['modelo']}: la tabla decía "
                      f"{row['atomos']} átomos y el archivo tiene "
                      f"{on_disk}; se corrige la tabla")
                row["atomos"] = on_disk
        rows = keep + rows
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n{len(rows)} modelos en {XYZ} y {IMG}")
    if BROKEN:
        print("\n" + "=" * 68)
        print(f"ATENCIÓN: {len(BROKEN)} coraza(s) fuera del rango sp². "
              "Estos modelos NO son estructuras válidas:")
        for label, verdict in BROKEN:
            print(f"  - {label}: {verdict}")
        print("Sirven como ilustración y no para relajar ni para DFT.")
        print("=" * 68)


def parse_args(argv: list[str] | None = None) -> dict:
    """Read the build's settings off the command line.

    Every entry of :data:`DEFAULTS` becomes an option of the same name
    with dashes, so adding a knob to that table is all it takes. The
    help text prints the default beside each one, because a sweep is
    only readable if the baseline is visible.
    """
    parser = argparse.ArgumentParser(
        description="Construye los modelos de FeSe@C para las "
                    "morfologías del SEM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name, value in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}",
                            dest=name, type=type(value), default=value)
    parser.add_argument("--solo", default="", metavar="NOMBRE",
                        help="construir sólo la familia cuyo nombre "
                             "empiece así (placa_H, barra_T, "
                             "particula_Fe3C, nanotubo, escena)")
    return vars(parser.parse_args(argv))


if __name__ == "__main__":
    OPT.update(parse_args())
    main()
