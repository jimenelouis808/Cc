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

from nanocarbon_lab import scene
from nanocarbon_lab.builders.capped_cnt import build_capped_cnt
from nanocarbon_lab.dopants.codoping import codope
from nanocarbon_lab.exports.xyz import write_render_bundle, write_xyz
from nanocarbon_lab.functionalize.attach import functionalize
from nanocarbon_lab.functionalize.groups import GROUPS, substitute
from nanocarbon_lab.nanoparticle import shapes as shape_module
from nanocarbon_lab.nanoparticle.carve import carve_nanoparticle
from nanocarbon_lab.nanoparticle.coat import GRAPHITIC_GAP, graphitic_coat
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
    "fe3c_a": 13.0, "fe3c_c": 18.0,   # Å, ellipsoid semi-axes
    "fe3c_capas": 2,
    # -- the one that goes in the tube --
    "fe3c_interior_a": 7.0, "fe3c_interior_c": 9.0,
    # -- the arm --
    "tubo_freq": 6,            # radius = 1.96 * freq Å, quantised
    "tubo_anillos": 10,
    "tubo_curva": 0.75,        # `waviness` of the arc centreline
    # -- the anemone --
    "brazos": 10,
    "brazos_separacion": 2.0,  # tube radii out from the centre
    # -- chemistry --
    "n_frac": 0.06, "se_frac": 0.025,
    # -- the coating --
    "hueco": 3.45,             # Å
    "ancla": 0.0,              # 0 now that the core is an obstacle
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
    for suffix, view in (("", (22.0, -60.0)), ("_eje", (90.0, -90.0))):
        save_structure_png(atoms, IMG / f"{label}{suffix}.png",
                           figsize=(5.5, 5.5), dpi=130, view=view)
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

    shell_list, coat_report = graphitic_coat(
        core, gap=gap, shells=shells, vacuum=VACUUM,
        seed=OPT["semilla"], wall_anchor=OPT["ancla"], roughness=0.0,
        grid_resolution=OPT["rejilla"],
        remesh_iterations=OPT["remallado"],
        relax_iterations=OPT["relajacion"])
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


def coated_tube(label: str, rows: list) -> dict:
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
        rows.append(record(whole, f"{label}_{version}", {
            "familia": label, "version": version, "nucleo": "—",
            "capas_grafiticas": 0,
            "veredicto_nucleo": "—",
            "veredicto_coraza": tube.info.get("quality_verdict", "—"),
        }))
        out[version] = whole
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
            f"deja {clearance:.2f} Å: no cabe dentro. Sube `TUBE_FREQ` o "
            f"encoge `CARBIDE_INSIDE`.")
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


def compositions(tube: Atoms, particle: Atoms, rows: list) -> None:
    """The arm and the anemone, as ILLUSTRATIONS.

    `scene.compose` places finished structures next to each other and
    does not bond them, so these are a tube and some particles in one
    coordinate system: right for a figure, and NOT something to relax or
    hand to a plane-wave code. The arm above is assembled directly
    rather than through `compose`, because its placement is measured.
    """
    print("\ncomposiciones (ilustraciones, no estructuras)")
    arm = arm_with_particle_inside(tube, particle)
    rows.append(record(arm, "brazo_particula_dentro", {
        "familia": "escena", "version": "ilustración", "nucleo": "Fe3C",
        "capas_grafiticas": 0, "veredicto_nucleo": "—",
        "veredicto_coraza": "ensamblaje: la partícula está colocada, no "
                            "enlazada; holgura medida",
    }))

    # The anemone: the same arm, rotated out of a common centre. The
    # arms start a tube radius out so their inner caps do not occupy the
    # same space -- five converging on one point came out 0.05 A apart,
    # which is five tubes inside each other and not an anemone.
    reach = 0.5 * float(max(np.ptp(arm.positions[:, axis])
                            for axis in range(3)))
    # Two tube diameters out, not one radius. The arms are ARCS, so
    # they bow toward each other along their length and the nearest
    # approach is mid-arm rather than at the inner caps: at one radius
    # four pairs came within 2.4-3.1 A, which on a render reads as
    # fused.
    start = (OPT["brazos_separacion"] * float(tube.info.get("radius", 8.0))
             + GRAPHITIC_GAP)
    placed = []
    for index in range(OPT["brazos"]):
        # Fibonacci directions, so the arms spread over the whole sphere
        # instead of lying in one plane: an anemone is not a flower.
        height = 1.0 - 2.0 * (index + 0.5) / OPT["brazos"]
        radial = float(np.sqrt(max(0.0, 1.0 - height * height)))
        angle = index * np.pi * (3.0 - np.sqrt(5.0))
        direction = np.array([radial * np.cos(angle),
                              radial * np.sin(angle), height])
        tilt = float(np.degrees(np.arccos(np.clip(direction[2], -1.0, 1.0))))
        azimuth = float(np.degrees(np.arctan2(direction[1], direction[0])))
        placed.append(scene.Item(
            arm, label=f"brazo{index}", rotation=(0.0, tilt, azimuth),
            translation=tuple(direction * (start + reach))))
    anemone, report = scene.compose(placed, vacuum=VACUUM, seams="drop")
    rows.append(record(anemone, "escena_anemona", {
        "familia": "escena", "version": "ilustración", "nucleo": "Fe3C",
        "capas_grafiticas": 0, "veredicto_nucleo": "—",
        "veredicto_coraza": f"composición: {OPT['brazos']} brazos, sin enlazar",
    }))
    for warning in report.warnings:
        print(f"    aviso: {warning}")


#: Shells whose carbon-carbon bonds left the sp2 range, collected so
#: the run ENDS by saying so.
#:
#: This exists because it already went wrong once and silently: the
#: per-model lines scroll past, the verdict was only in the JSON, and
#: nine models were written and handed over with walls carrying bonds
#: of 1.59-1.72 A against the ceiling of 1.55. A quality gate that
#: nobody reads is not a gate.
BROKEN: list[tuple[str, str]] = []


def main() -> None:
    rows: list = []

    # 8.9 A thick and 58 A across: 6.5 : 1, against the first version's
    # 3 : 1. What sets that floor is the COATING and not the crystal,
    # which is worth knowing before anyone asks for thinner: at 6.0 A
    # -- one P6_3/mmc cell, the thinnest stoichiometric slab there is --
    # the graphitic shell comes back BROKEN at every gap and anchor
    # tried (bonds to 1.73-1.76 A against the sp2 ceiling of 1.55), and
    # the builder's own message says why: "the surface has features
    # finer than a carbon ring". A 6 A rim is finer than the wall that
    # would have to wrap it. In a real sample that edge is where the
    # coating is defective or simply open.
    coated_particle("placa_H", CIF / "FeSe_hexagonal.cif", "prism",
                    {"radius": OPT["placa_radio"],
                     "half_height": OPT["placa_semigrosor"], "sides": 6},
                    shells=1, gap=OPT["hueco"], rows=rows)

    # And the thinnest plate the CRYSTAL allows, bare: one cell thick,
    # 58 A across, 9.8 : 1. Half a cell terminates Se-Se and comes out
    # Fe110Se200, which is not FeSe any more, so 6.0 A is the floor.
    # This is the one to compare against the micrograph.
    thin, _, _, (verdict, why) = core_from(
        CIF / "FeSe_hexagonal.cif", "prism",
        {"radius": OPT["placa_radio"], "half_height": 3.0, "sides": 6})
    thin.info["solid_indices"] = list(range(len(thin)))
    print(f"\nplaca delgada desnuda: {thin.get_chemical_formula()} "
          f"({len(thin)} átomos, {np.ptp(thin.positions[:, 2]):.1f} Å de "
          f"grosor) -> {verdict}")
    rows.append(record(thin, "placa_H_delgada_desnuda", {
        "familia": "placa_H", "version": "la más delgada posible, sin cubrir",
        "nucleo": thin.get_chemical_formula(), "capas_grafiticas": 0,
        "veredicto_nucleo": f"{verdict}: {why}", "veredicto_coraza": "—",
    }))

    # 85 x 9.4 A, 9 : 1 -- a needle rather than the first version's
    # stubby 3.2 : 1. The thickness stays at 9.4 A for the same reason
    # the plate's does: below about 9 A the graphitic shell has nothing
    # to wrap. A capsule and not a cylinder, because a cylinder's sharp
    # rim leaves the envelope nowhere to go and the coating refuses it.
    coated_particle("barra_T", CIF / "FeSe_tetragonal.cif", "rod",
                    {"radius": OPT["barra_radio"],
                     "half_length": OPT["barra_semilargo"]},
                    shells=1, gap=OPT["hueco"], rows=rows)

    coated_particle("particula_Fe3C", CIF / "Fe3C_cementita.cif",
                    "ellipsoid", {"a": OPT["fe3c_a"], "b": OPT["fe3c_a"],
                                  "c": OPT["fe3c_c"]},
                    shells=OPT["fe3c_capas"], gap=OPT["hueco"], rows=rows)

    # The small one, bare, for putting inside a tube. No graphitic shell
    # on it: inside the tube the TUBE is the graphite, and a shell would
    # be a second wall 3.35 A from the first with nothing between them.
    inside, _, _, (verdict, why) = core_from(
        CIF / "Fe3C_cementita.cif", "ellipsoid", {"a": OPT["fe3c_interior_a"],
         "b": OPT["fe3c_interior_a"], "c": OPT["fe3c_interior_c"]})
    inside.info["solid_indices"] = list(range(len(inside)))
    print(f"\npartícula interior: {inside.get_chemical_formula()} "
          f"({len(inside)} átomos) -> {verdict}")
    print(f"    {why}")
    rows.append(record(inside, "particula_Fe3C_interior", {
        "familia": "particula_Fe3C", "version": "para ir dentro del tubo",
        "nucleo": inside.get_chemical_formula(), "capas_grafiticas": 0,
        "veredicto_nucleo": f"{verdict}: {why}", "veredicto_coraza": "—",
    }))

    tubes = coated_tube("nanotubo", rows)

    compositions(tubes["C"], inside, rows)

    (HERE / "modelos.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(rows)} modelos escritos en {XYZ} y {IMG}")
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
