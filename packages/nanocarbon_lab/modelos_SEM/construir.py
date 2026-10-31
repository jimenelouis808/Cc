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

import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms

from nanocarbon_lab import scene
from nanocarbon_lab.builders.capped_cnt import build_capped_cnt
from nanocarbon_lab.dopants.codoping import codope
from nanocarbon_lab.exports.xyz import write_xyz
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

#: Substitutional co-doping of the carbon, as the user's material is:
#: N and Se together. The fractions are each against the ORIGINAL carbon
#: count, and each sits under its own ceiling in `dopants.chemistry`
#: (N 0.20, Se 0.03) -- Se is the binding one and 3 % is its limit, past
#: which the placement stops describing a doped sheet.
CODOPE = [("N", 0.06), ("Se", 0.025)]

#: "avoid", so no two heteroatoms end up bonded. N and Se are not a
#: donor-acceptor pair like B-N: there is no reason for them to sit
#: together, and letting them would concentrate the strain Se already
#: brings (it is 48 % larger than carbon and puckers its site).
AFFINITY = "avoid"

#: Surface groups for the third version, in the order they are grafted.
#: Coverage is deliberately low and falls along the list: each pass sees
#: the sites the previous ones left, and a real oxidised carbon carries a
#: few per cent of groups, not a monolayer.
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

#: Stiffness of the harmonic anchor that holds the shell on its
#: envelope. The default 0.7 let a handful of carbons fall to 1.55 A of
#: the core on the plate's sharp hexagonal rim -- a real rim reconstructs
#: there and this model does not, so the honest move is to keep the wall
#: on its envelope rather than to let it collapse somewhere arbitrary.
WALL_ANCHOR = 1.0

VACUUM = 10.0
SEED = 20251031


def group_for(name: str):
    """A shipped group, or one built from a shipped group by swap."""
    if name in EXTRA_GROUPS:
        base, swaps = EXTRA_GROUPS[name]
        return substitute(GROUPS[base], swaps, name=name)
    return GROUPS[name]


def core_from(cif: Path, shape: str, params: dict, centre: str = "cell"):
    """Carve one inorganic core and return it with its report."""
    crystal, crystal_report = read_crystal(cif)
    cut = shape_module.make_shape(shape, **params)
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
    out = codope(carbon, CODOPE, affinity=AFFINITY, seed=SEED)
    if version == "NSe":
        return out
    for index, (name, coverage) in enumerate(GRAFTS):
        try:
            out = functionalize(out, group_for(name), coverage=coverage,
                                face="outer", seed=SEED + index)
        except ValueError as exc:
            print(f"      {name}: no se injertó ({exc})")
    return out


def record(atoms: Atoms, label: str, notes: dict) -> dict:
    """Write the xyz and a PNG, and collect what the structure is."""
    XYZ.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    comment = f"{label} | " + " ".join(f"{k}{v}" for k, v in sorted(counts.items()))
    write_xyz(atoms, XYZ / f"{label}.xyz", comment=comment)
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
                    shells: int, gap: float, rows: list) -> dict:
    """One inorganic core in its graphitic shell, in the three versions.

    The core is carved once and the shells are built once; only the
    chemistry on the carbon changes between versions. Doing it the other
    way -- three independent builds -- would let the meshing differ
    between them and make the comparison meaningless.
    """
    print(f"\n{label}")
    t0 = time.time()
    core, crystal_report, carve_report, (verdict, why) = core_from(
        cif, shape, params)
    print(f"  núcleo: {core.get_chemical_formula()} "
          f"({len(core)} átomos, {time.time() - t0:.1f} s) -> {verdict}")
    print(f"    {why}")
    for warning in carve_report.warnings:
        print(f"    aviso: {warning}")

    shell_list, coat_report = graphitic_coat(
        core, gap=gap, shells=shells, vacuum=VACUUM, seed=SEED,
        wall_anchor=WALL_ANCHOR)
    print(f"  coraza: {sum(len(s) for s in shell_list)} C en {shells} capa(s), "
          f"{time.time() - t0:.1f} s")
    for warning in coat_report.warnings:
        print(f"    aviso: {warning}")

    # The bare core as well. Once the shell is on, the habit the whole
    # model is a hypothesis about is invisible -- the coating's envelope
    # is an offset of the solid and rounds every corner by the gap -- so
    # the picture that can be compared with the micrograph is this one.
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
    tube = build_capped_cnt(n_body_rings=14, freq=4, vacuum=VACUUM, seed=SEED)
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


def compositions(tube: Atoms, particle: Atoms, rows: list) -> None:
    """The tip and the anemone, as ILLUSTRATIONS.

    `scene.compose` places finished structures next to each other and
    measures the clearances; it does not bond them. So these two files
    are a tube and some particles in one coordinate system, which is
    what a figure needs and is NOT a structure to relax or to feed a
    plane-wave code. The reach of each object is measured first so the
    placement is in the structures' own units rather than guessed.
    """
    print("\ncomposiciones (ilustraciones, no estructuras)")
    half_tube = 0.5 * (tube.positions[:, 2].max() - tube.positions[:, 2].min())
    particle_reach = 0.5 * max(
        np.ptp(particle.positions[:, axis]) for axis in range(3))
    # The particle sits just past the cap: the "outside the tip" reading
    # of the micrograph. The gap is the graphitic one, because that is
    # what separates two carbon surfaces that are not bonded -- and it
    # has to be left explicitly, since `compose` places and does not
    # push apart. For the "inside" reading, move it to -half_tube and
    # shrink the particle below the tube's radius.
    offset = half_tube + particle_reach + GRAPHITIC_GAP

    tip, tip_report = scene.compose([
        scene.Item(tube, label="nanotubo"),
        scene.Item(particle, label="Fe3C@C", translation=(0.0, 0.0, offset)),
    ], vacuum=VACUUM, seams="drop")
    rows.append(record(tip, "escena_punta", {
        "familia": "escena", "version": "ilustración", "nucleo": "Fe3C",
        "capas_grafiticas": 0, "veredicto_nucleo": "—",
        "veredicto_coraza": "composición: dos objetos, sin enlazar",
    }))
    for warning in tip_report.warnings:
        print(f"    aviso: {warning}")

    # The anemone: tubes radiating from a common origin, each with its
    # particle at the far end. Five is enough to read as one and few
    # enough to stay under a hundred thousand atoms.
    placed = []
    for index in range(5):
        angle = np.radians(index * 72.0)
        direction = np.array([np.cos(angle), np.sin(angle), 0.6])
        direction /= np.linalg.norm(direction)
        axis_tilt = np.degrees(np.arccos(direction[2]))
        azimuth = np.degrees(np.arctan2(direction[1], direction[0]))
        # The tubes start a tube-radius out from the origin instead of
        # at it: five of them converging on one point put their inner
        # caps 0.05 A apart, which is five tubes occupying the same
        # space rather than an anemone.
        start = tube.info.get("radius", 8.0) + GRAPHITIC_GAP
        placed.append(scene.Item(
            tube, label=f"tubo{index}", rotation=(0.0, axis_tilt, azimuth),
            translation=tuple(direction * (start + half_tube))))
        placed.append(scene.Item(
            particle, label=f"punta{index}",
            translation=tuple(direction * (start + 2 * half_tube
                                           + particle_reach
                                           + GRAPHITIC_GAP))))
    anemone, anemone_report = scene.compose(placed, vacuum=VACUUM, seams="drop")
    rows.append(record(anemone, "escena_anemona", {
        "familia": "escena", "version": "ilustración", "nucleo": "Fe3C",
        "capas_grafiticas": 0, "veredicto_nucleo": "—",
        "veredicto_coraza": "composición: 10 objetos, sin enlazar",
    }))
    for warning in anemone_report.warnings:
        print(f"    aviso: {warning}")


def main() -> None:
    rows: list = []

    coated_particle("placa_H", CIF / "FeSe_hexagonal.cif", "prism",
                    {"radius": 18.0, "half_height": 6.0, "sides": 6},
                    shells=1, gap=3.45, rows=rows)

    coated_particle("barra_T", CIF / "FeSe_tetragonal.cif", "rod",
                    {"radius": 6.0, "half_length": 13.0},
                    shells=1, gap=3.45, rows=rows)

    carbide = coated_particle("particula_Fe3C", CIF / "Fe3C_cementita.cif",
                              "ellipsoid", {"a": 13.0, "b": 13.0, "c": 18.0},
                              shells=2, gap=3.45, rows=rows)

    tubes = coated_tube("nanotubo", rows)

    compositions(tubes["C"], carbide["C"], rows)

    (HERE / "modelos.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(rows)} modelos escritos en {XYZ} y {IMG}")


if __name__ == "__main__":
    main()
