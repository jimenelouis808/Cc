"""``nanocarbon`` command line entry point.

Sub-commands:

* ``cnt``        — build a single CNT and export it.
* ``graphene``   — build a graphene supercell.
* ``ribbon``     — build a graphene nanoribbon.
* ``foam``       — build a 3D carbon foam.
* ``cnt-cap``    — build a fully capped/defected CNT and export XYZ + a
  Blender-ready render bundle (see ``nanocarbon_lab/blender/``).
* ``coil``       — build a nanocoil whose rings follow the curvature.
* ``fullerene``  — build a closed cage (C60, C240, C540, ...).
* ``onion``      — build a carbon nano-onion (C60@C240@C540).
* ``junction``   — build a capped L/T/Y/X nanotube junction.
* ``schwarzite`` — build a periodic negative-curvature schwarzite unit cell.
* ``mwcnt``      — build a multi-wall nanotube from concentric shells.
* ``bundle``     — build a hexagonally packed rope of tubes.
* ``tmd``        — build an MX2 monolayer, bilayer or few-layer slab.
* ``tmd-bulk``   — build the bulk MX2 crystal (2H, 3R or AA).
* ``tmd-ribbon`` — build an MX2 nanoribbon with a chosen edge termination.
* ``tmd-tube``   — roll an MX2 monolayer into an (n, m) nanotube.
* ``tmd-coil``   — coil an MX2 nanotube onto a helix.
* ``tmd-schwarzite`` — MX2 on a triply periodic minimal surface.
* ``tmd-junction`` — a finite, closed MX2 L/T/Y/X tube junction.
* ``twist``      — two layers with a commensurate twist (moire).
* ``stack``      — aligned van der Waals stack of 2D layers.
* ``validate``   — run validation on an existing structure file.
* ``dopants``    — list the heteroatoms available for carbon and what each does.

Every carbon sub-command builds **pure carbon** unless ``--dopant`` is
given; doping is an edit applied afterwards, never a different material.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from ase import io as ase_io

from ..builders import (
    build_bundle,
    build_capped_cnt,
    build_carbon_foam,
    build_cnt,
    build_coil,
    build_fullerene,
    build_graphene_supercell,
    build_junction,
    build_multiwall_cnt,
    build_nano_onion,
    build_nanocoil,
    build_nanoribbon,
    build_schwarzite,
)
from ..defects import introduce_vacancies
from ..dopants import DOPANT_CHEMISTRY, DOPANT_ELEMENTS
from ..exports.lammps import write_lammps
from ..exports.qe import QESettings, write_qe_input
from ..exports.xyz import write_render_bundle
from ..hetero import available_layers, build_twisted_bilayer, build_vdw_stack
from ..jobs import DOPANT_SITES, Job, apply_doping
from ..tmd import (
    MATERIALS,
    build_tmd_bulk,
    build_tmd_coil,
    build_tmd_layers,
    build_tmd_nanotube,
    build_tmd_ribbon,
    build_tmd_schwarzite,
)
from ..tmd.curved import build_tmd_junction, schwarzite_quality
from ..tmd.materials import (
    available_chalcogens,
    available_metals,
    material_for,
)
from ..tmd.quality import geometry_report as tmd_geometry_report
from ..tmd.quality import tmd_quality
from ..utils.constants import MAX_DOPING_FRACTION, MIN_DOPING_FRACTION
from ..validation.checks import run_basic_checks
from ..validation.quality import sp2_quality


def _add_doping_arguments(p, seed_help: str | None = None) -> None:
    """Attach the doping flags every carbon sub-command shares.

    One helper rather than three copies: the choices used to be spelled
    out at each call site, so adding a dopant meant editing all of them
    and the lists had already drifted apart.
    """
    p.add_argument("--dopant", choices=list(DOPANT_ELEMENTS), default=None,
                   help="Heteroatom substituted into the carbon lattice after "
                        "the build. Omit for pure carbon, which is the "
                        "default. Run `nanocarbon dopants` for what each one "
                        "does and how much of it is realistic.")
    p.add_argument("--dopant-conc", type=float, default=0.0,
                   help=f"Substitution fraction, "
                        f"{MIN_DOPING_FRACTION:g}-{MAX_DOPING_FRACTION:g} for "
                        "the range that describes a real doped carbon. Higher "
                        "is allowed but warns.")
    p.add_argument("--dopant-site", choices=list(DOPANT_SITES), default="random",
                   help="Where the substitutions go. 'pentagon' puts them on "
                        "the five-membered rings, which is where a cap or cage "
                        "carries its curvature and its reactivity; the "
                        "fraction is then of the pentagon sites. Needs a "
                        "builder that records rings.")
    if seed_help:
        p.add_argument("--seed", type=int, default=0, help=seed_help)


def _add_material_argument(p) -> None:
    """Attach the three ways to name a dichalcogenide.

    ``--material`` takes the formula; ``--metal`` and ``--chalcogen``
    take the two elements separately, which is how the choice is usually
    actually made -- "a tungsten selenide" rather than "WSe2". They are
    alternatives, not a combination, and giving both is an error rather
    than a silent precedence rule.
    """
    p.add_argument("--material", default=None, choices=sorted(MATERIALS),
                   help="Compound by formula. Default MoS2. Alternatively "
                        "give --metal and --chalcogen.")
    p.add_argument("--metal", default=None, choices=list(available_metals()),
                   help="Transition metal (or Sn), used with --chalcogen "
                        "instead of --material.")
    p.add_argument("--chalcogen", default=None,
                   choices=list(available_chalcogens()),
                   help="Chalcogen, used with --metal instead of --material.")


def _resolve_material(args) -> None:
    """Turn --metal/--chalcogen into the formula the builders take.

    Done once here rather than in each of the seven dichalcogenide
    sub-commands, so a new one cannot forget it. A pair with no tabulated
    compound raises from `material_for`, whose message names what *is*
    available for each of the two elements.
    """
    if not hasattr(args, "metal"):
        return
    metal, chalcogen, formula = args.metal, args.chalcogen, args.material
    if (metal is None) != (chalcogen is None):
        raise SystemExit(
            "--metal and --chalcogen go together: give both, or give "
            "--material instead."
        )
    if metal is not None:
        if formula is not None:
            raise SystemExit(
                f"Give either --material {formula} or --metal {metal} "
                f"--chalcogen {chalcogen}, not both."
            )
        try:
            args.material = material_for(metal, chalcogen).formula
        except KeyError as exc:
            # material_for's message already names what is available for
            # each element; a traceback on top of it helps nobody.
            raise SystemExit(str(exc).strip("'")) from None
    elif formula is None:
        args.material = "MoS2"


def _dope(atoms, args):
    """Apply the doping flags, honouring the placement choice."""
    if not getattr(args, "dopant", None) or getattr(args, "dopant_conc", 0) <= 0:
        return atoms
    job = Job(mode="capped tube", dopant=args.dopant,
              dopant_conc=args.dopant_conc,
              dopant_site=getattr(args, "dopant_site", "random"),
              seed=args.seed)
    return apply_doping(atoms, job)


def _apply_post(atoms, args):
    atoms = _dope(atoms, args)
    if getattr(args, "vacancies", 0):
        atoms = introduce_vacancies(atoms, n_defects=args.vacancies, seed=args.seed)
    return atoms


def _export(atoms, outdir: Path, fmt: str, calculation: str, force: bool):
    if fmt in ("qe", "both"):
        write_qe_input(atoms, outdir / "qe",
                       settings=QESettings(calculation=calculation), force=force)
    if fmt in ("lammps", "both"):
        write_lammps(atoms, outdir / "lammps", force=force)


def _cmd_cnt(args):
    atoms = build_cnt(n=args.n, m=args.m, length=args.length,
                      bond=args.bond, vacuum=args.vacuum)
    atoms = _apply_post(atoms, args)
    _export(atoms, Path(args.out), args.format, args.calculation, args.force)
    return 0


def _cmd_graphene(args):
    atoms = build_graphene_supercell(nx=args.nx, ny=args.ny,
                                     bond=args.bond, vacuum=args.vacuum)
    atoms = _apply_post(atoms, args)
    _export(atoms, Path(args.out), args.format, args.calculation, args.force)
    return 0


def _cmd_ribbon(args):
    atoms = build_nanoribbon(width=args.width, length=args.length,
                             edge=args.edge, bond=args.bond,
                             vacuum=args.vacuum, passivate=args.passivate)
    atoms = _apply_post(atoms, args)
    _export(atoms, Path(args.out), args.format, args.calculation, args.force)
    return 0


def _cmd_nanocoil(args):
    atoms = build_nanocoil(
        n=args.n, m=args.m,
        coil_radius=args.coil_radius, pitch=args.pitch,
        n_turns=args.turns, bond=args.bond, vacuum=args.vacuum,
        stone_wales_density=args.sw_density, seed=args.seed,
    )
    atoms = _apply_post(atoms, args)
    _export(atoms, Path(args.out), args.format, args.calculation, args.force)
    return 0


def _cmd_foam(args):
    atoms = build_carbon_foam(box_size=args.box, n_flakes=args.flakes,
                              flake_radius=args.radius, seed=args.seed)
    _export(atoms, Path(args.out), args.format, args.calculation, args.force)
    return 0


def _parse_defect_specs(raw: list[str] | None) -> list[dict]:
    """Parse repeated ``--defect TYPE[:COUNT]`` flags into DefectSpec dicts."""
    specs = []
    for item in raw or []:
        if ":" in item:
            kind, count_s = item.split(":", 1)
            count = int(count_s)
        else:
            kind, count = item, 1
        kind = kind.strip()
        if kind not in ("stone_wales", "divacancy"):
            raise ValueError(
                f"Unknown --defect type {kind!r}; expected 'stone_wales' or 'divacancy'."
            )
        specs.append({"type": kind, "count": count})
    return specs


def _cmd_cnt_cap(args):
    atoms = build_capped_cnt(
        n_body_rings=args.rings,
        freq=args.freq,
        target_radius=args.target_radius,
        bond=args.bond,
        bend_angle=args.bend_angle,
        shape=args.shape,
        waviness=args.waviness,
        max_strain=args.max_strain,
        shape_points=args.shape_points,
        helix_turns=args.helix_turns,
        helix_radius=args.helix_radius,
        helix_pitch=args.helix_pitch,
        helix_handedness=-1 if args.helix_handedness == "left" else 1,
        helix_taper=args.helix_taper,
        roughness=args.roughness,
        defects=_parse_defect_specs(args.defect),
        relax_iterations=args.relax_iterations,
        seed=args.seed,
    )
    atoms = _maybe_dope(atoms, args)
    xyz_path, json_path = write_render_bundle(atoms, Path(args.out))
    g = atoms.info["geometry"]
    print(f"Wrote {xyz_path} and {json_path}")
    print(f"  n_atoms     = {len(atoms)}")
    print(f"  radius      = {atoms.info['radius']:.2f} A   length = {atoms.info['length']:.1f} A")
    print(f"  shape       = {atoms.info['shape']}  "
          f"(path strain {atoms.info['path_strain']:.1%})")
    print(f"  ring_counts = {atoms.info['ring_counts']}")
    print(f"  bond length = {g['bond_min']:.3f} / {g['bond_mean']:.3f} / {g['bond_max']:.3f} A"
          f"  (std {g['bond_std']:.4f})")
    print(f"  bond angle  = {g['angle_min']:.1f} / {g['angle_mean']:.1f} / {g['angle_max']:.1f} deg"
          f"  (std {g['angle_std']:.2f})")
    print(f"  close contacts (<2 A, non-bonded) = {g['n_close_contacts']}")
    _report_doping(atoms)
    verdict, why = sp2_quality(g)
    print(f"  sp2 verdict = {verdict.upper()}: {why}")
    return 0


def _add_surface_flags(p, anneal: bool | int = True):
    """Surface finish and doping, shared by the structure sub-commands.

    ``anneal=False`` for the seed-polyhedron builders (fullerenes,
    onions), whose topology comes from an exact seed rather than a
    remeshed surface -- there is nothing for flip annealing to clean up,
    so offering the flag would only imply a control that does nothing.
    An int sets a different default, which the schwarzites need: there
    annealing actively hurts (see ``build_schwarzite``), so it defaults
    to 0 rather than to the junction's 80.
    """
    if anneal is not False:
        default = 80 if anneal is True else int(anneal)
        p.add_argument("--anneal-sweeps", type=int, default=default,
                       help="Flip-annealing passes that remove pentagon-heptagon "
                            "pairs beyond those curvature needs (39->14 on a Y "
                            "junction). 0 keeps the as-grown, rougher wall. On a "
                            "schwarzite the 5-7 pairs are how the net covers the "
                            "saddle, so annealing stretches bonds and the default "
                            "there is 0.")
    p.add_argument("--roughness", type=float, default=0.0,
                   help="RMS out-of-plane corrugation (Å) for a CVD-grown "
                        "rather than ideal wall; 0.1-0.3 is realistic.")
    _add_doping_arguments(p, seed_help="RNG seed for defects, roughness and doping.")


def _maybe_dope(atoms, args):
    return _dope(atoms, args)


def _report_doping(atoms) -> None:
    """Print what was substituted, if anything.

    Silent on a pure carbon structure, which is the default and the
    common case. When there *is* doping the placement matters as much as
    the amount, so both are printed -- and for ring placement both
    fractions, since "10% of the pentagon sites" and "2.5% of the
    structure" are the same edit described two ways, and quoting either
    one alone reads as the other.
    """
    entries = atoms.info.get("dopants")
    if not entries:
        return
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["element"]] = (counts.get(entry["element"], 0)
                                    + len(entry["indices"]))
    formula = ", ".join(f"{n} {sym}" for sym, n in sorted(counts.items()))
    mode = atoms.info.get("doping_mode", "random")
    overall = sum(counts.values()) / max(1, len(atoms))
    print(f"  doping      = {formula}  ({overall:.1%} of all atoms, {mode})")
    if "doping_sites_available" in atoms.info:
        size = atoms.info.get("doping_ring_size", 5)
        print(f"                {atoms.info['doping_concentration']:.1%} of the "
              f"{atoms.info['doping_sites_available']} {size}-ring sites")


def _report_structure(atoms, xyz_path, json_path):
    """Shared summary printer for the structure-building sub-commands."""
    g = atoms.info["geometry"]
    print(f"Wrote {xyz_path} and {json_path}")
    print(f"  n_atoms     = {len(atoms)}")
    if "genus" in atoms.info:
        print(f"  euler       = {atoms.info['euler']}   genus = {atoms.info['genus']}")
    if all(atoms.get_pbc()):
        print(f"  periodic    = yes, cubic cell {atoms.cell[0][0]:.2f} A")
    print(f"  ring_counts = {atoms.info['ring_counts']}")
    print(f"  bond length = {g['bond_min']:.3f} / {g['bond_mean']:.3f} / {g['bond_max']:.3f} A"
          f"  (std {g['bond_std']:.4f})")
    print(f"  bond angle  = {g['angle_min']:.1f} / {g['angle_mean']:.1f} / {g['angle_max']:.1f} deg")
    print(f"  close contacts (<2 A, non-bonded) = {g['n_close_contacts']}")
    _report_doping(atoms)
    verdict, why = sp2_quality(g)
    print(f"  sp2 verdict = {verdict.upper()}: {why}")


def _report_tmd(atoms, xyz_path, json_path, stoichiometric=True):
    """Summary printer for the dichalcogenides.

    Separate from the carbon one because almost nothing carries over:
    there are no rings to count, the metal is six-coordinate rather than
    three, and the bond to check against is material-specific.
    """
    report = tmd_geometry_report(atoms)
    info = atoms.info
    print(f"Wrote {xyz_path} and {json_path}")
    print(f"  n_atoms     = {len(atoms)}  ({atoms.get_chemical_formula()})")
    print(f"  material    = {info['material']}  phase {info['phase']} "
          f"({info['coordination']})")
    if "stacking" in info and info["stacking"] != "n/a":
        print(f"  stacking    = {info['stacking']}, {info['n_layers']} layers")
    if all(atoms.get_pbc()):
        lengths = atoms.cell.lengths()
        print(f"  periodic    = 3D, a = {lengths[0]:.3f} A, c = {lengths[2]:.3f} A")
    if "edge" in info:
        print(f"  edge        = {info['edge']} / {info['termination']}, "
              f"width {info['width_angstrom']:.1f} A")
    if "chiral_indices" in info and "radius" in info:
        n, m = info["chiral_indices"]
        print(f"  tube        = ({n},{m}) {info['chirality']}, "
              f"R = {info['radius']:.2f} A, diameter {info['diameter']:.2f} A")
        print(f"  roll strain = {info['roll_strain']:.1%} on the outer "
              f"{info['chalcogen']} plane (goes as h/2R)")
    if info.get("structure_type") == "tmd_coil":
        n, m = info["chiral_indices"]
        print(f"  tube        = ({n},{m}) {info['chirality']}, "
              f"R = {info['tube_radius']:.2f} A")
        print(f"  coil        = R {info['coil_radius']:.1f} A, pitch "
              f"{info['pitch']:.1f} A, {info['turns']:g} turns, "
              f"{info['periods']} periods")
        if abs(info["coil_radius"] - info["requested_coil_radius"]) > 0.05:
            print(f"                (asked {info['requested_coil_radius']:.1f} A "
                  f"/ {info['requested_pitch']:.1f} A — the path is scaled to a "
                  "whole number of tube periods)")
        print(f"  roll strain = {info['roll_strain']:.1%}  (h/2R, from rolling)")
        print(f"  bend strain = {info['bend_strain']:.1%}  (R_outer * kappa, "
              "from coiling)")
        print(f"  total       = {info['total_strain']:.1%} on the outer "
              f"{info['chalcogen']} plane")
    print(f"  M-X bond    = {report['bond_min']:.3f} / {report['bond_mean']:.3f} "
          f"/ {report['bond_max']:.3f} A  (ideal {report['bond_ideal']:.3f})")
    print(f"  coordination= metal {report['metal_coordination_min']}-"
          f"{report['metal_coordination_max']}, chalcogen "
          f"{report['chalcogen_coordination_min']}-"
          f"{report['chalcogen_coordination_max']}")
    print(f"  X/M ratio   = {report['stoichiometry']:.3f}")
    verdict, why = tmd_quality(report, expect_stoichiometric=stoichiometric,
                               structure_type=info.get("structure_type"))
    print(f"  verdict     = {verdict.upper()}: {why}")
    if "phase_note" in info:
        print(f"  note        = {info['phase_note']}")
    return 0


def _cmd_tmd(args):
    atoms = build_tmd_layers(
        args.material, n_layers=args.layers, phase=args.phase,
        stacking=args.stacking, nx=args.nx, ny=args.ny, vacuum=args.vacuum,
    )
    return _report_tmd(atoms, *write_render_bundle(atoms, Path(args.out)))


def _cmd_tmd_bulk(args):
    atoms = build_tmd_bulk(args.material, phase=args.phase,
                           stacking=args.stacking, nx=args.nx, ny=args.ny)
    return _report_tmd(atoms, *write_render_bundle(atoms, Path(args.out)))


def _cmd_tmd_ribbon(args):
    atoms = build_tmd_ribbon(
        args.material, width=args.width, length=args.length, edge=args.edge,
        termination=args.termination, phase=args.phase,
    )
    return _report_tmd(atoms, *write_render_bundle(atoms, Path(args.out)),
                       stoichiometric=(args.termination == "mixed"))


def _cmd_tmd_tube(args):
    atoms = build_tmd_nanotube(args.material, n=args.n, m=args.m,
                               length=args.length, phase=args.phase)
    return _report_tmd(atoms, *write_render_bundle(atoms, Path(args.out)))


def _report_stack(atoms, xyz_path, json_path):
    info = atoms.info
    report = run_basic_checks(atoms)
    print(f"Wrote {xyz_path} and {json_path}")
    print(f"  n_atoms     = {len(atoms)}  ({atoms.get_chemical_formula()})")
    if info["structure_type"] == "twisted_bilayer":
        m, n = info["commensurate_index"]
        print(f"  stack       = {info['bottom_layer']} / {info['top_layer']}")
        print(f"  twist       = {info['twist_angle']:.4f} deg  (m,n) = ({m},{n})"
              f"   asked {info['requested_angle']:.3f}")
        print(f"  moire       = {info['moire_period']:.1f} A period, "
              f"{info['cells_per_layer']} cells per layer")
    else:
        print(f"  stack       = {' / '.join(info['layers'])}")
    print(f"  gap         = {info['interlayer_gap']:.2f} A between facing planes")
    strain = info.get("imposed_strain", 0.0)
    if isinstance(strain, list):
        print(f"  strain      = {max(abs(v) for v in strain):.2%} worst, "
              "imposed to share one cell")
    elif strain:
        print(f"  strain      = {strain:+.2%} on the top layer, imposed to "
              "share one cell")
    print(f"  cell        = {atoms.cell.lengths()[0]:.2f} x "
          f"{atoms.cell.lengths()[1]:.2f} A in plane")
    print(f"  validation  = {'OK' if report.ok else 'FAILED'}"
          + ("" if report.ok else f": {report.errors}"))
    return 0


def _cmd_twist(args):
    atoms = build_twisted_bilayer(
        layer=args.layer, target_angle=args.angle, max_index=args.max_index,
        gap=args.gap, top_layer=args.top_layer,
    )
    return _report_stack(atoms, *write_render_bundle(atoms, Path(args.out)))


def _cmd_stack(args):
    atoms = build_vdw_stack(args.layers, gap=args.gap, nx=args.nx, ny=args.ny)
    return _report_stack(atoms, *write_render_bundle(atoms, Path(args.out)))


def _cmd_tmd_junction(args):
    atoms = build_tmd_junction(
        args.material, kind=args.kind, tube_radius=args.tube_radius,
        arm_length=args.arm_length, blend=args.blend, parity=args.parity,
        phase=args.phase, seed=args.seed,
    )
    xyz_path, json_path = write_render_bundle(atoms, Path(args.out))
    info = atoms.info
    verdict, why = schwarzite_quality(atoms)
    print(f"Wrote {xyz_path} and {json_path}")
    print(f"  n_atoms     = {len(atoms)}  ({atoms.get_chemical_formula()})")
    print(f"  material    = {info['material']}  {info['junction_kind']} junction, "
          f"R = {info['tube_radius']:.1f} A arms {info['arm_length']:.1f} A")
    print(f"  genus       = {info['genus']}  (Euler {info['euler']})")
    rings = ", ".join(f"{k}:{v}" for k, v in sorted(info["ring_counts"].items()))
    print(f"  rings       = {rings}")
    expected = 6 * info["euler"]
    print(f"  sum(6-n)    = {info['ring_deficit']}  (6*chi = {expected})"
          f"  {'ok' if info['ring_deficit'] == expected else 'MISMATCH'}")
    print("                a capped junction is sphere-like, so this is "
          "POSITIVE — paid in squares, since MX2 has no pentagons")
    print(f"  parity      = {info['parity']}  ->  {info['odd_rings']} odd rings"
          + (f", {info['parity_splits']} splits" if info["parity_splits"] else ""))
    print(f"  antiphase   = {info['antiphase_bonds']}/{info['n_net_bonds']} bonds "
          f"({info['antiphase_fraction']:.1%}) are M-M or X-X")
    print(f"  M-X spread  = {info['bond_deviation_p95']:.1%} at p95, "
          f"{info['bond_deviation_max']:.1%} worst")
    print(f"  coordination= metal {info['graph_metal_coordination'][0]}-"
          f"{info['graph_metal_coordination'][1]}, chalcogen "
          f"{info['graph_chalcogen_coordination'][0]}-"
          f"{info['graph_chalcogen_coordination'][1]}  (from the bond graph)")
    print(f"  X/M ratio   = {info['stoichiometry']:.3f}")
    print(f"  verdict     = {verdict.upper()}: {why}")
    return 0


def _cmd_tmd_schwarzite(args):
    atoms = build_tmd_schwarzite(
        args.material, kind=args.kind, cell=args.cell, parity=args.parity,
        phase=args.phase, seed=args.seed,
    )
    xyz_path, json_path = write_render_bundle(atoms, Path(args.out))
    info = atoms.info
    verdict, why = schwarzite_quality(atoms)
    print(f"Wrote {xyz_path} and {json_path}")
    print(f"  n_atoms     = {len(atoms)}  ({atoms.get_chemical_formula()})")
    print(f"  material    = {info['material']}  on the "
          f"{info['schwarzite_kind']} surface, {info['cell']:.0f} A cell")
    print(f"  genus       = {info['genus']}  (Euler {info['euler']})")
    rings = ", ".join(f"{k}:{v}" for k, v in sorted(info["ring_counts"].items()))
    print(f"  rings       = {rings}")
    print(f"  sum(6-n)    = {info['ring_deficit']}  (6*chi = "
          f"{6 * info['euler']})"
          f"  {'ok' if info['ring_deficit'] == 6 * info['euler'] else 'MISMATCH'}")
    print(f"  parity      = {info['parity']}  ->  {info['odd_rings']} odd rings"
          + (f", {info['parity_splits']} splits" if info["parity_splits"] else ""))
    print(f"  antiphase   = {info['antiphase_bonds']}/{info['n_net_bonds']} bonds "
          f"({info['antiphase_fraction']:.1%}) are M-M or X-X")
    print(f"  M-X spread  = {info['bond_deviation_p95']:.1%} at p95, "
          f"{info['bond_deviation_max']:.1%} worst")
    print(f"  coordination= metal {info['graph_metal_coordination'][0]}-"
          f"{info['graph_metal_coordination'][1]}, chalcogen "
          f"{info['graph_chalcogen_coordination'][0]}-"
          f"{info['graph_chalcogen_coordination'][1]}  (from the bond graph)")
    print(f"  X/M ratio   = {info['stoichiometry']:.3f}")
    print(f"  verdict     = {verdict.upper()}: {why}")
    return 0


def _cmd_tmd_coil(args):
    atoms = build_tmd_coil(
        args.material, n=args.n, m=args.m, coil_radius=args.coil_radius,
        pitch=args.pitch, turns=args.turns, phase=args.phase,
        handedness=1 if args.handedness == "right" else -1,
    )
    return _report_tmd(atoms, *write_render_bundle(atoms, Path(args.out)))


def _cmd_junction(args):
    atoms = build_junction(
        kind=args.kind, tube_radius=args.tube_radius, arm_length=args.arm_length,
        blend=args.blend, bond=args.bond, grid_resolution=args.grid,
        remesh_iterations=args.remesh_iterations,
        anneal_sweeps=args.anneal_sweeps, roughness=args.roughness,
        seed=args.seed,
    )
    atoms = _maybe_dope(atoms, args)
    _report_structure(atoms, *write_render_bundle(atoms, Path(args.out)))
    return 0


def _cmd_schwarzite(args):
    atoms = build_schwarzite(
        kind=args.kind, cell=args.cell,
        thickness=args.thickness, bond=args.bond, grid_resolution=args.grid,
        remesh_iterations=args.remesh_iterations,
        anneal_sweeps=args.anneal_sweeps, roughness=args.roughness,
        seed=args.seed,
    )
    atoms = _maybe_dope(atoms, args)
    _report_structure(atoms, *write_render_bundle(atoms, Path(args.out)))
    return 0


def _cmd_mwcnt(args):
    atoms = build_multiwall_cnt(
        n_shells=args.shells, inner_freq=args.inner_freq,
        freq_step=args.freq_step, n_body_rings=args.rings,
        bond=args.bond, roughness=args.roughness, seed=args.seed,
    )
    atoms = _maybe_dope(atoms, args)
    _report_structure(atoms, *write_render_bundle(atoms, Path(args.out)))
    print(f"  shells      = {args.shells}, wall spacing "
          f"{atoms.info['wall_spacing']:.2f} A, closest approach "
          f"{atoms.info['geometry']['min_wall_separation']:.2f} A")
    return 0


def _cmd_bundle(args):
    atoms = build_bundle(
        n_rings_across=args.shells, freq=args.freq, n_body_rings=args.rings,
        gap=args.gap, bond=args.bond, roughness=args.roughness, seed=args.seed,
    )
    atoms = _maybe_dope(atoms, args)
    _report_structure(atoms, *write_render_bundle(atoms, Path(args.out)))
    print(f"  tubes       = {atoms.info['n_tubes']}, lattice "
          f"{atoms.info['lattice_constant']:.1f} A, closest approach "
          f"{atoms.info['geometry']['min_wall_separation']:.2f} A")
    return 0


def _cmd_coil(args):
    atoms = build_coil(
        coil_radius=args.coil_radius, pitch=args.pitch, turns=args.turns,
        tube_radius=args.tube_radius, bond=args.bond,
        handedness=-1 if args.handedness == "left" else 1, taper=args.taper,
        remesh_iterations=args.remesh_iterations,
        anneal_sweeps=args.anneal_sweeps, roughness=args.roughness,
        pin_ends=args.pin_ends, seed=args.seed,
    )
    atoms = _maybe_dope(atoms, args)
    _report_structure(atoms, *write_render_bundle(atoms, Path(args.out)))
    achieved_pitch = atoms.info["achieved_pitch"]
    # Pitch is only measurable above one full turn (see swept._measure_coil).
    pitch_text = (
        "n/a (needs more than one turn to measure)" if math.isnan(achieved_pitch)
        else f"{achieved_pitch:.1f} A"
    )
    print(f"  coil        = R {atoms.info['achieved_coil_radius']:.1f} A, pitch "
          f"{pitch_text} (asked R {args.coil_radius:.0f}, pitch {args.pitch:.0f})")
    return 0


def _cmd_fullerene(args):
    atoms = build_fullerene(
        freq=args.freq, family=args.family, bond=args.bond,
        roughness=args.roughness, seed=args.seed,
    )
    atoms = _maybe_dope(atoms, args)
    _report_structure(atoms, *write_render_bundle(atoms, Path(args.out)))
    print(f"  cage        = {atoms.info['formula']}, radius "
          f"{atoms.info['radius']:.2f} A")
    return 0


def _cmd_onion(args):
    atoms = build_nano_onion(
        n_shells=args.shells, inner_freq=args.inner_freq,
        freq_step=args.freq_step, family=args.family, bond=args.bond,
        roughness=args.roughness, seed=args.seed,
    )
    atoms = _maybe_dope(atoms, args)
    _report_structure(atoms, *write_render_bundle(atoms, Path(args.out)))
    g = atoms.info["geometry"]
    print(f"  onion       = {atoms.info['formula']}")
    print(f"  shells      = {atoms.info['n_shells']}, radii "
          f"{[round(r, 2) for r in atoms.info['shell_radii']]} A")
    print(f"  spacing     = {atoms.info['shell_spacing']:.2f} A "
          f"(closest approach {g['min_wall_separation']:.2f} A; graphite is 3.4)")
    return 0


def _cmd_validate(args):
    atoms = ase_io.read(args.path)
    report = run_basic_checks(atoms)
    print(report.summary())
    return 0 if report.ok else 1


def _cmd_dopants(args):
    """Print the dopant table, grouped by how the lattice takes them."""
    del args
    print("Host is always carbon; these substitute into it.\n")
    print(f"Interface range: {MIN_DOPING_FRACTION:.0%}-{MAX_DOPING_FRACTION:.0%}. "
          "Per-element ceilings below are where the placement stops\n"
          "describing a real material — past them you get a warning, "
          "not a refusal.\n")
    headings = {
        "planar": "PLANAR — fits the sp2 lattice, sheet stays flat",
        "puckered": "PUCKERED — substitutes but pulls its site out of plane; "
                    "relax before use",
        "vacancy": "VACANCY — single-atom sites; belong in a vacancy, usually "
                   "with N around them",
    }
    for site, heading in headings.items():
        members = [c for c in DOPANT_CHEMISTRY.values() if c.site == site]
        if not members:
            continue
        print(heading)
        for chem in members:
            print(f"  {chem.symbol:<3s} up to {chem.max_fraction:>4.0%}  "
                  f"r={chem.radius:.2f} Å ({chem.size_mismatch:+.0%} vs C)")
            print(f"      {chem.note}")
        print()
    print("Placement: " + ", ".join(DOPANT_SITES) +
          ".  'pentagon' needs a builder that records rings (capped tube,\n"
          "fullerene, nano-onion, junction, schwarzite, multi-wall, bundle).")
    return 0


def _add_common(p):
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--format", choices=["qe", "lammps", "both"], default="qe")
    p.add_argument("--calculation", default="scf",
                   choices=["scf", "relax", "vc-relax", "nscf", "bands"])
    p.add_argument("--bond", type=float, default=1.42, help="C-C bond length (Å).")
    p.add_argument("--vacuum", type=float, default=15.0, help="Vacuum padding (Å).")
    _add_doping_arguments(p)
    p.add_argument("--vacancies", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="Export even if validation reports errors.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nanocarbon",
                                     description="Generate and export nanocarbon structures.")
    sub = parser.add_subparsers(dest="command", required=True)

    cnt = sub.add_parser("cnt", help="Build a single-wall CNT.")
    cnt.add_argument("--n", type=int, required=True)
    cnt.add_argument("--m", type=int, required=True)
    cnt.add_argument("--length", type=float, default=10.0, help="Target length (Å).")
    _add_common(cnt)
    cnt.set_defaults(func=_cmd_cnt)

    gr = sub.add_parser("graphene", help="Build a graphene supercell.")
    gr.add_argument("--nx", type=int, default=2)
    gr.add_argument("--ny", type=int, default=2)
    _add_common(gr)
    gr.set_defaults(func=_cmd_graphene)

    rb = sub.add_parser("ribbon", help="Build a graphene nanoribbon.")
    rb.add_argument("--width", type=int, required=True)
    rb.add_argument("--length", type=int, required=True)
    rb.add_argument("--edge", choices=["armchair", "zigzag"], default="zigzag")
    rb.add_argument("--passivate", action="store_true")
    _add_common(rb)
    rb.set_defaults(func=_cmd_ribbon)

    nc = sub.add_parser("nanocoil", help="Build a helical carbon nanocoil.")
    nc.add_argument("--n", type=int, default=6)
    nc.add_argument("--m", type=int, default=6)
    nc.add_argument("--coil-radius", type=float, default=25.0,
                    help="Helix radius in Å (default 25).")
    nc.add_argument("--pitch", type=float, default=12.0,
                    help="Vertical advance per turn in Å (default 12).")
    nc.add_argument("--turns", type=float, default=1.0,
                    help="Number of helical turns (default 1).")
    nc.add_argument("--sw-density", type=float, default=0.0,
                    help="Stone-Wales defect density on outer wall (0-0.02).")
    _add_common(nc)
    nc.set_defaults(func=_cmd_nanocoil)

    fm = sub.add_parser("foam", help="Build a 3D carbon foam.")
    fm.add_argument("--box", type=float, default=30.0)
    fm.add_argument("--flakes", type=int, default=20)
    fm.add_argument("--radius", type=float, default=4.0)
    fm.add_argument("--seed", type=int, default=0)
    fm.add_argument("--out", required=True)
    fm.add_argument("--format", choices=["qe", "lammps", "both"], default="lammps")
    fm.add_argument("--calculation", default="relax",
                    choices=["scf", "relax", "vc-relax"])
    fm.add_argument("--force", action="store_true")
    fm.set_defaults(func=_cmd_foam)

    cc = sub.add_parser(
        "cnt-cap",
        help="Build a fully capped/defected CNT; export XYZ + Blender render bundle.",
    )
    cc.add_argument("--rings", type=int, default=8,
                    help="Body lattice rings (controls length, default 8).")
    cc.add_argument("--freq", type=int, default=3,
                    help="Geodesic subdivision frequency; sets the diameter "
                         "(radius ~= 1.96 x freq A). Default 3.")
    cc.add_argument("--target-radius", type=float, default=None,
                    help="Pick --freq automatically for this radius (Å). The lattice "
                         "quantises the radius, so the value obtained may differ by ~1 Å.")
    cc.add_argument("--bond", type=float, default=1.42, help="C-C bond length (Å).")
    cc.add_argument("--bend-angle", type=float, default=0.0,
                    help="Total elastic bend of the body, in radians (0 = straight, max 1.0).")
    cc.add_argument("--shape", default="straight",
                    choices=["straight", "arc", "s_curve", "helix", "random"],
                    help="Centreline the tube is swept along (default straight).")
    cc.add_argument("--waviness", type=float, default=0.7,
                    help="0-1: how far the centreline wanders before the strain "
                         "budget trims it (default 0.7).")
    cc.add_argument("--max-strain", type=float, default=0.08,
                    help="Outer-wall strain budget (default 0.08 = physical). "
                         "Up to ~0.15 stays intact for artwork; beyond that bonds "
                         "stretch out of the sp2 range.")
    cc.add_argument("--shape-points", type=int, default=9,
                    help="Centreline control points: more = more wiggles (default 9).")
    cc.add_argument("--helix-turns", type=float, default=1.5,
                    help="Turns, for --shape helix (default 1.5).")
    cc.add_argument("--helix-radius", type=float, default=None,
                    help="Coil radius (Å) for --shape helix. Given explicitly, "
                         "the coil is built to this dimension and the tube is "
                         "made long enough to complete the turns; a tight coil "
                         "warns about strain rather than being silently trimmed.")
    cc.add_argument("--helix-pitch", type=float, default=None,
                    help="Axial rise per turn (Å) for --shape helix. Requires "
                         "--helix-radius.")
    cc.add_argument("--helix-handedness", choices=["right", "left"], default="right",
                    help="Coil chirality (default right).")
    cc.add_argument("--helix-taper", type=float, default=1.0,
                    help="Ratio of final to initial coil radius: 1.0 is a "
                         "cylindrical coil, <1 a conical spring (default 1.0).")
    cc.add_argument(
        "--defect", action="append", default=[],
        help="Repeatable. 'stone_wales[:N]' (5-7-7-5 pairs) or "
             "'divacancy[:N]' (5-8-5 octagon), e.g. --defect stone_wales:2.",
    )
    cc.add_argument("--relax-iterations", type=int, default=3000,
                    help="L-BFGS iterations per relaxation cycle.")
    cc.add_argument("--roughness", type=float, default=0.0,
                    help="RMS out-of-plane corrugation (Å) for a CVD-grown "
                         "rather than ideal wall; 0.1-0.3 is realistic.")
    _add_doping_arguments(
        cc, seed_help="RNG seed for defect placement, roughness and doping.")
    cc.add_argument("--out", required=True,
                    help="Output path without extension (.xyz and .json are appended).")
    cc.set_defaults(func=_cmd_cnt_cap)

    jn = sub.add_parser(
        "junction",
        help="Build a capped L/T/Y/X nanotube junction (XYZ + render bundle).",
    )
    jn.add_argument("--kind", default="Y", choices=["L", "T", "Y", "X", "cross3d"])
    jn.add_argument("--tube-radius", type=float, default=6.0, help="Arm radius (Å).")
    jn.add_argument("--arm-length", type=float, default=22.0,
                    help="Centre-to-tip length of each arm (Å).")
    jn.add_argument("--blend", type=float, default=4.0,
                    help="Smooth-union radius at the branch (Å); sets how flared "
                         "the neck is and how many heptagons it takes.")
    jn.add_argument("--bond", type=float, default=1.42)
    jn.add_argument("--grid", type=int, default=70, help="Marching-cubes resolution.")
    jn.add_argument("--remesh-iterations", type=int, default=25)
    jn.add_argument("--out", required=True, help="Output path without extension.")
    _add_surface_flags(jn)
    jn.set_defaults(func=_cmd_junction)

    phases = ["2H", "1T", "1T'"]
    stackings = ["2H", "3R", "AA"]

    td = sub.add_parser(
        "tmd",
        help="Build an MX2 monolayer, bilayer or few-layer slab (MoS2, WS2...).",
    )
    _add_material_argument(td)
    td.add_argument("--layers", type=int, default=1,
                    help="X-M-X sandwiches: 1 = monolayer, 2 = bilayer.")
    td.add_argument("--phase", default="2H", choices=phases,
                    help="Metal coordination inside a layer: 2H is trigonal "
                         "prismatic (semiconducting, the group-6 ground "
                         "state), 1T octahedral (metallic). No TMD has a "
                         "tetragonal phase — all three sit on a hexagonal "
                         "lattice.")
    td.add_argument("--stacking", default="2H", choices=stackings,
                    help="How layers stack: 2H (AA', alternating 180 deg), "
                         "3R (rhombohedral), AA (eclipsed). Ignored for one "
                         "layer.")
    td.add_argument("--nx", type=int, default=1)
    td.add_argument("--ny", type=int, default=1)
    td.add_argument("--vacuum", type=float, default=15.0)
    td.add_argument("--out", required=True, help="Output path without extension.")
    td.set_defaults(func=_cmd_tmd)

    tb = sub.add_parser("tmd-bulk", help="Build the bulk MX2 crystal.")
    _add_material_argument(tb)
    tb.add_argument("--phase", default="2H", choices=phases)
    tb.add_argument("--stacking", default="2H", choices=stackings,
                    help="Sets the repeat: 2H is two layers per cell, 3R "
                         "three, AA one.")
    tb.add_argument("--nx", type=int, default=1)
    tb.add_argument("--ny", type=int, default=1)
    tb.add_argument("--out", required=True, help="Output path without extension.")
    tb.set_defaults(func=_cmd_tmd_bulk)

    tr = sub.add_parser("tmd-ribbon", help="Build an MX2 nanoribbon.")
    _add_material_argument(tr)
    tr.add_argument("--width", type=int, default=6, help="Lattice rows across.")
    tr.add_argument("--length", type=int, default=1, help="Repeats along the axis.")
    tr.add_argument("--edge", default="zigzag", choices=["zigzag", "armchair"])
    tr.add_argument("--termination", default="mixed",
                    choices=["mixed", "metal", "chalcogen"],
                    help="Zigzag edges only. MX2's two zigzag edges are "
                         "chemically different — the metal-terminated one is "
                         "metallic and magnetic — so a plain cut ('mixed') "
                         "gives one of each.")
    tr.add_argument("--phase", default="2H", choices=phases)
    tr.add_argument("--out", required=True, help="Output path without extension.")
    tr.set_defaults(func=_cmd_tmd_ribbon)

    tt = sub.add_parser(
        "tmd-tube",
        help="Roll an MX2 monolayer into an (n, m) nanotube.",
    )
    _add_material_argument(tt)
    tt.add_argument("--n", type=int, default=30)
    tt.add_argument("--m", type=int, default=0,
                    help="(n,0) zigzag, (n,n) armchair, else chiral.")
    tt.add_argument("--length", type=int, default=1,
                    help="Translational periods along the axis.")
    tt.add_argument("--phase", default="2H", choices=phases)
    tt.add_argument("--out", required=True, help="Output path without extension.")
    tt.set_defaults(func=_cmd_tmd_tube)

    layers = list(available_layers())
    tw = sub.add_parser(
        "twist",
        help="Stack two hexagonal layers with a commensurate twist (moire).",
    )
    tw.add_argument("--layer", default="graphene", choices=layers,
                    help="Bottom layer.")
    tw.add_argument("--top-layer", default=None, choices=layers,
                    help="Different top layer, making it a heterostructure. "
                         "Its lattice is strained onto the bottom one's.")
    tw.add_argument("--angle", type=float, default=5.0,
                    help="Wanted twist in degrees. Snapped to the nearest "
                         "commensurate angle -- nothing periodic exists in "
                         "between -- and the achieved value is reported.")
    tw.add_argument("--max-index", type=int, default=40,
                    help="Largest m searched. Small angles need large "
                         "indices (the 1.08 deg magic angle is (31,30)) and "
                         "the cell grows as m^2+mn+n^2.")
    tw.add_argument("--gap", type=float, default=3.35,
                    help="Separation between the facing atomic planes (A).")
    tw.add_argument("--out", required=True, help="Output path without extension.")
    tw.set_defaults(func=_cmd_twist)

    st = sub.add_parser(
        "stack",
        help="Stack aligned 2D layers with a van der Waals gap (no twist).",
    )
    st.add_argument("--layers", nargs="+", required=True, choices=layers,
                    help="Layers bottom to top, e.g. graphene hBN graphene.")
    st.add_argument("--gap", type=float, default=3.35)
    st.add_argument("--nx", type=int, default=1)
    st.add_argument("--ny", type=int, default=1)
    st.add_argument("--out", required=True, help="Output path without extension.")
    st.set_defaults(func=_cmd_stack)

    tj = sub.add_parser(
        "tmd-junction",
        help="Build a finite, closed MX2 tube junction (L, T, Y or X).",
    )
    _add_material_argument(tj)
    tj.add_argument("--kind", default="Y", choices=["L", "T", "Y", "X"])
    tj.add_argument("--tube-radius", type=float, default=12.0,
                    help="Arm radius in A. MX2 needs a wide tube: the "
                         "sandwich is ~3 A thick.")
    tj.add_argument("--arm-length", type=float, default=26.0)
    tj.add_argument("--blend", type=float, default=5.0,
                    help="Smoothing radius where the arms meet.")
    tj.add_argument("--parity", default="split",
                    choices=["none", "flip", "split"],
                    help="A junction is genus 0, so 'split' reaches exactly "
                         "zero M-M and X-X bonds -- unlike a schwarzite, "
                         "where homology sometimes prevents it.")
    tj.add_argument("--phase", default="2H", choices=phases)
    tj.add_argument("--seed", type=int, default=0)
    tj.add_argument("--out", required=True, help="Output path without extension.")
    tj.set_defaults(func=_cmd_tmd_junction)

    ts = sub.add_parser(
        "tmd-schwarzite",
        help="Build a periodic MX2 schwarzite (negative-curvature MX2).",
    )
    _add_material_argument(ts)
    ts.add_argument("--kind", default="primitive",
                    choices=["primitive", "diamond", "gyroid"])
    ts.add_argument("--cell", type=float, default=36.0,
                    help="Cubic cell length in A.")
    ts.add_argument("--parity", default="flip",
                    choices=["none", "flip", "split"],
                    help="How hard to push the net toward alternating M/X. "
                         "'none' keeps the best geometry, 'split' the best "
                         "chemistry (fewest M-M and X-X bonds), 'flip' is "
                         "between. The trade is real and monotone.")
    ts.add_argument("--phase", default="2H", choices=phases)
    ts.add_argument("--seed", type=int, default=0)
    ts.add_argument("--out", required=True, help="Output path without extension.")
    ts.set_defaults(func=_cmd_tmd_schwarzite)

    tc = sub.add_parser(
        "tmd-coil",
        help="Coil an MX2 nanotube onto a helix (elastic bend, no defects).",
    )
    _add_material_argument(tc)
    tc.add_argument("--n", type=int, default=30)
    tc.add_argument("--m", type=int, default=0,
                    help="(n,0) zigzag, (n,n) armchair, else chiral.")
    tc.add_argument("--coil-radius", type=float, default=220.0,
                    help="Helix radius in A. Bend strain is R_outer/this, so "
                         "a tight coil is a strained one.")
    tc.add_argument("--pitch", type=float, default=90.0,
                    help="Rise per turn in A.")
    tc.add_argument("--turns", type=float, default=0.5,
                    help="Turns of helix. The atom count scales with this "
                         "times the coil radius.")
    tc.add_argument("--phase", default="2H", choices=phases)
    tc.add_argument("--handedness", default="right", choices=["right", "left"])
    tc.add_argument("--out", required=True, help="Output path without extension.")
    tc.set_defaults(func=_cmd_tmd_coil)

    fu = sub.add_parser(
        "fullerene",
        help="Build a closed icosahedral fullerene cage (C60, C240, C540...).",
    )
    fu.add_argument("--family", default="C60", choices=["C60", "C20"],
                    help="'C60' is the class-II series GP(f,f) — C60, C240, "
                         "C540, radii stepping ~3.55 A. 'C20' is class-I "
                         "GP(f,0) — C20, C80, C180. C60 itself is not "
                         "reachable from the class-I seed at any frequency.")
    fu.add_argument("--freq", type=int, default=1,
                    help="Subdivision frequency; the cage has base*freq^2 "
                         "atoms (60, 240, 540... for --family C60).")
    fu.add_argument("--bond", type=float, default=1.42)
    fu.add_argument("--out", required=True, help="Output path without extension.")
    _add_surface_flags(fu, anneal=False)
    fu.set_defaults(func=_cmd_fullerene)

    on = sub.add_parser(
        "onion",
        help="Build a carbon nano-onion: concentric fullerene cages "
             "(C60@C240@C540).",
    )
    on.add_argument("--shells", type=int, default=3, help="Concentric cages.")
    on.add_argument("--inner-freq", type=int, default=1,
                    help="Frequency of the innermost cage (1 = C60).")
    on.add_argument("--freq-step", type=int, default=1,
                    help="Frequency step between shells; 1 gives ~3.5 A "
                         "spacing for the C60 family, the physical value.")
    on.add_argument("--family", default="C60", choices=["C60", "C20"])
    on.add_argument("--bond", type=float, default=1.42)
    on.add_argument("--out", required=True, help="Output path without extension.")
    _add_surface_flags(on, anneal=False)
    on.set_defaults(func=_cmd_onion)

    co = sub.add_parser(
        "coil",
        help="Build a nanocoil whose ring topology follows the curvature "
             "(implicit route; slower than 'cnt-cap --shape helix', but the "
             "bend is absorbed by pentagon-heptagon pairs instead of strain).",
    )
    co.add_argument("--coil-radius", type=float, default=30.0,
                    help="Helix radius (Å).")
    co.add_argument("--pitch", type=float, default=20.0,
                    help="Axial rise per turn (Å). Must clear two tube walls "
                         "plus a graphitic gap or the turns merge.")
    co.add_argument("--turns", type=float, default=1.5)
    co.add_argument("--tube-radius", type=float, default=6.0,
                    help="Radius of the tube itself (Å).")
    co.add_argument("--handedness", choices=["right", "left"], default="right")
    co.add_argument("--taper", type=float, default=1.0,
                    help="Ratio of final to initial coil radius (<1 = conical).")
    co.add_argument("--bond", type=float, default=1.42)
    co.add_argument("--remesh-iterations", type=int, default=25)
    co.add_argument("--pin-ends", action="store_true",
                    help="Restrain the end caps so the coil keeps its axial "
                         "length. Ring topology fixes curvature but not "
                         "torsion, so a free coil holds its radius and springs "
                         "open in pitch; pinning holds the pitch but strains "
                         "the network, so it is off by default.")
    co.add_argument("--out", required=True, help="Output path without extension.")
    _add_surface_flags(co)
    co.set_defaults(func=_cmd_coil)

    sz = sub.add_parser(
        "schwarzite",
        help="Build a periodic negative-curvature schwarzite unit cell.",
    )
    sz.add_argument("--kind", default="primitive",
                    choices=["primitive", "diamond", "gyroid"])
    sz.add_argument("--cell", type=float, default=36.0,
                    help="Cubic unit-cell length (Å). Bigger cells curve more "
                         "gently and relax cleaner; minimum 30 (primitive), "
                         "36 (gyroid, diamond).")
    sz.add_argument("--thickness", type=float, default=0.0,
                    help="Level-set offset; thins or thickens the channels.")
    sz.add_argument("--bond", type=float, default=1.42)
    sz.add_argument("--grid", type=int, default=64,
                    help="Grid points across one period; 64+ for a clean weld.")
    sz.add_argument("--remesh-iterations", type=int, default=25)
    sz.add_argument("--out", required=True, help="Output path without extension.")
    _add_surface_flags(sz, anneal=0)
    sz.set_defaults(func=_cmd_schwarzite)

    mw = sub.add_parser("mwcnt", help="Build a multi-wall carbon nanotube.")
    mw.add_argument("--shells", type=int, default=2, help="Concentric walls.")
    mw.add_argument("--inner-freq", type=int, default=3,
                    help="Subdivision frequency of the innermost shell.")
    mw.add_argument("--freq-step", type=int, default=2,
                    help="Frequency step between shells; 2 gives ~3.9 Å walls, "
                         "the closest the lattice allows to graphite's 3.4 Å.")
    mw.add_argument("--rings", type=int, default=10, help="Body rings per shell.")
    mw.add_argument("--bond", type=float, default=1.42)
    mw.add_argument("--out", required=True, help="Output path without extension.")
    _add_surface_flags(mw)
    mw.set_defaults(func=_cmd_mwcnt)

    bd = sub.add_parser("bundle", help="Build a hexagonally packed rope of tubes.")
    bd.add_argument("--shells", type=int, default=1,
                    help="Hexagonal shells around the central tube "
                         "(0/1/2 -> 1/7/19 tubes).")
    bd.add_argument("--freq", type=int, default=3, help="Tube diameter control.")
    bd.add_argument("--rings", type=int, default=10, help="Body rings per tube.")
    bd.add_argument("--gap", type=float, default=3.4,
                    help="Wall-to-wall van der Waals gap (Å).")
    bd.add_argument("--bond", type=float, default=1.42)
    bd.add_argument("--out", required=True, help="Output path without extension.")
    _add_surface_flags(bd)
    bd.set_defaults(func=_cmd_bundle)

    vl = sub.add_parser("validate",
                        help="Validate an existing structure file (CIF, XYZ, POSCAR…).")
    vl.add_argument("path", help="Path to a structure file readable by ASE.")
    vl.set_defaults(func=_cmd_validate)

    dp = sub.add_parser(
        "dopants",
        help="List the heteroatoms available for carbon, and what each does.")
    dp.set_defaults(func=_cmd_dopants)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _resolve_material(args)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
