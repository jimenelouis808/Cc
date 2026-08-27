"""``nanocarbon`` command line entry point.

Sub-commands:

* ``cnt``        — build a single CNT and export it.
* ``graphene``   — build a graphene supercell.
* ``ribbon``     — build a graphene nanoribbon.
* ``foam``       — build a 3D carbon foam.
* ``cnt-cap``    — build a fully capped/defected CNT and export XYZ + a
  Blender-ready render bundle (see ``nanocarbon_lab/blender/``).
* ``coil``       — build a nanocoil whose rings follow the curvature.
* ``junction``   — build a capped L/T/Y/X nanotube junction.
* ``schwarzite`` — build a periodic negative-curvature schwarzite unit cell.
* ``mwcnt``      — build a multi-wall nanotube from concentric shells.
* ``bundle``     — build a hexagonally packed rope of tubes.
* ``validate``   — run validation on an existing structure file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ase import io as ase_io

from ..builders import (
    build_capped_cnt,
    build_carbon_foam,
    build_cnt,
    build_graphene_supercell,
    build_bundle,
    build_coil,
    build_junction,
    build_multiwall_cnt,
    build_nanocoil,
    build_nanoribbon,
    build_schwarzite,
)
from ..defects import introduce_vacancies
from ..dopants import dope_random
from ..exports.lammps import write_lammps
from ..exports.qe import QESettings, write_qe_input
from ..exports.xyz import write_render_bundle
from ..validation.checks import run_basic_checks
from ..validation.quality import sp2_quality


def _apply_post(atoms, args):
    if getattr(args, "dopant", None):
        atoms = dope_random(atoms, args.dopant, args.dopant_conc, seed=args.seed)
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
    verdict, why = sp2_quality(g)
    print(f"  sp2 verdict = {verdict.upper()}: {why}")
    return 0


def _add_surface_flags(p):
    """Surface finish and doping, shared by the structure sub-commands."""
    p.add_argument("--anneal-sweeps", type=int, default=80,
                   help="Flip-annealing passes that remove pentagon-heptagon "
                        "pairs beyond those curvature needs (39->14 on a Y "
                        "junction). 0 keeps the as-grown, rougher wall.")
    p.add_argument("--roughness", type=float, default=0.0,
                   help="RMS out-of-plane corrugation (Å) for a CVD-grown "
                        "rather than ideal wall; 0.1-0.3 is realistic.")
    p.add_argument("--dopant", choices=["N", "B", "S", "P"], default=None)
    p.add_argument("--dopant-conc", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)


def _maybe_dope(atoms, args):
    if getattr(args, "dopant", None) and args.dopant_conc > 0:
        atoms = dope_random(atoms, args.dopant, args.dopant_conc, seed=args.seed)
    return atoms


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
    verdict, why = sp2_quality(g)
    print(f"  sp2 verdict = {verdict.upper()}: {why}")


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
    print(f"  coil        = R {atoms.info['achieved_coil_radius']:.1f} A, pitch "
          f"{atoms.info['achieved_pitch']:.1f} A (asked R {args.coil_radius:.0f}, "
          f"pitch {args.pitch:.0f})")
    return 0


def _cmd_validate(args):
    atoms = ase_io.read(args.path)
    report = run_basic_checks(atoms)
    print(report.summary())
    return 0 if report.ok else 1


def _add_common(p):
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--format", choices=["qe", "lammps", "both"], default="qe")
    p.add_argument("--calculation", default="scf",
                   choices=["scf", "relax", "vc-relax", "nscf", "bands"])
    p.add_argument("--bond", type=float, default=1.42, help="C-C bond length (Å).")
    p.add_argument("--vacuum", type=float, default=15.0, help="Vacuum padding (Å).")
    p.add_argument("--dopant", choices=["N", "B", "S", "P"], default=None)
    p.add_argument("--dopant-conc", type=float, default=0.0)
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
    cc.add_argument("--dopant", choices=["N", "B", "S", "P"], default=None)
    cc.add_argument("--dopant-conc", type=float, default=0.0)
    cc.add_argument("--seed", type=int, default=0,
                    help="RNG seed for defect placement, roughness and doping.")
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
    sz.add_argument("--cell", type=float, default=32.0,
                    help="Cubic unit-cell length (Å). Minimum depends on the "
                         "surface: 20 (primitive), 22 (gyroid), 30 (diamond).")
    sz.add_argument("--thickness", type=float, default=0.0,
                    help="Level-set offset; thins or thickens the channels.")
    sz.add_argument("--bond", type=float, default=1.42)
    sz.add_argument("--grid", type=int, default=64,
                    help="Grid points across one period; 64+ for a clean weld.")
    sz.add_argument("--remesh-iterations", type=int, default=25)
    sz.add_argument("--out", required=True, help="Output path without extension.")
    _add_surface_flags(sz)
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
