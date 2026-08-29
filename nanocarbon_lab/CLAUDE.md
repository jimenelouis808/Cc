# CLAUDE.md

This file gives instructions to Claude Code (or any assistant) working on this repository.

## Project scope
`nanocarbon_lab` is a modular Python framework to **generate, validate and export** nanocarbon structures (1D/2D/3D) for first-principles (Quantum ESPRESSO) and classical molecular dynamics (LAMMPS) simulations.

The framework must remain **scientifically valid**: physical bond lengths, correct coordination, no impossible geometries, reproducible dopant / defect placement.

## Repository layout
```
nanocarbon_lab/
├── builders/      # 1D/2D/3D structure generators incl. nanocoils (ASE-compatible Atoms)
│                  #   centerline.py: 3D path sweep (arc/S/helix/random) + strain budget
│                  #   implicit.py + remesh.py + junction.py: L/T/Y/X junctions and
│                  #   schwarzites, via SDF -> marching cubes -> isotropic remesh -> dual
│                  #   capped_cnt.py + fullerene_mesh.py: finite capped/defected
│                  #   "elongated fullerene" CNTs for rendering (see below)
│                  #   swept.py: coils/curved tubes via the implicit route, so
│                  #   ring sizes follow the curvature instead of straining
│                  #   assemblies.py: multi-wall tubes and bundles at the vdW gap
│                  #   fullerene.py: closed cages (C60, C240...) and nano-onions
├── dopants/       # substitutional dopants (N, B, S, P, co-doping)
├── defects/       # vacancies, Stone-Wales, topological defects
├── topology/      # networkx-based connectivity / coordination analysis
├── validation/    # bond lengths, coordination, density, vacuum checks
├── exports/       # Quantum ESPRESSO + LAMMPS writers, plain XYZ + Blender render bundle
├── relax/         # ASE optimizer wrapper + calculator-free harmonic pre-relax
├── viz/           # matplotlib 3D viewer
├── workflows/     # batch generation + ML dataset exporter
├── utils/         # constants, geometry helpers
├── cli/           # command line interface
├── gui/           # tkinter desktop app (build / preview / export / render)
├── tests/         # pytest unit tests
└── examples/      # runnable example scripts

blender/           # sibling to nanocarbon_lab/: bpy-based rendering pipeline,
                   #   run via `blender -b -P blender/render_cnt.py -- ...`
```

## Capped/defected CNT topology (fullerene_mesh)

`builders/fullerene_mesh.py` is the engine behind `build_capped_cnt`. It
works on the honeycomb's **triangulated dual** (each ring = one mesh
vertex; vertex degree 5/6/7/8 = pentagon/hexagon/heptagon/octagon), so
every ring edit (`edge_flip` = Stone-Wales 5-7-7-5, `contract_edge` =
divacancy 5-8-5, the seed polyhedron's poles = 6-pentagon caps) is a
provably Euler-consistent combinatorial operation, never a geometric
heuristic. **Do not** try to detect or edit rings by re-deriving them
from atom distances/`networkx.cycle_basis` on a curved/periodic shell --
that path was tried during development and produced silently wrong ring
counts (see the module docstring). If you add a new defect type here, add
it as a mesh-level operation with an Euler-invariant unit test (see
`tests/test_capped_cnt.py::TestFullereneMeshPrimitives`), not as a
post-hoc geometric edit.

## Golden rules for contributors / assistants
1. **Python >= 3.10**, type hints on all public functions.
2. **Docstrings** are mandatory. Each public function must document inputs, outputs, and (when relevant) the physical assumption.
3. Every builder must return an `ase.Atoms` with correct `pbc` and `cell`.
4. Every structure exported to QE or LAMMPS **must pass `validation.run_basic_checks`** first.
5. Random operations (dopants, defects, disordered foams) **must accept a `seed`** for reproducibility.
6. **No hardcoded paths**. Use `pathlib.Path` and user-supplied output directories.
7. **No monolithic scripts**: keep modules small and composable.
8. New features ship with a matching pytest test in `tests/`.

## Geometry quality (not just topology)

Correct ring counts are necessary but **not sufficient**: a shell can have
a perfect Euler budget and still be geometrically absurd. `build_capped_cnt`
relaxes against a valence force field (bond + true angle + non-bonded
repulsion, L-BFGS, exact analytic gradients) and records measured
statistics in `atoms.info["geometry"]`.

Two failure modes are already fixed here; do not reintroduce them:
1. **No real angle term.** Bond springs alone (or a 1-3 *distance* proxy)
   let the sheet pyramidalise and fold — this produced 66-164 deg angles
   with perfect-looking bond lengths.
2. **Taking the dual before smoothing.** Barycentric subdivision gives
   unequal triangles; atoms must be placed from a mesh already projected
   and Laplacian-smoothed onto the capsule, or structures beyond ~1000
   atoms fold through themselves.

Any change to the builder or relaxer must keep
`tests/test_capped_cnt.py::TestBuildCappedCNT::test_geometry_is_realistic_sp2`
passing: bonds 1.30-1.55 Å, angles 100-135 deg, zero sub-2 Å non-bonded
contacts, across straight, bent and defected cases.

**Curvature is limited by a strain budget**, not by taste: outer-wall
strain is `r_tube * kappa`, and `builders/centerline.py` trims a path's
amplitude until it fits (default 8%; >15% warns). Sweeping uses arc-length
parameterisation and a **rotation-minimizing frame** — never a Frenet
frame, whose normal flips 180 deg at every inflection point and would
shear a meandering tube apart.

Tube **radius is quantised** by the lattice (`R = 5*freq*sqrt(3)*bond/2pi`),
exactly as a real (n,m) tube's diameter is fixed by its indices. It is an
output, not a free input; `target_radius` picks the nearest realisable freq.

## Junctions and schwarzites (implicit route)

`junction.py` starts from a signed-distance field rather than a seed
polyhedron. The invariant to protect: **mesh vertex degree == carbon ring
size**, so the isotropic remesher in `remesh.py` is not cosmetic — without
it, marching cubes' degree-3 and degree-9 vertices become three-membered
rings and nine-membered holes. `_remove_low_degree_vertices` exists
specifically to kill degrees < 5.

Two subtleties already fixed; do not reintroduce:
1. **Edge collapse needs locking.** Adjacency is cached per pass, so after
   a collapse it is stale nearby; validating later collapses against it
   silently admits ones that tear the surface (this produced meshes with
   tens of boundary edges that happened to heal later).
2. **Fields must share units before combining.** The trigonometric
   schwarzite field is unitless; intersecting it with a ball's Å-valued
   SDF did nothing until `normalize_to_distance` was applied.

Periodic schwarzites add two requirements. Every geometric step —
remesh, dual, relaxation, geometry report — must use `minimum_image`, or a
bond across the cell seam reads as a cell-length stretch. And the **cell
must be relaxed with the atoms** (`CELL_RELAX_CYCLES`): holding it fixed
left Schwarz D with 6 Å bonds and 28 overlaps. `_finish` ends with a hard
quality gate that raises rather than returning a torn network — thresholds
are set far outside anything strain explains.

The Euler check is **genus-derived** (`deficit == 6 * chi`), never the
tube builder's hardcoded 12 — a schwarzite legitimately has a strongly
negative deficit. For an *assembly* (MWCNT, bundle) the budget is 12 per
disjoint shell, not 12 overall.

**Do not raise `anneal_sweeps` for schwarzites.** It defaults to 0 there
and to 80 for junctions, and that asymmetry is measured, not an
oversight. On a minimal surface the 5-7 pairs are how a hexagonal net
covers the saddle curvature; annealing them away forces the remaining
bonds to stretch. Schwarz P at 36 Å goes clean → strained → broken as
sweeps go 0 → 20 → 80, and the pattern held for every surface and cell
size tried. A high stray-pair count here means the surface is being
tiled correctly. `MIN_SCHWARZITE_CELL` was likewise raised to where the
*geometry* stops being broken, not merely where the mesh stops tearing.

## Curved tubes: swept vs implicit

There are two routes and they are not interchangeable. `build_capped_cnt(
shape=...)` sweeps a finished all-hexagon tube onto a centreline, so the
bend is carried as **elastic strain** — a geometric necessity, not a
relaxation failure, and no anchor tuning recovers it (measured: 6.5% path
strain gives 1.33–1.51 Å bonds at every anchor stiffness tried).
`builders/swept.py` meshes the curved surface implicitly instead, so ring
sizes follow the curvature (85 pentagons / 71 heptagons on a 1.5-turn
coil) and bonds return to graphitic length.

Two properties of that route must not be re-broken:
1. **Ring topology encodes curvature, not torsion.** A free coil keeps
   its radius (29.4 Å for a requested 30.0) but springs open along its
   axis (20 → 26.7 Å pitch). `_finish`'s `pin_near` can hold the ends, but
   it is **off by default and should stay that way**: at `k_pin=5` the
   same coil came out with 1.18–1.71 Å bonds and 3 overlapping pairs,
   failing the quality gate the free relaxation passes. Report the
   achieved pitch; do not hold a requested number at the cost of the
   chemistry.
2. **`build_coil` refuses a pitch below `2*tube_radius + 3.4`.** Below
   that the surface merges adjacent turns into one solid and the result
   is not a tube at all.

## Fullerene cages: the seed decides the cage

`builders/fullerene.py` is the `half_length = 0` limit of the capped tube
— a sphere, reusing the same dual/Euler/VFF machinery. Two seeds, and the
second is not optional: the icosahedron gives only the class-I series
GP(f,0) (C20, C80, C180), which **does not contain C60 at any
frequency**. C60 needs the pentakis dodecahedron (12 degree-5 + 20
degree-6 vertices), whose dual is the truncated icosahedron.

Both seeds are convex hulls of points on the unit sphere — the hull of
points on a sphere is their Delaunay triangulation, so connectivity
cannot be miswritten by hand. `_hull_mesh` re-orients every triangle
outward; mixed winding would make the dual order a ring's atoms into a
self-crossing polygon.

The class-II radius step (~3.5 Å per frequency) is what makes a graphitic
nano-onion possible; class-I's ~2.0 Å step reaches 3.4 Å at no
`freq_step`. Do not "simplify" the onion onto the class-I seed.

## Relaxation: the neighbour list needs a skin

`relax_shell`'s non-bonded list is frozen for a whole L-BFGS run. Built at
exactly `repel_cutoff`, two atoms further apart than that are invisible to
each other for thousands of iterations and pass straight through. Compact
shells never noticed; a 284 Å coiled tube drifted 6.5 Å per atom and fused
neighbouring turns, 370 sub-2 Å contacts between atoms 135 bonds apart.
`repel_skin` (default 5 Å) fixes it, with a **conservative Verlet
rebuild**: restart when the two largest displacements sum past the skin.
Do not tighten that to "any atom moved half the skin" — ordinary local
rearrangement is ~1 Å and would restart L-BFGS (discarding its history)
continuously, tripling the runtime.

## GUI: one job description, one killable process

`jobs.py` is the single mapping from "what to build" to builder
arguments, shared by the GUI and the CLI. The GUI used to carry its own
ninety-line `if mode == ...` chain duplicating it. Three features depend
on that mapping being written down once — the estimate, the
copy-as-command-line button, and handing work to a subprocess — so add
new modes there, not in `gui/app.py`.

`gui/worker.py` runs builds in a **process**, not a thread, because a
coil spends minutes inside numpy with nothing checking a cancel flag and
Python cannot safely interrupt a thread. Consequences to respect:

* Anything constructing `NanocarbonGUI` needs an
  `if __name__ == "__main__"` guard — `spawn` re-imports the parent's
  `__main__`. `gui/__main__.py` has one for exactly this reason.
* Errors cross the process boundary as `(repr, traceback)` **strings**;
  do not try to send exception objects, which may not round-trip.
* If spawning fails at all, the worker degrades to a thread rather than
  refusing to build; cancel is then advisory and `worker.degraded` says
  so. Keep that fallback.

The GUI never opens a modal dialog. `messagebox` is deliberately not
imported: a modal blocks the Tk event loop, which wedges a headless run
entirely, and it discards whatever the user was about to fix.

## Say whether the geometry is physical, not just what it measures

`validation/quality.sp2_quality` turns `atoms.info["geometry"]` into
`CLEAN` / `STRAINED` / `BROKEN` with a reason, and both the CLI and the
GUI print it. It exists because "0 close contacts" was being read as "the
structure is fine": an over-tight coil keeps its atoms apart while
stretching bonds to 1.69 Å, longer than any real C–C bond. Keep new
builders reporting it.

## Scientific guardrails
- Carbon bond length: default 1.42 Å (sp2). Accept anything in `[1.20, 1.80]` Å as bonded; anything in `(0, 0.9]` Å is a hard error.
- Expected C coordination: **2 (edge)**, **3 (sp2 bulk)**. Coordination >= 5 or == 1 in the bulk is rejected by validation.
- Dopants replace carbon atoms: N, B (sp2 compatible, coord 3), S, P (larger, often require local relaxation). The module flags clustered dopants as warnings, not errors.
- 2D structures: vacuum along the non-periodic direction must be `>= 12 Å` by default.
- 1D structures (CNT): vacuum in the two transverse directions must be `>= 10 Å` beyond the tube radius.

## Quick test commands
```bash
pip install -e .[dev]
pytest nanocarbon_lab/tests -q
pytest nanocarbon_lab/tests -q -m "not slow"   # skips the minutes-long coil builds
python -m nanocarbon_lab.cli.main cnt --n 6 --m 6 --length 10 --out out/cnt --format qe
python -m nanocarbon_lab.cli.main cnt-cap --rings 8 --freq 3 --defect stone_wales:1 --out out/cnt_cap/demo
# GUI (needs tkinter + matplotlib); headless GUI tests:
xvfb-run -a pytest nanocarbon_lab/tests/test_gui.py -q
```

## Where to add things
- New builder type → `builders/<name>.py` + export in `builders/__init__.py` + test in `tests/test_<name>.py`.
- New dopant chemistry → `dopants/<element>.py`, reuse `dopants.base.substitute_atoms`.
- New exporter → `exports/<backend>.py` implementing `write(atoms, outdir, **kwargs)`.

## What not to do
- Do **not** export structures that fail validation without explicit `force=True`.
- Do **not** introduce non-deterministic random state without a `seed` argument.
- Do **not** add dependencies outside the ones declared in `pyproject.toml` without justification.
