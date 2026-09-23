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
│                  #   network.py: periodic 3D nets of interconnected tubes
│                  #     (cubic, diamond), same implicit route as the schwarzite
├── tmd/           # MX2 dichalcogenides: materials.py (lattice constants),
│                  #   slab.py (mono/multi/bulk), ribbon.py, nanotube.py,
│                  #   coil.py (swept helical tubes), curved.py (schwarzites
│                  #   on a TPMS, with the M/X parity repair), modify.py
│                  #   (Janus, alloys, vacancies, antisites), quality.py.
│                  #   Deliberately NOT under builders/.
├── hetero/        # twisted bilayers and vdW stacks (moire.py). Above both
│                  #   builders/ and tmd/ because it composes them.
├── functionalize/ # surface groups grafted onto a finished structure:
│                  #   groups.py (a Z-matrix grammar, so an element swap
│                  #   rebuilds the bond lengths), attach.py (surface
│                  #   normals, site selection, steric placement)
├── dopants/       # substitutional heteroatoms in carbon: chemistry.py
│                  #   (which elements, what site, how much), rings.py
│                  #   (pentagon-selected placement), substitutional.py,
│                  #   codoping.py (several elements in one pass, each with
│                  #   its own fraction, and a seek/avoid correlation)
├── defects/       # vacancies, Stone-Wales, topological defects
├── topology/      # networkx-based connectivity / coordination analysis
├── validation/    # bond lengths, coordination, density, vacuum checks
├── exports/       # Quantum ESPRESSO + LAMMPS writers, plain XYZ + Blender render bundle
├── relax/         # ASE optimizer wrapper + calculator-free harmonic pre-relax
├── viz/           # matplotlib 3D viewer
├── workflows/     # sweep.py: Cartesian product over jobs.Job, so every
│                  #   mode is sweepable; batch.py + ml_dataset.py write
│                  #   the QE/LAMMPS inputs, features CSV and manifest
├── analyse/       # describe a structure this package did NOT build:
│                  #   rings.py (faces of the embedded surface, with
│                  #   shortest-path rings as the fallback), shape.py
│                  #   (dimensionality measured, not read off pbc),
│                  #   report.py (recorded vs measured vs inferred)
├── cell.py        # any structure -> a DFT-ready periodic unit cell
├── utils/         # constants, geometry helpers
├── cli/           # command line interface
├── gui/           # tkinter desktop app (build / preview / export / render)
├── tests/         # pytest unit tests
└── examples/      # runnable example scripts

nanocarbon_lab/blender/   # bpy-based rendering pipeline, shipped as package
                   #   *data* (it imports bpy, so it must not be a subpackage).
                   #   Run through a Blender app (`blender -b -P ...`) or
                   #   directly, when the `bpy` wheel is installed.
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

## Haeckelites: topology from the mesh, geometry from the honeycomb

`builders/haeckelite.py` designs 2D carbon allotropes by patterning
Stone-Wales rotations into graphene. It is the flat analogue of
`fullerene_mesh`, and the reason it needs its own module is that the flat
case breaks the rule the curved one taught.

**The Euler budget is the easy half.** A 2D periodic sheet is a torus, so
`sum(6-n) == 0` exactly, and an edge flip drops two mesh-vertex degrees and
raises two — paying zero. So **no pattern can break the budget**, however
many flips and wherever they go. A lattice nobody has published is as sound
topologically as R5,7, and the census is read off vertex degrees rather than
perceived. Do not re-derive it from distances; that is the failure
`fullerene_mesh` exists to prevent, and bare graph rewiring (swap one
neighbour between two atoms, degrees all stay 3) produces a graph that no
longer embeds in the torus at all — 2.3 Å bonds beside a perfect census.

**The geometry must not come from the mesh.** This is the new rule and it
contradicts `tmd/curved.py`'s "relax the site net, not the atoms", which is
right for a closed shell and wrong here. Equilateral triangles meeting five
at a vertex sum to 300°, seven to 420°, so a *flat* triangulation carrying
pentagons and heptagons **can never have equal edges**. Asked for them, the
mesh relaxation sat exactly still — the flipped lattice is a genuine
minimum of the edge springs, the long diagonals' forces cancelling to
2.6e-13 by symmetry — and the dual came out with 0.82 Å bonds no later
relaxation could fix. Smoothing first changed nothing (bit-identical at 0,
20 and 100 rounds), because the objective, not the optimiser, was wrong.

So geometry comes from the honeycomb side: graphene's exact dual, with each
rotated **dimer turned 90° in the plane** about its midpoint, which is how
a Stone-Wales rotation is actually drawn. A single defect built this way
relaxes to **1.322–1.481 Å and 103.5–136.7°** at graphene's own cell — the
published 5-7-7-5 geometry, and the one case where the engine checks
against the literature rather than itself. Three things are load-bearing:

* **The turn's sense is measured, not derived.** Only one of the two
  matches the rewiring the flip performed; the other puts each new partner
  at 2.40 Å instead of 1.51 — silently, since the topology is impeccable
  either way. `_turn_dimers` tries both per dimer.
* **Triangle indices move, so the atom behind them must be tracked.**
  `edge_flip` appends its two new triangles at the end rather than
  replacing in place. `apply_flips` carries a `labels` list; without it the
  rewired bonds and the rotated positions describe different atoms.
* **Rule A sets the density; Rule B is implied by it.** Rule A: each mesh
  vertex may be touched by at most one flip over the whole build, so every
  ring ends at 5, 6 or 7 (without it a run returned squares and nonagons).
  It also states the ceiling — touching all V vertices takes V/4 flips,
  which is a lattice with **no hexagons left**. Rule B: two rotations may
  not share a bond, or each turns an atom the other needs and the pair
  lands 3.4 Å apart. That failure was real, but under the *older* selection
  which kept only the flipped edges' endpoints disjoint. Once Rule A covers
  all four touched vertices it **implies** Rule B: triangles from
  vertex-disjoint flips share no vertex, and sharing an edge needs two.
  Measured across every pattern and cell size, Rule B defers nothing. The
  round loop behind it is a safety net that demonstrably runs once — do not
  describe it as what makes a dense lattice reachable.

* **The catalogue is solved, not asserted.** A zero-hexagon census needs
  the flips to *partition* the vertex set into groups of four — an exact
  cover, and one exists (14 search nodes at 4×4). But the census does not
  pin the arrangement: 24 covers enumerated there relaxed anywhere from
  1.195–1.685 Å to 1.321–1.520 Å. `CATALOGUE` stores the **lowest-strain**
  one in block coordinates and the builder tiles it, which reproduces
  identical geometry at 4×4, 8×4 and 4×8 — the sharpest available check on
  the tiling. Write an edge's far end as `i + 1`, never its wrapped index:
  `(3, j, 1)-(0, j, 1)` tiled correctly in j and silently stopped being an
  edge at all in i. **These cells are this framework's own relaxed
  numbers, not published lattice constants**; the docstring, the CLI and
  the GUI hint all say so, and they must keep saying so.

* **A catalogue entry refuses a cell its block does not divide.** The
  hexagons a partial block leaves behind are precisely the ones the lattice
  is defined by not having, so that is a different material, not a near
  miss. The GUI hint says this before the build rather than after.

* Measured: `r57` is 100% non-hexagonal at 1.324–1.524 Å and 101.4–138.6°;
  the generative `dense` reaches 66.7%. Every lattice either route produces
  passes the sp2 gate, so the gate in `build_haeckelite` is a guard rather
  than the mechanism.

* **The cell search must be fine, not wide.** The old ±36% first pass
  stepped in 9% jumps and straddled the optimum: the dense lattice scored
  1.217–1.558 Å under it and 1.284–1.479 Å at 3% steps, on identical
  topology. Granularity was the limiter, not physics — and the wide range
  was only ever needed because a crumpling sheet could pretend to want a
  far smaller cell.

**The sheet is kept flat on purpose.** The force field has bond and angle
terms but no flexural one — a sheet resists bending through its π system —
so long-wavelength wrinkling is nearly free in it, and given out-of-plane
freedom the cell search took it: a 4×4 R5,7 came back at **1.51 Å² per atom
against graphene's 2.619**, having bought low bond strain by crumpling into
a smaller footprint. Relaxing in-plane removes a degree of freedom the force
field cannot price. Do not reintroduce a buckling nudge and do not report a
buckling amplitude; it would be an artefact. Every pattern now lands at
2.62–2.68 Å² per atom, and a test pins that band.

**A catalogue entry is exact; a generative pattern is a request.** For
`r57` every rotation is applied and `n_flips_refused` is 0. For the
generative rules the gap is large — `dense` at nx=ny=6 applies 12 of the 36
edges it names. Both numbers go into `info`, the CLI prints the ratio and
labels which kind it was. Read the census, not the pattern name. (The
generative dense pattern was itself called `r57` before the catalogue
existed, which overclaimed it; it is `dense` now.)

**`sp2_quality` takes a `family`.** A regular heptagon's interior angle is
128.6° before any strain, so a haeckelite reaches 136–140° as geometry and
the hexagonal window called every sound one BROKEN — the same mistake
`tmd_quality` made counting a grafted hydrogen as a metal. Builders record
`info["quality_family"]` and `_report_structure` reads it. A new family with
its own legitimate angle range needs an entry in `QUALITY_WINDOWS`, not a
loosened sp2 window.

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

**Do not raise `anneal_sweeps` on any curved surface.** It defaults to 0
for the schwarzite, the network, the junction and the swept coil alike.
The junction and the coil defaulted to 80 for a long time on the strength
of the ring census, which annealing does improve; what it costs was only
measured later. On four junction kinds out of four the as-grown wall is
smoother, on the Y by more than a factor of two, and on three coil
geometries out of three the annealed tube comes out up to 55% too fat
while its helix springs open (a 4.5 Å tube walled at 6.96 Å, a requested
13 Å pitch achieved at 40.2). A spread-out population of 5-7 pairs lets
the net take up curvature everywhere at once; annealing most of them away
leaves the survivors carrying all of it, and each buckles the wall around
it. Judge these by the wall, not by the census -- and note that the sp2
verdict follows the census, so it moves the wrong way too: an L or a T
junction reads "strained" at 0 sweeps purely because a regular heptagon's
interior angle is 128.6 deg before any strain. On a minimal surface the 5-7 pairs are how a hexagonal net
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

## Nanocoils: two ways to break, with opposite cures

`builders/nanocoil.py` winds a finished `(n, m)` lattice along a helix,
so every ring stays a hexagon and the wall stays graphitic. That is the
reason to prefer it over `swept.py`'s meshed coil, which relieves
curvature with 5-7 pairs and comes back with ~90 non-hexagonal rings
where Euler needs 12 -- sound geometry, disordered wall. It is also two
orders of magnitude faster: a 4602-atom rolled coil takes 1.4 s against
167 s for a 2386-atom meshed one, because nothing is meshed or relaxed.

Bending a finished lattice can only **stretch** it, so two separate
things ruin the result and they must be reported separately:

* **Wall strain**, `r_tube * kappa`. Cured by widening the coil.
  `_clean_coil_radius` inverts the relation: a (5,5) tube at a 12 Å pitch
  needs about 42 Å before it fits the 8% budget. Small graphitic coils
  are not a tuning problem -- they do not exist, which is why real carbon
  nanocoils are tens to hundreds of Å across.
* **Turns colliding**, and this one **does not depend on the coil radius
  at all**. A (10,10) tube is 13.6 Å across, so at a 13 Å pitch it passes
  through its own next turn at every radius: measured at R = 120 Å, where
  the strain is a comfortable 5.6%, 5211 overlapping pairs and 0.69 Å
  bonds. Only raising the pitch above `2*r_tube + 3.4` helps. Widening
  the coil makes more of the collision, not less.

Both guards were absent until the builder was exposed in the GUI, and its
tests did not catch it: they asserted `1.2 < nearest_neighbour < 1.8`, a
window that accepts two carbons on top of each other. **All six** cases
that file built were torn. Judge with `sp2_quality` like everything else;
do not widen a tolerance to make a case pass.

Single-turn coils are a special case worth knowing about: with one turn
there is no next turn, only the two free ends, and whether those overlap
depends on the radius (R = 80 Å builds, 50 and 120 do not). Tests about
turn collision ask for two turns.

## A coil is a polygon, and its pentagons go outside

Two studies of single-wall coils, read against the periodic builder
(Popović et al., *Contemp. Mater.* III-1 (2012) 51; Liu et al.,
*Nanoscale Res. Lett.* 5 (2010) 478):

* **The surface is torus-like, so the Gaussian curvature is positive on
  the outer equator and negative on the inner one.** A pentagon is a
  +60° disclination and a heptagon a −60° one, so the pentagons belong
  on the outside and the heptagons on the inside — both papers say it in
  those words. It is the one structural claim about a coil that can be
  checked on a finished model without running anything, and
  `curvature_check` does. Measured on the smooth helix: 82 % of the
  non-hexagons land correctly, so the remesher gets the physics mostly
  right and not entirely. The ones on the wrong side are defects of the
  model and are reported as defects, not rounded away.
* **Seen down the axis a real coil is a POLYGON.** Liu et al. show the
  (6,6) coil's top view as a hexagonal torus and say it matches what is
  observed: the wall relieves its strain at a few knees, not everywhere
  at once. Measured at `coil_radius=8.75, pitch=9.6, tube_radius=3.0`
  (D/d = 3.92, inside the single-wall band), six sides per turn put
  **85% of the disclinations on their correct side** -- 9 of 10
  pentagons outside and 8 of 10 heptagons inside -- against **68%** for
  the smooth helix at the same parameters (8 of 11 and 7 of 11). The
  polygon is better, and it is better by a wide margin.

  **Quote these numbers with the geometry they came from.** An earlier
  version of this file claimed "8 and 8 with every pentagon outside",
  which does not reproduce: at D/d = 6.00 the hexagon is *worse* than
  the smooth helix (21 wrong against 15). The placement depends on the
  coil's proportions, so a census without its parameters is not a
  measurement of anything, and that is how the two got separated.
* **D/d is about 3.5.** Popović finds it for both of their classes; Liu's
  Table 2 gives 3.53 to 3.88 for the (5,5) through (8,8). A coil outside
  that band is not wrong — multi-wall coils reach ten and more — but it
  is not the single-wall geometry those calculations relaxed to, and the
  builder says so.
* **Tight is not impossible, and the guard conflated them.** Turns that
  INTERSECT cannot be built. Turns merely closer than a graphitic gap are
  tight, and the published single-wall coils ARE tight: Liu's relaxed
  (7,7) has a 12.11 Å pitch around a 9.52 Å tube, a 2.59 Å gap. The old
  `2*r_tube + 3.4` guard refused a structure somebody had already relaxed
  with DFT. It now refuses only `pitch <= 2*r_tube` and labels the rest.

### A polygon corner must not land on a sample

The centreline is sampled on a grid that starts at a turn boundary and
holds a whole number of samples per turn — both are needed for the field
to be periodic to machine precision, and the module's own docstring says
so. If the side count **divides** that sample count, a corner lands
exactly on a sample; the tangent there is two-valued, the swept frame
flips, and the seam does not weld. Measured: 5 and 10 sides tore against
4000 samples per turn while 6, 7, 8 and 12 welded, and **raising the
resolution made it worse** (208 boundary edges at 64, 332 at 96), which
is how it was told apart from a mesh that is merely too coarse. The grid
is chosen coprime with the side count. Rounding the corners instead does
not work: it shortens the path, the analytic spacing stops dividing one
turn, and every side count tears.

## The bend angle's limit is the tube's, not a constant

`build_capped_cnt(bend_angle=...)` used to reject anything past a flat
`MAX_PHYSICAL_BEND = 1.0` rad. The bend is imposed as an arc whose length
is the tube's own axial span, so the outer wall stretches by
`r_tube * angle / span` -- a number that depends on the tube as much as
on the angle, and a fixed cap is therefore wrong in both directions.
Measured: the default 8-ring freq-3 tube is already at 12% strain at 1
rad and tears, while a 30-ring freq-2 tube takes **150 deg** (8.3%) and
stays intact. The builder now warns past `DEFAULT_MAX_STRAIN` and lets
the existing quality gate refuse what actually tears; the GUI shows the
angle in degrees, because 1.0 rad reads as a mysteriously small number.

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

**Wrapping for `cKDTree(boxsize=...)` needs more than `np.mod`.** Two traps,
both of which crashed a sound structure from three frames away with a
message naming neither. A `0` edge means "not periodic along this axis"
(what a 2D sheet's vacuum direction needs), and `np.mod(x, 0)` is nan, which
arrives as "data must be finite". And for a tiny negative input `np.mod`
returns the edge **exactly** — `np.mod(-1e-18, 10.0)` is `10.0` — which the
tree rejects as outside the box, though an atom a hair below the cell origin
is perfectly ordinary. Both live in `relax_shell` and in
`capped_cnt.geometry_report`; the fix is a mask for the live axes plus a
clip to `np.nextafter(edge, 0)`. Any new `cKDTree(boxsize=...)` call needs
the same two.

## Dichalcogenides are not decorated carbon

`tmd/` is a separate package on purpose. An MX2 layer is a three-plane
X-M-X sandwich, so the carbon machinery does not transfer: the metal is
six-coordinate, the bond is 2.4 Å, there are no rings to count, and the
sp2 verdict's thresholds are all wrong. It has its own `quality.py`.

Rules that are easy to break here:

* **Geometry is `a` and `h`; the bond is derived.** Storing `d` as well
  invites the three to disagree. `d = sqrt(a^2/3 + h^2/4)` reproduces the
  literature 2.41 Å for MoS2.
* **The phase is only about where the bottom chalcogen plane sits** — 2H
  eclipsed (trigonal prismatic), 1T staggered (octahedral). No TMD has a
  tetragonal phase; do not add one.
* **2H stacking is a 6_3 screw, not a rotation about the metal.** Negating
  the fractional coordinates alone leaves the metal fixed and stacks metal
  on metal, which is AA. The `+(1/3, 2/3)` translation is what puts the
  metal over the chalcogen. 2H and 3R are identical as bilayers and
  diverge at the third layer.
* **1T' needs a doubled cell.** A cell with one metal has no partner to
  dimerise with. Double, then distort, then repeat.
* **Ribbon terminations are deliberately off-stoichiometry.** Pass
  `expect_stoichiometric=False` to `tmd_quality` for them.
* **Rolling strains the sandwich** by `h/2R`, unavoidably. That is why
  real MX2 tubes are tens of nm across; the builder warns past 10%.

**Coils sweep; they do not remesh.** `tmd/coil.py` bends a finished tube
onto a helix, so every ring stays a hexagon. The implicit route is not an
option for MX2: it absorbs curvature by introducing odd rings, and an odd
ring here forces an M-M or X-X bond. Report the roll and bend strains
*separately* -- they have opposite cures (widening the tube cuts `h/2R`
and raises `R_outer*kappa`), so their sum alone tells the user nothing
actionable. `sweep_along_path` rescales the path to the structure, so
pick the period count to match the arc and report the **achieved**
radius and pitch, never the requested ones.

**Schwarzites live in `tmd/curved.py`** and their reasoning is the part
most easily got wrong, in both directions.

An earlier version of this file said MX2 schwarzites were impossible
because pentagons are forbidden. Pentagons are forbidden, but that only
rules out the *sphere*: negative curvature wants **octagons**, which are
even. `sum(6-n) = 6*chi`, each octagon pays -2, Schwarz P wants twelve.
Do not reinstate the old claim.

Atoms are triangle centroids bonded across shared edges, so ring size ==
mesh vertex degree and M/X alternation == every degree even. Two repairs,
and the difference is not stylistic:

* **flip** toggles four degrees at once, so it can only shuffle sparse
  odd vertices and never reaches zero -- but it adds no vertices, so the
  geometry survives.
* **split** toggles exactly the two vertices opposite the edge. That
  weight-two move *does* reach zero, at one new vertex each; enough of
  them put more sites on the surface than its area holds at `a/sqrt(3)`.

Both are exposed as `parity` because the trade is real and monotone
(30 A Schwarz P: none 12.4% homoelemental / 6.2% p95 strain, split 0.0%
/ 13.4%). Do not quietly pick one.

Things already measured; do not re-derive them the hard way:

* **Even degrees are a sphere result.** At genus g there are 2g more Z/2
  classes, and even-degree meshes exist on both sides. So a perfectly
  bipartite cell is *reachable but not guaranteed* -- 30 A split gives
  zero homoelemental bonds and X/M = 2.0000; 36 A split gives 2.2%. What
  is left over is an inversion-domain boundary, so report the count.
* **Relax the site net, not the atoms.** The sites form a trivalent net
  identical to graphene's, so `relax_shell` at 120 deg and `a/sqrt(3)`
  applies directly and its angle term is what stops the sheet folding.
  Relaxing the finished MX2 atoms needs an angle target that fits both a
  3-coordinate chalcogen (~82 deg) and a 6-coordinate metal (several
  values at once); there is none.
* **`exclude_13=False` for the atom relaxation.** With `k_angle=0` and
  1-3 pairs excluded from the repulsion, two chalcogens on the same metal
  have *nothing* holding them apart -- that was 70 sub-2 A pairs, and
  keeping them in the repulsion made it zero. Carbon keeps the default
  True: its angle term owns those pairs.
* **Per-bond `equilibrium`.** Homoelemental defect bonds are not the M-X
  length; forcing them to it moved the worst bond from 13.6% to 24.2%.
* **Normals come from the relaxed net, sign-propagated across it.** The
  field gradient is wrong after relaxation (the sites have left the level
  set), and deciding each sign against the original triangle gives
  normals turning 178 deg. Do not smooth the normal field -- it averages
  normals that genuinely differ and made things worse.
* **Retry the grid.** Schwarz P at 42 A tears at resolution 64 (sites
  21 A apart) and is clean at 72. Validate the site spacing and retry on
  a shifted grid, as `build_schwarzite` does.
* **Judge with `schwarzite_quality`, not `tmd_quality`.** The latter
  finds bonds by distance; on a saddle a 2.4 A bond's cutoff reaches
  3.0 A and reads a sound cell as 4-8 coordinate metal.

**MX2 junctions** (`build_tmd_junction`) reuse the same machinery over
the junction field. The topology is *easier* than a schwarzite's and the
reason must not be forgotten: a capped junction is sphere-like at any arm
count, so `chi = 2`, the budget is `+12`, and it is paid in **squares**
(+2) against **octagons/decagons** (-2/-4) at the crotch. Genus 0 leaves
no homology classes for the parity repair to fight, so `split` reaches
exactly zero odd rings and the colouring is then exact -- which is why
the default parity here is `split`, not the schwarzite's `flip`. Measured
Y at r=12, arm 26: 3153 atoms, rings 4:205/6:664/8:169/10:15,
`sum(6-n)=12`, **0%** homoelemental, X/M = 2.000, M 6-6 / X 3-3,
p95 strain 7.3%. `schwarzite_quality` reads `info["genus"]` and phrases
the zero-antiphase case as guaranteed rather than lucky; keep that
branch. `tube_radius < 2*h` is refused -- the chalcogen planes would meet
on the axis.

## Nanotube networks: periodic, and the field must be too

`builders/network.py` builds 3D networks of interconnected tubes on a
crystallographic net (cubic, diamond) through the same implicit route as
the schwarzite, so the ring statistics stay **derived**: straight walls
come out all-hexagon, nodes come out with heptagons because a node is a
saddle, and `sum(6-n) = 6*chi` is checked against the net's own genus
(cubic 3 / -24, diamond 9 / -96). Do not "help" by placing rings.

Three things in `implicit.network_field` are load-bearing:

* **Struts are replicated into the 26 neighbouring images.** Not an
  optimisation: without it a strut leaving one face has no counterpart
  entering the opposite one, the periodic marching-cubes weld finds
  nothing to join, and the cell comes out torn. The test asserts the
  faces match to 1e-9, not approximately.
* **Only the nearest few struts may blend.** An exponential soft-min
  over all 27 images subtracts `blend*log(n)` wherever n struts are
  comparably close, and at 432 images that inflated the solid until it
  filled the entire cell. The blend is over the `n_blend` nearest, with
  the same polynomial smooth-min `smooth_union` uses.
* **Evaluation is chunked.** The point-by-strut array is 3.9 GB for a
  diamond cell on a 72^3 grid; unchunked, the process was *killed* --
  no traceback, no failure message, just a missing result. Same class of
  fault as the old quadratic `guess_bonds`.

`minimum_cell` is the honest floor: each node eats about
`tube_radius + blend` of either end of a strut, and below the cell where
one tube radius is left between them there is no tube -- only two nodes
touching. Refuse rather than return a sponge under a network's name.

The atom estimate uses a **measured constant per net** (`NETWORK_OVERLAP`,
cubic 0.71 / diamond 0.74), not the junction's linear law in the node
count: that law predicts diamond to 1% and cubic 27% low, because
coordination 6 hits its own floor.

## Supernetworks: the graph is the input, and it fixes the ring budget

`builders/supernetwork.py` generalises `network.py`: instead of choosing
between two hardcoded nets, it hangs a nanotube on every edge of a graph
handed to it. The catalogue (`SUPERLATTICES`) holds super-square,
super-graphene, super-cubic, super-diamond and super-fcc; a finite cage
comes from `icosahedral_cage()` or from `supergraph_from_atoms()`, which
turns any finished carbon structure into the skeleton of a bigger one --
a C60 whose bonds are tubes is the same construction as a cubic lattice
of them.

**The skeleton fixes `sum(6-n)` before anything is meshed.** The wall is
the boundary of a thickened graph, so `chi = 2*(V - E)` -- one handle per
independent cycle -- and the budget is `12*(V - E)`. That is
`SuperGraph.ring_budget`, and it is an **independent** check rather than
a restatement: `junction._finish` already tests the census against the
*mesh's* own Euler characteristic, which catches a torn mesh but cannot
catch a mesh that closed perfectly around the wrong graph. A blend wide
enough to merge two struts, or a grid coarse enough to pinch a neck shut,
gives a flawless surface of the wrong genus. Measured, the budget is met
exactly by every net built: square -12, graphene -24, cubic -24, diamond
-96, fcc -240, icosahedral cage -216. It also reproduces the two
constants `network.py` hardcodes (cubic `12*(1-3)`, diamond
`12*(8-16)`), so those are no longer separate knowledge. The check is
live, not decorative: a graph declaring three struts where its geometry
has two is refused at every resolution.

**Struts are stored as whichever image the edge search found first**, and
for a one-node net that can be the -x neighbour as easily as the +x one.
Geometrically the same infinite net -- and *not* the same field, because
the 27-image replication that makes the field periodic is then centred a
cell off. Measured against the proven cubic field, struts written
backwards moved the surface by 0.55 A near the faces and the periodic
weld failed with 264 boundary edges. `segments()` slides each strut so
its **midpoint** lies in the cell, which put the replication back on
centre: the difference against `network_field` is then 1.8e-15 (cubic)
and exactly 0 (diamond). Do not remove that canonicalisation.

`super-fcc` is in the catalogue because it is the densest sphere packing
and the obvious thing to ask for, and it is **reported rather than
recommended**: twelve tubes meeting at one point leave no room for a wall
between them at any radius that is still a tube. It does build (6890
atoms, 1.420 +- 0.023 A, zero contacts, budget met), which is the honest
answer -- read the node, not the verdict.

A finite cage needs a much roomier scale than looks necessary, and the
failure is not obvious: each vertex eats about `tube_radius + blend` of
*either* end of every edge it touches, so a C60 scaled by 7 has 9.9 A
edges against 12 A of appetite and is refused outright, while an
icosahedron at scale 9 passes the length test and then relaxes into
something the quality gate throws out. Scale the cage until there is real
tube left.

## Haeckelite tubes: a cylinder is developable, so nothing is re-derived

`builders/haeckelite_tube.py` rolls a finished haeckelite sheet. It is
the one curved builder here that does **not** derive its rings from a
mesh, and the reason is geometric rather than stylistic: a cylinder has
zero Gaussian curvature, so the roll is an *isometry* and the flat
lattice's census is carried over atom for atom. A sphere or a saddle has
curvature and must pay for it in pentagons or heptagons -- which is why
the fullerene, the schwarzite, the junction and the supernetwork all mesh.
A tube pays nothing, so re-perceiving its rings could only introduce an
error. The census is **carried, then verified**: the finished tube's own
faces are traced and checked, because the relaxation afterwards can still
tear. `sum(6-n)` is 0 on both sides, a periodic tube being a torus
exactly as the flat periodic sheet is.

* **`pattern="none"` is the control, not a degenerate case.** It goes
  down the same code path, so anything it gets wrong is the rolling and
  not the pattern -- and unlike a haeckelite it has an answer from
  outside this framework. An 8-cell roll is an (8,0): rolled radius
  3.13 A against the literature's 3.13, relaxing to 3.19, bonds
  1.417-1.419 A. That is the only check here that is not the builder
  marking its own work.

* **Which way the radius moves is not obvious, and both directions are
  real.** The chord is shorter than the arc by about `l^2/(24 R^2)`, so
  the waist starts under compression; the all-hexagon tube answers by
  expanding (3.13 -> 3.19 A), and every patterned one by contracting
  instead (4.89 -> 4.72, 8.00 -> 7.65), relieving the same compression
  through the axis, which shortens by 4.5% and 8.5%. The chord term is
  0.35% and 0.13% in those cases, so it is not what moves them. Neither
  radius is held; both are reported.

* **A cell search pinned at its own bound is reporting the bound.** The
  axial period is the one degree of freedom the roll leaves the cell
  holding (the radius re-fits itself, the atoms being free in x and y),
  and the first scan here was a fixed +-4% window. Every patterned
  lattice came back at exactly 0.96 -- its lower edge. The window now
  widens toward whichever end won until the minimum is strictly interior,
  and a test asserts it is. Do not put a fixed window back.

* **The radius floor is observed, not computed.** A valence force field
  barely sees curvature -- the real cost of a narrow tube is sigma-pi
  rehybridisation, which nothing here prices -- so it would happily
  return a 1 A tube with excellent bonds. `MIN_TUBE_RADIUS` is 2 A
  because that is where the narrowest nanotube ever observed sits, and it
  was grown inside a template holding it open. Do not replace it with a
  number this relaxer produced.

* **`roll="a"` and `roll="b"` are different tubes**, not one described
  twice: the lattice is anisotropic, so neither is a rotation of the
  other, and they come out at different radii and different axial
  periods from the same atom count.

* `r57` **is pentaheptite** -- pentagons and heptagons only, no hexagons
  -- which is the published name for that lattice. The catalogue entry
  predates the comparison and keeps its own name; the equivalence is
  worth knowing before adding a "pentaheptite" entry beside it.

## Heptanene: the Euler budget decides it before any geometry runs

`builders/heptanene.py` answers "may a trivalent net of nothing but
heptagons exist", and the answer comes from the same identity the rest of
this package checks against, `sum(6-n) = 6*chi`. All heptagons means `F`
faces paying -1 each, so `chi = -F/6`:

* **Elliptic** (sphere, chi=+2) needs F = -12. A negative face count is a
  contradiction, not a hard case. The same budget that gives a fullerene
  twelve pentagons rules heptagons out entirely: positive curvature is
  paid in faces *smaller* than six.
* **Euclidean** (a periodic sheet is a torus, chi=0) needs F = 0. The only
  flat all-heptagon net is the one with no heptagons. This is exactly why
  `haeckelite.py` must pair every heptagon with a pentagon -- on a flat
  sheet they are each other's payment.
* **Hyperbolic** (chi<0) gives F = -6*chi > 0. Possible, and only here.

The `{p,q}` test agrees independently: `{6,3}` is Euclidean at exactly 4
and `{7,3}` hyperbolic at 5. `admissible_geometries` takes a ring size so
the reasoning can be run on hexagons and pentagons as controls -- it must
return graphene for 6 and C20 for 5, and a test pins both.

**So heptanene is not a 2D material.** Hilbert's theorem forbids an
isometric embedding of the hyperbolic plane in three-space, so there is
no flat sheet at any size and no relaxation reaches one. What can exist
is a closed negatively-curved surface. Orientability forces F to be a
multiple of 12, giving genus 2 (F=12, V=28), **genus 3 (F=24, V=56)**,
genus 4 (F=36, V=84) -- and genus 3 is the genus of a triply periodic
minimal surface's primitive cell, which is what a periodic carbon crystal
has.

That genus-3 member is the **Klein quartic**, and `klein_quartic_map`
builds it from `PSL(2,7)` rather than transcribing a table: darts are
group elements, and vertices, edges and faces are cosets of the three
rotation subgroups, so 56/84/24 comes out with no arithmetic left over.
Do not replace this with a hardcoded adjacency list -- the derivation is
what makes the counts checkable.

**The lattice shift per bond is a cocycle, and its rank measures the
genus a second time.** A periodic realisation gives each bond an integer
translation, and the faces close only if those sum to zero round every
heptagon. Gauged to zero on a spanning tree, the remaining freedom is
`H_1`, rank `2g` -- and it comes out 6 **without ever using Euler's
formula**, which is an independent reading. It is solved over the
integers by Hermite reduction, not by an SVD: a floating null space spans
the right subspace but its vectors are not translations, and a shift of
0.9999 is not a shift.

**Most cocycles are not embeddings at all.** Placed barycentrically --
every atom at the mean of its neighbours, which is linear once the shifts
are fixed, and is the canonical crystallographic embedding -- eighteen of
the twenty three-of-six cocycles collapse, with atoms landing on each
other. Those nets are unstable in the crystallographic sense. Two survive.

**The geometry does not reach carbon, and the reason is room rather than
curvature.** Measured over both surviving cocycles and cells from 0.70 to
1.35 of the barycentric one: bonds **1.107-1.704 Å**, angles
**65.1-146.0 deg**, against the sp2 window 1.22-1.60 and 95-145. A
further 60 000 cocycles sampled from the lattice scored *worse*
(1.114-2.187 Å), so this is not a poor choice a wider search would fix.
`build_heptanene` therefore **raises** with the numbers attached, exactly
as `build_haeckelite` refuses a frustrated pattern; `strict=False`
returns it for inspection.

The diagnosis matters for whoever picks this up next. Heptanene's angular
excess is **25.7 deg per vertex, less than the 36 deg deficit C20 carries
at every one of its vertices -- and C20 exists**. So the curvature
magnitude is not the obstruction. What genus 3 cannot afford is surface:
56 atoms over three handles is 18.7 atoms each, a tube about two rings
around whose walls meet. The series climbs only slowly (21.0 at genus 4,
26.7 at genus 21, 28 in the limit), so the next thing to try is **a
larger member, which needs a larger Hurwitz group than PSL(2,7)** -- not
a better relaxation of this one.

## A disclination sits where the curvature puts it, and that is measurable

`analyse/curvature.py` generalises the one structural claim the coil
papers make that can be checked on a finished model without running
anything: a pentagon is a +60 deg disclination and a heptagon a -60 deg
one, so **pentagons belong in positive Gaussian curvature and heptagons
in negative**. `periodic_coil.curvature_check` tests it by asking which
side of the torus axis each ring falls on -- which needs an axis, so it
works on a coil and nothing else. The claim is about curvature, not
coils, and `disclination_check` states it intrinsically so the junction,
the schwarzite, the network and the supernetwork are all held to it. It
goes in through `junction._finish`, so every implicit-route builder
records it without asking, and `_report_structure` prints it, so every
one of them says so.

**The obvious measure is wrong in a way that looks right.** The discrete
Gaussian curvature of a polyhedron is the angular defect
`2*pi - sum(angles)` at a vertex -- but a carbon atom here is
**trivalent**, and three angles at a point can never sum past 360 deg
(the spherical triangle inequality). So that defect is never negative on
any structure: measured, it called Schwarz P -- a minimal surface,
negatively curved everywhere -- positive at every ring size. It measures
pyramidalisation. The curvature of a trivalent net is not at its atoms,
it is in the non-planar faces. So the surface is fitted instead:
`z = a x^2 + b xy + c y^2` in a frame aligned with the local normal over
each atom's two-bond neighbourhood, and `K` has the sign of `4ac - b^2`.
0.08 s at 1134 atoms.

**The flat haeckelite control is what makes the rest mean anything.** The
objection is circularity -- the patch around a pentagon is dominated by
that pentagon, so does the fit just recover the ring size? R5,7 settles
it: a *flat* lattice of 16 pentagons and 16 heptagons, every ring at
genuinely zero curvature, and the fit returns **exactly 0.000 for both**,
mean sign and median alike. Its agreement score is 0%, which is the
correct answer and not a failure -- there is no curvature for a
disclination to be on the right side of. **Keep that test.** Without it
the other numbers prove nothing.

Measured, as mean sign of `K` per ring size:

| surface | pentagons | hexagons | heptagons | agreement |
|---|---|---|---|---|
| C60 | +1.00 | +1.00 | -- | -- |
| graphene | -- | 0.00 | -- | -- |
| haeckelite R5,7 (flat) | 0.00 | -- | 0.00 | 0% |
| Y junction | +0.88 | +0.11 | -0.75 | 91% |
| X junction | +0.81 | +0.08 | -0.86 | 92% |
| Schwarz P | +0.51 | -0.51 | **-1.00** | 91% |
| super-square | +0.63 | -0.32 | -0.79 | 86% |
| periodic coil | +1.00 | -- | **-1.00** | **100%** |

A junction's hexagons sit near zero because its arms are cylinders, and
that is the row that shows the measure is reading shape rather than ring
size on a *curved* structure too.

**Where the two routes disagree, the intrinsic one is right.** The coil
scores 100% intrinsically against 68% (smooth helix) and 85% (hexagon)
from its own axis test. The axis test asks only which side of the torus
axis a ring falls on, so it misreads a ring sitting near the top or
bottom of the tube, where the real curvature is near zero. Those
"wrong side" counts are the crude criterion's artefact, not defects of
the model -- so do not quote them as model defects, which an earlier
version of this file effectively did. Both are recorded; the coil is the
one structure that carries both, and it is worth keeping that way
precisely because they can be compared.

Hexagons are **excluded from the agreement score**, not counted as
passes. A hexagon is the flat case and has no side to be on; scoring it
would inflate every number, and on a nanotube network -- mostly hexagons
-- it would hide the answer entirely.

## Toroids: the one census that can be predicted before the build

`builders/toroid.py` bends a tube until its ends meet. It is genus 1, so
`sum(6-n) = 6*chi = 0` **exactly**, and with only 5s, 6s and 7s available
that forces the pentagons and heptagons to come out in *equal numbers* --
the only curved builder here whose census is predictable rather than
merely measurable, and a test pins it. Measured at R=20, r=5: 68
pentagons, 560 hexagons, 68 heptagons, budget 0, and 93% of the
disclinations on the curvature side they belong on.

It is also the surface the coil papers are actually describing, with
nothing else going on: a torus has positive Gaussian curvature on its
outer equator and negative on its inner one, so the pentagons belong
outside and the heptagons inside and there is nowhere else for either to
go.

**The builder is deliberately thin, and should stay that way.**
`swept.build_swept_tube` already sweeps a tube along any centreline, and
a torus is just a **closed** one: the capsule sweep that would cap two
free ends instead overlaps itself and closes the surface. A second field
for it would be a second thing to keep right.

`R/r` is the physics. The wall bends by `r/R` at the inner equator, and
below `R = 2.5r` the hole is smaller than the tube is thick -- what comes
out is a dimpled sphere with a torus's name on it, so that is refused
rather than returned. The published toroidal carbons sit at 3 to 6;
outside that the structure is built and labelled, because it is not
wrong, only not the geometry those calculations relaxed to.

## Amorphous wall or crystalline: the mesh route always picks the first

This is the single biggest thing separating what this package draws from
what the literature draws, and it is not a bug in any one builder.

The implicit route -- field, marching cubes, remesh, dual -- **derives**
its topology, so the remesher picks whatever ring sizes the local
curvature calls for. Measured across every curved builder, the
disclinations land where they belong (85-100% on the correct side of the
curvature; see the table above), and there are **far too many of them**:
11-21% of all rings are non-hexagonal. That is a *sound* wall and an
**amorphous** one. Published pictures of toroids, cones and coils show
crystalline walls with a handful of disclinations in known places.

Where a crystalline route exists, offer it as a **separate mode**, not a
flag -- the two are different structures, and the names must say which:

| structure | mesh route | crystalline route |
|---|---|---|
| toroid | 68 pentagons + 68 heptagons at R=20 | `{6: 1100}` -- none at all |
| nanocone (112.9 deg) | 183 pentagons + 171 heptagons, 3800 atoms | `{5: 1, 6: 210}`, 470 atoms |
| coil | ~90 non-hexagons | `nanocoil.py`, all hexagons |

**The crystalline route always costs size, and the reason is the same
every time.** Bending or rolling a finished lattice can only *stretch*
it: there is no disclination to relieve the curvature with, which is
precisely what the mesh route buys with its 5-7 pairs. So a polyhex
toroid of a (5,5) tube needs R = 43 A and 2200 atoms to fit an 8% strain
budget, and the small round toroids in the literature are **not**
polyhexes -- they use knees with a pentagon outside and a heptagon
inside, which is a third route this package does not have yet.

**A developable surface is the easy case, and there are only two.** A
cylinder and a cone have zero Gaussian curvature, so unrolling them is an
isometry: bond lengths are exact by construction and no relaxation is
needed at all. `haeckelite_tube.py`, `toroid.build_polyhex_toroid` and
`nanocone.py` all exploit this. A sphere or a saddle cannot be unrolled,
which is why the fullerene, the schwarzite, the junction and the
supernetwork must mesh and must pay in disclinations.

### Nanocones: the apex angle is quantised, and only one is chemistry

`builders/nanocone.py` cuts a wedge of `N * 60` degrees out of graphene
about a **hexagon centre** and joins the edges. The apex hexagon loses N
of its six sectors, so `sin(theta/2) = 1 - N/6` -- the angle is an output.
Those five angles (112.9, 83.6, 60.0, 38.9, 19.2) are the ones Krishnan
et al. observed (*Nature* **388**, 451), so this is one of the few
builders with an external answer to check against.

* **Only `N = 1` is clean**: 470 atoms, `{5: 1, 6: 210}`, bonds
  **1.391-1.420 A**, angles 108.0-120.1. `N = 2` and `N = 3` put a square
  or a triangle at the apex -- sound topology, and 1.339 / 1.230 A bonds
  with 90 / 60 deg angles. Nature splits those disclinations into
  *separate* pentagons instead, which is how the 19.2 deg nanohorn works.
  `strict=True` refuses them; `N >= 4` cannot be a single apex ring at
  all (it would need a 2-gon) and is refused outright.
* **The cut must run through atoms for some N and between them for
  others.** Both are mirror lines of the hexagon, so both look equally
  valid -- and at 0 deg the `N = 1` cone gains a spurious four-membered
  ring at the seam, while at 30 deg the `N = 2` cone comes back with four
  squares instead of one (`sum(6-n) = +8` against +2). So the offset is
  **searched, not fixed**: the builder tries the candidates and keeps the
  one whose budget equals N. That test is exact -- a developable roll
  cannot create a ring -- which turns a magic number into a checked
  choice. Do not replace it with a constant; one was tried, and it was
  right for exactly one case.
* **The apex must be a hexagon centre, and that was verified by counting
  neighbours** rather than derived: with the basis at `(0,0)` and
  `(0,bond)` the centre is at `(0,-bond)`, and the origin of that basis
  is an *atom*, whose three-fold symmetry does not admit most wedges.
* **The rim is open on purpose** and recorded in `info["rim_atoms"]` and
  `info["terminal_atoms"]`, as a nanoribbon's edges are.

### The literature's own method, for when this is not enough

László, *Theor Chem Acc* **134**, 104 (2015) gives the general route:
**topological coordinates**. Cartesian positions come from the bi-lobal
eigenvectors of the adjacency or Laplacian matrix -- three for a sphere
(Fowler & Manolopoulos), **four** for a torus, and for junctions and
coils neither works: a 1165-atom junction needs 16, so there is no simple
rule. The general answer is a matrix `W`, built from a harmonic potential
over first and second neighbours, whose **null space contains X, Y and
Z**. It is worth knowing two things before reaching for it: `W` is
defined at the equilibrium geometry, so constructing it exactly needs the
coordinates it is meant to produce (the paper approximates it), and this
package already met the spherical case's limit from the other side --
`heptanene`'s Laplacian has an 8-fold degenerate first eigenvalue, so no
three of its eigenvectors embed it at all.

## Topological coordinates: positions from the bonds alone

`analyse/topological.py` implements Istvan Laszlo's review (*Theor Chem
Acc* **134**, 104, 2015). Certain eigenvectors of the adjacency matrix
are **bi-lobal** -- deleting the vertices where they vanish and the edges
across which they change sign leaves exactly two components -- and those
read as angles on the surface. Three of them place a fullerene (Fowler &
Manolopoulos); a torus needs **four**, because three always flatten it
(Graovac et al.), and a test here reproduces that failure rather than
merely citing it.

**Which four was measured, not assumed.** Take the toroidal polyhex this
package builds exactly -- a (5,5) over 60 periods, 1200 atoms, all
hexagons -- throw its coordinates away and keep the bonds:

| bi-lobal pair | eigenvalue | agrees with the ring angle | with the tube angle |
|---|---|---|---|
| (1, 2) | 2.9973 | **1.0000** | 0.0000 |
| (13, 14) | 2.8699 | 0.0000 | **1.0000** |
| (15, 16) | 2.8672 | 0.0019 | 0.0001 |
| (21, 22) | 2.8591 | 0.0000 | 0.0000 |

So it is the two highest-eigenvalue *degenerate* bi-lobal pairs, and the
agreement is exact up to an offset. Every other pair is orthogonal to
both. Re-deriving that costs one build, and it is the reason to believe
the recipe rather than the citation.

**The round trip is the test, and it has an answer.** Placing the torus
from its adjacency alone and then asking what bonds the *placement*
implies returns the 1800 bonds that went in, with the same length
distribution (1.267-1.570 A against 1.270-1.574). C60 likewise returns
its 90 bonds and its radius (3.532 A against 3.519). Its bonds come out
less uniform than the builder's -- 1.357-1.545 against a flat 1.420 --
which is the known character of the method: these are **placements, not
relaxations**, and they are excellent starting coordinates.

**There is no general case, and Laszlo says so.** A 1165-atom nanotube
junction needs sixteen bi-lobal eigenvectors; the method works only for
structures related to the sphere. The general answer is a matrix `W`
built from a harmonic potential over first and second neighbours, whose
null space holds X, Y and Z -- and constructing `W` exactly needs the
coordinates it is meant to produce. This package met that wall from the
other side: `heptanene`'s Laplacian has an **8-fold degenerate** first
non-trivial eigenvalue, so no three of its eigenvectors embed it at all.

## Perfect is two questions, and most builders answer only one

"Is it textbook-exact?" splits into **topology** (is the census the one
the structure is defined by?) and **geometry** (are the bonds carbon's?).
They are independent, and no builder here scores full marks on both by
accident. Measured:

| structure | route | bonds (A) | spread | non-hex |
|---|---|---|---|---|
| CNT (10,10) | exact lattice | placed, not measured | -- | 0% |
| **C60** | dual mesh + VFF | **1.420-1.420** | **0.000** | 37.5% (its twelve) |
| **nanocone 112.9** | sector cut | **1.391-1.420** | 0.029 | 0.6% (its one) |
| haeckelite R5,7 | honeycomb + rotations | 1.324-1.524 | 0.199 | 100% (by design) |
| polyhex toroid | bent lattice | 1.337-1.503 | 0.166 | **0%** |
| meshed toroid | implicit | 1.344-1.508 | 0.164 | 19.5% |
| Y junction | implicit | 1.366-1.484 | 0.117 | 16.6% |

Three things in that table are worth keeping in mind before claiming
anything is "perfect":

* **Only C60 and the nanocone are exact on both axes.** C60 relaxes to a
  flat 1.420 because a VFF on a sphere-like shell has an exact minimum;
  the nanocone never relaxes at all -- a cone is developable, so its
  bonds are 1.42 by construction and only the apex pentagon departs.
* **The polyhex toroid is topologically perfect and geometrically is
  not.** Zero disclinations, and bonds spanning 1.337-1.503 -- because
  bending a finished lattice stretches it, and there is no disclination
  to relieve that with. Do not read "all hexagons" as "ideal".
* **Its bond spread is 0.166 against the meshed toroid's 0.164.** The
  crystalline route buys a clean *census*, not better bonds, at this
  size. That is the honest summary of what the whole mesh-versus-lattice
  distinction is worth, and it is smaller than it looks.

Everything built through the implicit route -- junctions, schwarzites,
networks, supernetworks, the hypercube, the supertube, meshed coils and
toroids -- is sound and **amorphous-walled**, 11-21% non-hexagonal. For
most of those there is no published "perfect" version either: the ON-CNT
figures in the literature are idealised drawings, and real DFT junction
models carry 5-7 pairs too. Say "sound" for those, not "textbook".

## The Dunlap toroid is not reachable by meshing, and this is the evidence

A Dunlap toroid is a ring of straight tube segments joined at **knees**,
each knee an armchair-to-zigzag junction carrying one pentagon on the
outside and one heptagon on the inside. The arithmetic is clean: the
(n,n) and (2n,0) radii are in the ratio 2/sqrt(3) = 1.1547 for every n,
their chiral directions differ by 30 degrees, so a closed ring takes
**12 knees** -- and 12 is even, which it must be for the tube type to
come back to itself. The census is then exactly 12 pentagons and 12
heptagons, `sum(6-n) = 0`, genus 1.

**The implicit route cannot make one**, and it is worth having measured
rather than assumed. Sweeping a *polygonal* closed path, which is the
obvious way to ask for knees:

| centreline | census | disclinations near a knee |
|---|---|---|
| circular | 75 + 75 | -- |
| 12 sides | 72 + 72 | **10%** |
| 6 sides | 71 + 71 | 39% |

A true Dunlap toroid needs 12 + 12. The remesher **distributes**
curvature over the whole wall rather than concentrating it at the
corners, so the polygon changes almost nothing -- at 12 sides the
disclinations are, if anything, *further* from the knees than a uniform
scatter would put them. Do not ship a polygonal meshed toroid under
Dunlap's name.

### What the dual route reached, and exactly where it stops

The combinatorial route does work as far as the census, and then hits a
wall worth writing down.

A triangulated torus (24 x 12, 288 vertices) comes out with every degree
6 and `V - E + F = 0`. **Six edge flips, spaced round the ring, give
exactly the Dunlap census: `{5: 12, 6: 264, 7: 12}`, with all twelve
pentagons on the outer equator** -- which is right, because a flip
changes four degrees by (-1, -1, +1, +1) and six of them make twelve
fives and twelve sevens.

The wall is *where the heptagons are*. A flip always puts its two
sevens on neighbours of its two fives, so straight out of the flips the
heptagons sit one row either side of the pentagons -- rows 1 and 11
against row 0. A Dunlap knee needs the heptagon on the **inner** equator,
half a circumference away.

**Flips cannot separate them, and 56 attempts is the evidence.** Gliding
greedily toward the inner equator moved the pentagons too: the census
stayed perfect throughout and the pair ended interleaved across rows 3
to 9, still adjacent. That is not a search failure, it is the physics --
a 5-7 pair is a **dislocation**, a flip **glides** it, and glide
preserves the pair. Separating the two requires **climb**, which adds or
removes a row of atoms and so cannot be a flip at all: it needs
`contract_edge` or a split, changing the vertex count.

So the next step is precise rather than open-ended: create the pairs with
six flips (done, and exact), then **climb** each heptagon inward with
vertex-changing moves, checking after each that `sum(6 - deg)` is still
0 and no degree leaves {5, 6, 7}. What six flips alone produce is a
*Stone-Wales toroid* -- the right census, the wrong arrangement -- and it
should not be shipped under Dunlap's name.

### The row-count route: the third attempt, and the third wall

Climb was never reached, because a better idea got there first and then
failed for a reason worth more than the builder would have been. If the
disclinations must be *placed*, do not place them: **let the row counts
place them.** A torus's outer equator is longer than its inner one, so
triangulate it with a different number of vertices per row, proportional
to that row's own circumference. Every row-count change of one is a
disclination, and it lands where the curvature asks for it without
anybody choosing.

It reaches the census exactly. Scanning `(R, r)` under the constraint
that the mesh stay isotropic -- row spacing equal to in-row spacing
equal to `sqrt(3) * bond`, which is the only way the dual comes out at
graphene's bond length -- **44 cells give exactly `{5: 12, ..., 7: 12}`
with no degree outside 5-7**, all of them at `rows = 8` and
`r = 3.05-3.30 Å`, with `R` free from 12.5 to 59.5 Å. The relaxed nets
are good carbon: bonds 1.359-1.477 Å at `R = 12.5` and 1.376-1.472 at
`R = 36`, spreads of 0.118 and 0.096 against **0.164 for the meshed
toroid and 0.166 for the polyhex one**. By the bond statistics alone it
is the best toroid in the package.

It is still not a Dunlap toroid, and the measurement that settles it is
azimuthal rather than radial:

- **Radially it is right.** Pentagons sit at rho = 39.4 Å and heptagons
  at 34.4 Å on the `R = 36` cell -- outside and inside, which is what the
  "92% placed" figure in `disclination_check` was reporting.
- **Azimuthally it is wrong.** The gaps between consecutive pentagons
  round the ring run **0.1 deg to 60.9 deg**, not 30 deg. The twelve
  disclinations come out as **six pairs, not twelve isolated knees.**
- **The shape agrees.** Fourier-analysing the axis radius, the strongest
  harmonics are `n = 2` (1.52 Å) and `n = 6` (1.96 Å); `n = 12` is third
  and weak (0.51 Å). Six pairs of knees make a six-fold shape.

**This is structural, not a tuning failure, and that is the point.** A
row-count change happens at a *meridional band* -- constant angle round
the tube -- and a meridional band is a circle that runs all the way
round the ring. Its disclinations are therefore smeared round the ring
by construction. The route can never localise them into a knee, at any
`R`, any `r`, any row count. **The row-count route cannot produce a
Dunlap toroid**, and no search inside it will change that.

Two further traps met on the way, both worth keeping:

- **The census is not the structure.** The first version of this hit
  `{5: 12, 6: 1049, 7: 12}` and `sum(6-n) = 0` and looked finished. Its
  tube radius, measured, ran **0.17 to 12.14 Å** around a reported 3.01 --
  the wall had corrugated flat. A census can be perfect while the object
  is not a torus at all, so measure the shape as well, always.
- **`relax_shell` alone will wrinkle a wall.** The mesh and its dual were
  exact tori (`r = 4.75 +- 0.00`, dual bonds averaging exactly 1.420);
  free relaxation destroyed them, because equalising bond lengths is
  satisfied just as well by a corrugated surface as a smooth one. The
  cure already existed in the function: pass `anchor_normals` and the
  restraint acts only along the surface normal, holding the wall on its
  surface while every atom stays free within it. That cut the spread
  from +-2.55 Å to +-0.345.

**What is left is the lattice-exact knee**, which is what Dunlap actually
built: a polygon of straight `(n,n)` and `(2n,0)` segments, their axes 30
deg apart because that is the angle between the armchair and zigzag
directions, twelve of them closing the ring. A planar bevel cut does
*not* give it -- cutting a (5,5) tube on a plane and mirroring it tore
the lattice (dangling atoms at degree 1 and 2, triangles and squares at
the seam, measured at four bend angles), because the cut curve unrolls to
a sinusoid on the sheet rather than to a lattice mirror line. The knee
has to be built as the lattice object it is, one pentagon and one
heptagon matching an armchair rim to a zigzag one.

Until then **the package ships no Dunlap toroid**, which is the honest
state: two toroid routes that are what they say they are, and a third
that would not have been.

### Why the knee is hard, stated precisely

Two more constructions were tried and both hit the *same* wall, which is
worth stating once because it governs junctions and coils too.

Rows perpendicular to the tube axis fix the smearing problem completely:
a count change then lands on **one vertex at one azimuth**, and the
ring's phase places it to the degree -- measured 0.00 -> 194 deg,
0.25 -> 284 deg, 0.50 -> 14 deg, exactly 360 deg per unit of phase. So
azimuthal placement is solved and controllable.

What is not solved is **isolation**. Every move that changes the mesh
locally creates a 5 and a 7 *together*:

- A count step of one (`N, N, N+1, N+1, N`) gives `{5: 2, 6: 232, 7: 2}`
  with each 5 stacked directly on a 7, ~2 Å apart at the same azimuth.
- A shift band (both rings at `N`, the join offset stepping 0 -> 1 at one
  azimuth and back at another) gives `{5: 2, 6: 152, 7: 2}` with the 5 at
  0 deg and the 7 at 180 deg -- which *looks* like a knee until you notice
  the second pair sits at 165 and 345 deg, so each disclination is
  adjacent to an opposite one and they cancel.

Both are **dislocations**, which is the same object that defeated the
flip route: a 5-7 pair is a dislocation, and no local construction emits
one half of it. An isolated pentagon needs the rows to shrink
*progressively* -- a cone sector, which is exactly how `nanocone.py`
already gets its single clean pentagon (bonds 1.391-1.420, budget
exactly 1).

**So the route to a knee is two cone sectors joined, not a cylinder
edited.** That is a real build and it is not started. What *is* settled
is that a planar bevel cut cannot do it: the (5,5) and (10,0) tubes a
30 deg knee would join have radii 3.39 and 3.92 Å, so their cut ellipses
cannot coincide at any bevel angle, and cutting one tube and mirroring it
tears the lattice outright.

## A collapsed wall passes every check in this package

The periodic coil preset -- 8.75 Å coil, 9.6 Å pitch, **3 Å tube** --
comes back with its tube **pinched fully shut in places**, and nothing
in the package noticed. Measured against the helix centreline:

============================  ==========================
stage                         distance from the axis (Å)
============================  ==========================
dual, straight off the mesh   2.72 +- **0.31** (0.34-3.44)
after `relax_shell`           3.15 +- **1.59** (0.04-7.15)
============================  ==========================

Five times the spread, with atoms ending **0.04 Å from the axis** for a
requested 3.0. The control is the point: the mesh is fine and **the
relaxation is what collapses it**, for the same reason it wrinkles a
wall -- equalising bond lengths is satisfied exactly as well by a
collapsed tube as by a round one, and nothing in the force field knows
the difference.

**It passed the quality gate**: bonds 1.279-1.544 Å and *zero* close
contacts. That gate measures bond lengths and non-bonded distances, both
**local**, and a collapse is not local. Every bond was the right length;
the tube was gone.

`collapsed_wall()` is the check that catches it, and it is physical
rather than a tolerance: **no carbon has an angle sum below 328.4 deg**,
that being three tetrahedral angles. The coil reads **320.8**, which is
not a hybridisation at all. Sound structures clear it comfortably -- C60
at 348.0 is the most pyramidal thing here that is still carbon, and a
(5,5) tube reads 356.8.

### It was a flip with no length check, and five diagnoses were wrong first

**The flip tests were purely topological.** `_anneal_once` and
`place_disclinations` both checked degrees and whether the new edge
already existed, and **neither looked at where the vertices are**. So a
flip could join two points clear across the structure. On a 3 Å tube --
185 mesh vertices, under eight rings around its circumference -- that is
a flip straight through the tube, and the wall folds through itself.

`FLIP_MAX_EDGE` is the fix: a flip may not create an edge longer than
1.8x the target. The number is bounded on both sides and both bounds were
measured. Two equilateral triangles sharing an edge have a diagonal of
`sqrt(3)` times it, so a **legitimate** flip already needs 1.73x -- at the
splitter's 4/3 every flip is rejected and the annealer silently does
nothing, which is how the first attempt at this guard behaved. A flip
across a 3 Å tube makes a ~6 Å edge, 2.4x the target. On that tube: 1.8
leaves it sound at an angle sum of 329.9 deg, 2.0 and 2.5 collapse it
(319.2 and 321.4) and 3.0 tears it outright.

With the guard, annealing **helps** where it was documented as hurting:

==========  ==============  ==============  ==================
structure   placed, 0       placed, 80      bonds at 80 (Å)
==========  ==============  ==============  ==================
Y junction  94.3%           **97.6%**       1.361-1.483
L junction  90.0%           **100.0%**      1.319-1.503
coil (hex)  74%             **88%**         1.344-1.525
==========  ==============  ==============  ==================

**The `anneal_sweeps` table in `build_junction` predates this guard**, and
its conclusion -- that annealing buckles the wall -- was measuring flips
reaching across the structure rather than annealing itself. The other
builders' defaults are left at 0 all the same: re-measuring every one of
them is a separate decision, not a side effect of this fix.

**Five diagnoses were wrong before this one**, each costing a measured
experiment, and each kept because each looked right:

1. **"The relaxation collapses it."** The dual measured 2.72 +- 0.31 Å
   from the helix axis before relaxing and 3.15 +- 1.59 after, so the
   damage did appear there -- but only because the mesh handed to it was
   already ruined.
2. **"Hold the wall on its surface."** `anchor_normals` at fixed normals
   made it worse (1.19-1.75 Å bonds, 5 contacts); relax-then-reproject
   far worse (0.06-6.24 Å, 634 contacts). Both treated a symptom.
3. **"The turns merge."** At pitch 9.6 a 3 Å tube leaves a 3.6 Å gap.
   Refuted: at pitch 24 the gap is 18 Å and the mesh is as bad.
4. **"It skips the cell rescale `_finish` does."** True -- the dual
   arrives 12.3% off -- and fixing it changed nothing while breaking
   pitch 12.
5. **"Annealing is simply wrong here, set it to 0."** It does stop the
   collapse, and it costs the placement the annealing was for: 74%
   against 88%. Turning off the pass that exposes a bug is not fixing
   the bug.

What isolated it was comparing against builders that are sound on the
same machinery -- the junction (mesh edges 1.92-3.17), the relaxed coil
(1.81-3.67) and the **periodic** schwarzite (1.70-3.97) all remesh
cleanly -- and then remeshing the coil's **own captured mesh** with
annealing off, which gave 1.91-3.21 where the builder gave 1.94-12.90.
One parameter's difference, and it was not the periodic path.

**It is better, not perfect.** The tube still runs about 0.15-5.47 Å
around a requested 3.0. A 3 Å tube carries under eight rings around its
circumference, which is the floor of what this route can mesh at all.

## Holding a wall on its own surface, and where that pays

The implicit field is **zero on the wall**, so ``|field(atom)|`` measures
exactly how far an atom has left it -- no binning, no assumed axis, and
none of the systematic error that a hand-rolled roundness measure
carries. (One was tried first and was useless: it read the *raw
marching-cubes surface*, which is the exact tube by construction, as
11% collapsed. An angular wedge of a curved tube has its centroid off
the local axis. The pristine (5,5) control passed only because a
straight tube has no curvature to get wrong.)

Measured on the periodic coil, by stage:

============  ==============  ========  ==================
stage         mean off (Å)    max       beyond 0.5 Å
============  ==============  ========  ==================
raw surface   0.004           0.020     0.0%
dual          0.227           0.645     4.3%
**relaxed**   **0.801**       **2.84**  **56.5%**
============  ==============  ========  ==================

So the mesh and its dual are on the tube and the **relaxation walks the
atoms off it** -- for the same reason it does everything else here:
equalising bond lengths is satisfied as well off the surface as on it.

``wall_anchor`` passes ``anchor_normals`` built from the field's own
gradient, restraining each atom **along its normal only** so it still
slides freely within the wall. On the coils:

===========  =============  =============
coil         mean off, 0    mean off, 1.0
===========  =============  =============
preset       0.801 Å        **0.387**
hexagonal    1.010          **0.327**
R = 25       2.378          **0.361**
===========  =============  =============

**This was tried twice before and failed both times**, and the reason is
worth keeping: the mesh underneath was still torn by the unguarded
flips. A restraint on a broken mesh makes it worse, which is exactly
what was measured (1.19-1.75 Å bonds, contacts). It only became useful
once `FLIP_MAX_EDGE` had fixed the mesh.

### It pays where the wall is leaving, and costs where it is not

=================  ===============  ===============  ==============
structure          off-surface      angle sum        verdict
=================  ===============  ===============  ==============
periodic coil      0.801 -> 0.387   329.9 -> 334.0   **on by default**
superfullerene     --               **328.3 -> 333.0**  fixes a collapse
Y junction         0.790 -> 0.350   339.8 -> 338.5   off by default
X junction         0.838 -> 0.378   337.6 -> 338.9   off by default
=================  ===============  ===============  ==============

A junction's wall is **not** leaving its surface -- angle sums 337-340,
well clear of tetrahedral -- so the restraint buys fidelity nobody
needed and costs bonds (1.366-1.484 to 1.332-1.564) and placement
(94.3% to 92.0%). A thin coiled tube is the opposite case.

**The superfullerene was collapsed and is the clearest case for it**:
328.3 deg at the default, past tetrahedral, and 333.0 with the anchor.
It is still off by default there because the cost is real (bond spread
0.216 -> 0.363) and only the thin-tube cells need it -- so the collapse
warning names `wall_anchor` as the remedy rather than the builder
guessing.

### The flip guard helps every meshed builder, and the old defaults are stale

With `FLIP_MAX_EDGE` in place, annealing improves placement everywhere
it was measured, which is the opposite of what the `anneal_sweeps` table
records:

==================  ==========  ==========  ==================
structure           placed, 0   placed, 80  bonds at 80
==================  ==========  ==========  ==================
Y junction          94.3%       **97.6%**   1.361-1.483
L junction          90.0%       **100.0%**  1.319-1.503
schwarzite          79.2%       **87.5%**   1.323-1.577
super-graphene      89.6%       **94.3%**   **1.348-1.529**
superfullerene      95.0%       **99.1%**   1.296-1.604
==================  ==========  ==========  ==================

Super-graphene improves on **both** placement and bond spread, which no
reading of the old table would predict. **The other builders' defaults
are left at 0 regardless**: changing five builders' output is a decision
to take deliberately, with these numbers in hand, not a side effect.

## Every net and cage, audited rather than assumed

The defaults above were set from the handful of structures that happened
to be under the microscope. Running `collapsed_wall` over the whole
catalogue at its own defaults found **two more collapsed walls that
nobody had looked at**:

====================  =======  ==========  ===========  ==========  ======
net or cage           atoms    angle sum   verdict      placed      time
====================  =======  ==========  ===========  ==========  ======
super-square          664      331.6       sound        88.1%       23 s
super-graphene        1180     335.1       sound        89.6%       34 s
**super-cubic**       836      **327.8**   **COLLAPSED**  76.1%     38 s
super-diamond         --       --          refused at cell 40       --
super-fcc             3082     330.7       sound        87.1%       **1440 s**
super-icosahedron     4292     331.4       sound        93.9%       33 s
super-hypercube       6494     331.6       sound        92.9%       78 s
supertube-(4,4)       6728     334.1       sound        93.1%       447 s
supertube-(6,6)       5936     330.7       sound        91.6%       526 s
superfullerene-C60    7534     328.3       **COLLAPSED**  95.0%     73 s
====================  =======  ==========  ===========  ==========  ======

The table is the shipped presets, all ten rows, none extrapolated.
`wall_anchor=1` clears both collapses: super-cubic **327.8 -> 330.6** and
superfullerene **328.3 -> 333.0**; super-diamond's was a voxel problem
and is fixed outright (see below).
The cost is bonds and placement, and on super-cubic the placement cost is
steep -- 76.1% to 65.9%.

### Anchoring reached the cages and silently skipped the periodic nets

Worth keeping, because it looked like the fix simply failing. With the
anchor on, **super-cubic came back byte-identical** -- same atom count,
same bonds, same placement, same 37 s -- and still collapsed. A cage goes
through `_finish`'s finite branch, where the anchor was wired; a periodic
net goes through the variable-cell branch, where it had been left out
deliberately as "not worth getting subtly wrong".

Getting it right needs the **cumulative** scale: the field lives in the
mesh's original coordinates and that branch rescales cell and atoms
together each cycle, so the map back is `scaled_box / box`, tracked as
it goes. With that, super-cubic responds like everything else.

**A result identical to four decimal places is not a weak effect.** It
is a code path that never ran, and it should be read that way.

### super-fcc costs 24 minutes

Sound, and by far the slowest thing in the catalogue: 1440 s against
23-78 s for most, because an fcc node has **twelve** struts meeting at
it. Nothing is wrong with it; it is a size to know about before Build.

### A fixed grid over a growing cell

`grid_resolution` was a fixed 72 whatever the cell, so **the bigger the
structure the coarser its voxels** -- which is backwards, and it showed.
super-diamond is sound at scale 50 and collapses at 60:

===========  ==========  ==========  ============
cell         resolution  voxel (Å)   angle sum
===========  ==========  ==========  ============
scale 50     72          0.69        332.4 sound
scale 60     72          **0.83**    **327.4 COLLAPSED**
scale 60     100         0.60        **331.8 sound**
===========  ==========  ==========  ============

Nothing about a bigger cell makes the wall worse; only the voxel did.
`build_junction` has scaled its grid with its box since it was written
and this never did, so `grid_resolution` is now a **floor** and the
voxel is tied to the tube radius (`VOXEL_PER_TUBE = 0.12`, i.e. 0.60 Å
for a 5 Å tube -- the safe side of the boundary above). **The shipped
super-diamond preset was collapsed and now is not.**

It moves more than the arithmetic on `scale` suggests: the voxel follows
`max(box)`, and for a 2D-periodic sheet the box includes the vacuum
direction, so super-graphene went from resolution 72 to ~99 as well
(1180 atoms to 1148, angle sum 335.1 -> 338.6, placement 89.6% ->
85.7%). Mixed, and worth knowing rather than glossing.

### Correcting the audit above: the icosahedral preset is sound

The catalogue audit called super-icosahedron collapsed at 328.4. That
was **at the parameters the audit chose** (scale 20, tube 3.5), not at
the ones the window ships (scale 24, tube 4.0, blend 3.0), which read
331.4 and sound. A collapse found at invented parameters is a fact about
those parameters.

### Annealing really does roughen the wall, and the anchor pays for it

The `anneal_sweeps` table's claim held up when measured the right way.
Off-surface deviation, mean:

==================  ========  =========  ==========
structure           anneal 0  anneal 80  80 + anchor
==================  ========  =========  ==========
Y junction          0.711     0.751      --
super-graphene      0.939     **1.096**  **0.676**
super-cubic         0.990     **1.076**  **0.808**
==================  ========  =========  ==========

So annealing costs 6-17% more roughness -- and the anchor more than
repays it. On **super-cubic the combination is best on every axis that
matters**: the wall goes from collapsed (327.8) to sound (337.3),
placement from 76.1% to **88.0%**, and roughness below the baseline.
Only the bond spread widens, 0.230 to 0.254.

**Annealing alone also clears super-cubic's collapse** (338.1) with the
*tightest* bonds of the four combinations (0.185), which is the opposite
of what "annealing buckles the wall" would predict.

The defaults are still 0 and 0. These are two structures; the catalogue
has ten, several take minutes each, and changing what every meshed
builder emits deserves the full sweep rather than an extrapolation from
the two that were quick.

## Heptanene: sp3 makes it worse, and by a factor of six

The proposal was a 2D all-heptagon sheet with the sp2/sp3 mix as the free
parameter. The Euler accounting settles it. With all faces heptagons,
`n3` vertices of degree 3 (sp2, or sp3 with one bond leaving the net) and
`n4` of degree 4 (sp3 with all four in it):

    2E = 3*n3 + 4*n4,   7F = 2E,   V = n3 + n4
    chi = V - E + F = -(n3 + 6*n4) / 14

A 2D-periodic sheet is a torus per cell, so `chi = 0`, so
`n3 + 6*n4 = 0`, so `n3 = n4 = 0` -- no net at all. **And every
four-coordinate sp3 vertex counts six times an sp2 one**, so raising the
sp3 fraction drives `chi` further negative: it needs *more* handles, not
fewer. The refusal is not a limitation of the sp2 assumption.

What the accounting does give is where heptanene lives: `n3 = 56, n4 = 0`
gives `chi = -4`, genus 3 -- exactly one schwarzite unit cell. Heptanene
is a schwarzite, not a sheet.

**And no cell size rescues the embedding.** Expanding the Klein quartic's
cubic 3-torus and relaxing under periodic boundary:

======  ==========  =================  =========
scale   cell (Å)    bonds (Å)          contacts
======  ==========  =================  =========
x1.0    6.19        0.83-1.75          86
x1.3    8.05        1.05-2.03          12
x1.6    9.91        1.20-2.50          **0**
x2.2    13.63       1.56-3.50          0
======  ==========  =================  =========

The contacts go, and the bonds simply scale with the cell -- a 2:1 spread
at every size. There is no cell where it is carbon.

## Hybridisation is not curvature, and the flat haeckelite proves it

`analyse/hybridisation.py` measures sp3 character from the **angle sum**
at each carbon: 360 deg flat (sp2), 328.4 deg tetrahedral (sp3), and the
fraction is the linear interpolation. Validated against the knowns:

==================  ============  ================
structure           angle sum     sp3 character
==================  ============  ================
haeckelite (flat)   360.0 deg     **0.00**
CNT (10,10)         359.2         0.03
CNT (4,4)           355.0         0.16
C60                 348.0         0.38
heptanene           301.4-354.2   **1.03**
==================  ============  ================

**The haeckelite is the control that matters.** It is nothing but
pentagons and heptagons and it reads exactly zero sp3. Curvature lives in
the ring census; hybridisation lives in the angles, and a measure that
confused them would be reporting the census twice.

Note this is the same angle sum the curvature section warns is never
negative at a trivalent vertex. That is precisely why it cannot give a
curvature *sign* -- and precisely why it does give pyramidalisation,
which is what it is used for here. Heptanene reading past 1.0 is the
quantitative form of the refusal: its carbons are bent further than
tetrahedral, which no carbon is.

## Two display faults on structures with a tiny cell

**Edge-bond stubs can dominate a picture.** A crossing bond is drawn from
one atom toward the *image* of the other, so the stub reaches up to a
full cell outside the cluster. Heptanene has 29 of its 84 bonds crossing
a 6.19 Å cell -- 35% -- so the stubs reach twice as far as the structure
and are most of what is on screen, which is what "does not visualise
correctly" was. The limits now include the drawn segments rather than
only the atoms, and the preview says the crossing share when it is over
20% so the picture is explained rather than merely drawn.

**The superlattices were not a regression.** Rebuilt at HEAD, the
super-hypercube returns the identical census to the screenshot that
raised it -- `{5: 254, 6: 2539, 7: 398, 8: 24}`, 6494 atoms. What changed
is `set_box_aspect`: the old view squashed z by a quarter, so every
structure now shows depth it was previously hiding. That is a correction,
and it does make a dense superlattice look busier.

`refine_disclinations` is separately capped above
`LARGE_MESH_VERTICES`. The per-round cost is small and near-linear where
it was measured -- 0.10 s at 654 vertices, 0.21 s at 1153, so forty
rounds is seconds -- but a 12 592-atom superfullerene did not finish in
fifteen minutes with the pass on, and **that has not been isolated**. The
cap bounds the work; it does not explain the case.

## Where Dunlap's twelve comes from

Not from `4*pi*r/e`. That formula was written here first, depends on the
tube radius, and gives 17 or 27 -- it was never the right quantity.

On a torus `K dA = cos(phi) dphi dtheta`: **the radii cancel**. So
`integral K dA` over the outer half is `4*pi` for any `R` and any `r`,
and the disclination budget there is `3/pi * 4*pi = `**`12`**, with `-12`
on the inner half. **That is Dunlap's twelve**, and it is a topological
statement about the surface rather than a fact about carbon.

Two things follow, and both were measured:

- **A remeshed torus already satisfies it exactly.** Straight out of the
  remesher, the outer half carries `+12` and the inner `-12`. Nothing
  needed fixing about the budget.
- **What it lacks is the census.** It carries that budget as 73
  pentagons and 71 heptagons rather than 12 and 12. The surplus is
  neutral 5-7 pairs -- dislocations, which cost the budget nothing and
  are exactly the "scattered at random" the eye sees.

So a Dunlap toroid is not a different budget. It is the same budget with
the dislocations annealed away.

### Neither pass alone gets there

- **Census annealing** removes pairs and scrambles the split: 73/71 down
  to 21/17, outer charge falling `+12 -> +8`. `sum(|deg - 6|)` does not
  know which side of the equator a defect belongs on.
- **Placement** holds the split and stalls on the pairs: greedy descent
  cannot escape its own minimum, stopping at 47/43.

**Alternating them beats both**: 73/71 -> **17/15 with the split at
+11/-11**, converged by the fifth cycle. `refine_disclinations` is that
loop, and `place_curvature=True` now runs it rather than one greedy pass.
End to end through the builders:

============  =================  =================  =========  =========
structure     rings before       rings after        placed     placed
                                                    before     after
============  =================  =================  =========  =========
toroid        68 / 560 / 68      **22 / 653 / 22**  93.4%      **100%**
Y junction    50 / 443 / 38      **23 / 497 / 11**  94.3%      **100%**
============  =================  =================  =========  =========

No close contacts either way, and the toroid's bond spread improves
(0.164 -> 0.151). The toroid comes out two atoms heavier: the refinement
itself is verified to preserve `V`, `F` and every position exactly, so
that is the dual step downstream, not the flips.

**It is still not 12/12.** 22/22 is 68% of the way from 68/68, with every
one of them on the correct side of the curvature. The remaining ten pairs
are what greedy descent plus a cooled anneal could not find, not a
different budget.

## Four things the window got wrong about structures that were right

All four were reported from screenshots, and in every case the structure
was correct and the window was not. Worth keeping together, because the
common thread is that a display fault reads as a physics fault.

**A cubic box is not equal limits.** `_apply_limits` set the same span on
x, y and z and the C60 still rendered visibly squashed -- while its own
panel read bonds 1.420-1.420 Å and angles 108.0-120.0, which is a perfect
truncated icosahedron. Matplotlib's 3D box aspect defaults to `(4, 4, 3)`,
so equal limits still render flattened along z. `set_box_aspect((1, 1, 1))`
is the fix and it belongs beside every `set_xlim` trio.

**Colouring atoms by ring cannot work on a closed cage.** Every one of
C60's 60 atoms belongs to exactly one pentagon, so `_ring_of_atom`'s
priority rule gives all 60 the pentagon colour and the picture says "all
pentagons" about a structure that is 12 pentagons and 20 hexagons. This
is not a bug in the rule -- the rule is right and the question is
unanswerable per atom. Filling the **rings** is the view that answers it,
and it is also the only way to see at a glance whether the 5s, 7s and 8s
are scattered or placed. Rings spanning more than `MAX_RING_SPAN` are
left unfilled: a ring straddling the periodic boundary is a correct list
of atoms whose coordinates sit on opposite sides of the cell, and joining
them in order draws a polygon across the whole structure.

**`_update_info` runs after `_redraw`, so a KeyError there is silent and
wrong.** Heptanene's geometry dict had no `n_close_contacts` and the
panel indexed it directly, so the new structure was drawn while the panel
kept describing the *previous* build -- 64 atoms and a 16/16 census from
a haeckelite, against heptanene's real 56 atoms and `{7: 24}`. A stale
panel is worse than a blank one: it is a wrong reading that looks right.
The panel now uses `.get`, and heptanene reports contacts like everything
else (127 of them, which is itself evidence for the refusal).

**A GUI keyword the builder does not take is a TypeError on first tick.**
Wiring `place_curvature` through, the window passed it to
`coil (relaxed)` and missed `coil (periodic, DFT)`. Checking
`parameter_names` does not settle it either way: `build_coil` takes
`**kwargs` and forwards to the swept-tube builder, so it accepts the
keyword without declaring it, while a builder could declare it and never
reach the remesher. `test_place_curvature_wiring.py` reads the modes out
of `app.py` itself -- a hand-kept list would drift from the window, which
is the failure it exists to stop -- and checks both halves.

## Where a disclination belongs: the placement rule

The flip annealer's objective was `sum(|deg - 6|)`: every vertex wants to
be a hexagon, wherever it sits. Right on a flat sheet and wrong on
everything else here. The replacement is discrete Gauss-Bonnet read
locally -- over a region `sum(6 - deg) = (3/pi) * integral K dA`, so one
vertex's share is `3 * K(v) / pi`, which `vertex_curvature_targets()`
returns.

`K(v)` is the angle deficit `2*pi - sum(theta)` of the **triangulation**,
and it is geometric rather than a restatement of the degree: a degree-5
vertex on a flat sheet has five 72 deg angles summing to exactly `2*pi`
and asks for nothing. The suite pins that at `< 1e-9`. (Not the trivalent
net's deficit, which the curvature section above records is never
negative.) The rule is checked by the one identity that makes it a rule:
the targets sum to `6*chi` -- measured **11.951, 11.943, 11.865 against
an exact 12** on L, Y, X, and 12 on a sphere.

### Charge over a neighbourhood, never per vertex

The first version compared them **per vertex and was worthless**, in a way
worth keeping because it looks correct. Curvature is spread over area; a
disclination is a point. Per vertex every target lies under 0.2 while a
degree excess is +-1, so a heptagon costs 1.045 wherever it sits against
0.0005 for a hexagon -- **2067:1, and completely blind to position.** It
was a squared census with a rounding error, and it measured as one.

Summed over a **two-ring window** (19 vertices) the two become comparable
-- targets -1.29 to 2.98 against excesses -3 to 4 -- and correlate at
0.814. One ring (7 vertices) is too small to hold a disclination's worth
of curvature and correlates at 0.574; three (37) reaches 0.897 but smears
a junction neck into its arms. Two is the working choice, and
`curvature_misfit()` is that comparison summed.

### Why the placement pass recomputes instead of updating

`place_disclinations()` is greedy descent that **rebuilds the whole state
every round**. A faster incremental version was written first and did not
work, twice over, both failures worth recording:

- **Windows go stale across sweeps.** Flips change the adjacency, so
  neighbourhoods taken once no longer exist. Optimised against stale
  windows a Y junction ended at misfit 418 having *started* at 355 --
  worse than doing nothing, which a minimiser cannot be.
- **They go stale within a sweep too.** With the snapshot refreshed each
  sweep, 45 flips were accepted in one sweep, every one of them
  "improving" by its own delta, and the sweep still ended worse. Only the
  first was measured against the real graph.

Blocking each accepted flip's window is *not* enough either: two flips are
independent only when their windows are disjoint, which is **twice** the
window radius apart, not one. The pass therefore applies only
mutually-unreachable flips per round and then **verifies the true misfit
fell**, rolling the round back if it did not. The returned history is
monotonic by construction and the suite asserts it.

### The guard that makes it a placement rather than a rewrite

A flip preserves `sum(6 - deg)` but can still turn two hexagons into a
5-7 pair. Left free, the descent **buys misfit by manufacturing pairs**
to chase curvature finer than one disclination can represent -- fitting
noise. Measured with the guard off, the two measures moved in opposite
directions: misfit down 17-36% while the fraction of disclinations on the
correct side of the curvature went **95.7% -> 85.6%** on an X junction.
So the pass refuses any flip that raises the defect count. With it:

=========  ==========  =========  ===================  ==================
junction   placed      placed     rings before         rings after
           before      after
=========  ==========  =========  ===================  ==================
Y          94.3%       **98.4%**  50 / 443 / 38        37 / 469 / 25
L          90.0%       **93.8%**  41 / 313 / 29        38 / 320 / 26
T          92.3%       92.5%      58 / 422 / 46        46 / 447 / 34
X          95.7%       **97.9%**  64 / 539 / 50 + 1o   53 / 562 / 39 + 1o
=========  ==========  =========  ===================  ==================

All four improve, the disclination count falls as well, and no structure
gains a close contact. The bond spread moves both ways and by little
(0.117 -> 0.137 on a Y, 0.140 -> 0.126 on an X), so it is not a bond-
quality argument either way.

**The octagon survives the pass** on the X junction, which is the right
answer rather than a tolerance: `max_degree=8` admits it and the
objective keeps it where the curvature is most negative, at the neck. An
octagon carries -2 of budget, so it is worth two heptagons of negative
curvature and belongs only where that much is concentrated.

It is off by default (`place_curvature=False`) because turning it on
moves every structure that uses the remesher.

## Hypercubes and supertubes: the skeleton can be any graph at all

Two entries that cost almost nothing because `build_supernetwork` takes
a graph rather than a name:

* **`super-hypercube`** is the 4-cube: sixteen vertices, 32 edges,
  4-regular, budget `12*(16-32) = -192`, met exactly at 6522 atoms. Its
  geometry is the usual perspective projection along `w`, so the struts
  come out in **three different lengths** (ratio 2.31) -- that is what a
  tesseract looks like in three dimensions, not a defect, and `scale`
  sets the shortest because that is the one that has to leave a tube.
* **`supertube-(n,m)`** is super-graphene rolled. The super-graphene net
  *is* a honeycomb, so rolling it is the same operation as rolling
  graphene one scale up -- which means `build_cnt` already does the
  geometry, the seam closes by construction and the axial period is
  exact. Measured (4,4) over 2 periods: 4608 atoms, `pbc=(F,F,T)`, cell
  89.6 x 89.6 x 54.5 A, budget -192 met exactly. The `(n,m)` are the
  **super**-lattice's indices, not the wall's.

**Every cage reads `scale` as the strut length**, and the supertube
nearly broke that: writing all three axes as fractions of the axial
period made its struts read 0.3 A in the menu and the build refuse. The
two transverse axes are fractions of the strut and `z` of the period,
with the ratio carried in `shape`. A test pins the contract across all
of them -- at 2% rather than machine precision, because a *rolled*
skeleton's struts genuinely vary by about 1%, exactly as a real
nanotube's bonds do.

## `build` asks the signature; it does not assume a seed

`jobs.build` appended `seed=job.seed` to every carbon builder. An
exact-lattice builder -- the nanocone's sector cut, the polyhex toroid's
bend, heptanene's group-theoretic map -- has no randomness and no such
parameter, so the call raised `TypeError` before any geometry ran.
**Three modes shipped broken this way**, and in the window it surfaced
the instant their preset was picked, before anything had been drawn.

It now consults `parameter_names`. The test that guards it must **call
`build`**, not bind the signature: the first version bound `builder`
against `parameter_names`, which is exactly what `build` now consults, so
it agreed with itself and passed against the broken code. The three
deterministic modes are all under three seconds, which is what makes
calling them affordable in a unit test.

Applying a preset **builds immediately**, so a preset must not name a
combination the builder refuses. The heptanene preset set `strict=True`,
whose whole purpose is to refuse -- so picking it fired the refusal as an
error dialog. It sets `strict=False` now, and the panel's text carries
the finding instead.

## Presets are a catalogue of textbook structures, not a parameter sweep

`gui/app.PRESETS` is a reference shelf: one entry per structure worth
knowing, not several per structure at different sizes. It carried three
"meshed wall" nanocoils differing only in radius and turn count, which is
a sweep and belongs in `workflows/sweep.py`.

Two rules, and a test for each in `tests/test_gui_static.py`:

* **Every key must be a registered `_var` name and every `mode_kind` a
  real mode.** A preset is applied in a plain loop, so a typo is silent:
  the preset applies, that one field keeps its old value, and the
  structure built is not the one named.
* **No preset may anneal a curved surface.** Every junction, schwarzite,
  network and meshed-coil preset carried `anneal: 80` while this file
  said not to -- the documentation and the menu disagreed, and the menu
  is what people click. Measured on the Y junction preset's own
  parameters, 80 sweeps widen the bond spread from **0.0136 to 0.0175 Å**
  and the bond range from 1.366-1.484 to 1.319-1.506.

  Worth knowing, because it is the one thing annealing improves: it also
  tidies the census (88 non-hexagons down to 45) and takes the
  disclination placement from 94% to **100%**. The wall is still the
  deciding metric, as this file has always said -- but say it as a
  trade-off that was measured, not as a rule with no cost.

## Unit cells: pad only what does not repeat, measure only what is vacuum

`cell.to_unit_cell` turns any structure into `pbc=(True, True, True)`
with a real cell, because every plane-wave code and periodic viewer is
3D-periodic and none of them has a "molecule" setting.

Three rules, each of which was a bug first:

* **A periodic axis is never padded.** Its lattice vector is the
  physics; changing it changes the crystal rather than the box.
* **A non-periodic axis is rebuilt from the atom span, not padded.** A
  finite builder's cell is already a bounding box with padding in it, so
  padding that compounds on every call. `to_unit_cell` twice must equal
  `to_unit_cell` once, and a test pins it.
* **`image_separation` counts only images displaced along a vacuum
  axis.** In a real crystal an atom bonds to its image -- a nanotube is
  1.42 A from itself along its own axis, a schwarzite 1.37 A -- so
  measuring every image reported every correct periodic cell as
  unconverged. That is backwards: the contact *is* the structure. A cell
  with no vacuum direction returns `inf` and reports `None`, because a
  bulk crystal has nothing to converge and a number there would invite a
  meaningless comparison.

## A minimum-image bond list names the pair, not the cell

`info["bonds"]` is built from `guess_bonds`, which uses the minimum image
convention. That records `(i, j)` and throws away *which cell* j was in.
Anything that uses the pair without recovering the translation is wrong,
and wrong in a way that does not raise:

* **Drawing.** `pos[j] - pos[i]` for a bond crossing a face is as long as
  the cell -- up to 46 A on a 1502-atom gyroid, 131 times over. It looks
  like a rendering fault. It is a missing subtraction, and it was the
  single biggest source of visual noise in the preview.
* **Repeating.** `Atoms.repeat` copies `info` verbatim, so a supercell
  keeps one cell's bond list, ring list and ring census while having
  eight times the atoms. The preview then draws bonds on the first copy,
  the Blender bundle exports an eighth of the connectivity, and the Euler
  check calls a sound 2x2x2 gyroid BROKEN. Nothing about the atom count
  shows any of it.

`utils.geometry.bond_shifts` returns the integer translation per bond, and
`cell.supercell` is what the GUI's repeat button goes through: it
re-indexes the bonds copy by copy (verified equal to re-running
`guess_bonds` on the supercell), walks each ring bond by bond so one
straddling a face comes out whole, and scales the Euler budget by the
number of copies because it is a sum over rings.

Two things to keep: the shift search is a **candidate scan** around the
least-squares projection, not a rounding, because rounding is right for an
orthogonal cell and off by one for a sheared one. And repeating an
**aperiodic** direction raises rather than obeying -- the copies would sit
inside each other.

## The GUI cannot be tested without a display, so test what is not widgets

`tests/test_gui.py` skips itself entirely without tkinter, which is most
checkouts. `tests/test_gui_static.py` reads `gui/app.py` as text instead
and checks it against itself: every `command=self.x` names a method that
exists, and every `self.x` read is assigned somewhere. Both are attribute
lookups resolved at click time, so a linter sees neither.

When logic has to live in the widget module, keep it a function of its
arguments (`_describe`-style) or put it in a package module so it can be
tested directly. The bond-shift maths is in `utils.geometry` and the
supercell in `cell` for exactly that reason.

## Blender: the render must not depend on the subject's size

Three faults, and the first is the one that made the other two hard to
notice.

**The scripts must ship inside the package.** They lived beside it, so
pip left them out and the GUI's render button was dead for every
installed copy -- it reported "run the GUI from a full checkout". They
are declared as `[tool.setuptools.package-data]`, **not** a subpackage:
they import `bpy`, so an importable `nanocarbon_lab.blender.render_cnt`
would be a module that cannot be imported.

**Camera distance follows from the lens.** It was a hardcoded
`3.2 * radius`, which only fits a 50 mm lens; the styles use 50-90 mm
and four of five cropped the subject. The correct distance is
`radius / sin(atan(sensor / 2f))`, narrowed further by the aspect ratio,
because Blender maps the sensor onto the *longer* image axis and the
shorter one clips first. Do not put a constant back.

**Light positions and energies scale with the subject.** The `LightSpec`
coordinates are written against `NOMINAL_RADIUS` (10 Å); positions are
multiplied by `radius / NOMINAL_RADIUS` and energies by the square of
it, which holds irradiance constant. Without this a 90 Å structure
engulfed its own lighting and rendered at 0.04 luminance against a black
background. **A SUN is exempt** -- it is directional and infinitely
distant, so its irradiance does not fall off and scaling it over-lights.

`gui.has_bpy()` lets the GUI render through the `bpy` module when no
Blender application is installed, which removes the pipeline's single
most common failure. Check it with `find_spec`, never by importing:
importing Blender costs hundreds of megabytes, and it runs while drawing
a label.

## Doping: the host is carbon, and the elements are not interchangeable

`dopants/chemistry.py` is the authority on which heteroatoms may replace
a carbon. It replaced a four-element tuple in `utils.constants`, and the
reason is not that the tuple was short: a bare list made 10% Fe as easy
to ask for as 10% N, and only one of those is a material. Each entry
carries a **site type** and a **max_fraction**, and the warning fires
against the element's own ceiling.

* **planar** is N and B only. They are the only dopants within 0.15 Å of
  carbon and isoelectronic with it to within one electron, and the only
  ones that reach tens of per cent in real samples. Do not move anything
  else into this class on the grounds that it "should fit".
* **puckered** (P, S, Se, O, Si, Ge, Al) substitutes but leaves the site
  sp3 and out of plane. The builder places these on the **ideal lattice
  site** and cannot know how far they move; say so rather than implying
  the geometry is finished.
* **vacancy** (Mn, Fe, Co, Ni, Cu, Zn) is not a lattice substitution at
  all in reality -- these are M-N4 single-atom sites in a vacancy. One
  substituted onto a perfect lattice is a starting geometry, not the
  motif.

The halogens are absent on purpose: F and Cl bond *to* a carbon sheet
rather than replacing a carbon in it, so fluorographene is an adsorption
problem. Do not add them here.

Warnings, not errors, everywhere except an unknown element -- metastable
and computational structures are legitimate, and an element with no
entry has no radius and no coordination ceiling, so validation and
export would both misjudge it. **Adding a dopant means adding it to
`COVALENT_RADII` and `MAX_COORDINATION` too**; a test pins that.

**Ring-selected placement reads `info["rings"]`; it does not perceive
rings.** `dopants/rings.py` puts dopants on pentagons because those carry
a curved structure's curvature and its reactivity -- a fullerene's
chemistry is at its pentagons. Every mesh-based builder already records
the real atom indices per ring, so no geometry is needed. A structure
without that metadata **raises**; do not add a distance-based fallback,
which is the exact failure `builders/fullerene_mesh.py` exists to
prevent. Its concentration counts against the **sites of that ring
size**, and both that and the overall fraction are recorded, because on
a capped tube they differ by a factor of four and either alone reads as
the other.

`jobs.apply_doping` is the single placement policy; the GUI and the CLI
both go through it, as they do for everything else in `jobs.py`.

## Co-doping is one pass, not doping twice

`dopants/codoping.py` places several heteroatoms at once. It is not a
convenience wrapper around `dope_random`, and the two differences are the
whole module.

**Every fraction is of the same denominator.** Applied sequentially, each
species' concentration is taken against the carbons the *previous* one
left, so a 400-atom sheet asked for 5% N and 5% B came back with 5.00% and
4.75%. The error compounds with concentration and species count, and it was
silent: `info` recorded the fractions requested, not the ones placed. Counts
are now worked out together against the original carbon count by **largest
remainder** — independent rounding does not add up (four species at 0.1 on
95 carbons round to 10 each, which is 40 sites for a requested 38).
`jobs.Job` **refuses a `dopant` and a `codope` spec together** rather than
letting one silently redefine the other's fraction.

**Placement is correlated, because the chemistry is.** `affinity` takes
`"random"`, `"seek"` or `"avoid"`. In B,N co-doped graphene the species
prefer to sit next to each other — a B–N pair is isoelectronic with a C–C
pair — which is why real samples grow BN domains rather than a solid
solution; `"seek"` places dopants as bonded pairs of *unlike* species.
`"avoid"` leaves no two dopants bonded, for the dispersed case.

**The affinity's effect is measured, not asserted.** `info["codoping"]`
records `dopant_bonds` and `unlike_bonds`, and that is what the tests check.
Measured on a 400-carbon sheet at 5% N + 5% B: `seek` 22 dopant–dopant bonds
of which 20 unlike, `random` 3, `avoid` **0**. Do not replace these with a
claim in a docstring — the numbers are the only reason to believe the rule
did anything.

This is **not** an energy calculation: nothing here knows B–N is favourable.
The affinity is a placement rule the caller picks to match the sample they
mean, and the docstring says so. `"seek"` with a single species has nothing
to alternate with and produces like–like pairs, which is the honest answer
rather than a silent no-op.

`"avoid"` can run out of room — an independent set of the requested size
need not exist on a trivalent lattice — and then it **warns and records
`unplaced`** rather than filling the remainder in adjacent, which would give
up the one property that was asked for.

## Functionalisation adds atoms; doping replaces them

`functionalize/` is a third chemistry axis, not an alternative to the
other two. A dopant substitutes for a host atom, so it belongs to carbon
alone; a **group sits on top of a surface**, and carbon, MX2 and a vdW
stack all have one. `jobs.apply_grafting` therefore runs for every
family, after doping and after the MX2 edits, because it must see the
finished surface: a vacancy opens sites a group can reach and a Janus
face changes which element the anchor is.

**Groups are internal coordinates, never Cartesians.** That is the whole
design. `substitute(hydroxyl, {"O": "S"})` is a thiol with the *right*
geometry because the C-S bond is rebuilt at 1.81 Å; stored as
coordinates, the same swap leaves sulphur sitting at oxygen's 1.42 Å,
which is a 0.4 Å error in the one bond the group is defined by. Only
angles are stored, and every length comes from `COVALENT_RADII` times a
bond-order factor — which reproduces eleven literature bonds to within
0.03 Å, each pinned by a test. Swaps are restricted to the **same
valence**: an -OH whose oxygen became nitrogen is not a variant, it is a
group with a missing bond.

Angles in the registry are the **bond angle as chemistry quotes it**
(109.5 for tetrahedral). The Z-matrix works from the parent bond's
continuation, where that is a 70.5 degree deflection; the conversion
lives in `build_positions` so a new group is written with numbers a
chemist recognises.

Four things in `attach.py` are load-bearing:

* **The normal is where the bonds lean away from**, with the plane
  normal only as the fallback when they cancel. A pyramidal site has no
  bond plane — an MX2 chalcogen sits above three metals whose bond
  vectors are not coplanar — so the plane-normal branch returned
  whichever direction they varied least along and *every* group on
  *every* MoS2 surface was refused. The remaining sign ambiguity is
  fixed by propagating across the bond graph and then flipping **once
  per connected component** by the divergence theorem, so a multi-wall
  tube gets each shell oriented on its own instead of the inner one
  turned inside out.
* **Atoms within two bonds of the anchor are exempt from the steric
  test.** Their separation is a bond angle, not a contact: a hydroxyl's
  oxygen is 2.09 Å from the three carbons around its anchor and its
  hydrogen 1.96 Å from the anchor itself, no matter what. Scoring those
  as clashes refused literally every placement on every structure. The
  exemption holds **in every periodic image**, not just the home cell —
  a site at a cell face has its bonded neighbour stored on the far side,
  and restricting it to the home cell refused a third of the sites on a
  6x6 graphene cell.
* **Twelve rotamers are tried and the roomiest kept.** A group on one
  single bond spins freely, so refusing a site because the frame's
  arbitrary tangent aimed a hydrogen at the wall would measure the frame
  rather than the chemistry.
* **`face="both"` alternates by sublattice**, which is the chair
  conformation. Fluorographene reaches 100% coverage that way and 42% by
  shuffle order, because two fluorines on adjacent carbons on the same
  face are 1.42 Å apart. A site whose inner face is *blocked* always
  takes the outer one: an MX2 bond graph is bipartite between metal and
  chalcogen, so every chalcogen shares a colour and alternating aimed
  every group into the sandwich — zero coverage, blamed on sterics,
  which was true and useless.

Coverage is **measured, not assumed**, and the shortfall is reported
with its reason. Fluorine reaches 100% on graphene, a carboxyl 28%, an
epoxide about 30% of the bonds — the last being a maximum matching, since
no two bridges may share a carbon, not a steric limit.

`info["functionalization"]` is a **list**, appended to. Graphene oxide is
an epoxide graft followed by a hydroxyl graft, and with a single dict the
second call erased the first. `info["grafted_atoms"]` is the cumulative
list of added atoms, and `tmd.quality` excludes them: it calls anything
that is not a chalcogen a metal, so a grafted hydrogen counted as a metal
and a sound MoS2 slab read BROKEN with X/M = 1.73. Both are index lists
and both are remapped on deletion.

Grafting also **re-pads the non-periodic axes** to the vacuum the
structure was built with. Groups stick out, and a finite builder's cell
is a bounding box: a hydroxylated (6,6) tube built with 12 Å came back
with 8.55 and validation refused it. Periodic axes are never touched, and
the atoms are never moved — re-centring would silently shift every
pre-existing coordinate.

**Validation had three carbon-shaped assumptions**, all now element-aware,
and all the same mistake the coordination ceiling already fixed for
metals: 0.970 Å is the literature O-H bond and was warned about for being
below carbon's sp2 window (contacts are now judged against the pair's own
covalent radii); coordination 1 is a full valence for H and the halogens
(a correct CF monolayer produced fifty "dangling" warnings); and a
carbonyl oxygen has one neighbour and a full valence too, which
coordination counting cannot see, so builders declare
`info["terminal_atoms"]`.

**Two CLI post-processing helpers exist**, `_apply_post` and
`_maybe_dope`, and every post-build step must be in both. Grafting went
into the first alone, so `--graft` on a capped tube, fullerene, junction
or schwarzite did nothing at all — and did not even report an impossible
swap as impossible.

## Dichalcogenides are chosen by two elements, not by a formula

`material_for(metal, chalcogen)` is the lookup the GUI and CLI use, and
`available_metals` / `chalcogens_for` drive their dropdowns. It is
deliberately a **lookup, not a constructor**: an MX2 not in `MATERIALS`
is one whose lattice constants this package does not know, and deriving
them from covalent radii would produce a structure that looks
authoritative and is not. A missing pair raises with what *is* available
for each of the two elements.

Absences that are chemistry, not oversight: ReS2/ReSe2 distort into
diamond chains and NbTe2/TaTe2 into another pattern, so none is an ideal
1T or 2H cell; SnTe2 is not a layered MX2 at all. Do not "complete the
grid".

The platinum dichalcogenides have a van der Waals gap of ~2.4 Å against
MoS2's 3.0. That is real -- it is why PtSe2's gap depends so strongly on
layer count -- so the table's consistency test allows down to 2.3 Å.
Do not tighten it to make Pt look like the rest.

Adding a material means adding its metal to `COVALENT_RADII`,
`MAX_COORDINATION` and `HOMOELEMENTAL_BOND`. The last one feeds
`BOND_CUTOFF_OVERRIDE`, and a test asserts the resulting M-M cutoff
falls **below** that material's lattice constant -- above it, every metal
bonds to its six in-plane neighbours and reads as 12-coordinate.

## Analysing a file: keep what was recorded apart from what was guessed

`analyse/` describes a structure the framework did not build. Its whole
discipline is the three-way split the report prints: **recorded** (read
from `atoms.info`, so it is what a builder did), **measured** (true of
the coordinates -- a bond length, a span, an element count) and
**inferred** (a rule with thresholds behind it -- the bonds, the rings,
the shape). Blurring them would be worse than no report, since
"pentagons: 12" reads the same whether a builder placed twelve pentagons
or a distance cutoff guessed them.

**Rings are traced as faces, not searched as cycles.** For a trivalent
surface -- graphene, any tube, any cage, any schwarzite, any junction --
a ring is a *face of the embedded graph*, and ordering each atom's
neighbours by angle about its surface normal gives a rotation system
whose orbits are exactly those faces. Measured against the builders' own
recorded rings it reproduces them exactly, and the face count satisfies
`F = E - V + 2 - 2g` by construction.

Shortest-path rings are the fallback for anything that is not a
trivalent surface (an MX2 sandwich, a bulk crystal, a molecule), and the
report says which method ran, because they answer different questions.
SP rings **systematically miss large rings on a tiled surface**: a
heptagon every one of whose bonds also borders a hexagon is never the
smallest ring through any bond. Measured: a Y junction has 16 heptagons
and SP rings find 4; Schwarz P has 65 and SP rings find 10. Neither is
`networkx.cycle_basis`, which reports nine- and ten-membered rings on a
C60 -- a cycle basis counts independent loops, which is a different
question with a different right answer.

A cell so small that a pair of atoms is bonded through **more than one
image** cannot be written as a simple graph at all, so no census on it
describes it. That is detected and refused rather than answered.

**Shape is measured, never read off `pbc`.** Every plane-wave code writes
a slab as `pbc=(True, True, True)`, so trusting the flag would call it
bulk and quote a density for the padding. An axis counts as periodic only
if it also has no vacuum gap. For the finite part, spans classify a sheet
and a chain, but a tube and a cage both have three large spans -- what
separates them from a solid is that they are **hollow**, which the
relative spread of the radial distances measures without needing a length
scale. Two refinements were bugs first: the tube axis must be tried
against every principal axis *and* the periodic direction (a one-cell
MoS2 tube is 33 Å across and 4.6 Å long, so its longest principal axis is
a diameter and it read as a chain), and thickness must be measured along
the cell's non-periodic direction (on a 1x1 MX2 bilayer the smallest
principal span is *in plane*, so a 9.3 Å slab read as 0 Å thick).

Disjoint pieces are classified separately, and they win **only when they
disagree with the whole about dimensionality**. Two stacked MX2 layers
are two sheets and also one slab, both 2D, and the slab is the more
useful true statement because it accounts for every atom; two nested
tubes are 1D each while their union reads as a 0D branched shell, and
there the union is simply wrong -- the annulus between the walls is empty
space, not material.

**A decorated structure is not a different structure.** A tube with 72
carboxyls has 288 atoms at assorted radii, so its radial spread rises and
the whole reads as a solid cluster -- while every atom of its wall is
exactly where it was. Both the shape and the rings therefore fall back to
the **backbone**, the graph 2-core, which strips each pendant group one
atom at a time (the hydrogen, then the hydroxyl oxygen, then the carbonyl
oxygen, then the acid carbon) and leaves the wall. Faces traced on it
recover the tube's exact ring census, and the ring indices stay in the
original numbering.

**Hollowness is tested before flatness.** C20 is 3.9 Å across its second
principal axis -- narrower than `FLAT_SPAN` -- so the flatness test
claimed it as a chain before the shell test ever ran. A cage is hollow
whatever its size. A small-molecule test (at most 12 atoms *and* under
5 Å across, both conditions needed) runs before either, so water is a
molecule rather than a chain without C20 being swallowed with it.

Writing this package found three bugs elsewhere, which is what it is for:

* **`surface_normals` had `PYRAMIDAL_SUM` at 0.30**, so half of a
  Schwarz P cell's atoms were treated as pyramidal on the strength of a
  near-cancelling bond sum and were exempted from the sign propagation.
  431 of 1701 bonds joined atoms whose normals pointed opposite ways, and
  a quarter of any groups grafted onto a schwarzite would have gone
  through the wall. The threshold is now measured: every carbon surface
  is at most 1.07 (C20, the most pyramidalised sp2 carbon there is) and
  every MX2 chalcogen at least 1.79.
* **`coordination_numbers` counted node degrees in a networkx Graph**,
  which holds one edge per pair. In a 1x1 MoS2 cell an atom reaches the
  same neighbour through several images, so its metals read as
  two-coordinate and validation warned about dangling chalcogens in a
  perfect crystal. It counts from the bond list now.
* **`candidate_sites` offered atoms a previous graft had added.** A
  hydroxyl took the oxygen of an epoxide already on the sheet, giving a
  1.32 Å O-O peroxide bridge on a structure whose every other number
  looked right. Grafted atoms are excluded, and so is any atom already at
  its element's coordination limit.

## Removing an atom renumbers every atom after it

`utils/metadata.py` exists because two deletion paths -- carbon
`introduce_vacancies` and `chalcogen_vacancies` -- copied `atoms.info`
wholesale. The builders record `info["bonds"]` and `info["rings"]` as
**atom indices**, so after removing three atoms from a 240-atom capped
tube the bond indices still ran to 239 against 237 atoms, and every
index above a removed atom pointed at the wrong atom.

Nothing complained, which is what made it dangerous.
`coordination_numbers` prefers the recorded graph when it exists, so
validation read the corrupted one and passed; the render bundle writes
those same indices to JSON, so a defected tube drew bonds between atoms
that were never bonded.

Any function that deletes atoms must build its survivor list with
`keep_indices` and pass `atoms.info` through `remap_after_removal`.
Groups that lost a member are **dropped**, not repaired -- a bond with
one end missing is not a bond and a pentagon missing an atom is not a
pentagon -- and `ring_counts` is recomputed from the survivors rather
than carried, since a census contradicting the rings beside it is only
noticed after it has been plotted. **A new index-carrying `info` key
must be added to `INDEX_LIST_KEYS`**, not only to the builder writing it.

## MX2 quality is judged by role, not by the parent formula

`geometry_report` used to match bonds and count sublattices against
`info["metal"]` and `info["chalcogen"]`, which name the compound the
structure was *built* from. Every edit in `tmd/modify.py` introduces a
third species, so a Janus MoSSe had its Mo-Se bonds ignored and an
Mo(1-x)W(x)S2 alloy its W-S bonds: both read as under-coordinated, X/M
came out 1.00 and 3.60 against a true 2.00, and four correct structures
were reported BROKEN.

Classification is now by role -- `CHALCOGENS` is S/Se/Te and everything
else is the metal -- so a second metal or chalcogen counts without being
enumerated. Do not reintroduce a symbol comparison here.

Vacancies and antisites *are* off-composition; that is what they are.
The CLI drops `expect_stoichiometric` when `info["defect_log"]` is
present, exactly as it already did for a deliberately terminated ribbon.

`jobs.apply_tmd_chemistry` is the single policy, and `tmd_edit_amount`
is deliberately one field meaning a fraction, a count or a face
depending on the edit -- four fields of which three are always ignored
would be worse, and the GUI hint says which it is.

## Bond detection is element-aware, and must not be quadratic

Two faults here were load-bearing and are pinned by
`tests/test_validation_scaling.py`:

1. **`COVALENT_RADII` needs every element it will meet.** It held only C,
   N, B, S, P, H, O; everything else fell back to `MAX_CC_DISTANCE`
   (1.80 Å), so a 2.404 Å Mo-S bond was not a bond, every dichalcogenide
   validated as "isolated atoms", and **both exporters refused the entire
   tmd package**. Add radii when adding elements.
2. **Metal-metal pairs need `BOND_CUTOFF_OVERRIDE`.** Two metallic radii
   overshoot a layered compound's lattice constant -- Mo+Mo+0.30 is
   3.38 Å against MoS2's 3.16 -- so every metal picked up its six
   in-plane neighbours and read as 12-coordinate. The override cuts
   between the lattice repeat and a real 2.8 Å M-M bond, and it covers
   **pairs**, not just same-element ones, because an alloy puts Mo next
   to W.

`MAX_COORDINATION` is per element for the same reason: carbon's "5 or
more is unphysical" rejects a correct six-coordinate metal. Metals are
allowed 7 -- six ligands plus the 1T' dimer partner.

Neither `guess_bonds` nor `check_minimum_distances` may build the full
pairwise matrix. Both did, and it is O(N^2) in memory as well as time:
24 s and 79 MB at 3136 atoms, a gigabyte and unusable by the 11 164 atoms
of a magic-angle bilayer -- and validation runs on the path of every
export. Both now use `ase.neighborlist.neighbor_list`; the change was
107x faster at 3136 atoms with identical output.

`coordination_numbers` prefers `atoms.info["bonds"]` when the builder
recorded one. On a curved structure a distance cutoff is simply wrong: a
2.4 Å bond's cutoff reaches ~2.9 Å and sweeps up non-bonded neighbours,
so a schwarzite whose every metal has exactly six bonds reads as
ten-coordinate. Builders that know their bond graph should record it.

## Heterostructures: the twist is not a free parameter

`hetero/moire.py` stacks two hexagonal layers. Commensurate cells exist
only at `cos(theta) = (m^2+n^2+4mn) / (2(m^2+mn+n^2))`, holding
`m^2+mn+n^2` cells per layer -- (2,1) is 21.79 deg and (31,30) is the
1.0845 deg magic angle with 11 164 atoms. Snap the request and report
what was achieved; there is no periodic cell in between.

Three things already got this wrong; do not repeat them:

* **The two layers are 0 and theta, not +-theta/2.** A symmetric twist
  looks nicer and leaves the supercell commensurate with *neither* layer
  -- the fill then produced 242 atoms where 14 were required.
* **The sign matters.** `V = m*a1 + n*a2` is a lattice vector of a layer
  turned by `phi` exactly when `R(-phi)V` is one of the unrotated
  lattice, and it is `R(+theta)V` that lands on `n*a1 + m*a2`. Backwards
  gives 98 atoms instead of 14.
* **The honeycomb basis is (1/3, 1/3).** With `a2 = a(1/2, sqrt3/2)` --
  the 60-degree convention -- `(1/3, 2/3)` is the 120-degree form and
  puts sites `a/3` = 0.82 Å apart instead of 1.42.

`_fill_supercell` therefore asserts the atom count against
`cells * n_sites` rather than trusting the fill. All three bugs above
were caught by that assertion and would otherwise have produced
plausible-looking cells with the wrong number of atoms in them.

## Sweeps are built on jobs.py, not per mode

`workflows/sweep.py` takes a Cartesian product over `jobs.Job`, so every
mode is sweepable and a new mode in `jobs.py` gets a sweep for free.
Do not add a `batch_<mode>_sweep`; `batch_cnt_sweep` predates this and
stays only for compatibility.

`jobs.builder_for` is now the single builder table -- `build` uses it too,
so a mode cannot be buildable and un-sweepable at once -- and
`jobs.parameter_names` reads the real signature. The sweep checks names
against it **before building anything**: a typo used to be accepted in
silence, with the estimate falling back to the default and the mistake
surfacing as a TypeError on the first build, hours into a long run.

Each job gets `seed + index`, never a shared seed. The same seed across a
sweep puts the identical defect pattern in every structure, which is the
one property a training set must not have.

## GUI: one job description, one killable process

`jobs.py` is the single mapping from "what to build" to builder
arguments, shared by the GUI and the CLI. The GUI used to carry its own
ninety-line `if mode == ...` chain duplicating it. Three features depend
on that mapping being written down once — the estimate, the
copy-as-command-line button, and handing work to a subprocess — so add
new modes there, not in `gui/app.py`.

**Adding a mode is five edits, not one**, and forgetting the last two is
the single most repeated mistake in this repo's history --
`test_every_mode_has_a_sample` has now caught it four times. The list:
the family tuple in `MODES`, the `builders` dict in `build`, the atom
estimate, `_CLI_MAP`, and `SAMPLES` in `tests/test_jobs.py`. The sample
is not decoration: it is what drives the estimate, cost and
command-line-parses tests for that mode, so a mode without one is a mode
with no coverage at all. Run `pytest tests/test_jobs.py` before
committing a new mode -- it takes under a second and is exactly the
check that keeps being skipped.

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

The three columns are a **PanedWindow**, not fixed-width packs. Packed
at 268 px the parameter column clipped its own labels ("Subdivision freq
(diameter)") with no way to widen it, and a pixel width cannot be right
anyway -- it depends on the font, the theme and the platform. For the
same reason `ScrollableColumn._rewrap` re-wraps the explanatory labels
to the column's live width; a non-zero `wraplength` is what marks a
label as a hint, so new hints are picked up for free and ordinary labels
are left alone. Do not put a fixed `wraplength` back in as the final
word.

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
- Dopants replace carbon atoms; which ones and how many is `dopants/chemistry.py`, not a bare list (see below). Guardrails are warnings, not errors.
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
