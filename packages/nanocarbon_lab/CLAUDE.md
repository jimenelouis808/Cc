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
├── scene.py       # several finished structures -> one ILLUSTRATION for
│                  #   VESTA or Blender. Not a builder, and not a
│                  #   structure: see "A composition is an illustration"
├── render.py      # where Blender is, which script, what argv. The only
│                  #   part of the render pipeline an import can reach
├── utils/         # constants, geometry helpers
├── cli/           # command line interface
├── gui/           # tkinter desktop app (build / preview / export / render)
├── tests/         # pytest unit tests
└── examples/      # runnable example scripts

nanocarbon_lab/blender/   # the rendering pipeline, shipped as package
                   #   *data* (mesh_builder and render_cnt import bpy, so
                   #   it must not be a subpackage). styles.py and
                   #   bundle.py import no bpy and are the testable half.
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

## The Z path, and why the same request means three different things

`zigzag` is a centreline shape (`builders/centerline.py`), offered to
tubes, ribbons and bead chains. It is one name over three
representations, and **what it costs differs by representation, not by
how hard you ask**:

| Route | What a corner costs | Measured Z / length |
|---|---|---|
| Capped tube swept onto the path | elastic strain, trimmed to the budget | 8.2% at 1 period, **1.8%** at 2 |
| Ribbon folded out of its plane | nothing — an isometry | **25%** (15.6 Å over 63.4), limited by self-contact |
| Ribbon bent in its plane | strain, same budget as a tube | 2.5 Å against the fold's 15.6 |
| Bead chain | nothing — a smooth union of two necks | **43%** |

Three things follow and none is a preference.

**A swept tube's Z is gentle because the budget caps curvature, not
amplitude.** `max_strain / r_tube` is a radius of curvature — 74 Å for
the default 8% on a 5.9 Å tube — and a corner cannot be tighter than the
tube may bend. Halving the period therefore quarters the amplitude: more
periods gets you *less* Z. Lengthen the tube or use fewer periods;
raising `waviness` does nothing the budget will not take back.

**Bending a ribbon out of its own plane is free, and in its own plane is
not.** (Free does not mean unbounded: searched over `waviness` and
`turns`, a 6×30 ribbon tops out at 25% of its length, because the path's
arc length is normalised to the ribbon's span — a bigger swing shortens
the end-to-end distance it is measured against. What it is *not* bounded
by is strain.) The width rides the frame's binormal, which a
rotation-minimizing frame does not rotate, so every cross-section moves
rigidly: bonds come out 1.4197–1.4200 Å with no budget applied, and the
last 2 parts in 10⁴ is the frame interpolation rather than strain (it
shrinks with `frame_samples`). So `out_of_plane` is the default, and its
limit is the sheet **meeting itself** at the 3.4 Å graphitic contact —
which is checked, because nothing else would stop it. `in_plane` is the
tube's problem with the half-width in place of the tube radius, and goes
through the same `fit_to_strain_budget`.

**Sweeping a ribbon makes it finite.** A straight ribbon is periodic
along its length and its bond list names pairs across that seam; a bent
one has no translational period. The periodicity is dropped and the
bonds recomputed. This is the **sixth** place in this package where a
minimum-image bond list named the pair and not the cell — after the
census, `cell.supercell`, `scene`, the renderer and the bamboo
assembly. Assume it is the seventh next time.

**There is no `square_wave`, and the absence is deliberate.** Its
control points really are a square wave; the shape it returned was a
straight diagonal. `fit_to_strain_budget` interpolates toward the line
joining the path's **endpoints**, and a square wave does not end on the
axis — so trimming it does not flatten it toward a straight tube, it
flattens it toward a slanted one. Measured: a monotonic ramp at a raw
strain of 194–950%, an arc wearing a zigzag's name. A test builds those
control points by hand and pins the collapse, so the name cannot come
back by accident. For the same reason `zigzag` computes its **own**
sampling: four samples per period puts a control point on every corner
and every axis crossing, and a period count that is not a multiple of a
half aliases into a skewed sawtooth, so it is snapped rather than
honoured. `n_points` is ignored for this shape, and a test says so.

Neither is a Dunlap **knee**. A swept lattice carries its bend as
strain with no disclination to relieve it — see the table above.

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

## Bamboo: a sealed septum is not a manifold surface

`builders/bamboo.py` divides a tube into compartments, and the one
structural decision is **what the dividers are**. They are not part of
the tube's carbon network, and that is geometry rather than a modelling
convenience.

**Where a septum meets the wall, three sheets meet** -- the wall above
it, the wall below it, and the septum -- so a point on that rim has
three half-planes glued along a line for a neighbourhood, which is not
a disc. Marching cubes returns the boundary of a solid, and a boundary
is a manifold by construction, so **the implicit route cannot produce a
sealed septum at any resolution or blend**. Said in carbon: an atom on
that junction line would need four neighbours, and everything this
package meshes is a trivalent sp2 net.

Which is what the samples show anyway. A CVD-grown bamboo tube's
compartment walls are **separate graphitic cups** held against the wall
by van der Waals, not a covalent continuation of it. So this is an
**assembly**, as `assemblies.py` is for multi-wall tubes -- and it costs
nothing, because both pieces come from exact-lattice builders. The tube
is a rolled lattice and each cup a sector cut, so measured at
(16,16) with three compartments: **1840 atoms, `{5: 3, 6: 890}`, bonds
1.391-1.420 A**. One pentagon per cup and nothing else: no amorphous
wall anywhere, and the whole build is under a second because nothing is
meshed or relaxed.

**A bamboo tube has to be wide**, and that is the nanocoil's lesson
again rather than a default to taste. The cup must fit inside the wall
at a graphitic contact, so a **(10,10) tube -- radius 6.78 A -- leaves
3.38 A against a 3.40 A clearance and holds no cup at all**. Rim room
measured: (12,12) 4.74 A, (14,14) 6.09, **(16,16) 7.45** (the default),
(20,20) 10.16. Real bamboo tubes are multi-wall and tens of nm across,
which is the same answer from the sample's side.

**It stays periodic along its axis, and that is a bug fix rather than a
preference.** `build_cnt` returns a tube periodic in z whose bond list
records pairs across that seam. Assembled as a finite structure those
drew as lines the length of the cell -- measured, a **61.5 A "bond"**
against a real 1.42. That is the fifth place in this package the same
fact has bitten (the census, `cell.supercell`, `scene`, the renderer,
and now an assembly), so the cell keeps the lattice's own period and
the seam bonds stay true. A periodic cell of n compartments therefore
carries **n cups, not n - 1**: the divider at the far face is the one at
the near face.

And the compartment length is an **output**: the lattice quantises the
period, so 20.0 A asked comes back as 20.496 achieved, and both are
recorded. Neither is a prediction -- what sets it in a real sample is
the catalyst's feeding cycle, which nothing here models.

## Beads: a sphere is a capsule of zero length

`builders/beads.py` joins graphitic spheres with necks into **one**
closed surface -- the beads-on-a-string / interconnected-cage
morphology, which is neither a nano-onion (concentric shells of one
centre) nor a supernetwork (a *tube* on every edge, no sphere anywhere).

**It is one field, not two primitives unioned**, and that is what made
it small. A zero-length capsule is exactly a sphere, so a bead and a
neck are both segments handed to `implicit.graph_field` with different
radii, and the smooth union that already blends struts at a
supernetwork node is what flares the neck into the bead instead of
meeting it at a crease -- a crease being untileable by a hexagonal net,
which is why that union is smooth in the first place. `graph_field`
therefore takes **one radius per segment** now; the scalar path every
supernetwork and cage goes through is asserted **bit-identical**
(`np.array_equal`, not a tolerance), and the degenerate length is
clamped because the projection divides by it.

**The graph fixes the budget before anything is meshed**, by the
identity `supernetwork.py` already uses: `chi = 2*(V - E)`, so
`sum(6-n) = 12*(V - E)`. A chain is a **tree** -- `V - E = 1` -- so its
budget is **+12 at every length**: a sausage is still sphere-like, and
a test pins that at 2, 3, 5 and 9 beads. A ring closes, `V = E`, budget
**0**, a torus. Measured: chain of 3 gives `{5: 27, 6: 293, 7: 13,
8: 1}` summing to +12 against +12, ring of 6 gives +0 against +0, both
with zero contacts, and the chain scores **100% disclination
placement** -- all 41 on the correct side of the curvature, pentagons
mean sign +1.00 and heptagons -1.00. Only the periodic coil and the
knee nodes reach that elsewhere in this file.

`place_curvature` is **on** here and off for a junction, for the reason
`nanoparticle/coat.py` gives: a bead is convex, so pentagons belong on
it and stray heptagons are pure defects. A junction saddles and wants
the opposite.

### The zigzag layout: the same intersection, reached at a corner

A `zigzag` chain is a chain folded onto a Z path, so it is still a tree
and still carries +12 (measured at 5 beads: `{5: 37, 6: 357, 7: 21,
8: 2}`, 37 − 21 − 4 = 12). The step keeps its full length through a
turn, so every neck is the straight chain's neck and the spacing guard
goes on meaning what it meant.

What a turn does change is **which pairs are close**. The direction
swings by twice the leg angle, so the beads either side of a corner sit
`2 * spacing * cos(angle)` apart however long the legs are — a closed
form, tested at 30°, 45° and 60° — and that distance has to clear the
same `MIN_SPACING_FRACTION * bead_radius` floor the spacing does,
because it is the same two spheres intersecting. At 80° it is 5.83 Å
against a 14.00 Å floor and the build is refused before anything is
meshed.

**This is where a bead chain beats a swept tube and it is not a looser
limit, it is a different representation.** A corner here is a smooth
union of two necks, not a bent lattice, so there is no strain to pay:
measured, a 5-bead Z swings 43% of its length where a swept tube
reaches 8.2% at one period. See "The Z path" above for the table.

### Two spheres closer than twice their radius have no waist

The first version defaulted to `1.7 * bead_radius` and refused only
below `1.25`. Both are under `2.0`, where two spheres **intersect**, so
there is no waist for a neck to be -- and the render showed exactly
that: a sausage with dents, sound carbon, correct census, not a bead
chain. `MIN_SPACING_FRACTION` is 2.0 and the default 2.4.

**The dangerous case is the ring, and it is silent.** Measured at bead
7, neck 3, blend 3: a hole of **+4.9 A** closes as a torus, **+2.0 A**
is refused by `_finish`'s quality gate for features finer than a carbon
ring, and **+1.4 A closes as a SPHERE** -- every bond in range, zero
contacts, `chi = 2` and `sum(6-n) = +12` against a budget of 0. A sound
structure that is not the ring asked for. So there are two defences:
a geometric guard up front (the hole must clear the blend, a threshold
that sits inside the measured bracket), and an **independent census
check** after the build. `_finish` already tests the census against the
**mesh's** own Euler characteristic, which catches a torn mesh and
cannot catch a sound mesh that closed around the wrong graph.

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
| **Y junction** | 50 pentagons + 38 heptagons | `{6: N, 7: 6}` -- the six Gauss-Bonnet asks for |
| **Schwarz P** | 33 pentagons + 57 heptagons | `{6: 456, 7: 24}`, no pentagons |
| **Schwarz D** | -- | `{6: 608, 7: 96}`, no pentagons |
| **gyroid** | -- | `{6: 816, 7: 48}`, no pentagons |
| **super-graphene** | 89.6% placed, 11-21% non-hex | `{6: 432, 7: 24}`, bonds 1.408-1.436 Å |

The bottom three are the knee route (`builders/knee.py`), and they are
the rows where the crystalline side wins on **both** axes: a Y junction's
bond spread is 0.021 Å against the meshed Y's 0.117, at a third of the
atoms. That is the opposite of the rule below and the reason is that
nothing is being bent -- the node is assembled from straight arms and
pays its curvature in rings, exactly as the mesh route does, but with the
rings *placed by the construction* rather than chosen by a remesher.

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

## Goldberg-Coxeter: the literature's route, and the cleanest one here

`builders/goldberg.py` is the third way to an exact wall, after the knee
route and the mesh route, and it is the one the schwarzite literature has
used since Mackay & Terrones (*Nature* **352**, 762, 1991) and Koshino &
Aoki (*PRB* **93**, 041412(R), 2016): take a **coarse polyhedral map**
whose vertices already carry the ring sizes the topology needs, fill every
coarse triangle with a flat patch of the exact triangular lattice
(`subdivide` for `(m, 0)`, `root_three` for `(1, 1)`, composed for
`(m, m)`), and take the dual. Every new vertex has degree six, so the
only non-hexagons are the map's own vertices, `m` rings apart. It is the
C60-from-an-icosahedron construction on a surface of negative curvature.

| structure | route | atoms | census | bonds (Å) | verdict |
|---|---|---|---|---|---|
| Schwarz P | knee (20, 9) | 968 | `{6: 456, 7: 24}` | 1.417-1.540 | strained |
| Schwarz P | **GC(4,0)** | 768 | `{6: 368, 8: 12}` | **1.405-1.423** | clean |
| super-square | knee (12, 5) | 180 | `{5: 4, 6: 68, 7: 16}` | 1.374-1.505 | clean |
| super-square | **GC(3,0)** | 468 | `{5: 4, 6: 220, 8: 8}` | **1.404-1.436** | clean |
| Schwarz D | knee (18, 5) | 1440 | `{6: 608, 7: 96}` | 1.347-1.489 | clean |
| Schwarz D | **GC(2,0)** of C168's map | 896 | `{6: 336, 7: 96}` | **1.402-1.454** | clean |

Both Goldberg rows place 100 % of their disclinations on the right side
of the curvature, and their hexagons read a median `K` of zero: the
patches really are flat and the curvature really is at the defects.
P is clean at every size tried, GC(1,0) to GC(6,0) and (1,1), (2,2).

* **The P map is the muoctahedron {6,4|4}**: the hexagonal faces of the
  bcc truncated-octahedron packing, which separate the corner labyrinth
  from the body-centred one. 8 hexagons, 12 vertices, `chi = -4`. Fanned
  from their centres, the vertices come out degree 8 -- twelve octagons
  -- and nothing else is not six. It is written as its construction
  (`_truncated_octahedron`), not as a vertex table.
* **The super-square map is one LAYER of those octahedra**, side squares
  open as windows, top and bottom squares closed and split on a diagonal:
  `chi = -2`, census `{5: 4, 8: 8}`. The four pentagons are the poles of
  each crossing, which are genuinely convex, as in the knee X node.
* **Both have zero-length arms.** The crossings are fused octahedra, not
  nodes joined by tubes. Putting a straight square-section tube between
  two windows turns each window vertex from degree 8 into degree 7 --
  the knee route's `{5: 4, 7: 16}` comes straight back. So octagons and
  arms are exclusive on these maps, and that is topology, not tuning.
* **A coarse edge of exactly half the cell cannot be minimum-imaged.**
  The super-square's closing-square diagonals are `a/2`; the first build
  took the wrong image on some of them and returned 3-6 Å bonds while the
  census was perfect. The map is built in a 2 x 2 supercell, refined, and
  `fold_mesh` folds it back, checking every triangle appears exactly
  four times. That is why the super-square needs `m >= 2`.
* **Chiral `(m, n)` is refused**, not approximated: it needs a
  consistent cut of the triangular lattice across every coarse edge.

### The general version works on one net, and that is the honest result

`build_supernetwork(goldberg=k)` and `build_schwarzite(goldberg=k)`
remesh `k` times coarser and fill the coarse mesh with lattice patches
(`refine_remeshed`), projected back onto the field. Measured at the
presets' sizes:

| net | k = 1 | k = 2 |
|---|---|---|
| super-graphene (34/5/4) | 18.1 % non-hex, 94 % placed | **6.3 %, 100 % placed**, strained |
| super-cubic (40/5/4) | 19.8 %, clean | torn (bonds to 6.3 Å) |
| super-square (30/5/4) | 15.0 %, clean | torn |
| super-diamond (60/5/4) | 19.9 % | broken, 6.1 % |
| cube cage (24/3/2) | 20.3 % | broken |
| icosahedral cage (24/4/3) | 21.0 % | broken |

and super-graphene at `k = 3` is broken (bonds to 1.77 Å, collapsed).
A coarse mesh needs about five edges round each tube, so `k` is bounded
by the radius -- the window says so -- and even inside that bound most
nets tear. **It ships as an option with one preset (super-graphene,
k = 2), not as a default**, and the exact maps above are where the clean
walls are. Do not extrapolate the super-graphene row to the others.

### Schwarz D: Vanderbilt & Tersoff's C168 is the coarse map

`c168_atoms` rebuilds C168 (*PRL* **68**, 511, 1992) from the paper's own
Table I -- seven sites, the twelve rotations of T, inversion plus
`(a/4, a/4, a/4)` for the second joint, the four fcc translations -- and
checks it: 672 atoms in the 21.8 Å cubic cell, every atom
three-coordinate, bonds 1.388-1.454 Å. Its faces are perceived as the
shortest cycle at every angle and **checked against Euler** (each atom in
exactly three faces, sizes only 6 and 7) rather than trusted; that
shortcut is safe on C168 and is not a general ring perceiver.

Every C168 hexagon is ringed by alternating hexagons and heptagons, so it
touches exactly three heptagons, and those three are a triangle of the
coarse map: **Klein's {3,7}**, 96 degree-7 vertices and 224 triangles per
cubic cell (`diamond_heptagon_map`). C168's own dual
(`diamond_c168_map`) is GC(1,1) of it, and subdividing that gives the
class-II family without needing an oriented map.

| GC | atoms | census | bonds (Å) | verdict |
|---|---|---|---|---|
| (1,0) | 224 | `{7: 96}` | 1.401-1.442 | clean |
| (2,0) | 896 | `{6: 336, 7: 96}` | 1.402-1.454 | clean |
| (3,0) | 2016 | `{6: 896, 7: 96}` | 1.406-1.445 | clean |
| (1,1) = C168 | 672 | `{6: 224, 7: 96}` | 1.406-1.455 | clean |
| (2,2) | 2688 | `{6: 1232, 7: 96}` | 1.411-1.477 | clean |

Every row 100 % placed. The knee D at its default is 1440 atoms,
1.347-1.489 Å; the Goldberg D is tighter at every size.

### Heptanene exists on the D surface -- correcting this file

GC(1,0) of the Klein map is **all heptagons**: 56 atoms per primitive fcc
cell, 224 per cubic cell, 12.9 Å, the C56 Vanderbilt & Tersoff say they
built first and inflated into C168. Relaxed with the same force field the
heptanene section below uses: bonds **1.401-1.442 Å**, angles
116.4-122.6 deg, angle sums from 357.1 deg (0.04 sp3 character), no
contact under 2 Å, and the closest pair of atoms more than four bonds
apart at **3.92 Å** -- the walls do not touch. Density 2.07 g/cm³.

So the claim below that heptanene "is not carbon at any cell" is **wrong
as a statement about the net**. What `build_heptanene` measured is that
its own embeddings -- the Klein quartic placed barycentrically in a
*simple cubic* 3-torus, one copy per 6-14 Å cube -- do not reach carbon,
and that stands. The D-surface embedding is a different crystal: fcc,
four copies per cubic cell, positions inherited from a relaxed C168.
`build_heptanene` keeps refusing what it builds; the all-heptagon crystal
is `build_goldberg("diamond", 1, 0)`. Neither has been checked beyond the
valence force field; a DFT relaxation is what would settle it.

### Coils: the knee route already is the literature's construction

Chuang, Fan & Jin (*J. Chem. Inf. Model.* **49**, 361, 2009) classify
toroidal and helical tubes by unfolding a polyhedral unit cell onto
graphene, the non-hexagons sitting only at the polyhedron's vertices --
the same idea as this module, one dimension down. `knee.py`'s toroid and
coils are that construction for the Dunlap knee (two 5-7 pairs per knee,
everything else exact lattice) and are already clean at 100 % placement;
see the periodic coil section. What Chuang adds and this package lacks is
the **chiral** family (a tilted base vector), which needs the same
consistent lattice cut that keeps chiral GC(m, n) out of this module.

## Octahedral frames: the cages the knee route cannot build

`builders/frame.py` is the Goldberg construction on a thickened graph.
Every node is a **truncated octahedron** -- the bcc cell, six square faces
along <100> and eight hexagonal ones along <111> -- every arm a **prism**
of the face it leaves by, and every face without an arm is closed: a
hexagon fanned (a flat patch, no defect), a square split on a diagonal
(two short corners, two pentagons, on the outside). Counting triangles
round each coarse vertex then gives the census, and it is the minimum the
node allows, every defect isolated:

| node | census | same as |
|---|---|---|
| 6 <100> arms | 24 heptagons | knee P |
| 4 <100>, planar | 4 pentagons + 16 heptagons | knee X, super-square |
| 4 <111> | 12 heptagons | knee tetrahedral |
| **cube corner, 3 <100>** | **6 pentagons + 12 heptagons** | nothing -- new |
| square + hexagon arm on neighbouring faces | adds an octagon | (tesseract) |

The cube corner is the point. A convex node never balances, so the knee
route cannot build any finite cage (see the convex-hull note above); here
its three closed squares put six pentagons on the convex outside and its
twelve heptagons sit in the necks -- the right sign at every one, measured.

Measured, refined by GC(m, 0), 100 % placed in every row:

| frame | m, rows | atoms | bonds (Å) | verdict |
|---|---|---|---|---|
| cube cage | 3, 2 | 5616 | 1.343-1.517 | clean |
| cube cage | 2, 1 | 2112 | 1.334-1.510 | strained |
| adamantane cage | 2, 1 | 2400 | 1.350-1.486 | clean |
| super-cubic | 4, 2 | 1536 | 1.343-1.470 | clean |
| super-square | 3, 2 | 756 | 1.361-1.481 | clean |
| super-diamond | 3, 1 | 4320 | 1.367-1.497 | clean |
| tesseract | 3, 1 | 12960 | 1.289-1.574 | **broken** |

Two things were needed and both were measured:

* **Taubin smoothing of the refined mesh** (`goldberg.taubin_smooth`).
  The prisms' folds are 90 or 120 deg; left flat, the force field bends
  them out as bond strain.
* **A wide cell scan** (`finish_refined(cell_scan=True)`). A periodic cell
  cannot change size while relaxing, and the mean-bond rescale missed the
  best cell by **20 %** on super-cubic: scanned with full relaxations,
  0.78-0.82 of it gave 1.31-1.49 Å where the rescale gave 1.33-1.59. A
  +-10 % scan found nothing; the scan is 0.60-1.20 coarse and +-0.04 fine.
  It is most of the build time for large cells.

**The tesseract is not fixed.** Its inner corners send a square arm and a
hexagonal one out of neighbouring faces, which leaves an octagon between
them, and that node relaxes broken at every size tried. It stays in the
catalogue with no preset. The frames are bcc-lattice graphs only: an fcc
(<110>) arm needs a rhombic-dodecahedron node whose vertices are shared
by three or four faces, which would make nonagons and dodecagons -- so
super-fcc has no exact route here, and that is a consequence of twelve
tubes meeting at a point rather than a missing feature.

## Hull nodes on any graph, and the principle they exposed

`builders/hull.py` tries to lift the frame's restriction to bcc
directions: every tube an antiprism stack on a regular `k`-gon, every
node the **convex hull of its tubes' mouths** (so nothing can tear, at
any arm angles), the coarse census optimised by edge flips while the map
is small (`optimise_census`, objective curvature-aware through
`curvature_charges = 3K/pi`, which sums to `6 chi` exactly), then GC
inflation. It is **experimental and its geometry is not clean**, and the
reason is the most useful thing learned here.

The census half works. Coarse maps meet `12(V - E)` on every cage, and
the defect count is near the minimum -- cube cage `{5: 11, 7: 59}` for a
budget of -48 (the octahedral frame needs 144), tetrahedron 34, C60 cage
361 -- with 97-100 % placed. The geometry half does not: every cage
relaxes to about 1.22-1.66 Å, `broken`, and **none of these changed it**:
surface projection plus tangential smoothing (it dragged vertices to 5 Å
bonds where the hull and the implicit tubes disagree), Loop subdivision,
curvature-aware flips, antiprism instead of prism tubes, more relaxation.

**What every clean coarse map here has in common is equilateral
triangles.** The muoctahedron, C168's Klein map, the octahedral frames
and the knee nodes are all deltahedral: refined, each triangle becomes
the exact lattice, unsheared, and all the curvature sits at disclinations
and folds. A hull map's triangles run 0.30-2.65 edges long; refined, each
is an affine image of the lattice -- sheared graphene -- and no later
step removes a shear. `equilateralise` (springs to unit edges plus a
repulsion) only gets them to 0.72-1.45: with tubes five or six edges
round, the map is too small to have an equilateral realisation with the
degrees it carries.

So the rule for any future route is: **build a deltahedral carrier, then
inflate.** The obvious general one is a geodesic (icosahedral, all
near-equilateral) sphere as every node, each arm leaving through the
star of the vertex nearest its direction as a uniform antiprism. A star
around a degree-6 vertex rims with heptagons, the twelve degree-5
vertices left closed are pentagons on the convex side -- the right sign
for a cage corner -- and an icosahedral cage's arms lie exactly on its
own two-fold axes, which are geodesic vertices at frequency 2. Built: see below.

## Geodesic carriers: the deltahedral principle, applied

`builders/geodesic.py` is that carrier. Every node is a **geodesic
sphere** (icosahedron subdivided at `freq`, projected, unit mean edge);
each arm is assigned to the sphere vertex nearest its direction, the
star of that vertex is removed as the window, and the arm is a stack of
**uniform antiprism** rings -- each turned half a step so every strip
triangle is equilateral, the residual twist spread over the layers so
the last ring lands on the far window's rim. Then `equilateralise`,
GC(m, 0), Taubin smoothing and `finish_refined`. Pipeline: **deltahedral
carrier -> GC inflation -> smooth + VFF -> verify** (budget,
intrinsic placement, collapse).

**The frame is fitted, not assumed.** `fit_frame` rotates the golden
icosahedron to minimise the arms' miss (Nelder-Mead over rotation
vectors, worst miss as the score). The dodecahedral cage missed by
15.5 deg in the naive orientation and by **0.0** fitted; the
icosahedron, cube, dodecahedron and C60 cage are all exact at freq 2,
the tetrahedron and octahedron within 2.11 deg. At freq > 2 the freq-2
solution is added as a candidate, because the optimiser alone missed it.
An arm missing every vertex by more than `tolerance_deg` is refused.

Measured at m = 2, rows = 1, every row CLEAN and 100 % placed --
pentagons (the closed five-fold vertices) on the convex outside of every
node, heptagons (the window rims) in the necks, each one isolated:

| cage | freq | atoms | census | bonds (Å) |
|---|---|---|---|---|
| icosahedron | 2 | 5280 | `{5: 84, 7: 180, 8: 60}` | 1.376-1.463 |
| cube | 2 | 3136 | `{5: 48, 7: 96}` | 1.391-1.444 |
| dodecahedron | 2 | 7840 | `{5: 120, 7: 240}` | 1.394-1.442 |
| C60 of tubes | 2 | 23520 | `{5: 360, 7: 720}` | 1.391-1.465 |
| tetrahedron | 4 | 5120 | `{5: 42, 7: 66}` | 1.371-1.494 |
| octahedron | 4 | 7680 | `{5: 60, 7: 132}` | 1.361-1.498 |
| icosahedron | 4 | 16800 | `{5: 144, 7: 360}` | 1.409-1.436 |

Against the hull route's 1.22-1.66 Å *broken* on the same cages, with
the same census machinery: the only difference is that the carrier's
triangles are equilateral. **That is the principle confirmed**, and it
is what makes finite cages -- which the knee route proves impossible --
come out crystalline-walled rather than amorphous.

* **Tetrahedron and octahedron need freq 4.** Their arms are 60-90 deg
  apart, so at freq 2 neighbouring windows share rim vertices and those
  become octagons and nonagons (strained). `GEODESIC_CAGES` stores each
  cage's own freq and tolerance; the CLI and window read it.
* The icosahedron at freq 2 keeps 60 octagons for the same reason
  (its 30 two-fold windows touch); freq 4 removes them.
* `rows` is antiprism rows on the shortest arm; the scale is
  `(2 * node_radius + rows * sqrt(3)/2) / shortest edge`.

### Periodic nets: the same carrier in a cell

A periodic `SuperGraph` is built in its own cell (`_graph_frame`: zero
edge on an axis that does not repeat), the map's `_Builder` wraps into
that box, and `finish_refined(cell_scan=True)` scans the cell as for the
frames. Windows are keyed by **(edge, end)**, not by node: a one-node net
joins the node to its own image, so the node alone does not name the
window -- keyed by node, super-cubic's +x and -x windows overwrote each
other.

The arm directions fit golden-icosahedron spheres exactly: <100> arms are
two-fold axes (vertices at freq 2), <111> arms three-fold ones (face
centres, vertices at freq 3), and the honeycomb's 120 deg arms fit freq 2
as well. Measured at m = 2, rows = 1, every row CLEAN and 100 % placed:

| net | freq | atoms | census | bonds (Å) | time |
|---|---|---|---|---|---|
| super-cubic | 2 | 464 | `{6: 204, 7: 24}` | **1.401-1.442** | 16 s |
| super-square | 2 | 416 | `{5: 4, 6: 186, 7: 16}` | 1.381-1.459 | 17 s |
| super-graphene | 2 | 1568 | `{5: 24, 6: 708, 7: 48}` | 1.393-1.446 | 200 s |
| super-diamond | 3 | 5760 | `{5: 96, 6: 2576, 7: 192}` | 1.341-1.511 | 18 min |
| **super-fcc** | 4 | 5120 | `{5: 24, 6: 2232, 7: 264}` | 1.340-1.505 | 13 min |

* **super-cubic is the tightest P-type cell in the package**: the knee
  P is 1.417-1.540 (strained) and the frame 1.343-1.470, with the same
  `{7: 24}`.
* **super-fcc had no exact route at all** -- the frames section above
  explains why a rhombic-dodecahedron node cannot work. A frequency-4
  geodesic node takes twelve <110> arms within 2.1 deg, and the cell is
  clean. That paragraph's "consequence of twelve tubes meeting at a
  point" was about the octahedral node, not about the net.
* **Pentagons on a periodic net are not wrong here.** A geodesic node is
  convex between its windows, so super-graphene and super-diamond carry
  pentagons the knee route does not -- and every one is in positive
  curvature (100 % placed). The knee super-graphene (`{7: 24}`,
  1.408-1.436) is still the better sheet; the geodesic one is a
  different, fatter-noded structure.
* The cell scan dominates the large cells' time.
* **The two sheets were first measured through a bug**, and the numbers
  above are after it. `hull.equilateralise` clipped coordinates to the
  box on *every* axis before its neighbour search, and on an axis that
  does not repeat (box 0) that sends every coordinate to 0: the tree
  saw the sheet's thickness as nothing and paired vertices across it,
  and the "repulsion" -- with a negative gap -- pulled them together.
  The sheets survived it (1.367-1.473 and 1.342-1.506 before); the
  supertube, centred on its axis so that x and y both collapsed, did not
  (coarse edges 0.04-18 from a 0.65-1.90 input). Only the live axes are
  clipped now, the repulsion acts only inside its range, and
  `test_hull.py` pins the supertube's equalised edges to 0.8-1.25.

### Per-node frames, and windows only on six-fold vertices

A supertube (rolled super-graphene) or the projected tesseract has no
single frame: the best global fits miss by 7-16 deg. `fit_node_frames`
fits each node's sphere to its own three to five arms, and the antiprism
stacking already joins any two rims by an index shift and a residual
twist, so differently turned spheres need nothing else. The builder
uses one shared frame when it is within `tolerance_deg` and falls back
to per-node frames otherwise (`info["node_frames"]`).

**Windows open only on degree-6 sphere vertices** (`_six_fold`). The
first per-node fit put one end of a supertube arm on a five-fold vertex
-- a pentagonal mouth -- and the other on a six-fold one, and the arm
could not be stacked. The twelve five-fold vertices stay closed; they
are the node's pentagons.

| cage | freq | m, rows | atoms | bonds (Å) | verdict |
|---|---|---|---|---|---|
| cuboctahedron | 4 | 1, 1 | 3840 | 1.355-1.483, later 1.325-1.523 | clean, then strained |
| cuboctahedron | 4 | 1, 2 | 4128 | 1.345-1.504 | clean |
| hypercube, per node | 4 | 1, 1 | 7136 | 1.320-1.532 | strained, 98 % |
| hypercube, per node | **6** | 1, 1 | 14256 | 1.324-1.501 | strained, **100 %** |
| **hypercube, per node** | **6** | **1, 2** | 14784 | **1.403-1.439** | **clean, 100 %** |
| truncated cube | 4 | 1, 1 | 7680 | 1.299-1.546 | broken |
| truncated cube | 4 | 1, 2 | 8112 | 1.340-1.520 | strained |
| truncated cube | 4 | 2, 1 | 30720 | 1.354-1.493 | clean |

The hypercube fits **exactly** at frequency 6, node by node, and its
census `{5: 192, 7: 384}` meets `12(V - E) = -192` with every defect
placed; the frame route's tesseract was broken at every size. With
**two antiprism rows** it is clean, at 1.403-1.439 Å -- tighter than any
cage in the table above. One row leaves the two kinds of node too close
for the necks to relax; the census does not change, only the room.
The truncated cube says the same thing differently: broken at one row,
strained at two, clean at m = 2 -- the room comes from the tube or from
the arm, and either works.

**The supertube was broken by the equalise bug above**, not by its
frames, its rows or its seam: at one row and at two it relaxed to
1.00-2.22 and 0.75-2.65 Å, uniformly along the axis, and the coarse map
going *into* `equilateralise` was sound. Rebuilt after the fix, same
parameters (per-node freq 4, m = 1, one row): 10240 atoms,
`{5: 320, 7: 512}`, 1.309-1.537 Å, strained, 99.5 % placed -- from
broken to strained on that one change. **At two rows it is clean**:
10816 atoms, 1.358-1.483 Å, 100 % placed, 45 min. All seven rolls in
`CAGES` are in `GEODESIC_CAGES`; measured at the same parameters
(frequency 4, m = 1, two rows):

| supertube | atoms | census | bonds (Å) | verdict |
|---|---|---|---|---|
| (4,4) | 10816 | -- | 1.358-1.483 | clean |
| (5,5) | 13520 | `{5: 480, 7: 720}` | 1.381-1.448 | clean |
| (6,6) | 16224 | `{5: 456, 7: 744}` | 1.346-1.505 | clean |
| (6,0) | 16224 | `{5: 508, 7: 796}` | 1.316-1.504 | strained |
| (6,0), **frequency 6** | 35424 | `{5: 560, 7: 848}` | 1.345-1.478 | **clean**, 99.9 % |
| (8,8) | 21632 | `{5: 616, 7: 1000}` | 1.296-1.527 | broken |
| (8,8), **frequency 6** | 47232 | `{5: 736, 7: 1120}` | 1.345-1.487 | **clean**, 100 % |
| (8,0) | 21632 | `{5: 756, 7: 1140}` | 1.334-1.503 | clean |
| (10,0) | 27040 | `{5: 848, 7: 1328}` | 1.321-1.514 | strained |
| (10,0), three rows | 28480 | `{5: 848, 7: 1328}` | 1.320-1.525 | strained |

Neither chirality decides it: (5,5), (6,6) and (8,0) are clean at
frequency 4, (6,0) and (10,0) strained and (8,8) broken. The (6,0) and
the (8,8) are clean at frequency 6 (radius-1 windows, two rows), at more than twice
the atoms: the rule of the Archimedean audit again, a finer sphere for
arms that miss. A per-node frame fits each node to its own
arms within the 4.5 deg tolerance, and how far the arms miss differs
roll by roll. A third row does not help the (10,0): its arms miss, and a
longer oblique arm only shears more. Frequency 6 with radius-2 windows is refused on both
(an arm joins windows of different sizes).

### The Archimedean audit: room, not frames

Every cage in `supernetwork.CAGES` the carrier had not yet measured, at
m = 1 or 2 and one or two antiprism rows:

| cage | freq | m, rows | atoms | bonds (Å) | verdict |
|---|---|---|---|---|---|
| icosidodecahedron | 2 | 2, 1 | 12480 | 1.393-1.449 | clean |
| truncated tetrahedron | 4 | 1, 1 | 3840 | 1.307-1.526 | strained |
| truncated tetrahedron | 4 | 1, 2 | 4056 | 1.336-1.500 | clean |
| truncated octahedron | 4 | 1, 1 | 7680 | 1.319-1.522 | strained |
| truncated octahedron | 4 | 1, 2 | 8112 | 1.346-1.512 | clean |

| rhombicuboctahedron | 4 | 1, 2 | 8256 | 1.323-1.512 | strained (3 deg miss) |
| rhombicuboctahedron | 4 | 1, 3 | 8832 | 1.296-1.526 | **broken** |
| rhombicuboctahedron, per node | **6** | 1, 2 | 17856 | 1.338-1.492 | clean |

All 100 % placed where clean. The pattern of the hypercube and the
truncated cube holds: **a cage that is strained at one row is clean at
two** -- the census is fixed by the graph, and only the room between
nodes changes. **The exception is a miss.** The rhombicuboctahedron's
arms miss their frequency-4 windows by about 3 deg, and a third row made
it broken rather than clean: an arm that leaves its window obliquely
shears more the longer it is. The cure is a finer sphere (1.9 deg at
frequency 6), not a longer arm.

### New periodic nets, and zeolite-templated carbon

`GEODESIC_NETS` holds nets only this carrier has been measured on, kept
out of `SUPERLATTICES` because that catalogue feeds the meshed and knee
modes too: super-srs and its mirror (the gyroid's Laves graph, chiral,
nodes in four orientations, fitted node by node), super-bcc (eight
<111> arms, three-fold axes, freq 3), super-kagome (four in-plane arms
at 60/120 deg, freq 2) and super-lonsdaleite (hexagonal diamond, nodes
in two orientations, fitted node by node at freq 3). Measured at m = 2,
one row, CLEAN and 100 % placed:

| net | freq | atoms | census | bonds (Å) | time |
|---|---|---|---|---|---|
| super-bcc | 3 | 1440 | `{5: 24, 6: 588, 7: 96}` | 1.341-1.496 | 75 s |
| super-kagome | 2 | 2496 | `{5: 36, 7: 84, 8: 12}` | 1.388-1.450 | 7 min |
| super-srs | 2 | 3136 | `{5: 48, 6: 1416, 7: 96}` | 1.367-1.475 | 12 min |
| super-lonsdaleite | 3 | 5760 | `{5: 96, 6: 2576, 7: 192}` | 1.340-1.499 | 23 min |

* **Orthohexagonal cells**: `_fold_orthohexagonal` puts a hexagonal net
  in its `(a, a*sqrt3, c)` cell, because `minimum_image` takes only an
  orthorhombic box. Its basis is written with `a2` at **60 deg** to `a1`;
  the wurtzite fractions first typed were the 120 deg convention's, and
  lonsdaleite came out one-coordinate (the two "special points" are
  `(1/3, 1/3)` and `(2/3, 2/3)` at 60 deg).
* **Per-node fits need starts.** At 4 random starts lonsdaleite's
  tetrahedral nodes missed by 7.2 deg; at 16 they fit to 0.0. The
  default is 16. Changing it moved a one-row cuboctahedron from clean
  to strained: **a one-row cage sits on the edge and the frame decides
  it**, so the presets use two rows wherever one was marginal.

### The catalogue image

`examples/ex12_catalogue.py` builds every clean entry (cached as JSON,
so a second run only renders) and draws them with each ring filled by
its size -- pentagons red, heptagons blue, octagons green -- because
where the non-hexagons sit is the result, and an atoms-and-bonds view
hides it. Periodic cells are tiled (2x2x2, or 3x3 for a sheet, each ring
moved whole so its centre lies in the cell): one cell drawn alone looks
torn at its faces. Labels are in Spanish. `--quick` keeps the entries
that build in minutes.

**A zeolite-templated carbon lines the template's pores, so its net is
the pore net, not the framework**: FAU's supercages sit on a diamond
net (so does its framework, which makes the wall between them D-type),
LTA's alpha cages on a primitive cubic one (P-type), EMT -- FAU's
hexagonal polytype -- on lonsdaleite, SOD's beta cages on bcc. Measured
cubic cells of the clean Goldberg families:

| map | GC | atoms | a (Å) | density (g/cm³) |
|---|---|---|---|---|
| D | (1,1) = C168 | 672 | 21.80 | 1.29 |
| **D** | **(2,0)** | **896** | **24.74** | 1.18 |
| D | (2,1) | 1568 | 32.84 | 0.88 |
| P | (1,1) | 144 | 12.72 | 1.40 |
| P | (2,0) | 192 | 14.35 | 1.30 |
| P | (2,2) | 576 | 24.75 | 0.76 |

**D GC(2,0) relaxes to 24.74 Å, which is FAU's cubic cell** (NaY,
about 24.7 Å) -- a clean, 96-heptagon D wall commensurate with the
template. **It is not a ZTC model, and the carbon count says why**: the
cubic cell holds eight supercages, so 896 atoms is 112 C per supercage,
against the 63-72 reported for real FAU-templated carbons (Frontiers in
Materials 6, 104, 2019). And the field's own review of the atomistic
ZTC models (Taylor, Garman & Stadie) argues that synthesised ZTCs are
not schwarzites. No D member matches both: C168 is 84 per supercage at
21.8 Å, GC(1,0) 28 at 12.9 Å. The preset is called a
*FAU-commensurate D wall*, not a replica. LTA's 12.3 Å pseudo-cell is 3 %
under P GC(1,1). The geodesic lonsdaleite has EMT's axial ratio (66.8
over 40.9 against 28.4 over 17.4) but at m = 2 is 2.35 times its cell,
and m = 1 would still be about 18 % large: no EMT match yet. These are valence-force-field cells; whether a real
ZTC's wall sits on this surface, at this density, is for comparison with
the published ZTC models and DFT, not something this table settles.

### Compact nodes: the window grows with the sphere

The catalogue image made the cost visible: every node a ball. Measured,
the closed sphere between windows is **94 %** of a tetrahedral cage's
coarse triangles (frequency 4), 87 % of super-diamond's (3), 63 % of the
cube cage's, 45 % of the icosahedral cage's. The reason is that a window
was always the star of one vertex -- a mouth one edge across -- while the
frequency the arm angles need sets the sphere's radius (3.35 edges at
frequency 4). The tube stays thin and the node swells around it.

`window_radius` (`_patch`) takes every triangle within `r` rings of the
window vertex: a hexagonal mouth of `6r` rim vertices. The tube is
`6 * r * m` rings round, so for one tube diameter a larger `r` means a
smaller `m`, and the node becomes mostly mouth. The census is unchanged
-- no octagons appear at the hexagon's corners, measured -- and the
budget is met exactly. Windows of radius > 1 whose rims would touch are
refused; radius-1 rims may share a vertex (the frequency-2 icosahedron
does, cleanly).

At the same 12-ring tube (radius 2 at m = 1 against radius 1 at m = 2),
every row CLEAN:

| structure | freq | before | compact | bonds (Å) | placed |
|---|---|---|---|---|---|
| super-cubic | 4 | 464 | **320** | 1.405-1.443 | 100 % |
| super-square | 4 | 416 | **320** | 1.381-1.454 | 100 % |
| super-graphene | 4 | 1568 | **1280** | 1.394-1.447 | 100 % |
| super-srs | 4 | 3136 | **2560** | 1.366-1.473 | 100 % |
| cube cage | 4 | 3136 | **2560** | 1.391-1.443 | 100 % |
| dodecahedral cage | 4 | 7840 | **6400** | 1.393-1.442 | 100 % |
| tetrahedral cage | 6 | 5120 | **2874** | 1.376-1.480 | 99 % |
| octahedral cage | 6 | 7680 | **4308** | 1.352-1.490 | 98 % |
| super-diamond | 3 | 5760 | **1824** | 1.336-1.503 | **79 %** |

The diamond is the warning: at a third of the atoms the node has so
little sphere left that part of it saddles, and a fifth of its pentagons
sit in negative curvature. It is clean and it is not the right D -- the
knee D (1440 atoms, heptagons only) and the Goldberg D (896) are both
smaller and better placed. The icosahedral cage gains nothing (its best
compact option, frequency 6, is larger than frequency 2 at m = 2). The
presets carry a "compact" entry for every row but the diamond.

**Same tube, fewer atoms, is the comparison that matters.** At fixed `r`
the construction is self-similar -- doubling the frequency and the
window together changes nothing -- so the saving comes only from the
window taking a larger share of the sphere, which the arm angles bound:
the tetrahedron needs frequency 6 before radius-2 windows stop touching.

### The node can only be as small as its narrowest arm pair allows

The super-hypercube showed the problem at its worst: frequency 6 was
needed for its arms to land on window vertices, so its node was a
sphere 5.2 tube radii across. Two things fix it, and a third bounds it.

* **The largest window.** `largest_window` (and `window_radius=0`)
  finds the biggest radius whose windows do not touch. With a hexagonal
  window of `r` rings on a sphere of about `0.86 f` edges, the node is
  roughly `0.86 f / r` tube radii: the hypercube at frequency 6 goes
  from 5.2 (r = 1) to **2.6** (r = 2), clean, 16704 atoms,
  1.377-1.456 Å.
* **Bent arms.** `_stack_arm` runs each arm along a cubic Hermite curve
  that leaves each mouth along its window's own normal, rings on a
  rotation-minimising frame. A straight pair reproduces the old stack
  to 1e-6 in unrelaxed strain; a window that misses its arm by a few
  degrees no longer forces a finer sphere, the tube bends instead of
  shearing out of its mouth, and the map still meets `12(V - E)`
  (a test pins the hypercube at frequency 4, 7 deg off).
* **The bound.** A window's angular radius cannot exceed half the
  smallest angle between two arms of its node, so the node is at least
  about `1 / sin(theta_min / 2)` tube radii. The hypercube's projection
  has 54.7 deg between a radial arm and a cube edge: floor **2.2**, so
  2.6 is near it. Re-embedding the graph does not help -- it is C4 x C4,
  every pair of arms at a vertex lies in a square, and the torus
  placement measured worse (60 deg at best, edges 1:3). Only graphs whose
  arms are widely spread can have a node the size of the tube.

The most compact clean build of each, at m = 1 (largest window, every
row 100 % placed unless stated):

| structure | freq, r | node / tube | atoms | bonds (Å) |
|---|---|---|---|---|
| super-graphene | 6, 5 | **1.03** | 3048 | 1.363-1.481 |
| super-srs | 4, 3 | **1.15** | 2848 | 1.370-1.466 |
| super-cubic | 6, 4 | 1.29 | 684 | 1.394-1.437 |
| super-square | 6, 4 | 1.29 | 696 | 1.380-1.457 |
| cube cage | 6, 4 | 1.29 | 5616 | 1.390-1.457 |
| dodecahedral cage | 6, 4 | 1.29 | 14040 | 1.400-1.465 |
| tetrahedral cage | 8, 3 | 2.29 | 4902 | 1.374-1.464 (99 %) |
| octahedral cage | 8, 3 | 2.29 | 7244 | 1.354-1.478 (99 %) |
| hypercube | 6, 2 | 2.58 | 16704 | 1.377-1.456 |

Refused or worse at their most compact, and left as they were: the
tetrahedron and octahedron at frequency 2 (broken: their windows share
rims), the truncated cube at 3 (broken), super-fcc at 2 (broken),
super-lonsdaleite at (3, 2) (strained, 78 %) and super-diamond at (3, 2)
(clean, 79 %). The icosahedral cages, C60, the icosidodecahedron and the
Wythoff cages are already at 1.72, frequency 2 with r = 1, which is as
compact as their arms allow.

### Chiral Goldberg-Coxeter: the consistent cut

`goldberg.chiral_subdivide` builds GC(m, n) for any pair on any closed
oriented triangulation, so the refusal this file used to describe is
gone. Each coarse triangle sits in the Eisenstein lattice with corners
`0, w, w*omega` (`w = m + n*omega`); each lattice triangle goes to the
face its centroid lies in; a lattice point that falls outside its face
is carried into the neighbour by the rotation that unfolds the neighbour
onto this face -- a sixth root of unity about the shared edge, so
lattice points map to lattice points and both faces name every shared
vertex the same way. A centroid exactly on a coarse edge (at (1, 1) it
happens) goes to the face that runs the edge from its lower vertex
index. The triangle count is checked against `T * (m^2 + mn + n^2)`.

The cut needs a **consistently oriented** map, and C168's Klein map is
not one -- the class-II note above said it was built "without needing an
oriented map", and that is exactly why the first chiral D refused.
`orient_mesh` fixes the winding breadth-first and raises on a
non-orientable surface; the chiral cut calls it when the directed edges
do not pair.

| structure | GC | atoms | census | bonds (Å) |
|---|---|---|---|---|
| C140 (icosahedron) | (2,1) and (1,2) | 140 | `{5: 12, 6: 60}` | 1.419-1.422 |
| Schwarz P | (2,1) | 336 | `{6: 152, 8: 12}` | 1.413-1.437 |
| Schwarz P | (3,1) | 624 | `{6: 296, 8: 12}` | **1.417-1.432** |
| Schwarz D | (2,1) and (1,2) | 1568 | `{6: 672, 7: 96}` | 1.407-1.468 |
| Schwarz D | (3,1) | 2912 | `{6: 1344, 7: 96}` | 1.408-1.466 |
| super-square | (3,1) | 676 | `{5: 4, 6: 324, 8: 8}` | 1.404-1.443 |
| cube cage, geodesic | (2,1) | 5488 | `{5: 48, 7: 96}` | 1.402-1.454 |
| icosahedral cage, geodesic | (2,1) | 9240 | `{5: 84, 7: 180, 8: 60}` | 1.377-1.478 |

Every row CLEAN and 100 % placed; the census is the achiral one, as it
must be.

**Doubly chiral srs, and what the method cannot yet say.** srs is a
chiral net and GC(m, n) makes the wall chiral, so srs (2,1) and srs
(1,2) are diastereomers. Both are clean, 5488 atoms: (2,1) relaxes to
1.356-1.481 Å in a 77.26 Å cell, (1,2) to 1.373-1.475 in 78.09. The
control is the enantiomer -- the mirror net at (1,2) is the exact mirror
of srs (2,1) and must reproduce it -- and it **does not**: 80.57 Å,
1.359-1.481. The builder is not mirror-equivariant (the frame fit's
random starts and the arms' fixed reference axis are not mirrored with
the net), so two builds of mirror-image structures land 4 % apart.
The diastereomers' 1.1 % is inside that, and **no energetic difference
between them can be claimed** from these builds. Settling it needs the
enantiomer built by reflecting atoms rather than rebuilding, and an
energy from something better than a fixed-cell VFF scan whose own step
is 1 %.

**Measured: the diastereomers are indistinguishable at this level.**
Three frame seeds each, strain per atom (`strain_energy`):

| seed | srs (2,1) | srs (1,2) | difference |
|---|---|---|---|
| 0 | 0.03704 | 0.03614 | +2.5 % |
| 1 | 0.04132 | 0.04123 | +0.2 % |
| 2 | 0.04125 | 0.04133 | -0.2 % |

The seed moves the strain by 14 % (0.036-0.041) and the cell by 5 %;
at a fixed seed the two diastereomers agree to 0.2 % and the sign of the
difference changes. So **wall handedness against net handedness costs
nothing a valence force field can see**, and the 2.5 % of seed 0 was a
better frame, not a better diastereomer. If the two differ, the
difference is electronic -- the pi system's response to a helical wall
on a helical net -- and that is a question for tbkit (through an
extxyz exchange, never an import), not for this relaxer.

**Exact enantiomers, and the carrier's own handedness.**
`geodesic.enantiomer` reflects a finished structure through `x = L/2`:
every length and angle identical, `gc_indices` and the srs label swapped,
and a pseudoscalar (each atom's bond-vector triple product) flips sign --
a test pins all three. `strain_energy` is the relaxer's bond and angle
terms per atom, mirror-invariant by construction, for comparing builds.

What made the rebuilt mirror differ was traced to the **window rims**.
At frequency 2 a window around a two-fold vertex is a hexagon with only
two-fold symmetry. With one shared frame every rim has the same
orientation and arms join like rims: the Wythoff cube's GC(2,1) and
GC(1,2) then agree to **1e-15**, the truncated icosahedron's to 4e-9.
Carry the frames by the cube's group instead and a [4,3] mirror -- not a
symmetry of the icosahedral sphere -- turns a rim 90 deg about its arm;
the arm then needs a +-1.5-step twist, picks one, and the carrier itself
becomes chiral (1.9 % apart). Per-node frames have the same exposure,
which is why srs, whose nodes need them, could not compare its
diastereomers. `_choose_frames` therefore prefers a shared frame, then
group frames (exact under [5,3], whose every element is a symmetry of the
sphere; 7e-5 on the truncated cuboctahedron), then per-node fits, and
records which in `info["node_frames"]`. An even row count was tried as a
fix for the twist and changed nothing on the cube, so it is not there.

### Wythoff's kaleidoscope

`builders/wythoff.py` generates every Platonic and Archimedean cage (the
snubs aside) from three mirrors at `pi/p`, `pi/2`, `pi/q` and a ring
pattern: the generator lies on each unringed mirror and equidistant from
the ringed ones, its orbit is the vertex set, the shortest pairs are the
edges. Sixteen solids, every vertex, edge and degree count checked
against the tables in `test_wythoff.py`, none of them written down in the
module. `wythoff_frames` fits the node frame once at the generator and
carries it by the group element that carries the vertex, which fits
every [5,3] solid **exactly** at frequency 2 -- including the three no
other catalogue here had: the truncated dodecahedron, the
rhombicosidodecahedron and the truncated icosidodecahedron. Built, all
CLEAN and 100 % placed:

| cage | freq | m, rows | frames | atoms | bonds (Å) |
|---|---|---|---|---|---|
| truncated icosidodecahedron | 2 | 1, 1 | shared | 11760 | 1.361-1.450 |
| truncated cuboctahedron | 4 | 1, 2 | group | 16224 | 1.375-1.457 |
| rhombicosidodecahedron | 2 | 2, 1 | shared | 24960 | 1.387-1.458 |
| truncated dodecahedron | 2 | 2, 1 | shared | 23520 | 1.393-1.451 |

**Shared frames open windows on any sphere vertex.** A shared frame is
centrosymmetric, so the two ends of an arm sit on antipodes of one
degree; only differently oriented spheres need the six-fold rule. The
rule had been applied to the shared case too, and it lost super-fcc (its
<110> arms fit mixed sites to 2.1 deg at frequency 4, six-fold ones to
4.7). But dropping it everywhere was worse: mixed pentagonal and
hexagonal mouths left the cuboctahedron and the truncated tetrahedron
strained and the truncated octahedron broken, where six-fold windows
had them clean. So `_choose_frames` prefers six-fold windows (shared,
then group, then per node) and opens any vertex only as a last resort,
which only super-fcc needs. The tetrahedron and octahedron now take the
six-fold shared frame, `{5: 40, 7: 64}` and `{5: 48, 7: 120}`, both
clean -- slightly different from the mixed-window rows in the first
table, which were built before the rule existed. GC(3,1) Schwarz P is the tightest P cell in the package. `n` is
in the Goldberg and geodesic frames of the window and on both
subcommands; `(m, n)` and `(n, m)` are mirror images.

### Periodic superstructures: the Euclidean kaleidoscope

Which superstructures can be made periodic has an exact answer. A cage
tiles space only if its point group is a crystallographic one, so **the
icosahedral cages cannot** -- no lattice has a five-fold axis -- while
the cubic family can: the uniform honeycombs of `[4,3,4]` are built from
exactly those cages, and each is one periodic superstructure.
`wythoff.euclidean_wythoff` takes the same three-mirror construction to
the affine groups `[4,3,4]`, `[4,4]` and `[6,3]` (a mirror is a plane
`n.x + d = 0`, the generator solves `n_i.x + d_i = rings_i * t`), grows
the orbit **unfolded** in a window of five cells and cuts it to one.
Folding as it grows is wrong for `[6,3]`: the orthohexagonal cell is not
closed under its 60 deg rotations, and the orbit came out half-filled.
The 15 distinct patterns' vertex counts and coordinations are pinned in
`test_wythoff.py`; `WYTHOFF_NETS` names the twelve no other catalogue
holds (cubic, square, honeycomb and kagome are super-cubic,
super-square, super-graphene and super-kagome already).

| net | made of | freq, r, rows | atoms | bonds (Å) | verdict |
|---|---|---|---|---|---|
| triangular 3^6 | -- | 2, 1, 1 | 232 | 1.367-1.439 | clean |
| 4.8.8 | -- | 4, 2, 1 | 1272 | 1.363-1.476 | clean |
| 3.12.12 | -- | (being re-measured) | | | see below |
| 4.6.12 | -- | 4, 1, 2 | 8112 | 1.359-1.482 | clean (broken at 1 row) |
| truncated cubic | truncated cubes + octahedra | 4, 1, 2 | 2100 | 1.341-1.502 | clean (strained at 1 row) |
| rectified cubic | octahedra + cuboctahedra | 6, 2, 1 | 2148 | 1.347-1.491 | clean, 99 % (broken at f2) |
| **bitruncated cubic (SOD)** | truncated octahedra | 4, 2, 1 | 3792 | 1.353-1.493 | clean (broken at f2) |
| cantitruncated cubic | truncated cuboctahedra, ... | 4, 2, 1 | 7680 | 1.377-1.464 | clean, 96 % |
| omnitruncated cubic | truncated cuboctahedra + octagonal prisms | 4, 2, 1 | 15360 | 1.356-1.488 | clean, 95 % |
| cantellated cubic | rhombicuboctahedra, cuboctahedra, cubes | **6, 2, 1** | 8640 | 1.373-1.460 | clean (broken at f4; 4.5 deg tolerance, the arms bend) |
| runcitruncated cubic | truncated cubes, rhombicuboctahedra, ... | **6, 2, 1** | 17280 | 1.361-1.481 | clean, 99.9 % (strained at f4) |
| 3.4.6.4 | -- | **6, 2, 1** | 8616 | 1.377-1.465 | clean (broken at f4) |
All at m = 1 and the largest window that fits, so each node is as small
as its arms allow. The truncated-octahedron cage tiles space face to face
-- its periodic version is the **sodalite net**, and it is clean.
The rule of the Archimedean audit holds again: strained at one row,
clean at two; and where the arms miss the frequency-4 windows (the
nets that hold rhombicuboctahedra or 3.4.6.4 vertices) more rows do not
help, a finer sphere does: at frequency 6 the 3.4.6.4 sheet and the
runcitruncated honeycomb are clean. The cantellated honeycomb still
misses its frequency-6 windows by 3.2 deg, and builds clean when the
tolerance is 4.5 deg: the arms bend into their windows (`_stack_arm`)
instead of shearing out of them. Every uniform honeycomb of `[4,3,4]`
and every Wythoff tiling of `[4,4]` and `[6,3]` is now clean. The tetrahedral cages' honeycombs (the
quarter-cubic and alternated ones) need the `[3^[4]]` group and are not
built.

**Correcting this table: the first "3.12.12" was a honeycomb.** It was
built from rings (0, 1, 1) of `[6,3]`, which put the generator on the
mirror through the six-fold point -- the truncated triangular tiling,
which is the hexagonal tiling again, at a third of the edge. It has 12
vertices of coordination 3 per cell, exactly as 3.12.12 does, so the
count test passed; the arms were at 120, 120, 120 deg instead of 60, 150,
150. The 4272-atom "clean" row was therefore a super-graphene. 3.12.12 is
rings (1, 1, 0), and `test_wythoff.py` now pins every named tiling's
vertex angles, not only its counts.


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

### The same node makes the junction, and there the census is exact

`builders/knee.py` builds a junction the way it builds the schwarzite
cell: arms trimmed by **dominance** -- a vertex keeps the arm whose axis
it lies furthest along, and the surface separating two arms is then the
bisector plane of their axes, which is a plane for **any** pair and not
only for perpendicular ones. So nothing new is needed off the cube axes,
and `node_mesh` takes any directions at all.

**The budget is fixed before anything is meshed.** A node of `c` arms is
a sphere with `c` holes, so `chi = 2 - c` and `sum(6-n) = 6(2-c)`. A
junction saddles everywhere, so it can carry **no pentagon**, and the
whole budget is paid in heptagons:

| kind | arms | chi | census | bonds (Å) | placement |
|---|---|---|---|---|---|
| `y` | 3 | -1 | `{6: N, 7: 6}` | 1.410-1.431 | **100%** |
| `tetrahedral` | 4 | -2 | `{6: N, 7: 12}` | 1.415-1.428 | **100%** |

Measured at every circumference from 10 to 20 and every arm length
tried, exactly. Six is the number the published Y junctions carry. The
implicit route at a comparable size returns **50 pentagons and 38
heptagons** -- sound, and an amorphous wall. This is the same table row
the "mesh route versus crystalline route" section has for the toroid and
the coil, and here the crystalline route costs nothing: the Y's bond
spread is **0.021 Å against the meshed Y junction's 0.117**, so for once
it buys a clean census *and* better bonds.

The placement column is `analyse/curvature.disclination_check`, the
intrinsic measure, and it is not a restatement of the census: the surface
is fitted over each ring and the sign of `K` read off. Every heptagon is
in negative curvature, mean sign exactly **-1.00**. Only the periodic
coil reaches that elsewhere in this file.

#### Correcting this file: the planar X, and what `chi = -4` really was

An earlier version of this section said a planar X "closes a tunnel
through the middle" and left it out of `JUNCTION_AXES`. **That diagnosis
was wrong**, and the right one makes the X buildable.

A node is a union of trimmed cylinders, and the dominance trim covers
only the directions its arms span. **Coplanar arms leave the two poles
uncovered**, and the hole there has one edge per arm. The Y had them too
-- three coplanar arms leave a *triangle* at each pole, which
`fill_triangular_holes` was already closing, which is why nobody noticed.
Four coplanar arms leave a **square**, measured at every circumference
and arm length, and nothing was closing those. So the node was a sphere
with **six** holes -- four mouths and two open poles -- and `chi = -4` is
exactly `2 - 6`. There was no tunnel.

`hole_size` is the rule (`len(axes)` when the axes are rank-deficient, 3
otherwise) and `fill_node_holes` closes anything up to that size. With
the poles shut the X comes out at `chi = -2 = 2 - 4` and:

| kind | census | bonds (Å) | placement |
|---|---|---|---|
| `x` (4 arms, 90 deg, planar) | `{5: 4, 6: N, 7: 16}` | 1.395-1.464 | **100%** |

**Its pentagons are correct, which no other node here has.** The poles of
a four-way planar crossing are a pillow above and below the crossing
point and are *genuinely positively curved*, while the four crotches
saddle -- so four pentagons belong at the poles and sixteen heptagons in
the crotches. That is not a story told after the fact: the intrinsic
`disclination_check` puts **all twenty** on the correct side, with the
pentagons' mean sign of `K` at exactly **+1.00** and the heptagons' at
**-1.00**, and a test pins both.

So the census gate is `{5, 6, 7}` summing to the budget, not `{6, 7}`.
`sum(6-n)` alone cannot tell a correct pentagon from a wrong one -- the
gyroid's k=18 cell pays its exact -48 with sixteen pentagons that are
wrong -- so the gate is on the **ring set plus the budget**, and
`info["pentagons"]` reports the count whichever case it is.

#### Every kind carries its own shape, and one default was a bug

`DEFAULT_JUNCTION_SHAPE`, `DEFAULT_SCHWARZITE_SHAPE` and
`DEFAULT_SUPERNET_SHAPE` are per kind. That is not tidiness: the junction
builder shipped with a single default of `(14, 9)` for all kinds, which
builds a Y and **refuses a tetrahedral node outright** -- it closes at
k = 10 and 18 and not at 14. A kind whose own default cannot be built is
a kind nobody can pick from a menu, and that is what the window offered.

| kind | default | census |
|---|---|---|
| `y` | (14, 9) | `{6: N, 7: 6}` |
| `tetrahedral` | (18, 9) | `{6: N, 7: 12}` |
| `x` | (20, 9) | `{5: 4, 6: N, 7: 16}` |

#### super-square closes too, and it is a sheet of those crossings

With the poles shut, `super-square` -- one node of four coplanar arms per
cell -- comes out at `sum(6-n) = -12 = 12(1 - 2)`, `chi = -2`, with no
broken edge at every shape tried. The shipped `(12, 5)`: **180 atoms, a
14.6 Å cell, `{5: 4, 6: 68, 7: 16}`, bonds 1.374-1.505 Å, CLEAN, every
one of the twenty disclinations on its own side.** Most circumferences
pay the poles with octagons instead of heptagon pairs; only k = 12 and 20
give the five-six-seven census.

#### The 4-8 dipole, and why climb is the only move that removes it

The welded seams each come out carrying a neutral **4-8 pair sitting side
by side**. It is neutral -- `(6-4) + (6-8) = 0` -- so it costs the budget
nothing and **no census check sees it**: `sum(6-n)` reads -6 with the
dipoles there and -6 with them gone.

**No edge flip can remove it, and that is arithmetic rather than a search
failure.** A flip drops its two endpoints a degree and raises its two
opposites, so taking the 8 down takes a neighbouring hexagon down with it
and bringing the 4 up brings another hexagon up: `sum|deg - 6|` is 4
before and 4 after, whatever the flip lands on. Measured, greedy descent
over every flip finds not one improving move, and a plateau anneal across
four seeds stays put. This is the **same wall the Dunlap toroid met** --
gliding a dislocation preserves it, and separating or annihilating one
needs **climb**, which changes the vertex count and so cannot be a flip.

`collapse_degree_four` is that climb: delete the degree-4 vertex and fill
the quadrilateral its link leaves with two triangles. One vertex, four
edges and four faces go and two faces come back, so `chi` is untouched,
and all four neighbours drop a degree. **The diagonal decides which two
get it back, and choosing it is the whole move.** The link comes out in
the cyclic order `6, 7, 6, 8`; the diagonal joining the two hexagons
returns their degree and leaves the 8 at 7 and the 4 gone:

| diagonal | result at k=14 |
|---|---|
| the two hexagons | `{4: 5, 6: 235, 7: 6, 8: 5}` -- the dipole gone, nothing else touched |
| the 7 and the 8 | `{4: 5, 5: 2, 6: 232, 7: 6, 8: 6}` -- two pentagons invented |

Both are tried, a diagonal that is already an edge is skipped, and the
move is taken only when `sum|deg - 6|` over the **interior** strictly
falls. It is a **no-op** on a mesh that is already clean, which the
Schwarz P cell pins.

#### Three things measured and refuted on the way

Worth keeping, because each looked like the answer:

* **"It is a phase."** Rotating every arm's ring stack by half a step
  does change the census -- and for the worse: `{4: 6, 6: N, 8: 6, 9: 2}`,
  which trades the six heptagons for two **nonagons** and keeps the
  dipoles. No phase in a scan over four values and six circumferences
  reaches `{6: N, 7: 6}`.
* **"It is chirality."** `_ring_stack` climbs helically, so a neighbouring
  arm is a rotation of this one rather than its mirror -- and with three
  pairwise-adjacent arms there is no 2-colouring to fix that. It reads as
  a clean explanation and it is wrong: the census is **byte-identical
  across all eight per-arm mirror combinations**. The mirror changes the
  geometry and not the topology here.
* **"Anneal it out."** The flip annealer *worsens* it, from
  `sum|deg-6| = 24` to 30-37, and invents twenty pentagons where there
  were none, because its objective counts rim vertices. Turning a pass on
  is not the same as it helping.

**The fill is what puts the budget on the surface, not a tidying step.**
Where three arms meet, the trim leaves a *triangular hole* rather than
the shared degree-3 vertex the cube node leaves. Closing it adds no edge
-- all three already exist -- so no vertex changes degree and only `F`
rises. But the six vertices that stop being boundary come out at
**degree 7**: they *are* the junction's heptagons. Before the fill the
interior carries none at all.

### Every triply periodic surface is its net thickened, so one routine does all

The D surface is the diamond lattice thickened into a wall, so its piece
is the **four-arm node the junction already builds**. `diamond_cell_mesh`
puts one on each of the eight sites of the conventional cubic diamond
cell -- the A sublattice looking out along +(1,1,1) and its family, the B
sublattice along the negatives -- and glues each mouth to its neighbour's.

**Nothing about the census is fitted.** Gluing two boundary circles adds
nothing to `chi`, a circle having `chi = 0`, so a cell of `n` nodes of
`c` arms has `chi = n(2-c)`: here `8 * (2-4) = -16`, genus 9, and
`sum(6-n) = 6*chi = -96`. The cell edge is not fitted either -- two
mouths meeting make the diamond bond, so `2*reach = a*sqrt(3)/4` and
`a = 8*reach/sqrt(3)`.

Measured at the shipped shape (`circumference=18, arm_rows=5`): **1440
atoms in a 30.0 Å cell, `{6: 608, 7: 96}`, `sum(6-n) = -96`, bonds
1.347-1.489 Å, angles 109.1-125.0, zero contacts, CLEAN**, and every
heptagon in negative curvature. Ninety-six heptagons and nothing else --
no pentagon, and no square or octagon either.

**This is the conventional cell, not the primitive one.** The
rhombohedral primitive cell holds two nodes and is genus 3, which is what
D216 is. It is not orthorhombic, and `minimum_image` takes an
orthorhombic box only, so a bond across its seam would read as a
cell-length stretch. The cubic cell is four primitive cells of the same
surface and is the one that can be measured correctly.

**The circumference decides it, and the node says so first.** Scanned
over k = 8 to 24 at three arm lengths:

| k | verdict | why |
|---|---|---|
| 8, 14, 16, 20, 22 | refused | the node's seams do not pair up |
| 10 | builds | **broken** at 5 rows, strained at 7 and 9 |
| 12 | refused | `{6: N, 9: 32}` -- nonagons, which is the tetrahedral node's own census at that size |
| **18** | **clean at 5, 7 and 9 rows** | 0.0176-0.0199 Å bond spread |
| 24 | refused | `{5: 96, ...}` -- pentagons, on a surface that has no positive curvature |

So the default is `(18, 5)`, the smallest clean one, and
`DEFAULT_SCHWARZITE_SHAPE` is per kind rather than shared: what suits a P
cell (20, 9) does not suit a D one, because eight nodes in one cube leave
far less room between them than a single node does. A shape argument left
out takes the kind's own default; the refusal still names the pairs that
close.

**`build_knee_schwarzite` takes a `kind`, it is not a second builder.**
`SCHWARZITE_CELLS` holds `(arms, nodes, genus)` per kind and
`schwarzite_budget` derives `sum(6-n)` from those three alone, so adding
a surface is a table entry plus its net, not a copy of the finishing, the
rescale and the census gate.

#### Only the sites are written down

`SCHWARZITE_NETS` holds eight fractional coordinates per net and nothing
else. `net_geometry` derives each node's **arm directions from its
nearest neighbours** under the minimum image, and the bond as a fraction
of the cell edge from the same distances, so a net cannot carry an axis
table that disagrees with its own sites. The derivation is checked rather
than trusted: every site of a net must come out with the same
coordination, which is what catches a mistyped site.

The srs sites are written as their construction -- Wyckoff 8a of
I4(1)32 plus the body centring -- rather than as eight transcribed
triples, for the same reason.

#### The gyroid's node is the junction's Y, turned

srs (the Laves graph, (10,3)-a) is 3-coordinate, and **three unit vectors
with pairwise 120 deg angles sum to zero and are coplanar, with no choice
about it.** So a gyroid node *is* the planar Y `JUNCTION_AXES` already
holds, and the four distinct plane normals are the four <111>
directions. The junction and the gyroid are one object at two scales, and
a test asserts the coplanarity by the triple product rather than citing
it.

Eight of them per conventional cubic cell gives `chi = 8(2-3) = -8`,
genus 5, budget **-48**. Measured at the shipped shape
(`circumference=20, arm_rows=5`): **1744 atoms in a 39.0 Å cell,
`{6: 816, 7: 48}`, bonds 1.395-1.508 Å, angles 114.7-125.3, zero
contacts, CLEAN**, every heptagon in negative curvature, sp3 0.0%.

**The gyroid wants a fatter tube than P or D, and that is its node rather
than a preference.** Its arms leave at 120 deg against the diamond node's
109.47, so the saddle between them is tighter. Scanned:

| k | tube radius (Å) | verdict |
|---|---|---|
| 10, 12, 14, 22, 24 | -- | refused, the node's seams do not pair |
| 8 | 3.05 | **broken** -- 1.324-1.722 Å |
| 16 | 5.40 | **broken** -- 1.383-1.553 Å |
| 18 | -- | refused: `{5: 16, 6: N, 7: 32, 8: 16}` |
| **20** | **6.37** | **clean** -- 1.395-1.508 Å |

The k=18 row is the one worth keeping: it closes at **exactly the right
budget**, `sum(6-n) = -48`, and pays part of it with sixteen pentagons
against sixteen octagons. A minimal surface saddles everywhere, so a
pentagon on one is never right -- and `sum(6-n)` cannot see the
difference. The gate is on the **census**, not on the budget, and this is
the case that shows why it has to be.

#### Lonsdaleite and bcc: two more cells, and a node that depends on its frame

`SCHWARZITE_CELLS` also holds `lonsdaleite` (hexagonal diamond, the D
surface's hexagonal polytype) and `bcc` (the I-WP surface's labyrinth),
both measured at their default shape, every row 100 % placed and no
contact under 2 Å:

| cell | shape (k, rows) | atoms | census | bonds (Å) | sphere route |
|---|---|---|---|---|---|
| diamond | 18, 5 | 1440 | `{6: 608, 7: 96}` | 1.353-1.451 | super-diamond 5760 |
| **lonsdaleite** | 18, 5 | 1440 | `{6: 608, 7: 96}` | **1.376-1.431** | super-lonsdaleite 5760 |
| bcc | 36, 13 | 3624 | `{5: 48, 6: 1632, 7: 120}` | 1.370-1.515 | super-bcc **1440** |

**`node_mesh` depends on the node's orientation.** Each arm's rings are
laid out against the global z axis, so one node in two orientations is two
meshes, and whether the seams pair up depends on which. Lonsdaleite's
nodes are the diamond node in two mirror-related orientations, and meshed
where they stand no circumference from 8 to 24 closed (`seam 0-2 has 7
vertices on one side and 8 on the other`). Built as the diamond node and
turned rigidly onto their arms (`NET_NODE_FRAME`), circumference 18
closes at every arm length with exactly D's census. Diamond and the gyroid
are left meshed in place, because they already close and turning them
would change measured structures. A test pins the in-place failure, so
the frame cannot be dropped as redundant.

**bcc is the exception to "no pentagons on a minimal-surface cell".**
Eight `<111>` arms leave a convex pillow at each of the node's six `<100>`
poles, as the planar X junction does at its two. Scanned from 8 to 48
rows round, the only 5-6-7 census is at k = 36 (13 rows), and all 48
pentagons sit at the poles (mean sign of K **+1.00**) and all 120 heptagons
in the necks (**-1.00**). `SCHWARZITE_PENTAGONS` allows them for that cell
only, and every knee schwarzite now records its placement and raises if
its pentagons are not in positive curvature. Every other circumference
closes at the right budget, -72, with squares and octagons. **It is
sphere-free and not smaller.** Eight arms 70.5 deg apart need a fat tube,
and the geodesic super-bcc is 1440 atoms on a 12-ring one.

Also scanned and not shipped: the sodalite net (bitruncated cubic) closes
at k = 20, 7 rows with `{5: 48, 7: 192}` (about 5500 atoms against the
geodesic 3792); kagome, the triangular tiling and the rectified cubic
honeycomb never come out 5-6-7 from k = 8 to 30. The supertubes cannot use
this route at all: on a rolled sheet the three arms of a node do not sum
to zero (0.20-0.45) and the struts differ in length, and a knee node needs
both.

### The superstructure is the same law on a graph, and it agrees with the meshed route

`build_knee_supernetwork` hangs the **node** on every vertex of a net
instead of hanging a tube on every edge, so the rings are placed by the
construction rather than chosen by a remesher. `super-graphene` is a
honeycomb of tubes -- which is to say a **periodic sheet of the Y
junctions** the same module builds.

**The budget is one law reached from two directions, and they agree to
the integer.** `supernetwork.py` derives `12*(V - E)` from
`chi = 2*(V - E)`, one handle per independent cycle of the graph.
`knee.node_budget` derives `6*(2 - c)` from "a node of `c` arms is a
sphere with `c` holes". Summed over a graph the second gives
`sum_v 6(2 - deg v) = 12V - 6*2E = 12(V - E)` -- the first. Checked
against every entry of that catalogue (super-square -12, super-graphene
-24, super-cubic -24, super-diamond -96, super-fcc -240, icosahedral
cage -216), and a test pins it.

Measured on `super-graphene`: `{6: N, 7: 24}` and `sum(6-n) = -24` at
`chi = -4`, at **every** circumference from 8 to 28 and every arm length
tried, with no pentagon and no boundary edge. The geometry is not
shape-independent and picks the default: the bond spread runs 0.0543 Å at
k=18 (broken) down to **0.0036 at k=22** and back to 0.0210 at k=28, so
the shipped `(22, 5)` is a minimum rather than an edge. At it: 920 atoms,
bonds **1.408-1.436 Å**, angles 117.5-121.6, zero contacts, every
heptagon in negative curvature. That is the tightest geometry anything in
this package produces after C60.

#### A node comes out pure exactly when its arms balance

This is the rule the whole route turns on, and it was found by measuring
the cases that failed:

| node | arms sum to | census |
|---|---|---|
| Y, 3 arms at 120 deg | **0** | `{6: N, 7: 6}` |
| tetrahedral, 4 at 109.47 | **0** | `{6: N, 7: 12}` |
| octahedral, 6 arms (Schwarz P) | **0** | `{6: N, 7: 24}` |
| cube vertex, 3 perpendicular | (1,1,1) | never pure -- squares, pentagons or a nonagon at every shape |
| tetrahedron vertex, 3 at 60 deg | 2.449 | does not close: the dominance trim eats the arm |

An unbalanced node still pays its budget **exactly** -- the cube vertex
reads `sum(6-n) = -6` at every shape that closes -- it just pays it in a
different coin. So `sum(6-n)` cannot tell the two apart, and
`net_geometry` checks the balance itself and refuses rather than
returning a sheet with squares in it.

#### So there is no finite knee superstructure, at all

A convex polyhedron's vertex lies on its hull, so every edge at it points
into the supporting half-space and their sum has a strictly positive
component along the inward normal: it **cannot** balance. And every
finite graph has a vertex on its convex hull. So no finite cage of knee
nodes exists at any size, for any polyhedron -- a fact about geometry,
not a limitation here, and a test pins both halves (no Platonic cage has
a balanced vertex; every periodic super-net is balanced to machine zero).

Measured, the cages behave exactly as that predicts. The cube closes at
the right budget (`sum(6-n) = -48`, `chi = -8`) at twenty-nine shapes and
at **none** of them with a pure census; the tetrahedron, octahedron,
icosahedron and dodecahedron do not close at all. `supernetwork.py`'s
meshed cages -- the icosahedral cage, superfullerene-C60 -- remain the
route for a finite superstructure, and they are amorphous-walled by
construction. That is not a gap to be closed.

#### Images are enumerated, not minimum-imaged

`net_geometry` derives each node's arms from its neighbours, and doing
that under the **minimum image is wrong on a honeycomb**: two sites there
are neighbours through more than one image -- a Y node's three partners
are two images of the same site -- and the minimum image keeps only the
nearest. Measured, that returned a two-armed node and read a net that
balances perfectly as one that does not. The offsets are enumerated over
the live axes instead. (This is the same fact `analyse/rings.py` refuses
a too-small cell over, met from the other side.)

### The periodic coil, and the sign that hid it

**Correcting this file.** An earlier version of this section said the
periodic coil "does not close", with the holonomy measured and about 8000
radii swept and none working. The holonomy part is right and the
conclusion was wrong: the sweep was measuring **a sign error**.

At the wrap knee the *incoming* arm is the last one, whose boundary
already sits one period along the axis. The seam code added the period to
that side instead of subtracting it, pushing it to two periods. Every
other knee has an offset of zero, so nothing else could show the fault,
and the symptom -- "the two boundaries did not land on each other" -- is
exactly what a real holonomy mismatch looks like. With the sign flipped
the coil welds, and `knee_path_mesh`'s `period` does what it was written
to do: the wrap is a genuine mitre knee rather than two rims welded
afterwards.

**The census is predictable before anything is built.** A periodic cell
of a coil is a **torus** -- the tube closes on itself through the
boundary -- so `chi = 0`, `sum(6-n) = 0`, and with only 5s, 6s and 7s
available that forces equal numbers. The law fixes them: two pairs at
each knee, so a turn of `s` sides carries `2s` of each. At the shipped
shape (6 sides, k=10, pitch 15 Å, R=14): **672 atoms, `{5: 12, 6: 312,
7: 12}`, bonds 1.376-1.495 Å, angles 107.0-123.5, zero contacts, CLEAN,
`D/d = 3.52`** -- inside the 3.5-3.9 band the single-wall coil papers
report -- and **100% of the disclinations on the side they belong on**,
every pentagon outside and every heptagon inside, which is the one
structural claim those papers make that can be checked without running
anything.

#### Most radii do not close, and closing is not obeying the law

Two separate facts, and the first is the one the old note got half right.

The mitre reflections carry a frame once round the turn and return it
rotated -- compose them and the map **fixes the first arm**, so it is a
rotation about it by an angle the structure did not ask for. That angle
is continuous in the radius and the pitch (`-23.7` to `-124.2` degrees
over R = 8-20 Å and pitch 8-16 Å) and goes to zero as the pitch does,
which is the flat ring, which is the toroid. The wrap pairs up only where
it is close to a whole lattice step. Measured at 6 sides, k=10, pitch 15,
over R = 12 to 17 in 0.05 Å steps: **19 of 101 radii close**, in two
windows, 13.90-14.35 and 15.05-15.45 Å. Windows, not points -- but not a
free parameter either.

And inside a window the pair count still moves: 13.90-14.20 gives the
law's twelve, 14.25-14.35 gives **thirteen**. Thirteen is a sound torus,
`sum(6-n)` is 0 either way, and **no census check sees the difference** --
which is precisely why the builder warns rather than passing it in
silence, and why `clean_periodic_coils` filters on the law while the
builder only requires the budget.

A radius that fails the weld and a radius that welds into the wrong
census give **different refusals**, and a test pins both: R = 12.5 never
pairs, R = 13.0 welds and comes out `{5: 16, ..., 7: 12, 8: 2}`.

**The finite coil is untouched.** The offset only applies when a period
is given, so `build_knee_coil` cannot have moved, and a test asserts it
still has its two rims.

### One script builds every exact structure and checks it

`examples/verify_exact_structures.py` is the thing to run when the
question is "does any of this actually hold". It builds each structure
and prints the measured census beside **the budget its skeleton fixed
before anything was meshed** -- so the "law" column is a comparison, not
a restatement.

```
structure                 atoms  census              sum(6-n)  law  verdict  placed
toroid, 6 knees            1032  5:12 6:492 7:12           +0   ok  clean      100%
coil, finite                406  5:14 6:167 7:14           +0   --  clean      100%
coil, periodic cell         672  5:12 6:312 7:12           +0   ok  clean      100%
junction Y                  536  6:240 7:6                 -6   ok  clean      100%
junction X (planar)         780  5:4 6:328 7:16           -12   ok  clean      100%
junction, diamond node      756  6:328 7:12               -12   ok  clean      100%
sheet, super-square         180  5:4 6:68 7:16            -12   ok  clean      100%
sheet, super-graphene       920  6:432 7:24               -24   ok  clean      100%
Schwarz P / D / gyroid           (pass --all: -24, -96, -48)
```

Every row `clean` and every row at 100% placement, which is the column
worth reading twice: it is measured by **fitting the surface over each
ring** and taking the sign of `K` there, not read off the ring size, and
the flat-haeckelite control in `analyse/curvature.py` is what makes that
number mean anything.

`tests/test_verify_example.py` runs the script, because a silent breakage
there would report "All laws met" about a table it never built.

### A saddle has no pentagons, so its doping had no site

`DOPANT_SITES` offered `"pentagon"` and nothing else in the ring-selected
family. That is the right site on a fullerene or a capped tube -- their
chemistry *is* at the pentagons -- and it is a site that **does not
exist** on anything the knee route builds past the toroid: a schwarzite,
a junction or a knee supernetwork is hexagons plus heptagons, and the
reactivity is on the heptagons.

`dopants/rings.py` always took the ring size; only the policy layer in
`jobs.py` hardcoded five. `DOPANT_RING_SIZES` now maps `"pentagon"`,
`"heptagon"` and `"octagon"` onto 5, 7 and 8, and the window and the
command line read `DOPANT_SITES` so both pick them up.

**Which ring size is the interesting one follows the sign of the
curvature**, which is the same rule `analyse/curvature.py` states: a
pentagon is a +60 deg disclination and a heptagon a -60 deg one, so a
positively curved shell does its chemistry on pentagons and a saddle on
heptagons. Measured on a Y junction at 20% heptagon N: every nitrogen
lands on a heptagon, and the count is 20% of the **heptagon sites**, not
of the 272 atoms.

Asking for pentagons on a saddle raises with what rings the structure
*does* have, which is the useful refusal: `"This structure has no
5-membered rings to dope. It has 6-rings: 272 sites, 7-rings: ..."`.

The other two chemistry axes were already fine and are now pinned:
codoping places both species on every knee structure, and grafting adds
atoms to all of them (hydroxyl 2 per site, carboxyl 4, achieved coverage
within 0.001 of the request).

### The five Platonic cages, and why their walls are amorphous

`platonic_cage` puts a meshed tube on every edge of any of the five
solids; `scale` is the **strut length**, as for every cage, because that
is what decides whether a tube survives between two vertices at all.

| solid | V | E | degree | budget | genus | measured |
|---|---|---|---|---|---|---|
| tetrahedron | 4 | 6 | 3 | -24 | 3 | 812 atoms, met exactly, 0 contacts |
| cube | 8 | 12 | 3 | -48 | 5 | 1748 atoms, met exactly, 0 contacts |
| octahedron | 6 | 12 | 4 | -72 | 7 | 1510 atoms, met exactly, 0 contacts |
| dodecahedron | 20 | 30 | 3 | -120 | 11 | 4424 atoms, met exactly, 0 contacts |
| icosahedron | 12 | 30 | 5 | -216 | 19 | (already shipped) |

All at strut 24 Å, tube 3 Å, blend 2 Å, and all in under a minute.

**Their walls are amorphous and that is not a choice.** A cage of exact
knee nodes cannot exist, for the reason stated above: a node's census
comes out clean only when its arms sum to zero, and a convex
polyhedron's vertex lies on its own hull. A test asserts it directly on
all five -- the worst vertex imbalance is 1.73 and the best still over
0.5, against machine zero for every periodic net.

### The window exposes the shape, or the presets would be the only sizes

The knee modes passed nothing but the kind, so the circumference and the
arm length were unreachable from the window -- the presets would have
been the *only* structures those modes could make. They are spinboxes in
the `Knee node` frame now, and picking a kind writes **that kind's own**
default into them (they do not share one; see above) and leaves them
editable.

That is the honest shape of the claim: the **census** is exact over a
whole family of sizes -- a Y junction is `{6: N, 7: 6}` at every
circumference from 10 to 20 and every arm length tried -- while the
**geometry** is not, and that is what the default picks. A test builds a
Y at four circumferences away from its default and asserts the census
each time.

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

### A fixed grid over a growing cell -- and the fix that was worse

super-diamond is sound at scale 50 and collapses at 60, on nothing but
the voxel: 0.69 Å against 0.83. At resolution 100 the same cell, voxel
0.60, comes back sound at 331.8. That much is real.

**The fix inferred from it was not.** Tying the voxel to the tube radius
on EVERY build (`VOXEL_PER_TUBE = 0.12`) generalised one measurement to
ten structures, and the cages paid for it:

==================  =========  ======  =========  ==========
preset              box (Å)    wanted  got        cost
==================  =========  ======  =========  ==========
super-hypercube     77.2       184     160 (cap)  **11x**
supertube-(6,6)     110.2      307     160 (cap)  **11x**
superfullerene-C60  98.9       275     160 (cap)  **11x**
super-icosahedron   70.8       148     148        **8.7x**
==================  =========  ======  =========  ==========

Eleven times the cost **and clipped**, so they did not even get what the
rule demanded -- and they came back collapsed, which is what the user
saw. The error: the voxel was tied to `max(box)`, and **a cage's box is
mostly the vacuum around it**. A 3 Å tube in a 98.9 Å box was being
asked for a 0.36 Å voxel because of empty space.

Reverted. `grid_resolution` is a fixed floor again and the finer grid
moved into the rescue, where only a wall that actually collapsed pays
for it. The three cages returned to their audit numbers exactly --
331.6, 331.4, 330.7 -- at 88, 35 and 434 s.

**One measurement is a measurement, not a rule.** The general form is
the thing to distrust.

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

## The builder now refuses to hand back an impossible wall

A collapsed wall is not a poor structure, it is an **impossible** one --
an angle sum under 328.4 deg is past tetrahedral, which no carbon
reaches. `rescue_collapsed_wall` rebuilds it once with the wall held on
its own surface, the same pattern this module already uses for a weld
that fails: vary one thing and re-measure. Only a wall that actually
collapsed pays, because `wall_anchor` widens bonds and would be a loss
on a wall that was already sound.

**One anchor strength does not fit all**, and assuming it did left the
job half done. super-cubic clears at 1.0. The superfullerene gets
**worse** at 1.0 (327.6 -> 325.8) and needs 2.0 to clear, so a single
attempt returned it collapsed and looked like the fix simply not
working. `RESCUE_ANCHORS = (1.0, 2.0, 4.0)` escalates and keeps whichever
rebuild has the highest minimum angle sum.

**And "not collapsed" is not a stopping condition.** Stopping the moment
`collapsed_wall` passes left the superfullerene at **328.4 deg, the
threshold itself to one decimal** -- a wall that would read collapsed
again on the next change to anything. `RESCUE_MARGIN = 1.5` makes it
keep going, and one more step reached 330.8.

===================  =========  ==========  ============
structure            before     after       anchor used
===================  =========  ==========  ============
super-cubic          327.8      **330.6**   1.0
superfullerene-C60   327.6      **330.8**   4.0
super-graphene       338.6      untouched   --
===================  =========  ==========  ============

The cost is real and bounded by **time, not atom count**, which was
tried first and is the wrong measure: a super-fcc cell is 3082 atoms and
takes 1440 s while a superfullerene is 7534 atoms and takes 79. An atom
limit that spares the fcc from an hour and a half of rebuilds also
refuses the superfullerene, which could be rescued three times over
inside four minutes. `RESCUE_TIME_BUDGET` counts seconds instead, and
the ladder stops when they run out.

The ladder carries a finer grid as well as anchor strengths, because
super-diamond's collapse is a resolution problem no anchor fixes.

## Both switches are in the window

`anneal_sweeps` already had its slider; `wall_anchor` now has a
checkbox beside the placement one, and the hint says what each buys and
costs, because **they pull in opposite directions**: annealing places
the rings better and leaves the wall rougher, and the anchor is what
pays that back. Both reach all seven meshed modes, and
`test_place_curvature_wiring.py` asserts they reach the *same* seven --
a mode offering one and silently dropping the other is a wiring slip,
which is exactly how `coil (periodic, DFT)` was missed the first time.

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

(**Correction:** on the D surface it does embed -- see "Heptanene exists
on the D surface" above. What follows is about this builder's simple
cubic embedding only.)

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

## Eleven cages, and the three that were written and taken out again

`platonic_cage` gave the five Platonic solids and `archimedean_cage`
gives six more. **What they add is not a new node**: the tetrahedron,
the cube and the dodecahedron all give the SAME three-arm node, so what
changes across a catalogue of cages is the arrangement and the density --
how many nodes sit on one shell and how many arms meet at each. That is
the axis that decides how much of a cage's wall is curved, and curvature
is what chemisorption follows (`analyse/curvature.py`,
`analyse/hybridisation.py`). Nothing here computes an adsorption energy;
what it does is build the cage and measure its pyramidalisation.

Audited as **graphs** before any was meshed -- vertex and edge counts,
degree, the first-to-second neighbour shell gap that
`edges_from_positions` needs (tightest 1.41 against a tolerance of
1.10), and the budget `12(V-E)`:

| solid | V | E | degree | budget | gap |
|---|---|---|---|---|---|
| truncated tetrahedron | 12 | 18 | 3 | -72 | 1.73 |
| cuboctahedron | 12 | 24 | 4 | -144 | 1.41 |
| truncated cube | 24 | 36 | 3 | -144 | 1.85 |
| truncated octahedron | 24 | 36 | 3 | -144 | 1.41 |
| rhombicuboctahedron | 24 | 48 | 4 | -288 | 1.41 |
| icosidodecahedron | 30 | 60 | 4 | -360 | 1.62 |

And built, at strut 30 / tube 4.0 / blend 3.0. **All six meet their
budget EXACTLY with zero close contacts**, bonds inside 1.31-1.55 Å,
verdict `strained` -- which is the honest reading of a meshed wall, a
regular heptagon's interior angle being 128.6° before any strain:

| cage | atoms | bonds (Å) | time |
|---|---|---|---|
| truncated tetrahedron | 4344 | 1.326-1.535 | 37 s |
| cuboctahedron | 5250 | 1.338-1.523 | 57 s |
| truncated cube | 8684 | 1.328-1.548 | 162 s |
| truncated octahedron | 8940 | 1.332-1.526 | 113 s |
| rhombicuboctahedron | 10608 | 1.332-1.530 | 101 s |
| icosidodecahedron | 13108 | 1.335-1.540 | 165 s |

- **The strut is 30 and not the Platonic presets' 24, and that was
  measured.** At 24/3.0/2.0 the cuboctahedron comes back **broken** --
  1.311-1.589 Å, census `{5:117, 6:1181, 7:214, 8:19, 9:3}` -- while
  meeting its budget exactly and reporting no contact. Four arms at a
  vertex **on a sphere** is a tighter node than four on a cube's
  diagonal, and it is the cage with the worst vertex imbalance here, 2.00
  against the truncated octahedron's 1.00.
- **Their walls are amorphous and that is not a choice**, for the reason
  the Platonic cages' are: a cage of exact knee nodes cannot exist,
  because a node's census comes out clean only when its arms sum to zero
  and a convex polyhedron's vertex lies on its own hull. Measured, the
  imbalance runs 1.00 to 2.00 against **machine zero** for every periodic
  net, and a test asserts it on all six.
- **The truncated tetrahedron is built by truncation**, not from
  "permutations of (0, 1, 3) with an even number of minus signs". That is
  the usual statement and it is ambiguous -- the sign of a zero is not a
  sign -- and the ambiguous reading gives **24 vertices of degree 2**
  instead of 12 of degree 3. The vertex count is what caught it.
- **Two fullerene entries were written and taken out.**
  `superfullerene-C20` would have been one cage under two names: a C20
  cage is twelve pentagons and no hexagon, so its skeleton has exactly
  the dodecahedron's 20 vertices and 30 edges at the same budget.
  `superfullerene-C240` is four times the C60's wall, which is already
  7534 atoms. The duplicate test is on the **adjacency spectrum** and
  not on (V, E, degree), which was the first version and fails on two
  cages that are genuinely different -- the truncated cube and the
  truncated octahedron are both 24 vertices, 36 edges and 3-regular, and
  what separates them is their faces.

### `LARGE_AREA` and `PRESET_MAX_AREA` answer different questions

The static preset guard was pinned to `LARGE_AREA` when nothing on the
shelf exceeded it, and four of these cages do -- while building in 101 to
165 s against the superfullerene's 79 s at 24_090 Å². `LARGE_AREA` still
asks "is this one of the big ones", and the hint should keep answering
yes for them; what a preset guard protects against is a shelf entry out
of proportion with the rest. So they are two constants now, and the
limit **cannot** be a time: `build_cost_note` already records the three
attempts that failed, and the ranking is not monotone in area -- a cage
closes on itself while a periodic cell has its faces welded to their
opposite numbers, so super-fcc is 12_796 Å² in 413 s while the
superfullerene is 24_090 in 79.

## A supertube is armchair or zigzag, and the narrow one is the bad one

`(n,n)` is an **armchair** roll of super-graphene and `(n,0)` a
**zigzag** one -- the same distinction a real nanotube's indices make,
one level of scale up, because the super-graphene net *is* a honeycomb.
The catalogue held two armchairs and no zigzag at all.

- **They are two structures, not two names for one.** The (6,6) and the
  (6,0) share 48 vertices and 72 struts and have axial periods of 69.3
  and 120.0 Å at a 20 Å strut. Built: `supertube-(6,0)` gives 5916
  atoms, budget -288 met exactly, bonds 1.311-1.549 Å, zero contacts,
  strained, 215 s -- beside the (6,6)'s 5936 atoms.
- **The struts come out short by a law, not by an error.** Rolling
  replaces each arc by its chord, so a strut is **never longer** than
  asked and is shorter by about `18.3/n²` per cent for an armchair roll
  and `30/n²` for a zigzag one, the circumference in units of the strut
  differing between the two by √3. Measured at a 20 Å strut: 2.02 % at
  (3,3), 1.15 % at (4,4), 0.50 % at (6,6), 0.30 % at (8,8); 0.85 % at
  (6,0) and 0.20 % at (12,0). The test asserts the law rather than a flat
  tolerance -- a flat 2 % was the first version and the (3,3) failed it
  by two hundredths of a point, which is the law and not a defect.
- **So the NARROW roll is the distorted one**, which is the opposite of
  the usual worry about size. `(3,3)` is out of the catalogue and it was
  measured rather than guessed at: it builds -- 2940 atoms, budget -144
  met exactly, no close contact -- and comes back **broken**, bonds
  1.292-1.570 Å, at the same 14/3/2 shape where the (6,0) is merely
  strained.
- **A chiral roll is out for the other reason: cost, not geometry.**
  (6,3) needs 168 vertices over two periods in a 317 Å cell and (9,3) 312
  in 433 Å, against 48 for the (6,6), which is already minutes.
  `supertube_graph` takes any indices for whoever wants one.
- Of the seven registered, the (4,4) and (6,6) were shipped and measured
  before, the (6,0) is measured above, and the (5,5), (8,8), (8,0) and
  (10,0) are **bracketed** by those rather than each built -- the chord
  law saying the wider ones are the safer ones. The docstring says which
  are which.

### The sheet has no armchair/zigzag choice, and the supertube is why

Asked for super-graphene "in armchair instead of zigzag", the answer is
that **a 2D periodic net has no such choice**: the honeycomb's
rectangular cell is `a` by `a√3`, so rotating it 90° gives `a√3` by `a`
-- the same cell with its axes exchanged, and the same net. Adding a
second catalogue entry for it would be a false distinction of exactly
the kind the C20 duplicate was.

Where the distinction is real is anywhere the net has an **edge or a
roll**: that is the supertube's `(n,n)` against `(n,0)`, which is now in
the catalogue both ways. A finite **super-ribbon** -- a strip of
super-graphene with open armchair or zigzag edges -- would be the other
real case and is **not** built: it is a new builder, not an orientation
flag, and half of one would be worse than none.

### Knees on a superstructure already exist, and are a separate mode

`build_knee_supernetwork` hangs the exact **knee node** on every vertex
of a net instead of hanging a meshed tube on every edge, so the rings are
placed by the construction rather than chosen by a remesher. It is a
separate entry point from `build_supernetwork`, with its own GUI modes
and its own shape defaults, and neither route touches the other. It
covers the two nets whose cells have an aperiodic axis -- `super-graphene`
(`{6: N, 7: 24}`, bonds 1.408-1.436 Å, the tightest geometry in this
package after C60) and `super-square` (`{5: 4, 6: 68, 7: 16}`) -- and it
is the meshed route, not this one, that reaches the cages, because no
finite cage of knee nodes exists at all.

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

## A net with no preset can only be built with someone else's numbers

The `Net` combobox writes `sn_graph` and nothing else -- `_update_sn_hint`
is its only listener -- so picking a net there leaves the cell length,
tube radius and blend at whatever the **last preset** put in them.

`super-fcc`, `super-cubic` and `super-square` had no preset, so the
combobox was the only way to reach them. Choosing `super-fcc` after
"Super-diamond (tubes at 109.47°)" therefore built it at `scale=60,
tube_radius=5, blend=4`, and a user reported it still going after ninety
minutes. Nothing was wrong with it: 42 Å struts, 24 Å of free tube, every
geometry check passed and the hint read perfectly healthy. It was simply
an enormous thing to ask for, and nothing said so. Its own preset --
`scale=40, tube_radius=3, blend=2` -- builds **sound at 334.0°** in 413 s,
the best minimum angle sum of any 3D periodic net in the catalogue.

`tests/test_gui_static.py` now fails if any entry of `SUPERLATTICES` has
no preset of its own, if a preset would be refused by the builder's own
strut check, or if one asks for a build of tens of minutes.

## The rescue must never re-arm the rescue

`rescue_collapsed_wall` is armed by `wall_anchor <= 0`, and one of its
own rungs rebuilds on a finer grid with **`wall_anchor=0`** -- exactly
that condition. Nothing but the private `_rescue` flag stops that rung
from arming a second rescue, which arms a third.

Traced on superfullerene-C60 it nested five deep, multiplying
`grid_resolution` by `RESCUE_GRID` at every level:

```
72 -> 100 -> 141 -> 197 -> 274
```

55x the voxels of the grid asked for, one rebuild at that size taking
2005 s, and four rungs at each level. `RESCUE_TIME_BUDGET` does not
catch it: each level measures `seconds_spent` inside its own fresh
build, so every level starts its budget again at zero.

So `again()` passes `_rescue=False` and the rescue block reads
`if wall_anchor <= 0.0 and _rescue:`. `tests/test_wall_rescue.py` pins
all three parts -- the flag in the rebuild, the flag in the guard, and
an exercised stub asserting the nesting depth stays at 1.

## Size can be stated before a build; time cannot

`SuperGraph.wall_area` is `2*pi*r` times the total strut length, known
before anything is meshed, and the supernetwork hint shows it. It is a
**size**. Three attempts to turn it into a predicted build time all
failed, and each failure is pinned by a test in `test_build_cost.py` so
it is not tried again.

| net | kind | area Å² | atoms | build |
|---|---|---|---|---|
| super-cubic | 3D periodic | 3 204 | 836 | 37 s |
| super-graphene | 2D periodic | 3 700 | 1 180 | 36 s |
| super-fcc | 3D periodic | 12 796 | 2 982 | 413 s |
| super-diamond | 3D periodic | 13 059 | 3 752 | ~400 s |
| supertube-(6,6) | 2D periodic | 18 964 | 5 922 | 434 s |
| super-fcc | 3D periodic | 21 340 | 4 694 | 483 s |
| super-fcc | 3D periodic | 31 989 | 6 862 | 566 s |
| super-icosahedron | cage | 18 096 | 4 292 | 35 s |
| super-hypercube | cage | 21 802 | 6 494 | 88 s |
| superfullerene-C60 | cage | 24 090 | 7 534 | 79 s |

* **Area ranks the cages backwards.** The hypercube carries more wall
  than the super-diamond cell (21 802 against 13 059) and builds in a
  sixteenth of the time. A cage closes on itself; a periodic cell has
  its faces welded to their opposite numbers, and that is the cost.
* **Area flattens among the periodic cells.** `grid_resolution` is 72
  whatever the cell is, so a larger cell is sampled more coarsely and
  does not carry proportionally more mesh. From 12 800 to 32 000 Å²
  the build goes 413 s to 566 s -- 2.5× the wall for 1.4× the time.
* **Cost per unit area is not monotone** (9.7, 11.5, 22.9, 30.6, 32.3,
  22.6, 17.7), so it cannot order anything.
* **Atom count does not separate a cage from a cell**, the same reason
  it was wrong as a rescue budget: 7534 atoms in 79 s against 2982 in
  413.

A bad prediction here is worse than none. Bands fitted to the first
five rows called super-fcc at scale 60 "tens of minutes to build"; it
builds **sound at 331.9° in 566 s**. That is the hint talking someone
out of pressing a button that works.

**So the ninety minutes a user reported has a cause that is not size.**
At 60/5/4 with the defaults the same net builds in under ten minutes.
The window passes `anneal_sweeps` from `var_anneal`, which goes stale
across a mode switch exactly as the cell length does -- that is the
open suspect, and it is being measured.

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

`render.has_bpy()` lets the pipeline render through the `bpy` module when
no Blender application is installed, which removes its single most common
failure. Check it with `find_spec`, never by importing: importing Blender
costs hundreds of megabytes, and it runs while drawing a label.

## The render pipeline has four files, and three of them must not import bpy

`nanocarbon_lab/blender/` is package data rather than a subpackage
because `bpy` only exists inside Blender. That is a reason to keep the
*unimportable* part as small as possible, and it had grown too large:
the colour decision, the style catalogue, Blender discovery and the
command line were all behind that wall, so none of it could be tested on
an ordinary machine and the whole pipeline was reachable only from the
window's button.

| file | imports bpy | what it owns |
|---|---|---|
| `nanocarbon_lab/render.py` | no | where Blender is, which script, what argv |
| `blender/styles.py` | no | styles, materials, backgrounds, palettes |
| `blender/bundle.py` | no | reading a bundle, and which slot each atom takes |
| `blender/camera.py` | no | which way to look from, and how far back |
| `blender/effects.py` | no* | effect specs, arc geometry, the glare pass |
| `blender/mesh_builder.py` | **yes** | spheres and cylinders |
| `blender/render_cnt.py` | **yes** | world, floor, lights, camera, render |
| `blender/nanocarbon_blender.py` | **yes** | build *and* render inside Blender |

\* `effects.py` imports `bpy` **inside its builders**, never at module
level, so the specs, the parser, the arc geometry and the bloom are all
importable and testable outside Blender while the object creation is
not. That is the pattern to follow for anything new here: if a thing can
be decided without Blender, decide it where it can be tested.

`render.py` is the only one an ordinary import reaches; the rest are
loaded by path, which is what `render.style_catalogue()`,
`effect_catalogue()` and `camera_module()` do once each and cache.
Everything that is not geometry belongs on the left-hand side of that
table.

**A module loaded by path must make its own sibling imports work.**
`styles.py` imports `effects` and `bundle.py` imports `styles`, and
neither is a package, so both insert their own directory on `sys.path`
at the top. Without it they import fine inside Blender -- where
`render_cnt` has already done it -- and fail everywhere else, which is
exactly backwards from where the tests run.

**Blender's own Python has numpy and nothing else of ours.** No scipy,
no ase, no networkx, no scikit-image. So anything in the render path
that reaches for scipy is a bug: `camera.convex_hull_area` is a
monotone chain written out by hand and `effects._gaussian_blur` is three
box passes, both for that reason and not for speed.
`nanocarbon_blender.py --check` reports what the running Blender is
missing and prints the `pip install` line with **its own** interpreter
path filled in, because the failure is otherwise an ImportError from
inside a 3D application with no hint that the fix is one install.

### The camera measures the best angle, and frames the box not the sphere

Two separate faults, and the first made the second hard to see.

**Framing used the bounding sphere.** Right for a cage and badly wrong
for a composition: a row of structures 77 Å long and 12 Å deep has a
bounding sphere of radius 39 Å, so the camera backed off far enough to
fit a 78 Å ball and the subject came out filling **0.56 of the frame
width against the 0.86 asked for**. `camera.frame_camera` frames the
**projected bounding box**, horizontally and vertically against their
own half-angles, and adds the depth half-extent because the near face of
a subject is closer than its centre.

**The direction was the constant `(1, -1.4, 0.6)`** -- a decent
three-quarter view of a compact object and an arbitrary one of anything
else. It looked down the length of a row and at the end of a tube.
`camera.best_view` projects the atoms along each candidate and keeps the
largest **silhouette area**, which is the direction that shows the most
structure. Measured: a row along x is viewed from mostly +z, a tube
along z from perpendicular to it, and a sphere from a three-quarter
angle.

Three restrictions on the search, each about what a figure wants rather
than what scores highest: only from above, never from straight overhead
(a pole view flattens anything that is not a sheet), and **the named
views are not in the candidate set**. That last one matters -- an exact
elevation is an exact optimum, and no tie-break bonus pulls away from
one. They stay reachable as `--view front`, `top`, `iso`, `hero`.

### The floor belongs under the subject's lowest point, not under its centre

Third time the bounding sphere has been the wrong measure, after the
camera's framing and the effects' scale -- and the first two are already
written down two paragraphs apart in this file, which is what makes this
one worth recording rather than merely fixing.

`_add_floor` placed the plane at `centre.z - radius * 1.04`. That is
correct for a sphere and its own docstring said so ("the lowest atom of
a sphere"). **A composition is never a sphere**, and the error is the
gap between the bounding-sphere radius and the actual drop to the
subject's base:

| subject | radius | height | spherical floor | measured floor | error |
|---|---|---|---|---|---|
| row of three cages | 65.0 | 39.8 | -40.7 | 4.4 | **45.1 Å** |
| one cage | 34.5 | 39.8 | -9.9 | 4.7 | 14.6 Å |
| one tall tube | 32.8 | 63.4 | 3.6 | 4.7 | **1.1 Å** |

So a row of 39.8 Å cages had its floor **further below them than they
are tall**, and they levitated out of their own reflection. Measured on
the finished frame, the band under the subject was a smooth gradient --
0.0876 rising monotonically to 0.1124 -- with no trace of the cages in
it. A reflection carries the subject's shape; a gradient is the
backdrop.

**The error scales with how far from spherical the subject is, which is
why it hid.** On a single tall tube the spherical formula is 1.1 Å out
and the reflection looks perfectly fine -- that case was tested first
and gave false confidence. Do not validate a floor on a compact subject.

`camera.floor_z` now takes the subject's **lowest point**, and lives in
`camera.py` rather than in the render script for the reason that table
of file responsibilities gives: it is arithmetic, it needs no Blender,
so it belongs where a test can reach it. The test pins the *property* --
the plane clears the bottom without levitating, and sits more than 40 Å
above where the spherical formula would put that row -- rather than the
constant, which is what the old test pinned and why it passed through
the whole bug.

### Effects are decoration, and they are measured in subject radii

`effects.py` is the part of a cover that is **not** the structure:
`laser`, `arc`, `resonance`, `orbit`, `halo`, `fog`, `glare`. None of it
is chemistry, none of it is measured from the structure, and it must
never be confused with the structure's own geometry.

Every distance is in **subject radii**, never Ångström, so an effect
means the same thing on a 3 Å cage and a 90 Å network -- the property
the light rig already has, for the same reason.

Four things were wrong first and each is a rule now:

* **Scale effects by what the camera FRAMES, not by the bounding
  sphere.** On a row the sphere is the half-diagonal while the camera
  frames the much tighter projected box, so rings written as "1.1 radii"
  came out larger than the frame and cut across the picture. The camera
  is set up first and returns the framed half-extent for exactly this.
* **An emissive object is a light source.** A dozen bright arcs do not
  only look like lightning, they *light everything*, and with any haze
  in the air they wash the frame out. `_set_light_contribution` turns
  off their diffuse, glossy and scatter rays, so an arc is drawn *over*
  the picture while a laser -- which should illuminate what it hits --
  keeps `light=1`.
* **Random pairs on a sphere are antipodal on average.** Arcs joining
  them were as long as the structure is wide, which on a composition is
  a zigzag right across the frame. `span` caps the angle between an
  arc's two ends so it hugs the surface, which is also what a corona
  discharge does.
* **A beam aimed at the centre goes straight through.** `stop` lands it
  on the surface, which is the difference between something striking the
  structure and a line drawn over the picture.

### The bloom is an array pass, because the compositor returns black

Blender 5 moved compositing to `scene.compositing_node_group`. Assigning
one in a headless `bpy` build returns a **black frame** -- 0.0003 mean
against 0.2125 for the same scene without it, reproduced on a bare
emissive sphere with nothing else in the scene. A feature that silently
turns a render black is worse than no feature.

So `effects.glare_image` does it over the finished image: threshold,
blur, add back. It works in every Blender, costs nothing in sample
count, and -- unlike a node graph -- is testable in an ordinary
interpreter. Two details are load-bearing: the bright pass is **capped**
as well as floored (`maximum`), because an emissive filament is many
times brighter than white in linear light and without a ceiling its glow
swamps the frame; and the glow keeps the **hue** of what is glowing, a
white bloom around a green laser being a lens flare from a different
photograph.

### Stars are Voronoi, not noise

`space` and `nebula` are procedural, so nothing has to be shipped or
found at render time and a 6000 px cover still has stars that are
points. The stars are a **Voronoi** field thresholded just above zero:
noise gives soft blobs that read as dust, while Voronoi has one sharp
minimum per cell, so the cut leaves a scatter of points of genuinely
different brightness.

### Two bpy traps worth not meeting twice

* **`nodes.remove()` can leave other node references dangling.** Taking
  the output node *before* deleting the rest and then using it gives
  `KeyError: 'Surface'` on a node that plainly has a Surface input.
  `_fresh_tree` clears the tree and makes a new output instead.
* **`foreach_set` wants float32.** A float64 array raises
  `TypeError: incorrect sequence item type: d`, which names neither the
  array nor the dtype.

**Nothing may keep its own copy of the style names.** The window did, as
a hardcoded list of five, and it had been wrong since the sixth style was
written -- a menu silently offering half the catalogue, with nothing to
show the rest existed. `BLENDER_STYLES` reads `styles.py`, and a test
asserts the window's source contains no style name.

### Three axes, and fusing them was the bug

A style used to be the whole look: character, colouring and backdrop in
one object, and ring colouring was therefore not optional.

* **`--style`** is the *character* of the surfaces -- metallic,
  roughness, clearcoat, which shader, the lights, the lens.
* **`--colour-by`** is what the hue *means*: `ring`, `element`,
  `structure` or `uniform`. Left out, each style's own
  `default_colour_by` decides, so a cover style keeps ring colouring and
  a figure style takes element. **Do not send a concrete value because
  that is what a menu happened to display** -- it overrides every style's
  own choice without anybody asking it to, which is why
  `render_command` leaves a `None` off the command line entirely rather
  than passing a default.
* **`--background`** is the backdrop, and it is its own object so that
  "this style on white for the paper" and "this style on black for the
  cover" are two uses of one style rather than two styles.

`MaterialSpec.with_colour` is what makes the split work: a style's
character survives a recolour, so element colouring inside
`gold_nanotech` gives a gold-finish oxygen rather than a flat primary. An
emissive material's *glow* is recoloured with it -- a red-hot accent that
glows blue is not a colour choice.

Two consequences follow from the colour mode and neither is cosmetic:

* **Bonds split in half when the colour belongs to the atom.** A C-O bond
  drawn in carbon's grey says the oxygen is carbon. `element` and
  `structure` therefore draw two half-cylinders per bond; `ring` keeps
  one, a ring being a property of neither end. `styles.SPLIT_BONDS` holds
  the rule and a test pins that its keys are exactly `COLOUR_MODES`.
* **Atom radii follow the element when the colour does.** Equal spheres
  are right for an all-carbon lattice and wrong the moment a hydrogen and
  a gold atom are in the same picture.

**Ring colouring says little on a closed cage and nothing on a doped
one**, which is the reason this was asked for: every one of C60's atoms
sits on a pentagon, so the picture said "all pentagons" about twelve
pentagons and twenty hexagons. That is the same fact the preview's
ring-fill view exists for, met from the renderer's side.

**And element colouring says nothing on a pure-carbon structure**, which
is the other half of the same point: a C240 cage or a bare nanotube
coloured by element is one colour throughout, correctly. Element mode
earns its keep on a doped, functionalised or encapsulated structure --
the showcase is a cage at 8 % N, 8 % B with hydroxyls, which is five
elements and reads instantly. Neither mode is the right default for
everything, which is why `default_colour_by` is per style.

### A periodic structure's own bonds cross its cell, and the renderer drew them

The same trap as "a minimum-image bond list names the pair, not the
cell", met a third time -- and this time in the **renderer**, on a
single structure, with nothing to do with compositions. A bond list
records `(i, j)` and throws away which image `j` was in, so drawn from
the raw coordinates a bond across a cell face is a straight line as long
as the cell. Measured on a 920-atom super-graphene cell: **70 of its
1380 bonds, up to 41.1 Å** against a real bond of 1.41 -- five per cent
of the bonds drawn as girders across the whole picture. That is what
"saturated with bonds" looks like, and it was the worst-looking defect
the pipeline had.

The sidecar carries the **cell**, so `bundle.split_seam_bonds` can
recover the image: for each bond, the integer combination of lattice
vectors that makes it shortest. Zero is an interior bond; anything else
crossed a face. `--seams` then chooses `stub` (a half-bond pointing the
way the structure continues, which is what a crystallographic viewer
draws at a cell face -- the default), `drop`, or `keep`, which
reproduces the defect on purpose.

The shift is a **candidate scan** over {-1, 0, +1} per live axis, not a
rounding of the fractional coordinate: rounding is right for an
orthogonal cell and off by one for a sheared one. One cell is always
enough -- a bond longer than that is not a bond.

This is the third place the same fact has bitten: `analyse` for the
census, `cell.supercell` for repetition, `scene` for compositions, and
now the renderer. Anything that uses a recorded bond pair **as a pair of
positions** needs the shift.

### A draft is the cover with more noise, and that is measured

A 4096 px cover at 1024 samples is **874x the sampling work of a
draft**, which is too much to spend on a picture nobody has seen yet.
`--estimate` prints the resolution, the samples and that ratio and
renders nothing, so the cover is a decision rather than a click.

It states **work and ratios, never seconds**, for the reason "Size can
be stated before a build; time cannot" gives: the render path has the
same two-term shape, where building the scene does not depend on the
resolution at all (measured: **291 s of a 348 s** draft render of a
1532-atom particle) and only sampling scales. A single number would
hide that 874x the sampling is nowhere near 874x the wall clock. The
ratios are against the DRAFT and not against whatever was asked for --
compared to the request, every cheaper preset prints `0.00x` at cover,
which tells nobody anything.

**Iterating at draft is only worth anything if the draft predicts the
cover**, so that was measured rather than argued. The same scene at
`draft` (640 px, 48 samples) and `preview` (1280, 128), the second
box-downscaled to the first:

```
mean |diff| 0.0072   p99 0.0951   max 0.2882
draft         silhouette bbox x 50-568 y 85-524  centroid (330.5, 311.5)
preview-down  silhouette bbox x 26-568 y 84-524  centroid (325.7, 314.0)
```

The centroid moves 5 px in 640 and the subject's extent is the same, so
what changes is the noise. That holds by construction and each piece is
a rule worth keeping: the glare radius is a **fraction of the short
side**, sphere detail follows the **atom count**, every effect is
measured in **subject radii**, and `--quality` scales the **long** side
carrying the aspect. Anything added here that is sized in pixels would
break the guarantee and make cheap iteration meaningless.

### Depth colouring cannot read through a hollow subject

Two separate faults were found asking for `--colour-by depth` on a C240
cage, and the first one hides the second.

**A round subject has no longest axis.** The ratio of the first two
singular values is **1.0000** for the cage and **1.0123** for a coated
spherical nanoparticle, against 1.73 for a sheet, 3.01 for a row of
three structures and 6.29 for a long net. At 1.0 the principal axis is
decided by rounding, so the gradient runs along a direction a tiny
perturbation rotates. `--depth-axis view|x|y|z` fixes that, and
`depth_axis_quality` reports it (`DEGENERATE_AXIS`).

**And fixing it does not make the picture readable.** Graded along the
view direction with a monotone ramp, the cage comes out with its
silhouette at mid-ramp and its centre carrying the two ends at once --
which is correct, and unreadable. A hollow shell projects its near wall
and its far wall **onto the same place on screen**, so every band
appears at the middle of the disc and only the silhouette sits at an
unambiguous depth. No choice of axis changes that; it is what seeing
through something means.

So `depth` is a mode for an **extended or porous** subject -- a sheet, a
net, a composition -- where the structure's own thickness is spread
across the frame. For a closed cage, `ring` and `element` are the modes
that say something, and the note the renderer prints says so rather than
recommending the axis flag as a cure.

An earlier version of that note did recommend it as a cure, which
overclaimed: the axis was measured and the legibility was assumed.

### Haze does not sit behind a gradient, it replaces it

`porous_depth` read as a grey wash rather than as depth colouring, and
the lights were not the cause. A fog volume **scatters the key light
across the whole frame**, so at `fog` 0.0035 under a 2600 W key the
backdrop lifts and the gradient loses the contrast it is made of. Every
colour was still present and none of them could be read.

A trace (0.0008) separates layers, which is what the style is for. Past
that, haze is not a depth cue behind the subject -- it is a veil in
front of it. Judge a fog strength against the backdrop's brightness in
the finished frame, not against how misty the scene looks in isolation.

### A depth ramp that cycles in hue is a cover, not a measurement

`DEPTH_RAMP` is viridis-like and **monotone in brightness**, so a figure
printed in greyscale keeps its ordering. `Style.depth_ramp` overrides it
per style and defaults to empty.

The three cover ramps are not interchangeable with it and the tests say
so in Rec. 709 luma rather than in prose: `PLASMA_RAMP` climbs the whole
way and is asserted **sorted**; `IRIDESCENT_RAMP` and `AURORA_RAMP`
cycle and are asserted **not** sorted. Keep that second half -- it is
what stops a cover ramp being quietly recommended for a figure that has
to survive a black-and-white print.

### `star_density` is a Voronoi scale, not a fraction

Written as 0.22, read as "22 per cent", one Voronoi cell covers the
whole sky and a `space` background renders **starless** while its style
promises far stars. The shipped values are 220-300. Nothing raises, the
frame simply looks like a plain black backdrop, and the only way to
catch it is to read the style's own description against its render --
so a test now walks every `space` and `nebula` background and refuses a
`star_density` under 10.

### The surface model could not make a hole, by construction

`build_smooth_surface` hardcoded `skin_radius` at **1.6 A** and no style
could reach it. A graphene hexagon's circumradius is **1.42 A**, so a
1.6 A skin on every strut is wider than the ring it has to leave open:
on any carbon net every pore welds shut, necessarily and at every size.
The model could only ever return a solid, and `membrane` -- whose
description promises "the shape and the holes through it" -- rendered
with no holes at all, contradicting itself.

`Style.skin_radius` defaults to 1.6, so every style that existed before
renders exactly as it did; `membrane` sets 0.55. The test measures the
constraint that makes an open pore possible (under half a hexagon's
circumradius) rather than pinning the value that happened to be chosen.

**A hardcoded value no caller can reach is not a default, it is a
limit.** This one was invisible because the model's output looked
deliberate.

### Style colours are LINEAR; the frame is written as sRGB

A `MaterialSpec` or `Background` colour goes to Blender as a linear
value and the PNG is encoded sRGB, so a linear **0.09 displays near
0.33**. An "ink" backdrop set at `(0.03, 0.06, 0.10)` comes out mid
blue-grey on screen -- correctly, and it reads as a bug.

That cost a cycle here: the mid-grey backdrop was diagnosed as the glare
pass blooming an overexposed subject across the frame. It is not.
Measured on a synthetic bright blob over an ink field, `glare_image`
leaves a far corner at **0.0633 before and 0.0633 after** for subjects
at 0.6, 1.0, 2.0 and 4.0 -- the blur is local and cannot reach that far
-- and a render with `--no-style-effects` shows the same backdrop. Both
the bloom theory and the "wrong colour" reading were wrong; the only
real fault was exposure.

So judge a colour by what the encoded frame shows, and set it for that.
Do not re-investigate the glare pass over a lifted background.

### Depth is the colour mode a superstructure has

On an all-carbon net, `element` is one hue and `ring` is nearly all
hexagons -- neither says anything. `depth` grades each atom by its
position along the structure's **own longest axis**, found by principal
component rather than by bounding box, so a sheet lying in xy and a tube
standing in z both get a gradient across their length and a diagonal
structure is still graded along it.

The ramp is viridis-like and **monotone in brightness**, which a rainbow
is not: a figure printed in black and white must not lose its ordering,
and a test pins that the greys are sorted. It is quantised into
`DEPTH_BANDS` = 24, which is a real limit rather than a setting --
material slots are discrete -- and 24 is past where the eye sees
stepping.

`surface` also became a **model** rather than a separate `--mode`, so
every representation is one control: at a thousand atoms the thicket of
sticks hides the shape, and the shape is what is left to say.

### The spheres were eighty triangles, at every size

`sphere_subdiv` was fixed at 2. An icosphere at subdivision 2 is
**80 triangles** -- not 320, which is subdivision 3; the count is
`20 * 4^n` -- and that is visibly faceted the moment an atom is more
than a few per cent of the frame, which a close-up of a small molecule
always is.

It cannot simply be turned up: the cost is linear in the atom count, and
2000 atoms at subdivision 5 is ten million triangles before a single
bond is drawn. `DETAIL_LEVELS` holds the four steps (80 / 320 / 1280 /
5120 faces) and `detail_for` picks from the atom count, because the
right answer genuinely depends on the subject -- a 60-atom cage fills
the frame with twenty-pixel spheres and needs every triangle, while in a
2000-atom supernetwork an atom is three pixels across and subdivision 5
is invisible. `--detail` overrides it.

### Superstructures are a different problem from objects

At 900-2000 atoms the subject is a **lattice** rather than a thing, and
three defaults have to change together -- which is why
`lattice_dark`, `lattice_depth`, `lattice_blueprint` and
`lattice_studio` exist instead of a note telling the user to adjust
three sliders:

* **The atoms have to shrink** (`atom_scale` around 0.75). Anything that
  fattens an atom closes the holes the eye reads depth through, and the
  picture becomes a wall of spheres.
* **The depth of field has to open** (f/5.6 or deeper, or off). A
  lattice photographed at f/2 is a blur with a sharp band through it,
  which reads as a mistake rather than as a choice.
* **The colour cannot come from the element.** An all-carbon net
  coloured by element is one grey mass; curvature is what varies.

`lattice_blueprint` goes further and drops the spheres entirely
(`licorice`), because at that size the atoms are what hide the topology
and the topology is the result. `lattice_depth` carries haze, which is
the only depth cue that still works when a structure is mostly its own
repetition.

### A finish is a third axis, and it keeps the colour

`--style` is the character, `--colour-by` is what the hue means, and
`--background` is the backdrop. A **finish** is the fourth: it replaces
every material's character while keeping its colour, so "element
colours, in glass, on the cosmos backdrop" is one render rather than a
style nobody wrote. `FINISHES` has sixteen entries and `"style"` -- the
default -- leaves a style's own character alone, because its author
chose it.

Two things in `apply_finish` are measured rather than tasteful:

* **Glass lifts a dark colour toward white** (`GLASS_TINT = 0.25`).
  Transmitted light is multiplied by the base colour at every bounce, so
  a glass version of carbon's element grey, through a cage's near wall
  and its far wall, comes out nearly black: it reads as soot, not as
  glass. Lifting the tint keeps *which element* an atom is while letting
  the light through.
* **Transmission goes on the Principled BSDF, not the Glass node.**
  Principled keeps the base colour tinting the transmitted light and
  EEVEE renders it properly; a raw Glass node does neither as well. The
  `glass_cryo` style still uses the Glass node directly, which is a
  second look rather than a duplicate.

`gold` is the one finish that **sets** the colour instead of keeping it,
because gold is a material and not a hue to apply to someone else's
colour -- a blue gold is a contradiction.

### Ball-and-stick or space-filling, and the mixed case is the point

`--model` is a fifth axis. `ballstick` draws spheres at **display**
radii joined by sticks, which is right for a lattice where the
connectivity *is* the subject. `spacefill` draws them at their **van der
Waals** radii with no sticks at all, which is right for a metal
particle. `licorice` and `wireframe` drop the spheres.

**`auto` is why this exists.** It asks per *atom*: metals come out
space-filling and everything else ball-and-stick, in the same picture.
A carbon-coated gold particle is then a solid metal core inside an open
carbon cage -- which is what it is, and which neither pure model gets
right about more than half of the structure.

Three rules behind it:

* **The vdW radii are the real ones** (Bondi, Alvarez 2013 for the
  metals), not display radii scaled for taste. A space-filling model is
  *defined* by atoms touching at their vdW contact, so tuning them would
  stop it being one. `SPACEFILL_SCALE` is 0.98, which is only to stop
  z-fighting where a lattice puts two spheres in exact contact.
* **`auto` draws no stick into a space-filling atom.** It would start
  inside a sphere of its own radius and come out the far side -- and
  between two metals it is an invention anyway: a gold cluster's atoms
  are packed, not bonded in pairs. That is what makes clusters on a
  sheet read correctly: solid clusters, open lattice, nothing joining
  them.
* **`SPACEFILL_ELEMENTS` is the metals only.** Carbon, the dopants and
  the graft elements stay ball-and-stick whatever else is in the
  structure, because the lattice is the subject.

A style may declare a `default_model` -- a model-kit look is
ball-and-stick by definition, a supported-catalyst look is mixed -- and
an explicit `--model` still wins.

### The bonds are decided separately from the atoms

They were not. Whenever the colour belonged to the atom, the bonds
simply reused the atoms' material list, so asking for glass atoms gave
glass bonds and there was no way to say otherwise -- and a figure often
wants the opposite, element-coloured atoms on a neutral frame. That is
two decisions, so `bond_materials` makes them two: `--bond-finish` and
`--bond-colour` (`atoms` / `style` / `neutral` / `auto`).

`auto` reproduces the old behaviour and is the default. `atoms` on a
ring-coloured render **falls through to the style's bond** rather than
inventing a rule: an atom on a pentagon and a hexagon has two ring
colours and picking one would be arbitrary.

The split-bond materials are built fresh rather than reusing the atoms'
datablocks, because the bonds may carry a different finish and sharing
the datablock would silently apply it to the atoms too.

### Rim light is the one thing a lamp cannot do

`MaterialSpec.rim` adds emission where the surface turns away from the
camera, through `LayerWeight.Facing`. A rim *lamp* draws a bright edge
on one side; this draws one on **every silhouette the viewer can see**,
including the far wall of a cage seen through its near one, which is
what makes a dark cover read at thumbnail size. Wired as an **Add
Shader**, not a Mix: a rim is light arriving on top of the material, and
mixing would dim the body to pay for the edge.

It is on in the dark cover styles and off in the figure styles, where a
glow is decoration on something a reviewer has to measure.

### The window says what a preset decides

A preset is a bundle of a dozen choices and its name says none of them.
`style_summary` prints the ones that change what you see -- backdrop,
colour mode, model, lens, samples, atom and bond scale, and which
effects it brings -- and the window shows it under the style menu,
followed by whatever the user has overridden. `MODEL_HELP`,
`COLOUR_HELP` and `BACKGROUND_HELP` are one line per option, so a
control explains itself rather than being guessed at from its name.

A test asserts every option of every one of those three has a help line.
A control whose choices are not explained is a control nobody uses.

### An effect is light, and light is not a line

Three rewrites, each because the effect "looked basic", and each the
same underlying mistake -- drawing a shape where there should be light.

* **A constant-width stroke reads as a wire.** Every filament now
  carries a `taper_profile`: thin at both ends for an arc, thin at one
  for a spark, a slight waist for a beam. Blender's spline *radius*
  channel does it, so it is still one curve object.
* **One emissive tube is a glowing wire, not a glow.** `_glow_pair`
  builds a hot thin core inside a wide, nearly transparent sheath,
  because a single material cannot be both opaque at the centre and
  faint at the edge. The core keeps the hue -- lifting it most of the
  way to white, which was the first version, turned every coloured
  filament into a white wire and threw away what the colour parameter
  controls.
* **A random walk is not lightning.** `fractal_offsets` is midpoint
  displacement: set a segment's midpoint to the mean of its ends plus a
  random kick, halve the kick, recurse. That gives large bends with
  smaller kinks on them and smaller ones on those. A walk's
  displacement is the *sum* of its steps, so it drifts smoothly and the
  result is a meander -- which is precisely what the arcs looked like.
  The offsets are normalised, so `jitter` means the same thing at every
  segment count; otherwise a finer arc is also a wilder one.

**And the halo has to be a volume.** A surface shell keeps the sphere's
silhouette however it is tuned: uniform emission gives a frosted disc,
and Fresnel at the limb gives a hard ring -- a circle somebody drew
round the structure. Both were built and both were rejected on sight. A
volume has no silhouette. One catch, which cost a white frame with the
structure a ghost inside it: **volume emission accumulates along the
ray**, so a shell twice as wide glows twice as brightly for the same
setting. The brightness is divided by the path length, which restores
the rule that an effect means the same thing at every size.

### Five looks worked back from references, not invented

`model_kit`, `clusters_on_sheet`, `clear_glass`, `neon_violet` and
`teal_lab` each copy a specific kind of picture, and each description
says which -- a preset whose provenance is "it looked nice" cannot be
corrected later. What they add beyond colour:

* **A floor that reflects** (`Background.floor_metallic`). A product
  shot stands its subject on a mirror; a shadow catcher is not the same
  thing.
* **Bokeh as geometry**, not as a texture. `_build_bokeh` puts real
  emissive spheres behind the focal plane so the camera's own aperture
  blurs them -- painted discs stay the same size and shape wherever the
  camera goes, while these grow, shift and overlap when the lens or the
  f-stop changes. The style that ships them also ships a wide aperture,
  because bokeh with everything in focus is a field of dots.
* **Matte rods under glossy balls** (`model_kit`). A kit's spheres are
  injection-moulded gloss and its sticks are matte white plastic, and
  getting that wrong is most of why a render reads as a diagram rather
  than as a photograph of an object. This is the case the separate bond
  material exists for.

`clear_glass` carries 640 samples because glass is the slowest thing
here: every sphere refracts twice and what is *behind* it is the
picture.

### Cycles is the default and EEVEE will abort if you let it

Every style declares `engine="CYCLES"` and always has; `--engine` is an
override, not a correction. Three facts about that worth not
rediscovering:

* **Cycles is an add-on**, so in a stock `bpy` wheel it is missing from
  `render.engine`'s enum listing -- the listing is built before add-ons
  register. Assigning `"CYCLES"` works anyway, and `addon_utils.check
  ("cycles")` returns `(True, True)`. Do not conclude from the enum that
  Cycles is unavailable.
* **EEVEE's identifier moved.** It was `BLENDER_EEVEE_NEXT` through 4.x
  and is `BLENDER_EEVEE` in 5. Both are tried in order.
* **EEVEE is a rasteriser and needs a GPU context.** Without EGL it does
  not raise and does not return an error: it **aborts the process** with
  SIGABRT after printing one line about `libEGL.so.1`, which cannot be
  caught from Python once it starts. `_can_rasterise` rules it out
  beforehand and falls back to Cycles with a note. It checks for **EGL
  specifically** -- the first version accepted any of EGL/GL/OpenGL, and
  this container has `libGL.so.1` and no libEGL, so it passed the check
  straight through to the abort it exists to prevent.

EEVEE's `samples` are temporal anti-aliasing passes rather than light
paths, so the number means something different and is clamped to
16-128: a Cycles-tuned 512 would make the fast preview slow.

### Resolution and samples are one decision

`QUALITY_PRESETS` sets both, because a 6000 px frame at 64 samples is a
big noisy picture and a 600 px one at 1024 is a slow thumbnail. `print`
is 300 dpi across a single journal column with room to crop; `cover` is
300 dpi across a full page. The preset scales the **longer** side and
carries the aspect ratio, so a cover asked for at 2000x2400 stays 5:6
rather than going square.

It lives in `styles.py`, not in the render script, so it can be
imported rather than scraped -- the first test of it parsed the dict out
of the source text and broke on its own closing brace.

### The pipeline renders anything with a bundle, not just carbon

Worth stating because it keeps being asked: the renderer reads a
**render bundle**, not a structure type. Nanoparticles, compositions,
dichalcogenides and anything else a builder writes go through the same
path, and `--colour-by element` is where a nanoparticle earns its keep
-- an Au490 core inside an N-doped graphitic shell is three elements and
reads instantly, where the same mode on a pure-carbon cage is one
colour.

### A shader is not a parameter, and a floor is not decoration

Four of the five entries in `styles.SHADERS` cannot be expressed by
Principled parameters at all -- glass needs refraction through a closed
surface, toon needs banded rather than smooth falloff, emission needs no
surface response, and clay needs the specular lobe *gone* rather than
merely rough -- so `render_cnt._shader_node` builds each as its own node
and a test asserts every shader a style asks for has a setup there. A
missing one raises at render time, which is after the slow part.

The floor is a tri-state question pretending to be a checkbox: some
styles ship one and most do not. The window therefore *initialises* the
checkbox from the style whenever the style or the background changes,
rather than defaulting to off -- left off it would silently strip
`clay_studio` of the floor the style exists for. A **shadow catcher** is
invisible but still receives shadow, which is what keeps a transparent
background's contact shadow; without it a composited subject reads as
pasted rather than placed. The plane is added *after* the subject's size
is measured, or a plane two dozen radii across would be what the camera
frames.

### Opening Blender is a different operation from rendering

`--save-blend` writes the assembled scene and `--no-render` skips the
render, so `--open` (no `-b`) hands a live Blender window over with the
lights, camera, materials and background already built. No amount of
style presets replaces moving a camera by hand, and this is the point
where the pipeline stops guessing.

It needs the **Blender application**: the `bpy` wheel is a full Blender
build in this interpreter and has no window, so asking for one through it
must say why rather than rendering headless and reporting success. The
window launches it with `Popen` rather than `run` -- an interactive
session lives as long as the user wants it to, and waiting for it in a
thread would leave the GUI reporting a render that is not happening.

## A composition is an illustration, not a structure

`scene.py` arranges several finished structures into one file for VESTA
or Blender. It is **not** a builder and it is deliberately an action in
the window's right-hand column beside export and render, not a build mode
on the left: nothing here builds anything, and what comes out is marked
`info["is_illustration"] = True` so every consumer can branch on it. The
items are placed where they look right, there is no relaxed interface
between any two of them, and nothing about the gaps is physics. The
report's first warning says exactly that, every time.

* **Bonds and rings are carried and re-indexed, never re-guessed.** A
  bond search over the whole scene would invent bonds between two
  structures that merely sit 4 Å apart, which is the one thing a
  composition must not claim. `Atoms.set_tags` carries the item index per
  atom, which is all `--colour-by structure` reads, and a test asserts no
  bond and no ring spans two tags.
* **A bond that left its item's periodic cell cannot be drawn as a
  bond.** Same trap as "a minimum-image bond list names the pair, not
  the cell", met where there is no cell to shift by: a composition is
  finite, so the atom at the far end is not in the scene. Measured on a
  (5,5) tube whose every real bond is 1.42 Å, drawing them from the raw
  coordinates gave a **12.4 Å** line -- and the span alone cannot
  identify them, a (6,0) tube's axial period being 4.26 Å, so
  `_crossing_pairs` uses `bond_shifts`.

  There are three honest answers and the default is the useful one.
  `seams="stub"` draws a **half-bond** pointing the way the structure
  continues, which is both true and what a crystallographic viewer does
  at a cell face -- the tube ends in bonds going somewhere rather than
  in a ring of bare atoms. `seams="drop"` leaves them out, for an item
  that really is finite and whose periodicity was only ever a
  convenience for the DFT cell. And `Item.repeat` makes the seam
  **interior** by repeating the item through `cell.supercell`, which
  re-indexes bonds copy by copy: a (5,5) tube at `repeat=(1,1,3)` goes
  from 150 bonds with 14 at the seam to 436 bonds with 14 at the seam,
  all of them 1.410-1.419 Å. The report names the count either way.

  The stubs are stored as `info["seam_stubs"]`, one entry per *end* --
  `[atom, dx, dy, dz]`, a single index and an offset rather than a pair
  -- so anything that reads a bond as two indices is right to ignore
  them. The offset is in the **placed** frame: returning it in the
  source frame was the first version, and it put every stub on a rotated
  item pointing the wrong way, which looks like a rendering fault rather
  than a missing rotation. Rings straddling the seam are dropped for the
  reason the preview leaves them unfilled.
* **Spacing is by the item's *oriented* bounding box.** The first version
  measured bounding-sphere radii and ignored each item's rotation: a
  requested 8 Å gap came out at **17.9 Å** in a row and **23.3 Å** in a
  grid. `row`, `stack` and `grid` are per-axis and hit the request to
  about a tenth of an Ångström; `circle` is a radius rather than a pitch,
  so for it the gap is a floor and not a target.
* **`Item.oriented()` returns the item centred on the origin**, so
  `translation` is where its centre goes rather than an offset from
  wherever the builder happened to leave it, and `rotation` is about the
  item's own centre -- about the scene origin it would also *move* the
  item, and the layout has just placed it.
* **A scaled item is recorded as scaled and warned about.** Scaling
  changes every bond length. That is a legitimate choice for a figure and
  it is not a structure any more.
* **`items_from_files` reads the `.json` sidecar beside each file**, and
  that is not a convenience: an `.xyz` carries no connectivity, so
  without it a composition has no bonds at all and the renderer falls
  back to guessing -- the thing this module exists not to do. The
  sidecar's atom count is checked first, a bond list indexed for a
  different structure being worse than none.

Composing replaces the window's preview, so Export and Render then apply
to the composition with no further wiring, and the sidecar carries
`item_per_atom`.

### `best_view` is wrong for a composition placed in depth

`camera.best_view` keeps the direction whose silhouette has the largest
area, which is the one that shows the most structure -- right when the
subject's shape is the point, and **backwards for items strung along a
line**: the largest silhouette of three objects on a line is the view
that puts them side by side, which is exactly the depth the composition
was placed for, destroyed. No tie-break fixes it, because the view that
reads as depth scores *worst* by the measure.

So `NAMED_VIEWS` carries `depth` = (0.35, -1.0, 0.30), nearly down the
line and shallow rather than dead-on, so the ones behind are seen past
the one in front rather than eclipsed by it. It is a name and not a
search for the same reason `top` and `front` are: an exact answer no
measure of area will return. It reaches `render_cnt --view depth` by
being in the table, and nothing else had to change.

The placement itself is not the camera's problem and must not be solved
there: a field of items at any real density will interpenetrate under
random orientations, and `compose` measures every pair and names the
ones under `TOUCHING` (2.0 A) and under `CLOSE` (4.0 A). Tune against
that report, not by eye. Where the density is the point -- a heap of
needles -- tuning a grid pitch and a jitter by hand reaches a floor
(measured: two interpenetrating pairs, then one, then no further);
**draw, measure and reject** instead, keeping the first draw that clears
a stated gap against the items already placed. That keeps the density
and removes the collisions, which widening the field cannot do.

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

## Nanoparticles: the one input that is somebody else's file

`nanoparticle/` cuts a particle out of an imported crystal and optionally
wraps it in graphitic carbon. It is **not** under `builders/`, for the
same reason `tmd/` is not: what comes out is gold, or magnetite, or iron
carbide, and the carbon machinery does not transfer -- there are no rings
to count on a metal core and the sp2 windows mean nothing there. What *is*
reused is everything about surfaces rather than about carbon: the
signed-distance fields, marching cubes, the isotropic remesher and the
valence force field. That reuse is why the coating is a few hundred lines
and not a few thousand.

Four modules and one entry point: `crystal.py` reads the file,
`shapes.py` is the catalogue and the algebra that composes it, `carve.py`
does the cut, `coat.py` wraps the result, `quality.py` gives the verdict
and `build.py` is the single call all three consumers go through (the
CLI, the window and `jobs.Job`).

### A shape is a signed distance, and the algebra is the point

Twenty-three shapes, in four groups that mean different things. The
**round** ones are cuts in the laboratory frame, so the cut exposes
whatever facets it happens to -- a real particle (quenched, milled,
ablated) and not a facetted one. The **facetted** habits' faces are
lattice planes and need the crystal to construct. The **five-fold**
envelopes (decahedron, icosahedron) belong to particles that are really
*twinned*, and a single-crystal cut reproduces the outline and not the
twin boundaries; both say so in their docstrings and in `info`. The
**compositions** are unions and differences of the rest.

Because a shape is a function and not a class hierarchy, a shape nobody
anticipated is a composition rather than new code: `hollow` is a
difference, a star is a smooth union, a nanocage is `hollowed(name, ...)`
for *any* of the twenty-three rather than a catalogue entry per habit.

* **A polyhedron's extent comes from its vertices, enumerated.** Not from
  sampled directions: a regular tetrahedron with faces at 16 Å reaches
  48 Å (a regular tetrahedron's circumradius is three times its
  inradius), and sampling 512 directions returned 44.4 -- *short of the
  solid*, so the sampling box would have cut the four corners off and
  nothing downstream could have told. A sharp vertex is exactly what
  sampling misses, and a facetted particle is nothing but sharp vertices.
  `polyhedron_vertices` solves every triple of faces, which is thousands
  of 3x3 solves and runs once per shape.
* **`exact_outside` is recorded, not assumed.** A convex polyhedron's
  field is `max_i(n_i.r - d_i)`: the sign is exact everywhere, the
  *distance* only away from the edges. That matters because the coater
  offsets the field, and offsetting an under-estimate rounds the edges
  with radius equal to the offset. **That is the answer, not an
  approximation to apologise for**: a graphitic sheet cannot fold along a
  zero-radius crease, and a real encapsulated cube is round at the
  corners in every micrograph of one.
* **A half-space has infinite extent and says so.** A finite number there
  would quietly clip whatever it was intersected with. `intersection`
  takes the minimum, so it costs nothing once it meets something closed.

### Miller indices are not Cartesian components

`crystal.py` converts `(hkl)` through the **reciprocal** lattice,
`n = h b1 + k b2 + l b3`. For a cubic cell that comes out parallel to
`(h, k, l)` and for nothing else: in a calcite-like rhombohedral cell the
(104) normal -- the cleavage plane every calcite rhomb is bounded by --
is **15.5 degrees** away from what its indices would point if read as
Cartesian components, and a 15-degree error in a facet normal is a
different facet.

The symmetry-equivalent set comes from the **lattice's point group**,
found by brute force over integer matrices preserving the metric tensor.
Both halves are exact statements -- an operation maps the lattice onto
itself iff it is an integer matrix in the cell's basis, and it is a
rotation or reflection iff it preserves the metric -- so no spglib and no
tolerance on the symmetry itself. Verified: cubic 48, **fcc primitive 48**
(it looks rhombohedral and is cubic, which is the case that makes
deriving it from the metric rather than the cell's shape worth doing),
hexagonal 24, rhombohedral 12, tetragonal 16, orthorhombic 8, triclinic 2.

What that gives is the **holohedry**, which may be higher than the
crystal's own point group -- any ordered alloy, wurtzite's polar axis, a
Jahn-Teller distortion. Then a "family" contains faces that are
crystallographically different and may have different surface energies.
The report says which group was found and how large, so the assumption is
visible rather than buried.

`wulff()` ships **no table of surface energies** and will not. They
depend on the termination, on adsorbates and on the method, and a builder
that supplied its own would be making the construction's most important
decision for the user. Pass `family_normals_for(cell)` with it or a Wulff
polyhedron on a hexagonal cell comes out cubic-shaped: measured, the same
energies give 8 faces with the lattice's families and 12 with the cubic
fallback.

### The cut decides four things, and the caller did not

The selection is one line. Everything else in `carve.py` is about what
that line leaves open:

* **Where the centre is.** A sphere cut around a lattice point and the
  same sphere cut around the hole between four of them are different
  particles: different composition, different surface, and for a
  compound different *charge*. In rock salt, centred on the cation the
  cube comes out cation-rich and centred on the anion, anion-rich. There
  is no neutral default; `"cell"` is merely the one needing no extra
  information.
* **What the cut leaves hanging.** A singly-coordinated surface atom is
  not a facet, it is an artefact: no counterpart on a real particle, and
  it dominates any later relaxation. Trimmed by default, **iteratively**,
  because removing one lowers its neighbour's count and on a thin plate
  that cascades.
* **Stoichiometry, and therefore charge.** Always measured;
  `stoichiometric=True` additionally trims. That trim is the standard
  recipe and it is a *heuristic* -- least bound by neighbour count is not
  least bound by energy -- and it says so.
* **Whether a facet landed on a lattice plane.** This one is a trap, not
  a choice. A 4a cube of fcc gold centred on a lattice point has its six
  faces exactly on planes of atoms, and changing the edge by **one part
  in a million** moves the count between **171, 256 and 365 atoms** -- a
  factor of 2.1. Three different terminations, all real; what would be
  wrong is not saying so. Offset the cut so the faces fall *between* the
  planes and the count is the lattice's own: exactly `4n^3` for an
  n x n x n cube of fcc, for every n tried. `TIE_TOLERANCE` counts the
  atoms on the surface and the report warns. **It does not move the
  cut**: which plane terminates a facet is a statement about the
  particle, and nudging it silently would answer a chemical question
  with a numerical convenience.

**The surface is measured against the bulk, not against a threshold**, so
"surface fraction" means the same thing for a metal, an oxide and a
layered sulfide with no per-material tuning.

**And the neighbour cutoff is derived from the crystal, not looked up.**
ASE's `natural_cutoffs` are covalent radii, and in MgO the Mg-Mg distance
(2.978 Å) falls inside any usable slack on two magnesium covalent radii,
so a radius-based cutoff reports magnesium as **18**-coordinate in rock
salt -- twelve magnesiums alongside the six oxygens. An ionic crystal has
no covalent radii to speak of and a metal has no bonds in that sense at
all. `NEIGHBOUR_SLACK * d_min` reproduces the textbook number for every
structure tried: fcc 12, rock salt 6 and 6, rutile 6 and 3, diamond 4.
bcc comes out **14 rather than 8**, because bcc's second shell is only
15% beyond its first and no single slack separates them -- which is why
bcc is described as 8+6 anyway. The crystal's cutoff is then used for the
particle too: two coordination numbers measured with different cutoffs
are not comparable, and the whole surface test is that comparison.

### The coating is the existing implicit route, with three measured defaults

A graphitic shell is a closed carbon surface, so it goes through exactly
the junction/schwarzite pipeline and shares `junction._finish` rather
than copying it. Nothing about the census is placed: the field's topology
fixes it and the Euler check inside that function verifies it. Measured
`sum(6-n) = +12` exactly on all six shapes tried (sphere, cube, rod,
octahedron, disc, hexagonal prism), because an envelope is sphere-like
whatever it wraps.

**The field is the core's own van der Waals envelope**,
`min_i |r - r_i| - gap`, not the shape it was carved to. So it works for
a core from anywhere (a relaxed particle, an imported cluster, something
this framework did not build), it follows the *atoms* rather than the
ideal shape, and it rounds every edge at radius `gap` automatically.

Three defaults, each from a measurement, and two of them are the
**opposite** of the junction's:

1. **`place_curvature=True` here, `False` for a junction.** A junction
   saddles: negative curvature everywhere, pentagons do not belong on it,
   and a spread-out population of 5-7 pairs is how the net covers it. An
   envelope is the offset of a solid, so it is **convex everywhere**:
   pentagons wanted, heptagons pure defects. Placement takes the census
   from 39/25/1 to **17/5/0** -- deficit 12 either way, 22 disclinations
   instead of 65 -- and the wall comes out smoother and better bonded. It
   costs about twice the remesh time.
2. **`anneal_sweeps=0`.** Annealing *does* cut the stray-pair count and
   makes the wall worse at every anchor tried (1.620 Å bonds against
   1.546). The same lesson the junction records, reached from the other
   side: annealing optimises the **census**, and the census is not what a
   wall is judged by.
3. **`wall_anchor=0.7`**, on here and off for a junction. A junction's
   wall has nothing behind it; a coated particle's has the core in the
   way, so a free relaxation that lets the wall move inward is relaxing
   it through solid matter -- measured, free relaxation gives excellent
   carbon (1.335-1.494 Å) with an atom **0.87 Å** from a gold one. It is
   a monotone trade and 0.7 is the stiffest setting whose bonds stay
   inside the sp2 window (0.7: 1.90 Å clearance, 1.335-1.550 Å; 1.5:
   2.13 Å, 1.322-1.582). `relax_iterations` past 3000 changes none of
   these to three decimals.

**There is no setting that makes a coated wall as good as a free cage's,
and there should not be.** A shell conforming to a facetted core is
strained, and "strained" is the honest verdict rather than a failure.
What would be a failure is carbon inside the metal.

**What is NOT there, and was built to find out:** a one-sided constraint
cannot set the radius. The obvious scheme -- relax freely, push back the
atoms that entered the core, anchor only those, repeat -- oscillates:
pushing one atom out pulls its neighbours in, and asked to hold the whole
wall at 0.8 of a 3.35 Å gap it ended with 133 of 594 atoms still
violating, a 1.36 Å closest approach and 1.70 Å bonds -- worse on every
count than doing nothing. At a stiff anchor it has nothing to do and at a
soft one it destroys what the anchor achieved. A closed network's radius
is already fixed by its ring count; a repair loop cannot renegotiate it.

**Report the tails, not just the extremes.** The gap goes out as three
numbers (mean, 5th percentile, minimum) and the bonds as two (p95 and
max), because the extreme over a thousand bonds is a tail statistic. A
wall with p95 1.48 Å and max 1.58 Å is a good wall with a strained
corner, and judged on its maximum alone it reads as a failure.

**3.35 Å is a van der Waals contact, not a bond.** The carbide formers
(Fe, Co, Ni, Ti, W...) chemisorb near 2.1 Å, which is a different
interface with a different electronic answer. Both are offered, the
warning fires either way, and nothing here computes which one a given
system prefers.

### The chemistry goes on the shell, never on the core

Substituting a nitrogen for an iron is not doping, it is a different
compound, and `dopants/` is about what an sp2 carbon lattice tolerates.
So the dopant and the graft are applied **inside** the build rather than
after it -- the shell does not exist until the coating has run -- and
`jobs.build` has its own branch for that rather than routing them through
`apply_doping`. Groups go on the **outermost** shell's outer face: the
3.35 Å between two shells holds nothing, and a hydroxyl pointing into a
gold particle is not a structure.

Co-doping goes the same way, and it is wired because the window shows the
control: a control that is visible and silently ignored is worse than one
that is absent. `dopant` and `codope_spec` are refused together, for the
reason `Job` already gives -- running one substitution after another makes
the second fraction a fraction of what the first left.

### A cage is judged as a cage

A hollow particle is almost all surface **by design**. Judging one by the
solid thresholds is the mistake `sp2_quality` made when its hexagonal
window called every sound haeckelite BROKEN: the window was right and the
family was wrong. `hollow()` marks itself in `info` rather than being
inferred -- guessing from `operation == "difference"` would also catch the
concave cube, which is solid.

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

### A rim is not evidence against a surface, and a threshold said it was

`is_surface_net` gated face tracing on "**at least 90%** of bonded atoms
have degree 3". That is a tolerance standing in for a question, and it
was deciding real cases on its margin:

| structure | degrees | trivalent | old verdict |
|---|---|---|---|
| planar X junction | `{2: 80, 3: 700}` | 89.74% | **fell back** |
| tetrahedral node | `{2: 72, 3: 684}` | 90.48% | traced |
| nanocone | `{2: 50, 3: 420}` | 89.36% | **fell back** |
| nanoribbon 6x6 | `{2: 12, 3: 60}` | 83.33% | **fell back** |

The tetrahedral node passing by 0.48 of a point was luck. A **nanocone
and a nanoribbon are defined by having an edge**, and this file's own
nanocone section says so ("the rim is open on purpose"). Falling back is
not harmless: the X junction's census came out `{5: 4, 6: 328}` with
`sum(6-n) = +4` where its true census is `{5: 4, 6: 328, 7: 16}` and
`-12`. **All sixteen heptagons, silently** -- the exact failure mode two
paragraphs up, met on a structure this package builds and whose census it
already knows.

The test is now the question that distinguishes a surface: **no atom may
have more than three neighbours, and at least one must have three.**
Degree 1 or 2 is a rim, which an open surface legitimately has; degree 4
or more is a net that does not tile a surface at all. What must stay
excluded still does -- a bulk MX2 reaches degree 15 by the bond list, and
a carboxylated tube has degree-4 anchors so it still goes down the
backbone branch.

`trace_faces` needed no change: it always returned the orbits it
discarded for being too long, and an open surface's rim is exactly one of
those.

#### But a rim can be short enough to pass as a ring

`trace_faces` tells a face from the boundary by **size**, which is exact
only while the rim is longer than a ring may be. A ribbon is periodic
along its length, so it is an annulus -- `chi = 0`, therefore
`faces = E - V`:

| ribbon | V | E | E - V | faces | discarded |
|---|---|---|---|---|---|
| 6 wide, 3 long | 36 | 51 | 15 | **17** | **0** |
| 6 wide, 6 long | 72 | 102 | 30 | 30 | 2 |
| 8 by 8 | 128 | 184 | 56 | 56 | 2 |
| 10 by 10 | 200 | 290 | 90 | 90 | 2 |

Every row but the first discards its two rims and lands on Euler exactly.
The 6-by-3 ribbon's rims come out **six atoms long** -- a hexagon's size
-- so both are counted as rings and the census is two hexagons heavy.

There is no way to tell them apart by size, so the report **says so**
rather than guessing: a structure that has boundary atoms and discarded
*nothing* gets a caveat and `reliable = False`, the same signal a
collapsed cell gets. Nothing is subtracted, because which two of the
faces are the rims is exactly what is not known.

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

## A ring count cannot colour an atom

`ring_census` traced the faces of the graph and returned a **histogram**,
throwing the indices away. `build_cnt` and `build_nanoribbon` recorded
that count and nothing else, so `info["rings"]` was absent on both -- and
the window's `_ring_of_atom` falls back to 6 for every atom it finds in
no ring, so ring colouring painted the whole structure the hexagon
colour. On a pristine tube that looks right, which is why it survived;
on a defected one it is wrong and still looks deliberate.

`ring_faces` is the same trace with the indices kept, and `ring_census`
is now a histogram over it. Both builders record `rings` beside their
count, and a test asserts the two **agree** -- a census that contradicts
the rings beside it is only noticed after it has been plotted.

**This is face tracing, not ring perception.** Each directed bond has
exactly one successor round the local surface normal, so the walks are
the orbits of a permutation; it is what `analyse/rings.py` does and what
the census already did. It is *not* the distance / `networkx.cycle_basis`
route `fullerene_mesh` exists to prevent. The check that it is sound is
external rather than self-referential: a periodic tube's census is
**exact** -- Euler on the torus gives `F = E - V = N/2` -- and the trace
returns exactly that (60 faces for a 120-atom (6,6)).

### The refusal moved, and the new one is better

A test asserted that a plain tube raises "no ring metadata" when asked
to dope its pentagons. That refusal was right about the outcome and
wrong about the reason, and the reason is what a user acts on: a tube
does not lack ring information, it lacks pentagons. With the rings
recorded, `ring_sites` returns **empty** -- deliberate, and its docstring
already said so -- and `jobs.apply_doping` raises the useful form,
`"This structure has no 5-membered rings to dope. It has 6-rings: 96
sites."` The old message is still reachable and still tested: a plain
`build_graphene` sheet records no rings at all.

**Check which layer refuses before concluding a guard is gone.** The
first reading of this was that the change had removed a refusal, because
`ring_sites` returned quietly where it used to raise. The build still
refuses; it refuses one layer up, with a better message.

## Two panels, one attribute name, and a control that did nothing

The preview's colour box stopped working and nothing raised. The cause
is not in the colour code at all:

```python
self.var_colour_by = tk.StringVar(value="element")          # preview bar
...
self.var_colour_by = tk.StringVar(value=STYLE_DEFAULT_LABEL)  # render panel
```

The render panel is built **after** the preview, so its assignment
replaces the attribute. The preview's combobox holds its own reference
to the first `StringVar`, so clicking it still updated that object and
still fired `_redraw` -- while `_atom_colours` read
`self.var_colour_by`, which by then was the render panel's, whose value
is `"(style default)"`. Neither `"ring"` nor `"element"`, so every atom
fell through to the final `return [PLAIN_ATOM_COLOUR] * n`.

**And `PLAIN_ATOM_COLOUR` is `#5b6472`, which is exactly
`RING_COLOURS[6]`.** So the frame was indistinguishable from correct
ring colouring of an all-hexagon structure. Element colouring was
equally dead and nobody noticed, because an all-carbon structure is one
colour either way. A control that is visible and silently ignored is
worse than one that is absent -- the same rule the nanoparticle section
states for co-doping.

`test_every_attribute_read_is_assigned` cannot catch this: the
attribute **is** assigned, twice. The new test asks a different
question -- how many *methods* create a given `var_*` -- and an AST scan
of the whole window found exactly one clash, so the fault was bounded.
The preview's is `var_preview_colour` now. They answer different
questions anyway: the preview offers element/ring/plain and the render
panel the renderer's modes plus "(style default)".

## An imported file arrives with coordinates and nothing else

`analyse` computed the bond pairs and traced the rings for its report,
then returned the ring **summary** and dropped the indices
(`{k: v for k, v in rings.items() if k != "rings"}`). Nothing was
written back to `atoms.info`. So after opening somebody else's
`.extxyz` the panel showed 720 bonds and a full census while the
structure held neither: the preview drew a point cloud, ring colouring
and ring faces had nothing to read, and the render bundle would have
exported no connectivity. **The report was right and the structure was
empty**, which is why it reads as two unrelated faults.

`attach_inferred_graph` writes them under the keys every consumer
already reads, and the **provenance flag is the point**. This package's
discipline is that *recorded* and *inferred* must never be blurred, and
a guessed bond list under a builder's key would blur it exactly. So it
sets `info["graph_inferred"]`, `_recorded` reports `has_bond_graph` and
`has_ring_list` as **False** when that flag is present, and
`rings_are_recorded` stays False on a re-analysis. A test pins both
halves, with a builder-made tube as the control that nothing moved for
structures that really do record their graph.

Order matters and the call site says so: **analyse first, annotate
after**. `_recorded` reads `info`, so attaching beforehand would make
the report claim the file supplied what this module had just guessed.

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

## The relaxer had no door, for two releases

`relax_structure` -- the valence force field on a structure that already
exists -- and `write_relaxation_request` were written, tested and
reachable by **nothing but their own tests**. No subcommand, no button.
A structure somebody imported could be read, described and exported, and
not fixed. Worth recording because the gap was invisible from inside:
every test passed, the engine was correct, and the feature did not
exist for anyone who was not writing Python against the package.

The doors are `nanocarbon relax` / `nanocarbon relax-request` and the
**Relax geometry** panel under *Analyse a file*, which is where an
imported structure arrives.

Three things about that wiring are load-bearing:

* **The relaxation goes through the build worker, tagged.** It spends a
  minute inside L-BFGS on tens of thousands of atoms checking no cancel
  flag, which is the same reason `gui/worker.py` exists for builds.
  Requests now carry a kind (`BUILD` / `RELAX`); a **result** is still
  `done` or `error`, so nothing reading `poll()` changed. The dispatch
  lives in `run_request`, outside the child's loop, so a test can call
  it -- and an unknown kind **raises** rather than falling back to a
  build, because a build returned for a relax request looks exactly like
  a correct answer.
* **The window has to know which it asked for.** A build and a
  relaxation both come back as an `Atoms` under kind `done`, so
  `_poll_worker` branches on `self._task`. Without it a relaxation
  reported "Build complete" and wiped the before/after numbers that are
  the only reason to have pressed the button.
* **The angle targets are CARBON's, and now the engine says so.** Three
  neighbours means 120 deg here. On a dichalcogenide that is false -- a
  chalcogen with three metal neighbours sits near 82 deg and a
  six-coordinate metal has no single angle at all, which is why
  `tmd/curved.py` relaxes the site net and not the atoms. Pulling an
  MX2 to carbon's targets does not raise; it deforms the sandwich and
  leaves the bond lengths looking right, which is this package's
  signature failure mode. `relax_structure` warns when it sees an
  element that is not C or H and no `angle_override` was passed, and
  both doors surface it. The warning is in the library rather than in
  each caller, so a third door gets it for free.

Measured end to end on a 540-atom capped tube shoved off its geometry by
a parabola: bonds **1.282-1.546 -> 1.415-1.423 Å**, angles
**98.5-127.8 -> 107.8-119.9 deg**, zero contacts throughout, largest
single move 0.93 Å.

**The relaxed file does not carry its bond list**, and the command says
so instead of writing something that will not reopen: a list this long
in an extxyz comment line is what breaks ASE's own reader -- the same
fact `write_relaxation_request` already strips `info` for. Rerunning
`relax` on the output guesses the bonds from the distances, which on
relaxed geometry is reliable.

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

## A chalcogenide core is a solid, and its chalcogen is not a metal

`--model auto` draws space-filling spheres for the elements in
`SPACEFILL_ELEMENTS` and ball-and-stick for everything else, and that
table is **metals**, for the reason the constant records: a metal has no
directional bonds to draw. An FeSe or an MoS2 core is a solid too, and
its chalcogen is not a metal, so it came out open with sticks to the
iron -- which is not what the particle is.

* **`--spacefill-element Se` is a per-render declaration, not a table
  entry, and the asymmetry is the whole point.** The same selenium can
  be in two places at once: in the FeSe core, where it is solid, and
  substituted INTO the carbon shell around it, where ball-and-stick is
  right. **No per-element rule can tell those two populations apart**,
  so a structure carrying both cannot be rendered correctly this way at
  all, and one carrying only the first is free to say so. Adding Se to
  the table would silently inflate every Se-doped sheet in the package.
* It threads through `is_spacefill_element`, `model_radius` and
  `bond_is_drawn` as an `extra` argument with an empty default, so every
  existing call keeps its behaviour exactly; `atom_radii` and
  `build_ball_and_stick` pass it on. The tests pin both halves: a
  declared element grows and stops drawing sticks, and carbon beside it
  does not move.

## A shell relaxing against nothing sinks into the core it wraps

`graphitic_coat` wraps a core in a graphitic wall and then relaxes that
wall against the sp2 valence force field. **Nothing in that force field
knew the core was there.** The bond, angle and non-bonded terms all run
over the shell's own atoms, so equalising bond lengths is satisfied just
as well by a wall with half of itself inside the metal.

Measured on the FeSe and Fe3C particles, with the wall left free:
**0.15-0.28 Å** from core atoms, while reporting **1.50-1.53 Å** bonds.
Neither number trips a gate -- the bonds are perfect and the contact
check runs over the shell, not across the interface -- so it shipped.

* **`wall_anchor` was the only thing holding the wall out, and it pays
  for that in bonds.** Holding the wall on its own envelope along the
  normal keeps it clear of the core and stretches it: the same five
  structures came back at **1.59-1.72 Å**, outside the sp2 range, and
  the builder correctly called every one of them broken. Swept over gap
  x anchor x seed x geometry, **no combination satisfied both** -- which
  is the signature of a missing term rather than a tuning problem.
* **`obstacles` is that term**: fixed points the relaxing atoms are
  pushed off, which do not bond, do not move and take no gradient. The
  core goes in as obstacles and the anchor can then be zero. The five
  structures go to **1.504-1.526 Å**, four of five inside the sp2 range.
* **The obstacles must be rescaled into the relaxation's own frame**,
  and this is the part that looked like the term merely being weak.
  `_finish` rescales the dual onto the target bond length before
  relaxing (`positions * scale`), so obstacles handed over in the
  field's Ångström land where the core is *not*. `scale` is near 1 and
  not 1 -- the dual can arrive over 10 % off -- so the push was landing
  slightly beside its target and the effect read as a weak constant.
* **Only the core, not the finished shells.** Adding each shell to the
  obstacle list for the next one looks like the obvious completion and
  makes things worse: the shells are meshed independently, so
  layer-to-layer registry is accidental, and pushing every too-close
  pair apart fights the bonds. The two-shell carbide came back at
  1.27-1.96 Å and the builder refused it. The interlayer spacing already
  keeps shells apart; nothing keeps the first one out of the core.
* **The default is unchanged.** `wall_anchor` still defaults to 0.7, so
  no existing structure moves; a caller that passes the core as
  obstacles is the one that should also pass `wall_anchor=0`.

### And a second fault, localised but NOT fixed

With the bonds clean the core-shell distance is still wrong, and for a
different reason: **the shell is undersized against the envelope it was
meshed on.** Measured, the 86 Å rod's wall reaches 44.7 Å where its
envelope is at 47.0, and about **10 % of its atoms** then sit inside the
1.8 Å floor (171 of 1818; 168 of 1296 on the carbide).

That is the same `scale` step seen from the other side. The dual's mean
bond arrives off target, the whole wall is scaled to fix it, and a wall
with a fixed number of atoms has a fixed area -- so it can have the
right bond length or sit on its envelope, not both. The cure is to mesh
the envelope at the right density to begin with, so `scale` is already
1, rather than to rescale afterwards. `remesh` takes
`target_edge=sqrt(3)*bond` for exactly this and is evidently not
reaching it. **Do not "fix" this by undoing the scale**: that restores
the envelope and gives the bonds back their error.

## Which atoms are a solid is a region, not an element

`--model auto` draws space-filling spheres for the elements in
`SPACEFILL_ELEMENTS`, and `--spacefill-element` adds more for one
render. Neither can describe a **carbon-coated chalcogenide**, and the
reason is not a gap in the table: the same selenium is in the FeSe core,
where it is part of a solid, *and* substituted into the carbon shell
wrapping it, where it is one atom in a lattice. Asked by element, both
come out the same, and drawn as spheres the shell's dopant reads as a
crust of blobs on the graphite.

* `write_render_bundle` carries `info["solid_indices"]` into the sidecar
  as `solid`, and `atom_radii` and the bond loop read it. **A per-atom
  declaration beats the per-element one** where both are present, which
  is the right precedence: the builder knows which atoms it carved and
  which it meshed, and the element never will.
* It travels as **indices**, because that is what survives a file. The
  core is written first, so it is `range(len(core))` -- and the
  composition's arm, where the particle goes in last, is
  `range(len(tube), len(arm))`.
* The element rule stays as the fallback for a structure that carries no
  list, and a test pins that both seleniums then read alike -- which is
  the ambiguity, stated rather than hidden.
