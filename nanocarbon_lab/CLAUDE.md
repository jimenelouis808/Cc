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
├── tmd/           # MX2 dichalcogenides: materials.py (lattice constants),
│                  #   slab.py (mono/multi/bulk), ribbon.py, nanotube.py,
│                  #   coil.py (swept helical tubes), curved.py (schwarzites
│                  #   on a TPMS, with the M/X parity repair), modify.py
│                  #   (Janus, alloys, vacancies, antisites), quality.py.
│                  #   Deliberately NOT under builders/.
├── hetero/        # twisted bilayers and vdW stacks (moire.py). Above both
│                  #   builders/ and tmd/ because it composes them.
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
