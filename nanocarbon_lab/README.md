# nanocarbon_lab

Modular Python framework to **generate, validate and export** realistic
nanocarbon structures (1D, 2D, 3D) for first-principles (Quantum ESPRESSO)
and classical molecular dynamics (LAMMPS) simulations.

Everything that is stochastic — dopant placement, vacancy choice, foam
construction — is **seeded and reproducible**. Every exporter runs a
validation pass before writing.

## Features

| Module          | What it does                                                                 |
|-----------------|------------------------------------------------------------------------------|
| `builders`      | CNT (armchair / zigzag / chiral), graphene, nanoribbons, **nanocoils**, 3D carbon foam, **capped/defected fullerene-CNTs** |
| `dopants`       | Substitutional N, B, S, P and co-doping, random / edges / bulk / cluster     |
| `defects`       | Mono- and divacancies, Stone-Wales, local random distortion                  |
| `topology`      | networkx-based bond graph, coordination, connectivity, ring statistics       |
| `validation`    | Minimum distances, coordination sanity, density, vacuum, cell consistency   |
| `exports`       | Quantum ESPRESSO `pw.x` input, LAMMPS data + script (AIREBO default), XYZ + Blender render bundle |
| `relax`         | ASE optimizer wrapper + calculator-free harmonic pre-relaxation              |
| `viz`           | Matplotlib 3D viewer / PNG exporter                                          |
| `workflows`     | Batch sweeps + ML-ready dataset exporter (XYZ + features CSV + manifest)     |
| `cli`           | `nanocarbon` command-line entry point                                        |
| `gui`           | `nanocarbon-gui` desktop app: sliders, live 3D preview, export, Blender render |
| `implicit`/`remesh`/`junction` | L/T/Y/X nanotube junctions and periodic schwarzite unit cells from implicit surfaces |
| `swept`         | Coils and arbitrary curved tubes whose ring topology is **derived from the curvature** |
| `fullerene`     | Closed cages (C60, C240, C540, C20, C80, …) and carbon nano-onions |
| `assemblies`    | Multi-wall nanotubes and hexagonally packed bundles, at the van der Waals gap |
| `tmd`           | **Transition-metal dichalcogenides** (MoS2, WS2, …): monolayers, stacks, bulk crystals, ribbons, rolled nanotubes, helical coils and **schwarzites**; Janus layers, alloys, vacancies and antisites |
| `hetero`        | **Twisted bilayers and van der Waals stacks**: commensurate moiré cells (including the 1.08° magic angle), graphene/hBN/MX2 in any combination |

## Installation

```bash
git clone <this repo>
cd nanocarbon_lab
pip install -e .[dev]
```

Python 3.10+ required. Dependencies: `numpy`, `scipy`, `ase`, `networkx`,
`scikit-image` (marching cubes, for junctions and schwarzites).
For the desktop app add `pip install -e ".[gui]"` (matplotlib) — plus
`python3-tk` on Linux, see [Graphical interface](#graphical-interface).

## Quick start — Python API

```python
from nanocarbon_lab.builders import build_cnt
from nanocarbon_lab.dopants   import dope_random
from nanocarbon_lab.defects   import introduce_vacancies
from nanocarbon_lab.exports.qe     import write_qe_input, QESettings
from nanocarbon_lab.exports.lammps import write_lammps

cnt = build_cnt(n=6, m=6, length=12.0)           # (6,6) armchair, ~12 Å
cnt = dope_random(cnt, "N", 0.03, seed=42)        # 3% N substitutional
cnt = introduce_vacancies(cnt, n_defects=1, seed=42)

write_qe_input(cnt, "out/cnt/qe", settings=QESettings(calculation="relax"))
write_lammps  (cnt, "out/cnt/lammps")
```

## Quick start — CLI

```bash
# (6,6) armchair CNT, 12 Å long, export to QE and LAMMPS
nanocarbon cnt --n 6 --m 6 --length 12 --out out/cnt --format both --calculation relax

# 4x4 graphene supercell, 3% N doping, write QE input
nanocarbon graphene --nx 4 --ny 4 --dopant N --dopant-conc 0.03 --out out/gr --format qe

# Zigzag nanoribbon, 6 wide, 3 long, passivated
nanocarbon ribbon --width 6 --length 3 --edge zigzag --passivate --out out/ribbon --format qe

# Carbon nanocoil: (6,6) SWCNT wound into a helix, R=25 Å, pitch=12 Å, 1.5 turns
nanocarbon nanocoil --n 6 --m 6 --coil-radius 25 --pitch 12 --turns 1.5 --out out/coil --format both --force

# 3D carbon foam, LAMMPS only (relaxation recommended before DFT)
nanocarbon foam --box 30 --flakes 25 --radius 4 --seed 0 --out out/foam --format lammps

# Validate any ASE-readable structure file
nanocarbon validate out/cnt/qe/pw.in
```

Structures for rendering (XYZ + Blender render bundle, plus CIF for the
periodic ones). All of these share `--anneal-sweeps`, `--roughness`,
`--dopant`, `--dopant-conc` and `--seed`:

```bash
# Capped tube swept onto a left-handed conical coil, CVD-rough, 2% N-doped
nanocarbon cnt-cap --rings 6 --freq 3 --shape helix \
  --helix-radius 90 --helix-pitch 30 --helix-handedness left --helix-taper 0.7 \
  --roughness 0.2 --dopant N --dopant-conc 0.02 --out out/coil_swept

# Nanocoil whose rings follow the curvature (slower, no elastic strain)
nanocarbon coil --coil-radius 30 --pitch 20 --turns 1.5 --tube-radius 6 --out out/coil

# Y junction, annealed smooth
nanocarbon junction --kind Y --tube-radius 6 --arm-length 22 --out out/junction

# Periodic gyroid schwarzite unit cell (writes .cif too)
nanocarbon schwarzite --kind gyroid --cell 26 --out out/gyroid

# C60, and a C60@C240@C540 nano-onion
nanocarbon fullerene --family C60 --freq 1 --out out/c60
nanocarbon onion --shells 3 --out out/onion

# Two-wall nanotube and a 7-tube rope
nanocarbon mwcnt --shells 2 --inner-freq 3 --rings 10 --out out/mwcnt
nanocarbon bundle --shells 1 --freq 3 --rings 10 --out out/rope
```

Dichalcogenides (MoS2, WS2, …) have their own sub-commands, since none of
the sp2 flags above mean anything for them:

```bash
# MoS2 monolayer and bilayer, 2H (trigonal prismatic) or 1T (octahedral)
nanocarbon tmd --material MoS2 --layers 1 --phase 2H --out out/mos2_mono
nanocarbon tmd --material MoS2 --layers 2 --stacking 2H --out out/mos2_bi

# Bulk crystal, 2H or 3R
nanocarbon tmd-bulk --material MoS2 --stacking 2H --out out/mos2_bulk

# Nanoribbon with a chosen zigzag edge chemistry
nanocarbon tmd-ribbon --material MoS2 --width 8 --termination metal --out out/mos2_rib

# Rolled (40,0) nanotube
nanocarbon tmd-tube --material MoS2 --n 40 --m 0 --out out/mos2_nt

# Helical coil — a swept tube, so the atom count follows the coil radius
nanocarbon tmd-coil --material MoS2 --n 30 --coil-radius 220 --pitch 90 \
    --turns 0.25 --out out/mos2_coil

# Schwarzite on a minimal surface. --parity trades M/X alternation
# against bond strain; 'split' makes every ring even.
nanocarbon tmd-schwarzite --material MoS2 --kind primitive --cell 36 \
    --parity flip --out out/mos2_sw
```

Stacks of 2D layers — twisted or aligned, and mixing families freely:

```bash
# Twisted bilayer graphene. The angle is snapped to the commensurate
# series and the achieved value reported; 1.1 gives the 1.0845° magic angle.
nanocarbon twist --layer graphene --angle 21.8 --out out/tbg
nanocarbon twist --layer graphene --angle 1.1 --max-index 40 --out out/magic

# Graphene on hBN, twisted. The 1.8% mismatch is strained onto the top layer.
nanocarbon twist --layer graphene --top-layer hBN --angle 7.3 --out out/g_hbn

# Aligned stack, no moiré, so the cell stays primitive
nanocarbon stack --layers graphene hBN graphene --nx 2 --ny 2 --out out/sandwich
```

Every one of these prints an **sp2 verdict** alongside the measured bond
and angle statistics — `CLEAN`, `STRAINED` or `BROKEN`, with the reason.
It exists because "0 close contacts" is not the same as "physical": an
over-tight coil keeps its atoms apart while stretching its bonds to
1.69 Å, which is longer than even an sp3 C–C bond.

## Nanocoils

`build_nanocoil` generates a helical CNT by mapping a straight `(n, m)`
segment onto a helix with configurable **coil radius** `R`, **pitch** `P`
and **number of turns**. The arc length is set exactly to
`n_turns · √((2πR)² + P²)` so the underlying CNT is neither stretched nor
compressed on average; bond-length distortion stays below ~`r_tube/R` (a
few percent for `R ≥ 25 Å`).

Parameters:

| Argument              | Meaning                                                        |
|-----------------------|----------------------------------------------------------------|
| `n, m`                | Chirality of the underlying CNT                                |
| `coil_radius`         | Helix radius (Å), must be ≥ 2× the tube radius                 |
| `pitch`               | Vertical advance per turn (Å)                                  |
| `n_turns`             | Number of helical turns (float — 1.5 gives 1½ loops)           |
| `stone_wales_density` | Fraction of bonds to SW-rotate, biased to the outer wall (≤0.02) |

Post-construction, apply `relax.harmonic_pre_relax` or an ASE calculator to
relieve the bending strain before DFT / MD.

```python
from nanocarbon_lab.builders import build_nanocoil
from nanocarbon_lab.relax    import harmonic_pre_relax

coil = build_nanocoil(n=6, m=6, coil_radius=25.0, pitch=12.0, n_turns=1.5,
                      stone_wales_density=0.005, seed=0)
harmonic_pre_relax(coil, steps=300)
```

## Pre-relaxation

`relax.relax_with_calculator(atoms, calc, algorithm='lbfgs', fmax=0.05)`
runs BFGS / L-BFGS / FIRE against any ASE calculator (LAMMPS via
`LAMMPSlib`, DFT, M3GNet / MACE / GAP…).

`relax.harmonic_pre_relax(atoms)` is calculator-free: it fixes the
topology once (bond graph via covalent radii), then minimises a
Hooke-spring potential toward `bond = 1.42 Å`. Handy for foams and
coils before handing the structure to a proper force field.

## ML dataset export

`workflows.write_ml_dataset(jobs, root)` runs a batch of jobs and writes:

* `structures/<name>.xyz` (extended XYZ per structure),
* `features.csv` with one row per structure — composition, density,
  dimensionality, mean coordination, ring statistics (3 through 8) and
  per-element counts — ready for scikit-learn / pandas ingestion,
* `manifest.json` with validation outcomes and `atoms.info` payloads.

## Visualisation

```python
from nanocarbon_lab.viz import save_structure_png
save_structure_png(atoms, "coil.png", view=(20, 45))
```

Matplotlib-based 3D scatter with bond lines. Intended for quick QA —
for publication rendering export to `.xyz`/`.cif` and use OVITO / VESTA.

## Typical workflow

1. **Build.** Pick a builder and parameters; obtain an `ase.Atoms` with
   correct `pbc` and vacuum padding.
2. **Customise.** Compose `dopants.*` and `defects.*` — they return new
   `Atoms` objects and accept a `seed` for reproducibility.
3. **Validate.** `nanocarbon_lab.validation.run_basic_checks(atoms)` returns
   a structured report (errors, warnings, info dict). Exporters call this
   automatically and refuse to write bad structures unless `force=True`.
4. **Export.** Use `exports.qe.write_qe_input` or `exports.lammps.write_lammps`.
   Dimensionality, vacuum and k-mesh are inferred from `pbc` and the cell.
5. **Scale up.** Build a list of `workflows.BatchJob` (or use the
   `batch_cnt_sweep` helper) and call `workflows.write_dataset(jobs, root)`.
   You get one folder per structure and a `dataset.json` manifest with
   geometry, formula, validation outcome and export paths.

## Integration with QE and LAMMPS

### Quantum ESPRESSO

The writer produces a complete `pw.in` with `&CONTROL`, `&SYSTEM`,
`&ELECTRONS`, and, when relevant, `&IONS` / `&CELL`. `ATOMIC_SPECIES`,
`ATOMIC_POSITIONS`, `CELL_PARAMETERS` and `K_POINTS` cards are included.

Auto-detected from the structure:

| Detected                         | Effect                                |
|----------------------------------|---------------------------------------|
| 0 periodic axes                  | 1×1×1 k-mesh, `assume_isolated='mp'`  |
| 1 periodic axis                  | nₖ×1×1 along that axis                |
| 2 periodic axes                  | nₖ×nₖ×1, `assume_isolated='2D'`       |
| 3 periodic axes                  | full MP mesh                          |

Defaults: `ecutwfc=60 Ry`, `ecutrho=480 Ry`, `smearing='mv'`, PAW pseudos
(edit `QESettings.pseudopotentials` to override).

### LAMMPS

The writer produces a `data.lammps` (box, tilt factors if needed, masses,
atomic coordinates) and an `in.lammps` that performs:

1. Minimisation (`minimize`),
2. NVT equilibration (Nosé-Hoover),
3. Optional NPT stage.

Default pair style for pure-carbon systems: `airebo 3.0 1 1` with the
`CH.airebo` parameter file. Customise via `LAMMPSSettings`.

## Running the test suite

```bash
pytest -q
```

105 tests covering builders (including nanocoils and capped/defected
fullerene-CNTs), dopants, defects, topology, validation, exporters,
workflows, the GUI, and the bonus modules (relax / viz / ML dataset).

The capped-CNT tests assert **physical** quality, not only topology:
bond lengths in 1.30-1.55 Å, bond angles in 100-135 deg, and zero
non-bonded contacts below 2 Å, across straight, bent and defected cases.
GUI tests run headlessly under `xvfb-run` and skip cleanly when tkinter
or a display is unavailable.

## Capped & defected CNTs for rendering (Blender / journal-cover art)

Everything above builds **open, infinitely-periodic** tubes -- correct for
DFT/MD, but with no ends and no way to host a genuinely closed-shell
defect. `builders.build_capped_cnt` instead builds a **finite, fully
closed shell**: a straight or gently bent cylindrical body terminated at
both ends by hemispherical fullerene domes, with pentagon/heptagon/octagon
defects composed in on request -- meant for exporting to `.xyz` and
rendering in Blender (or another external tool) for illustration/cover
art, not for DFT input decks.

### Graphical interface

```bash
pip install -e ".[gui]"
nanocarbon-gui            # or: python -m nanocarbon_lab.gui
```

Eight structure types from one dropdown — **capped tube**, **coil
(relaxed)**, **fullerene**, **nano-onion**, **junction**, **schwarzite**,
**multi-wall** and **bundle** — with only the panels that apply to the
current one shown, in a scrolling column so nothing falls off a laptop
screen.

**Every number is typeable.** Each parameter is a slider *and* an entry
box, and the box accepts values outside the slider's comfortable range,
up to the builder's real limit. A coil radius of 37.5 Å is a perfectly
good request that no slider detent will land on; so is 640 Å.

**Builds can be cancelled.** They run in a separate process, not a
thread, precisely so that Cancel can terminate one — a coil spends
minutes inside numpy with nothing checking a flag, and Python cannot
safely interrupt a thread. If the environment cannot start a process
(a frozen executable, a restricted sandbox), the GUI falls back to a
thread automatically and says that Cancel is then only advisory.

**You are told the cost before you commit.** The estimate line predicts
the atom count — exactly for the seed-polyhedron modes, within a few per
cent for the meshed ones — and flags whether you are asking for a
tenth of a second or six minutes.

Twelve **presets** (C60, nano-onion, gyroid, Y junction, seven-tube rope,
CVD-rough tube, …) are one click away, and any parameter set can be saved
to and loaded from JSON. **Copy equivalent CLI command** puts the exact
`nanocarbon …` invocation on the clipboard, so anything found by dragging
sliders can be scripted, reproduced, or pasted into a methods section.

The preview colours atoms by ring type and can **hide ring sizes** —
switching the hexagons off is the fastest way to see where the curvature
actually went, since on a junction or a coil the 5s and 7s are a handful
of atoms buried in thousands. A **session history** keeps every build so
a promising result is not lost the moment you nudge the next parameter.

The readout reports ring counts, the Euler check, wall or shell spacing
where it applies, achieved coil dimensions, the measured
bond/angle/contact statistics and the **sp2 verdict**. Errors appear in a
panel that can be read and copied, never a modal dialog — a modal blocks
the event loop and throws away the parameters you were about to fix.

Live hints do the arithmetic before you build: the coil panel reports how
much tube a given radius/pitch/turns consumes and what outer-wall strain
that implies (green / amber / red), taper included — a conical spring is
judged at its **tightest** end, since that is where the wall gives way.

`tkinter` is part of the standard library but ships separately on some
Linux distributions -- `sudo apt install python3-tk` (Debian/Ubuntu) or
`sudo dnf install python3-tkinter` (Fedora). The python.org installers for
Windows and macOS already include it. Everything the GUI does is also
available from `nanocarbon cnt-cap`.

#### Windows, step by step

1. Install **Python 3.10+** from [python.org](https://www.python.org/downloads/)
   (not the Microsoft Store build). Tick **“Add python.exe to PATH”** on the
   first screen of the installer. tkinter is included.
2. Unzip the project, then in PowerShell:

   ```powershell
   cd path\to\nanocarbon_lab
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -e ".[dev,gui]"
   nanocarbon-gui
   ```

   If PowerShell blocks the activation script, either run
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use
   `cmd.exe` with `.venv\Scripts\activate.bat` instead.
3. Optional, for rendering: install **Blender** from
   [blender.org](https://www.blender.org/download/). The Windows installer
   does **not** add Blender to `PATH`, so the GUI also searches
   `C:\Program Files\Blender Foundation\Blender *\blender.exe`
   automatically. If you installed it somewhere else (portable build, Steam),
   use the **“Locate Blender…”** button, or set a `BLENDER` environment
   variable pointing at `blender.exe`.

Command line on Windows is the same, with backslashes in paths:

```powershell
nanocarbon cnt-cap --rings 10 --target-radius 7.8 --defect stone_wales:2 --out out\demo
blender -b -P blender\render_cnt.py -- --xyz out\demo.xyz --json out\demo.json --style nature_dark --out out\cover.png
```

### Why the ring topology is guaranteed correct

A capped nanotube is topologically an **elongated fullerene**: a closed,
3-coordinate carbon shell. Euler's polyhedron theorem forces
`sum(6 - ring_size)` over every ring to equal exactly `+12`, always. In
this framework:

* **pentagons** carry positive curvature → **convex** points (the two end
  caps, 6 pentagons each, 12 total for a plain capped tube);
* **heptagons** carry negative curvature → **concave/saddle** points
  (paired with pentagons at a Stone-Wales 5-7-7-5 defect, or at a bend);
* **octagons** appear at a reconstructed divacancy (5-8-5: one octagon
  flanked by two pentagons).

Rather than editing the honeycomb lattice directly (easy to silently
build an invalid ring pattern), `builders.fullerene_mesh` works one level
down on the honeycomb's **triangulated dual** (each ring = one mesh
vertex; a degree-5/6/7/8 vertex *is* a pentagon/hexagon/heptagon/octagon).
Local defects become simple, providably-correct combinatorial edits
(`edge_flip` = Stone-Wales, `contract_edge` = divacancy) that can never
produce a topologically inconsistent structure -- see the module
docstring for the full construction and `tests/test_capped_cnt.py` for
the Euler-invariant assertions.

### Why the geometry is physically realistic

Correct topology is not enough: a structure can have perfect ring counts
and still be geometrically absurd. The shell is therefore relaxed against
a **valence force field** -- bond stretching toward 1.42 Å, *true* angle
bending toward 120 deg, and short-range non-bonded repulsion -- minimised
with L-BFGS using exact analytic gradients (verified against finite
differences to ~1e-9).

Two details matter, and both were found by measuring rather than
assuming:

* **A real angle term is essential.** Bond springs alone, or a 1-3
  *distance* proxy for angles, leave the sheet free to pyramidalise and
  fold: an earlier fixed-step relaxer produced near-perfect bond lengths
  alongside 66-164 deg bond angles and dozens of sub-2 Å contacts.
* **The mesh is smoothed before the dual is taken.** Barycentric
  subdivision yields quite unequal triangles; taking the dual first and
  moving atoms afterwards leaves hexagons so uneven that beyond ~1000
  atoms the optimiser cannot recover and the shell folds through itself.
  Projecting and Laplacian-smoothing the *mesh* onto the capsule first
  fixes this.

A clean capped tube now relaxes to **1.415-1.423 Å** bonds and
**107.8-120.0 deg** angles -- the 107.8 deg floor being exactly the
interior angle of the cap pentagons, which is the correct physical answer.
Every structure carries its own measured statistics in
`atoms.info["geometry"]`, so you can assert on quality instead of
trusting the builder.

### Diameter is quantised, not free

The body circumference must fit a whole number of hexagons, so the radius
follows from `freq`: `R = 5 * freq * sqrt(3) * bond / (2*pi)`, about
`1.96 * freq` Å. This is the same constraint that fixes a real `(n, m)`
nanotube's diameter from its chiral indices. Pass `target_radius` to have
the nearest realisable `freq` chosen for you, then read back
`atoms.info["radius"]`.

```python
from nanocarbon_lab.builders import build_capped_cnt
from nanocarbon_lab.exports import write_render_bundle

cnt = build_capped_cnt(
    n_body_rings=12,          # body length
    target_radius=7.8,         # Å -- picks freq=4; exact value reported back
    bend_angle=0.4,             # radians, 0 = straight (max 1.0)
    defects=[
        {"type": "stone_wales", "count": 2},  # 5-7-7-5 pairs
        {"type": "divacancy", "count": 1},     # 5-8-5 (octagon)
    ],
    seed=7,
)
print(cnt.info["ring_counts"])  # e.g. {5: 18, 6: 938, 7: 4, 8: 1}
print(cnt.info["geometry"])     # bond/angle/contact statistics
write_render_bundle(cnt, "out/cnt_cap/demo")  # writes demo.xyz + demo.json
```

Or from the CLI:

```bash
nanocarbon cnt-cap --rings 12 --target-radius 7.8 --bend-angle 0.4 \
  --defect stone_wales:2 --defect divacancy:1 --seed 7 \
  --out out/cnt_cap/demo
```

### Curved, coiled and randomly meandering tubes

`shape=` sweeps the whole tube along a 3D centreline — `"arc"`,
`"s_curve"`, `"helix"` or a seeded `"random"` meander (`"straight"` is the
default). `waviness` (0–1) sets how far the path wanders and
`shape_points` how many wiggles it has.

```bash
nanocarbon cnt-cap --rings 40 --freq 2 --shape random \
  --waviness 1.0 --shape-points 10 --seed 11 --out out/wavy
```

**How curved a tube can be is set by physics, not by taste.** The outer
wall of a bent tube is stretched by roughly `r_tube × κ`, so the builder
computes that strain and trims the path's amplitude until it fits a
budget, rather than letting you request something the lattice cannot
survive. Measured on a 3200-atom tube:

| strain | outcome |
|---|---|
| ~8% (default) | clean sp2: 1.33–1.51 Å bonds, no close contacts |
| 12–16% | bonds stretched but structure intact — fine for artwork |
| ≥20% | bonds past 1.6 Å; no longer physically meaningful |
| ~100% | 2.2 Å "bonds" and overlapping atoms |

Raise `max_strain` past 0.15 and the builder warns you that you have left
the physical regime. Since strain is `r_tube × κ`, **a thinner, longer
tube curves far more dramatically at the same strain** — for a strongly
meandering cover image use a low `freq` and a high `rings` (e.g.
`--freq 2 --rings 40`) rather than turning `waviness` up on a fat tube.

Two implementation details matter and are unit-tested: the path is
parameterised by **arc length** (otherwise the tube stretches where the
spline runs fast), and the cross-section is carried by a
**rotation-minimizing frame** rather than a Frenet frame — the Frenet
normal flips 180° at every inflection point, and a random meander is full
of them, which would shear the tube apart.

Bending via `bend_angle` (a single planar arc, kept for compatibility) is
imposed as an arc-length-preserving sweep after the straight relaxation,
then re-relaxed with both caps restrained. Past roughly
0.6 rad the outer wall is visibly stretched -- which is real elastic
strain -- and beyond 1.0 rad the request is rejected outright, because a
real nanotube buckles into a localised kink rather than straining
uniformly, and this smooth-arc model does not represent that.

This writes `demo.xyz` (plain XYZ, readable by any molecular viewer) and
`demo.json` (a sidecar with explicit bonds and per-atom ring membership,
consumed by the Blender pipeline below to colour pentagons/heptagons/
octagons differently).

### Rendering in Blender

`blender/` (repo root, sibling to `nanocarbon_lab/`) is a small,
self-contained Blender Python pipeline -- run **through Blender itself**,
not a normal `python` interpreter, since `bpy`/`bmesh` only exist there:

```bash
blender -b -P blender/render_cnt.py -- \
  --xyz out/cnt_cap/demo.xyz --json out/cnt_cap/demo.json \
  --style nature_dark --mode ballstick \
  --out out/cnt_cap/cover.png --resolution 2000 2400 --samples 256
```

* `--mode ballstick` builds coloured spheres + cylinders (good for
  close-ups where the lattice/defects should read clearly);
  `--mode surface` instead skins the bond graph into one continuous
  glossy tube (Blender's Skin + Subdivision modifiers -- good for wide
  macro shots where the tube's silhouette is the hero); `--mode both`
  does both.
* `--style` selects a full look (materials + world background + lighting
  rig + camera lens/DOF) from `blender/styles.py`. Five presets ship out
  of the box -- run `blender -b -P blender/render_cnt.py -- --list-styles`
  to print their descriptions:

  | style | mood |
  |---|---|
  | `nature_dark` | matte-black void, single dramatic key + cool rim light, near-black graphite body, glowing defect accents |
  | `acs_nano_vivid` | saturated blue-magenta gradient backdrop, glossy clear-coated colour, punchy 3-point lighting |
  | `small_minimal` | clean white seamless backdrop, large soft studio lights, matte pastel body |
  | `blueprint_technical` | deep navy void, thin glowing emissive bonds/defects, circuit-diagram mood |
  | `gold_nanotech` | warm near-black backdrop, polished gold/bronze metal, 3-point studio rig |

  All five colour pentagons/heptagons/octagons with a style-appropriate
  accent so curvature and defects are visually legible against the
  hexagonal body -- edit `blender/styles.py` (plain dataclasses, no
  Blender dependency, so it's editable/testable outside Blender) to add
  your own or tweak an existing one.
* Pass `--transparent-background` for a PNG you can composite over other
  art; drop `--samples` to use the style's own default (Cycles path
  tracing, GPU device selected automatically when available).

## Junctions (L, T, Y, X) and schwarzites

Branched and sponge-like carbon comes from a different route than the
tube builder: an **implicit surface**, meshed and remeshed, with the
topology left to follow from the geometry.

```bash
nanocarbon junction --kind Y --tube-radius 6 --arm-length 22 --out out/yj
nanocarbon schwarzite --kind gyroid --cell 32 --out out/gyroid
```

```python
from nanocarbon_lab.builders import build_junction, build_schwarzite

y = build_junction("Y", tube_radius=6.0, arm_length=22.0, blend=4.0)
print(y.info["ring_counts"])   # e.g. {5: 50, 6: 443, 7: 38}
print(y.info["genus"])          # 0 — capped, so sphere-like

g = build_schwarzite("gyroid", cell=32.0)
print(g.info["genus"])          # 5 — five handles
print(all(g.get_pbc()))          # True — a real periodic cell
```

The pipeline is: signed-distance field → marching cubes → **isotropic
remeshing** → dual → valence-force-field relaxation. The remeshing step is
the one that matters. In the dual, **a mesh vertex of degree d becomes a
carbon ring of size d**, and raw marching cubes produces degrees from 3 to
9 — three-membered rings and nine-membered holes. Repeated
split/collapse/flip/smooth passes pull the degrees onto 6, leaving 5s and
7s only where curvature demands them.

Nothing tells the code that a Y junction needs heptagons. The branch is a
saddle, saddles carry negative Gaussian curvature, and that emerges from
the remesher as degree-7 vertices. Euler's theorem then holds on its own:

| structure | genus | `sum(6 - ring_size)` | character |
|---|---|---|---|
| capped tube | 0 | +12 | 12 pentagons close the two caps |
| L / T / Y / X junction | 0 | +12 | pentagons at arm tips, heptagons at the neck |
| gyroid unit cell | 5 | −48 | saddles everywhere; heptagons outnumber pentagons |
| Schwarz D unit cell | 9 | −96 | more handles, more heptagons |

Because a schwarzite is *supposed* to break the "+12" rule, the builder
reads the expected budget from the mesh's own Euler characteristic
instead of assuming it, and refuses to emit a structure that disagrees.

### Periodic schwarzite unit cells

`build_schwarzite` returns a **genuinely periodic** `ase.Atoms`
(`pbc=True`, cubic cell), not a clipped fragment: one period of the
surface is meshed and then welded onto itself across the cell faces, so
the tubes run out of one face and back in the opposite one exactly as in
the published structures. Tile it 2x2x2 to see the network continue.

Getting the textbook genus out of the mesh is the check that the weld
really closed:

| surface | genus | `sum(6 - ring_size)` | minimum cell |
|---|---|---|---|
| Schwarz P (`primitive`) | 3 | −24 | 30 Å |
| gyroid | 5 | −48 | 36 Å |
| Schwarz D (`diamond`) | 9 | −96 | 36 Å |

**Two corrections here, both from measuring rather than assuming**, and
together they are what fixed schwarzites coming out visibly defective.

*Annealing makes them worse, not better.* On a junction, a stray
pentagon–heptagon pair is genuine disorder and flip annealing helps
(39 → 14 on a Y). On a triply periodic **minimal** surface it is not
disorder at all: the surface saddles everywhere, and 5–7 pairs are the
mechanism by which a hexagonal net covers that Gaussian curvature.
Anneal them away and the remaining lattice has to stretch to cover the
same curvature instead. Sweeping every surface and cell size, the
`sp2 verdict` got monotonically worse with annealing:

| Schwarz P, 36 Å | 0 sweeps | 20 | 80 |
|---|---|---|---|
| stray 5–7 pairs | 41 | 12 | 9 |
| longest bond | 1.519 Å | 1.534 Å | 1.575 Å |
| verdict | **clean** | strained | broken |

So `build_schwarzite` defaults to `anneal_sweeps=0`, and the GUI turns
the shared annealing slider off when you switch to schwarzite mode. A
high stray count on these structures is a sign the surface is being
tiled correctly, not something to polish out.

*The minimum cells were too low.* They had been set at the point where
the mesh stopped tearing, which let through cells that passed the tear
gate and still relaxed to bonds well outside the sp2 range — Schwarz P
at 24 Å gives 1.328–1.560 Å, Schwarz D at 30 Å gives 1.343–1.586 Å. The
limits above are now the smallest cell whose *geometry* is not broken.
Bigger is better throughout: larger cells curve more gently, so if a
result still looks strained, widen the cell before anything else.

Two things make the periodic path work, and both were found by measuring
rather than assuming. **Everything downstream is minimum-image**: the
remesher, the dual, the relaxation and the geometry report all measure
across the cell seam, or a bond wrapping the boundary reads as a
cell-length stretch. And the **cell is relaxed along with the atoms** —
holding it fixed leaves the network unable to reach 1.42 Å bonds, which
on the denser Schwarz D surface showed up as 6 Å "bonds" and 28 atomic
overlaps.

Whether a given neck is resolved also depends on how the surface falls
between marching-cubes sample points, so an isolated `(cell, resolution)`
pair can fail where both its neighbours are fine — the gyroid at 26 Å did
exactly that. That is discretisation, not physics, so the builder retries
with a shifted grid before giving up.

### Coils with real dimensions

`shape="helix"` takes `helix_radius` and `helix_pitch` in **Å**, and sizes
the tube to the coil's arc length rather than squeezing the coil onto
whatever tube you asked for — so the dimensions you request are the ones
you get, and `n_body_rings` is derived.

```python
coil = build_capped_cnt(
    shape="helix", helix_radius=60.0, helix_pitch=25.0,
    helix_turns=1.5, freq=2,
)   # 570 Å of tube, 6.6% wall strain
```

Because the other shapes are qualitative, they get trimmed to the strain
budget; a coil given explicit dimensions is a specification, so it is
honoured and the resulting strain is **reported and warned about**
instead. That matters here: outer-wall strain is `r_tube / R_coil`, so a
25 Å coil around a 4 Å tube is already at 16% and visibly degrades. Real
carbon nanocoils have coil radii of hundreds of Å for exactly this
reason.

### Two ways to bend a tube, and when to use each

`build_capped_cnt(shape=...)` **sweeps** a finished all-hexagon tube onto
a curved centreline. The lattice is unchanged, so the bend is carried
entirely as elastic strain — and that is a hard geometric fact, not a
relaxation failure: a pure-hexagon tube bent onto an arc *must* have a
longer outer wall than inner. On a 90 Å coil around a 5.9 Å tube, 6.5%
path strain leaves bonds spanning 1.33–1.51 Å, and weakening the
positional restraints only lets atoms wander (1 Å drift) without
recovering a hundredth of an Ångström of bond length.

`build_coil` / `build_swept_tube` take the **implicit** route instead —
the same one junctions and schwarzites use. The curved tube is built as a
signed-distance surface, meshed, and remeshed; ring sizes then follow the
curvature, with pentagons on the compressed inner wall and heptagons on
the stretched outer wall, exactly as real coiled nanotubes relieve the
strain. Bonds come back to graphitic length.

```python
from nanocarbon_lab.builders import build_coil

coil = build_coil(coil_radius=30.0, pitch=20.0, turns=1.5, tube_radius=6.0)
coil.info["ring_counts"]            # e.g. {5: 85, 6: 1854, 7: 71, 8: 1}
coil.info["achieved_coil_radius"]   # measured off the relaxed atoms
```

| | swept (`shape="helix"`) | implicit (`build_coil`) |
|---|---|---|
| speed | seconds | minutes |
| caps | exact 6-pentagon domes | derived from the surface |
| tight curvature | bonds stretch out of range | absorbed by 5–7 pairs |
| dimensions | honoured exactly | radius honoured, pitch relaxes |

Use the swept route for gentle curves and when you need exact dimensions
fast; use the implicit route when the curvature is tight enough that the
`sp2 verdict` (printed by the CLI and shown in the GUI) reports
`BROKEN`. Budget minutes, not seconds: a 2800-atom coil takes about
6 minutes to mesh, remesh and relax. The GUI runs it on a worker thread
so the window stays responsive, and the tests that build one are marked
`slow` (`pytest -m "not slow"` skips them).

Two things about the implicit coil are worth knowing. Its pitch must
clear two tube walls plus a graphitic gap or the turns merge into one
solid, and the builder refuses rather than emitting that.

And ring topology encodes a tube's **curvature** but not its **torsion**.
So the coil radius comes back as asked — 29.4 Å measured against a
requested 30.0 — while the pitch is a soft mode free to move. How far it
moves depends on how hard the coil is working: a tight one (30 Å coil,
6 Å tube) opens from 20 Å to 26.7, while a gentler one (34 Å coil, 4.5 Å
tube) barely shifts, 20 Å to 22. `pin_ends=True` holds the axial length by
restraining the end caps, but it fights the relaxation where the network
is most distorted: the same coil then came out with 1.18–1.71 Å bonds and
three overlapping atom pairs, failing the quality gate the free
relaxation passes cleanly. The default is therefore to let the coil find
its own pitch and report it in `atoms.info["achieved_pitch"]`, rather
than hold a requested number at the cost of the chemistry.

**One honest limitation.** The remesher settles into a local minimum
containing scattered pentagon–heptagon pairs along the arms, beyond the
ones curvature requires — roughly 39 spurious pairs on a Y junction.
These are dislocations: topologically neutral, geometrically sound (bond
and angle statistics stay in range), and genuinely present in
CVD-grown junctions, so the structures are realistic rather than
idealised. Raising `remesh_iterations` barely helps (38 → 35 pairs for 5×
the work), but **flip annealing** does: `anneal_sweeps=80` takes a Y
junction from 39 pairs to 14–15 in about a second. Set `anneal_sweeps=0`
to keep the as-grown wall.

### Surface finish: smooth or CVD-rough, on purpose

Two independent knobs, on every builder and in the GUI's *Surface finish*
panel:

* `anneal_sweeps` — Metropolis flip annealing of the mesh. `0` keeps the
  as-remeshed defect population (rougher, as-grown); `80` removes the
  strays curvature does not require.
* `roughness` — RMS out-of-plane corrugation in Å, applied *after*
  relaxation by displacing each atom along its local surface normal and
  re-settling. Normal-direction displacement is the soft one; isotropic
  jitter merely strains bonds and gets undone. Topology is untouched.

Measured on a capped tube: σ = 0 → 0.001 Å radial RMS; σ = 0.3 → 0.133 Å
with bonds still at 1.387–1.469 Å; σ = 0.5 → 0.193 Å at 1.365–1.496 Å.
The wall looks CVD-grown and stays chemically valid throughout.

### Fullerene cages and nano-onions

A fullerene is the `half_length = 0` limit of the capped tube — a sphere
instead of a capsule — so it reuses the same dual/Euler/force-field
machinery. What is specific is the **seed**, because on a sphere the seed
decides which cage you get:

```python
from nanocarbon_lab.builders import build_fullerene, build_nano_onion

c60 = build_fullerene(freq=1, family="C60")     # 12 pentagons, 20 hexagons
onion = build_nano_onion(n_shells=3)            # C60@C240@C540, 840 atoms
```

| family | seed | series | atoms | radius step |
|---|---|---|---|---|
| `"C60"` | pentakis dodecahedron | GP(f,f), class II | 60, 240, 540, 960 | ~3.5 Å |
| `"C20"` | icosahedron | GP(f,0), class I | 20, 80, 180, 320 | ~2.0 Å |

Two seeds rather than one, because **C60 is not in the class-I series at
any frequency** — an icosahedron seed cannot produce the one fullerene
everybody actually wants. Both seeds are built as the convex hull of
points on the unit sphere, which is the honest way to triangulate a point
set on a sphere: the hull *is* the spherical Delaunay triangulation, so
the connectivity cannot be got wrong by hand-listing faces.

Measured, relaxed: C60 comes out at radius 3.52 Å (literature 3.55) with
every bond at 1.420 Å and angles spanning exactly 108.0°–120.0° — 108°
being the pentagon's interior angle, which is the right answer rather
than a coincidence.

The radius step is why `"C60"` is the default for onions. At ~3.5 Å per
frequency it lands within a tenth of an Ångström of graphite's 3.4 Å, so
C60@C240@C540 nests at the physical spacing with no fudge: **shell
spacing 3.48 Å, closest approach 3.37 Å**. The class-I series steps by
only ~2.0 Å and no `freq_step` combination reaches 3.4. (Contrast the
multi-wall *tube* below, where the lattice quantises radius in 1.96 Å
steps and 3.9 Å is the closest realisable wall spacing — the spherical
family is the luckier geometry.)

As with multi-wall tubes, shells are relaxed independently and then
nested: the covalent force field has no dispersion term, so what holds an
onion together is precisely the term the model lacks, and relaxing the
assembly as a whole would collapse the cages into one another. The
achieved spacing is measured and reported.

### Multi-wall tubes and bundles

```python
from nanocarbon_lab.builders import build_bundle, build_multiwall_cnt

mwcnt = build_multiwall_cnt(n_shells=3, inner_freq=3, n_body_rings=10)
rope = build_bundle(n_rings_across=1, freq=3, n_body_rings=10)   # 7 tubes
```

Neither is a new topology — each shell or tube is an ordinary capped
tube, so each pays its own 12-pentagon Euler budget and the assembly's
counts are simply their sum. What makes them their own objects is the
**van der Waals spacing**, which the covalent relaxation knows nothing
about, so the builders place independently-relaxed shells and *measure*
the separation rather than assuming it.

Measuring it correctly took two attempts worth recording. Excluding only
bonded pairs reported 2.29 Å (that is a pentagon's 1–3 diagonal);
excluding 1–2 and 1–3 pairs reported 2.79 Å (a hexagon's 1–4 diagonal).
Both are intra-wall distances that have nothing to do with the gap. The
separation is now measured **between known shell index ranges**, giving
3.41 Å for a MWCNT and 3.43 Å in a rope, and `nan` for a lone tube —
which is the honest answer when there is no second wall.

The lattice quantises tube radius in ~1.96 Å steps, so a MWCNT's wall
spacing cannot land on graphite's 3.35 Å exactly; `freq_step=2` gives
3.9 Å, the closest realisable value, and the achieved spacing is reported
in `atoms.info["wall_spacing"]`.

### Other free/open tools for this pipeline

Blender is the right choice specifically because the request was for
*artistic* control (arbitrary shaders, compositing, non-photorealistic
looks) -- but depending on what you need, these are worth knowing about
too, all free and open source:

* **[OVITO](https://www.ovito.org/)** (Basic edition is free) reads
  `.xyz` directly with zero setup and has a built-in ambient-occlusion +
  Tachyon ray-traced renderer that already produces publication-quality
  stills -- the fastest path from `.xyz` to a clean figure if you don't
  need Blender-level artistic control.
* **[Molecular Nodes](https://bradyajohnston.github.io/MolecularNodes/)**
  is a free, actively-maintained Blender add-on (geometry-nodes based)
  purpose-built for importing molecular/crystal structures with
  ball-and-stick or surface representations and node-based styling. This
  repo ships its own minimal `bpy`/`bmesh` pipeline instead (no add-on
  dependency, full control over the ring-type colouring described above),
  but Molecular Nodes is a strong alternative or complement if you want a
  GUI-driven workflow inside Blender.
* **VMD** + its Tachyon renderer is the traditional computational-chemistry
  choice, scriptable in Tcl, and widely used for MD trajectory rendering.
* **POV-Ray** is a free ray tracer some structure tools (incl. OVITO) can
  export directly to, useful for scripted/batch rendering without a full
  3D DCC tool.

## Transition-metal dichalcogenides (MoS2, WS2, …)

A TMD is **not** a decorated graphene. It is a three-plane X–M–X sandwich:
a metal layer between two chalcogen layers, held to its neighbours by van
der Waals forces alone. Almost nothing from the carbon side transfers —
the metal is six-coordinate rather than three, the bond is 2.4 Å rather
than 1.42, and there are no rings to count — so this lives in its own
`nanocarbon_lab.tmd` package with its own quality verdict.

```python
from nanocarbon_lab.tmd import (
    build_tmd_monolayer, build_tmd_layers, build_tmd_bulk,
    build_tmd_ribbon, build_tmd_nanotube,
)

mono   = build_tmd_monolayer("MoS2", phase="2H")        # 2D
bilayer= build_tmd_layers("MoS2", n_layers=2, stacking="2H")
bulk   = build_tmd_bulk("MoS2", stacking="2H")          # 3D
ribbon = build_tmd_ribbon("MoS2", width=8, termination="metal")   # 1D
tube   = build_tmd_nanotube("MoS2", n=40, m=0)          # 1D
```

Thirteen materials ship with measured lattice constants: MoS2, MoSe2,
MoTe2, WS2, WSe2, WTe2, NbSe2, TaS2, TiS2, ZrS2, HfS2, PtS2, SnS2.

### Geometry is two numbers, and the third is derived

Each material stores the in-plane lattice constant `a` and the
chalcogen–chalcogen separation `h`. The M–X bond follows,
`d = sqrt(a²/3 + h²/4)`, rather than being stored alongside them — three
stored numbers is an invitation for them to disagree. For MoS2 that gives
**2.404 Å against the 2.41 Å usually quoted**, and the built bulk cell
comes out at a = 3.160 Å, c = 12.294 Å against the measured 12.29.

### Phases: what actually differs

| phase | metal coordination | chalcogen planes | typical |
|---|---|---|---|
| `2H` | trigonal prismatic | eclipsed, both over the same column | MoS2, WS2 (semiconducting) |
| `1T` | octahedral | staggered by 60° | TiS2, PtS2 (metallic) |
| `1T'` | distorted octahedral | staggered, metals dimerised | metastable MoS2 |

That is the whole difference: **where the bottom chalcogen plane sits**.
1T' adds a metal–metal pairing on top, and the builder reproduces the
2.80 Å Mo–Mo dimer against the undistorted 3.16 Å.

**On "tetragonal": no TMD has one.** 2H, 1T and 1T' all sit on a
hexagonal (or, for 1T', a slightly distorted hexagonal) lattice. What
varies between them is the coordination polyhedron around the metal, not
the crystal system — and the 2H/1T distinction is almost certainly what
is meant when a paper contrasts "the hexagonal phase" with another one.

### Stacking

`2H` (AA′) alternates a 180° rotation, `3R` shifts each layer by (1/3,
2/3) without rotating, `AA` is eclipsed. 2H and 3R both put the metal of
one layer over the chalcogen of the next and are **indistinguishable as
bilayers** — they diverge at the third layer, ABAB against ABCABC, which
is what makes them different crystals rather than different slabs.

### Ribbons: the edge is a chemical choice

Graphene's zigzag edge is just a zigzag edge. MX2's comes in two forms,
metal-terminated or chalcogen-terminated, and they differ in nearly
everything that matters — the metal edge is metallic and magnetic, and it
is what gives CVD-grown MoS2 its triangles. So `termination` is an
explicit argument, and a terminated ribbon is **deliberately
off-stoichiometry** (Mo₁₂S₂₀, not Mo₁₂S₂₄); the quality check is told to
expect that rather than calling it broken.

### Nanotubes: why MX2 tubes are big

Rolling graphene costs only bond bending. Rolling a TMD puts the outer
chalcogen plane on a larger circle than the inner one, and the strain is
geometric and unavoidable:

    outer plane strain = h / 2R

For MoS2 (h = 3.13 Å) that is **15.6% at R = 10 Å and 5.2% at R = 30 Å**.
This is the reason real MoS2 nanotubes are tens of nanometres across
where carbon ones are one — the same curvature costs far more. The
builder warns past 10% and the verdict reports the resulting bond spread.

### MX2 coils

A coil is a nanotube swept onto a helical centreline with a
rotation-minimizing frame. Carbon has two routes to a coil — sweep a
finished tube and carry the bend as elastic strain, or mesh the curved
surface so ring sizes follow the curvature. For a dichalcogenide only the
first is available, and that is chemistry rather than a missing feature:
the implicit route absorbs curvature by introducing odd rings, and an odd
ring in MX2 forces an M–M or X–X bond. Sweeping keeps every ring a
hexagon and every bond M–X, and pays in strain.

Two strains stack, and they have opposite cures:

| strain | formula | set by |
|---|---|---|
| roll | `h / 2R_tube` | the tube's own radius |
| bend | `R_outer × κ` | how tightly it is coiled, `κ = R/(R² + (p/2π)²)` |

Widening the tube cuts the first and **raises** the second, so they cannot
be minimised together — the coil radius is the free parameter. A (30,0)
MoS2 coil at R = 220 Å spends 10.4% on the roll and 7.6% on the bend; a
coil under 6% total is genuinely enormous, which is why the builder
reports both numbers separately rather than their sum. The atom count
follows the helix arc length, so it grows with the coil radius rather than
with anything that reads as a size: that same coil is 11 340 atoms at a
quarter turn.

### MX2 schwarzites

MX2 on a triply periodic minimal surface, periodic in all three
directions. `nanocarbon tmd-schwarzite --material MoS2 --kind primitive
--cell 36 --parity flip --out out/mos2_sw`.

An earlier version of this file said this was impossible because
pentagons are forbidden. Pentagons *are* forbidden — M and X alternate,
so only even rings are allowed — but that argument only rules out the
*sphere*. Negative curvature never needed pentagons: it needs
**octagons**, which are even. `sum(6 - n) = 6χ`, each octagon pays −2, so
Schwarz P (χ = −4) wants twelve. h-BN schwarzites in the literature are
built exactly this way.

Atoms sit at triangle centroids and bond across shared mesh edges, so
ring size equals mesh-vertex degree and the M/X question becomes "is
every degree even". Two moves get there, and they trade against each
other:

* an **edge flip** toggles four degrees at once, so with odd vertices
  sparse it can only shuffle them — it never reaches zero, but it adds no
  vertices and the geometry survives;
* an **edge split** toggles exactly the two vertices opposite the edge.
  That is the weight-two move that *does* reach zero, at one new vertex
  each, and enough of them crowd the surface with more sites than its
  area holds at spacing `a/√3`.

Measured on Schwarz P, MoS2 (all with zero atomic overlaps and
coordination 6/3 throughout):

| cell | `--parity` | atoms | homoelemental | X/M | M–X p95 | worst |
|---|---|---|---|---|---|---|
| 30 Å | `none` | 758 | 12.4% | 1.893 | 6.2% | 12.2% |
| 30 Å | `split` | 921 | **0.0%** | **2.0000** | 13.4% | 30.6% |
| 36 Å | `none` | 1066 | 11.1% | 1.978 | 3.9% | 8.4% |
| 36 Å | `split` | 1298 | 2.2% | 1.991 | 11.5% | 25.0% |
| 48 Å | `none` | 1856 | 10.4% | 1.994 | 2.4% | 4.6% |
| 48 Å | `split` | 2264 | 3.1% | 1.995 | 7.9% | 24.8% |

Two things worth reading off that table. **A perfectly alternating,
exactly stoichiometric MoS2 schwarzite exists** — the 30 Å split cell has
zero homoelemental bonds and X/M = 2.0000. And it is not guaranteed: even
degrees are a sphere result, and at genus 3 there are six more Z/2
classes that even-degree meshes sit on both sides of. When the repair
lands on the wrong side, the leftover bonds are a real **inversion-domain
boundary** — the line defect seen in grown MoS2 and h-BN — so the count
is reported rather than assumed away.

Larger cells curve more gently and cut the bond strain (p95 6.2% → 2.4%
from 30 to 48 Å) without much helping the parity, so the two knobs are
worth turning independently.

The verdict here does **not** go through the usual `tmd_quality`. That
finds bonds by distance, which is right for a slab and wrong on a saddle:
a 2.4 Å bond's cutoff reaches 3.0 Å and sweeps up non-bonded neighbours,
reading a sound cell as 4–8 coordinate metal. The builder keeps the real
bond graph and judges from that.

### Twisted bilayers and moiré superlattices

Two hexagonal sheets share a periodic cell only at special angles. For
integers `m > n ≥ 0`, `m·a1 + n·a2` in one layer is exactly as long as
`n·a1 + m·a2` in the other, so

```
cos θ = (m² + n² + 4mn) / (2(m² + mn + n²))
```

and the cell holds `m² + mn + n²` primitive cells **per layer**. That is
not a curiosity: `(2,1)` is 21.79°, and `(31,30)` is **1.0845° — the
magic angle** of twisted bilayer graphene, where the moiré bands flatten
and the system superconducts.

So a twist is not a free parameter. Ask for 1.1° and you get 1.0845°,
with 11 164 atoms and a 129.9 Å moiré period, because nothing periodic
exists in between. The builder snaps and reports both.

| request | achieved | (m,n) | cells/layer | atoms |
|---|---|---|---|---|
| 21.8° | 21.7868° | (2,1) | 7 | 28 |
| 13.2° | 13.1736° | (3,2) | 19 | 76 |
| 7.3° | 7.3410° | (5,4) | 61 | 244 |
| 1.1° | **1.0845°** | (31,30) | 2791 | **11 164** |

Layers are described uniformly, so any pair stacks: graphene, hBN, and
every dichalcogenide. Unlike layers have different lattice constants, so
a common cell needs one of them strained — graphene on hBN is 1.8%, which
is routinely modelled; graphene on MoS2 would be 28%, and that is refused
rather than silently returned.

`stack` is the untwisted counterpart: with every layer aligned there is
no moiré, the cell stays primitive, and an arbitrary number of layers
costs nothing.

### MX2 chemistry after the build

`tmd.modify` edits a finished structure's chemistry without touching its
geometry, so they compose in any order — build a tube, make it Janus,
alloy it, knock sulphur out of it:

* **Janus** (`make_janus`) — one chalcogen species on top, another
  underneath, as in MoSSe. Breaks the mirror symmetry, which switches on
  an out-of-plane dipole and piezoelectricity neither parent has. Which
  face a chalcogen belongs to is decided from its own outward direction,
  not from a global z, so this works on a **rolled tube** too: `side=+1`
  is the outer wall and `-1` the inner one.
* **Chalcogen vacancies** (`chalcogen_vacancies`) — the dominant point
  defect in grown MoS2. `paired=True` removes both sides of a site.
* **Alloys** (`alloy`) — Mo(1−x)W(x)S2 and friends, on either sublattice.
  The **achieved** fraction is reported, since nine metal sites cannot be
  split in half.
* **Antisites** (`antisites`) — a metal on a chalcogen site, as in
  sulphur-poor growth.

### Not yet: Y junctions in MX2

The remaining gap. Same machinery as the schwarzite — an implicit surface
meshed and decorated — so it is now a wiring job rather than a research
one, but the junction field needs its caps handled and it is untested
here.

## Repository layout

```
nanocarbon_lab/
├── builders/      # structure generators (CNT, graphene, ribbon, nanocoil, foam,
│                  #   capped/defected fullerene-CNTs: capped_cnt.py + fullerene_mesh.py)
├── dopants/       # substitutional chemistry
├── defects/       # vacancies, Stone-Wales, distortion
├── topology/      # networkx connectivity / rings
├── validation/    # structural checks
├── exports/       # QE + LAMMPS writers, plain XYZ + Blender render bundle
├── relax/         # ASE optimizers + harmonic pre-relaxation
├── viz/           # matplotlib 3D viewer / PNG exporter
├── workflows/     # batch generation + metadata + ML dataset
├── utils/         # constants, geometry, RNG
├── cli/           # command line
├── gui/           # tkinter desktop app (build, preview, export, render)
├── tests/         # pytest suite
└── examples/      # runnable example scripts

blender/           # Blender-side rendering pipeline (run via `blender -b -P ...`)
├── styles.py       # journal-cover style presets (pure Python, no bpy)
├── mesh_builder.py # bpy/bmesh: XYZ+JSON -> coloured ball-and-stick / smooth-surface mesh
└── render_cnt.py   # CLI driver: world/lighting/camera + render to PNG
```

## License

MIT.
