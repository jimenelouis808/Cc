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

## Installation

```bash
git clone <this repo>
cd nanocarbon_lab
pip install -e .[dev]
```

Python 3.10+ required. Dependencies: `numpy`, `scipy`, `ase`, `networkx`.
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

Sliders for length, diameter, bend and bond length; spinboxes for how many
Stone-Wales and divacancy defects to scatter in; a live 3D preview coloured
by ring type; a panel reporting ring counts, the Euler check and the
measured bond/angle/contact statistics; and buttons to save the
`.xyz` + `.json` bundle or drive Blender directly. Builds run on a worker
thread, so the window stays responsive while a few-thousand-atom shell
relaxes.

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

Bending is imposed as an arc-length-preserving sweep after the straight
relaxation, then re-relaxed with both caps restrained. Past roughly
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
