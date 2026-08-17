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
| `builders`      | CNT (armchair / zigzag / chiral), graphene, nanoribbons, **nanocoils**, 3D carbon foam |
| `dopants`       | Substitutional N, B, S, P and co-doping, random / edges / bulk / cluster     |
| `defects`       | Mono- and divacancies, Stone-Wales, local random distortion                  |
| `topology`      | networkx-based bond graph, coordination, connectivity, ring statistics       |
| `validation`    | Minimum distances, coordination sanity, density, vacuum, cell consistency   |
| `exports`       | Quantum ESPRESSO `pw.x` input, LAMMPS data + script (AIREBO default)         |
| `relax`         | ASE optimizer wrapper + calculator-free harmonic pre-relaxation              |
| `viz`           | Matplotlib 3D viewer / PNG exporter                                          |
| `workflows`     | Batch sweeps + ML-ready dataset exporter (XYZ + features CSV + manifest)     |
| `cli`           | `nanocarbon` command-line entry point                                        |

## Installation

```bash
git clone <this repo>
cd nanocarbon_lab
pip install -e .[dev]
```

Python 3.10+ required. Dependencies: `numpy`, `scipy`, `ase`, `networkx`.

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

89 tests covering builders (including nanocoils and capped/defected
fullerene-CNTs), dopants, defects, topology, validation, exporters,
workflows and the bonus modules (relax / viz / ML dataset).

## Capped & defected CNTs for rendering (Blender / journal-cover art)

Everything above builds **open, infinitely-periodic** tubes -- correct for
DFT/MD, but with no ends and no way to host a genuinely closed-shell
defect. `builders.build_capped_cnt` instead builds a **finite, fully
closed shell**: a straight or gently bent cylindrical body terminated at
both ends by hemispherical fullerene domes, with pentagon/heptagon/octagon
defects composed in on request -- meant for exporting to `.xyz` and
rendering in Blender (or another external tool) for illustration/cover
art, not for DFT input decks.

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

```python
from nanocarbon_lab.builders import build_capped_cnt
from nanocarbon_lab.exports import write_render_bundle

cnt = build_capped_cnt(
    n_body_rings=12,       # body length
    freq=4,                 # diameter / lattice detail
    radius=7.0,              # Å
    bend_angle=0.4,           # radians, 0 = straight
    defects=[
        {"type": "stone_wales", "count": 2},  # 5-7-7-5 pairs
        {"type": "divacancy", "count": 1},     # 5-8-5 (octagon)
    ],
    seed=7,
)
print(cnt.info["ring_counts"])   # e.g. {5: 20, 6: 938, 7: 4, 8: 1}
write_render_bundle(cnt, "out/cnt_cap/demo")  # writes demo.xyz + demo.json
```

Or from the CLI:

```bash
nanocarbon cnt-cap --rings 12 --freq 4 --radius 7.0 --bend-angle 0.4 \
  --defect stone_wales:2 --defect divacancy:1 --seed 7 \
  --out out/cnt_cap/demo
```

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
├── tests/         # pytest suite
└── examples/      # runnable example scripts

blender/           # Blender-side rendering pipeline (run via `blender -b -P ...`)
├── styles.py       # journal-cover style presets (pure Python, no bpy)
├── mesh_builder.py # bpy/bmesh: XYZ+JSON -> coloured ball-and-stick / smooth-surface mesh
└── render_cnt.py   # CLI driver: world/lighting/camera + render to PNG
```

## License

MIT.
