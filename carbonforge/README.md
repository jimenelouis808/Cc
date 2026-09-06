# carbonforge

Build realistic nanocarbon structures (1D, 2D, 3D) and prepare **complete
first-principles calculations** for them — band structure, Raman and infrared
spectra, spin-orbit coupling — targeting **Quantum ESPRESSO**, **SIESTA** and
**LAMMPS**.

Two properties hold throughout:

* Everything stochastic (dopants, vacancies, foams) is **seeded and
  reproducible**.
* Nothing is written without validation. Beyond geometry, carbonforge checks
  the **physics of the calculation itself** and refuses setups that cannot
  work — Raman on a metallic nanotube, spin-orbit with scalar-relativistic
  pseudopotentials, `vc-relax` on a slab whose vacuum would collapse.

## Start here

```bash
carbonforge presets
carbonforge ribbon --width 6 --length 3 --edge zigzag --preset bands --out run
```

That builds a nanoribbon, **notices its edges are magnetic**, switches on
spin polarisation with an antiferromagnetic guess, relaxes the geometry,
computes the bands on the *relaxed* structure, sizes the k-point pools for
your core count, and writes down every decision in `DECISIONES.txt`.

A preset adapts to the structure it is given. On an armchair ribbon it leaves
spin off, because that one does not need it.

## Features

| Module          | What it does                                                                 |
|-----------------|------------------------------------------------------------------------------|
| `builders`      | CNT (armchair / zigzag / chiral), graphene, nanoribbons, **nanocoils**, 3D carbon foam |
| `dopants`       | Substitutional N, B, S, P and co-doping, random / edges / bulk / cluster     |
| `functionalization` | Functional groups (-NH2, -NO2, -OH, -COOH, epoxide…) and the four nitrogen lattice configurations |
| `defects`       | Mono- and divacancies, Stone-Wales, local random distortion                  |
| `topology`      | networkx-based bond graph, coordination, connectivity, ring statistics       |
| `validation`    | Geometry checks **plus** calculation-level physics checks                    |
| `exports`       | Quantum ESPRESSO (`pw.x`/`ph.x`/`bands.x`), SIESTA `.fdf`, LAMMPS           |
| `calculations`  | Band paths, phonon/IR/Raman, DOS, spin/vdW/hybrids, spin-orbit               |
| `relax`         | ASE optimizer wrapper + calculator-free harmonic pre-relaxation              |
| `viz`           | Matplotlib 3D viewer / PNG exporter                                          |
| `results`       | Parse and plot finished runs: bands, DOS/PDOS, IR/Raman spectra              |
| `exports.pseudos` | Which pseudopotentials a run needs, and whether you have them              |
| `workflows`     | Presets, chained relax→property pipelines, sweeps, ML datasets               |
| `gui`           | Tkinter desktop app with live 3D preview (`carbonforge-gui`)                  |
| `cli`           | `carbonforge` command-line entry point                                        |

> **¿Primera vez?** La [**Guía rápida en español**](GUIA_RAPIDA.md) explica
> paso a paso cómo instalarlo y usarlo en tu portátil, incluida la interfaz
> gráfica. Para modificar el código, la
> [**Guía de desarrollo**](DESARROLLO.md) cubre tests, dónde tocar cada cosa
> y qué falta validar contra datos reales.

## Installation

One command, on either platform:

```bash
./install.sh          # Linux / macOS
install.bat           # Windows (double-click or run from cmd)
```

Both create a `.venv`, install the package, verify Tkinter is present for the
GUI and run the test suite. Manual route if you prefer:

```bash
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

Python 3.10+ required. Dependencies: `numpy`, `scipy`, `ase`, `networkx`,
`matplotlib`.

## Graphical interface

```bash
carbonforge-gui
```

A desktop app with two tabs. **Construir estructura** picks a structure type,
tunes its parameters, shows a live 3D preview plus the geometry and physics
reports, and exports to QE / SIESTA / LAMMPS / XYZ / CIF. **Analizar
resultados** opens a finished calculation — a band file or `dynmat.out` — and
plots it inline, with the same warnings the CLI gives.

The build tab exposes functional groups and lattice nitrogen in their own
panel, kept visually separate because they are different chemistry.

Structures are built on a worker thread, so the window stays responsive on
large models. Tkinter is required — it ships with Python on Windows and
macOS; on Linux install `python3-tk` (it cannot be installed with pip). The
app prints platform-specific instructions if it is missing.

## Quick start — Python API

```python
from carbonforge.builders import build_cnt
from carbonforge.dopants   import dope_random
from carbonforge.defects   import introduce_vacancies
from carbonforge.exports.qe     import write_qe_input, QESettings
from carbonforge.exports.lammps import write_lammps

cnt = build_cnt(n=6, m=6, length=12.0)           # (6,6) armchair, ~12 Å
cnt = dope_random(cnt, "N", 0.03, seed=42)        # 3% N substitutional
cnt = introduce_vacancies(cnt, n_defects=1, seed=42)

write_qe_input(cnt, "out/cnt/qe", settings=QESettings(calculation="relax"))
write_lammps  (cnt, "out/cnt/lammps")
```

## Quick start — CLI

```bash
# (6,6) armchair CNT, 12 Å long, export to QE and LAMMPS
carbonforge cnt --n 6 --m 6 --length 12 --out out/cnt --format both --calculation relax

# 4x4 graphene supercell, 3% N doping, write QE input
carbonforge graphene --nx 4 --ny 4 --dopant N --dopant-conc 0.03 --out out/gr --format qe

# Zigzag nanoribbon, 6 wide, 3 long, passivated
carbonforge ribbon --width 6 --length 3 --edge zigzag --passivate --out out/ribbon --format qe

# Carbon nanocoil: (6,6) SWCNT wound into a helix, R=25 Å, pitch=12 Å, 1.5 turns
carbonforge nanocoil --n 6 --m 6 --coil-radius 25 --pitch 12 --turns 1.5 --out out/coil --format both --force

# 3D carbon foam, LAMMPS only (relaxation recommended before DFT)
carbonforge foam --box 30 --flakes 25 --radius 4 --seed 0 --out out/foam --format lammps

# Validate any ASE-readable structure file
carbonforge validate out/cnt/qe/pw.in
```

## Calculations

Beyond writing a structure, carbonforge prepares the calculation you actually
want to run.

### Band structure

```python
from carbonforge.exports.qe import write_qe_bands
write_qe_bands(atoms, "out/bands")     # scf -> bands -> bands.x + run script
```

The high-symmetry path is chosen from the Bravais lattice via ASE, and is
dimensionality-aware: a CNT gets Γ-X along its periodic axis, hexagonal
graphene gets Γ-M-K-Γ, and an *orthogonal* graphene supercell correctly gets
Γ-X-S-Y-Γ instead — a distinction that silently ruins hand-written paths.

### Raman and infrared

```python
from carbonforge.calculations import raman_setup
from carbonforge.exports.qe import write_qe_spectroscopy
write_qe_spectroscopy(atoms, "out/raman", raman_setup())  # scf -> ph.x -> dynmat.x
```

Three prerequisites are checked before anything is written, because each
one otherwise kills the job hours into a queue:

| Requirement | Why |
|---|---|
| A band gap | `epsil=.true.` (Born charges) is undefined for metals. Armchair CNTs and pristine graphene fail here. |
| Norm-conserving pseudopotentials | QE's DFPT Raman does not support PAW or ultrasoft — which are the defaults. |
| q = Γ | IR and Raman intensities are only defined at the zone centre. |

Frequencies alone (`phonon_setup()`) have none of these restrictions and work
fine on metals with PAW.

### Spin-orbit coupling

```python
from carbonforge.calculations import soc_setup
from carbonforge.exports.qe import QESettings, write_qe_input
write_qe_input(atoms, "out/soc", settings=QESettings(spinorbit=soc_setup()))
```

Sets `noncolin` / `lspinorb` and rewrites the pseudopotential names to their
`rel-` counterparts. Two honest caveats are raised automatically: scalar
pseudopotentials give **exactly zero splitting with no error**, and SOC in
pure carbon is ~10⁻² meV — far below what a routine DFT run resolves. It
becomes interesting with heavy adatoms (Au, Bi, Pb).

### SIESTA

```python
from carbonforge.exports.siesta import SiestaSettings, write_siesta
write_siesta(atoms, "out/siesta", settings=SiestaSettings(run_type="bands"))
```

A complete `.fdf`: species, lattice, coordinates, k-grid (1 along vacuum
axes), basis, functional, band lines. Note SIESTA has **no DFPT**: phonons
come from frozen force constants (`MD.TypeOfRun FC` + the `vibra` utility),
and there is no Raman implementation — for that, use Quantum ESPRESSO.

## Functional groups and nitrogen

Two different chemistries, deliberately kept apart:

**Attached groups** hang off a carbon, at an edge or on the basal plane:

```bash
carbonforge groups                       # list what is available
carbonforge ribbon --width 6 --length 3 --group NH2 --group-count 2 \
                   --task bands --out out/amino --format all
```

Available: `-H`, `-OH`, `-NH2`, `-NO2`, `-C≡N`, `-COOH`, `-CHO`, `-CONH2`,
`=O`, `-SH`, `-CH3` and the bridging epoxide. Edge attachment is the ordinary
case; `--group-site basal` forces the anchor carbon to sp3, which is what
graphene oxide is.

**Lattice nitrogen** sits inside the ring system, and is *not* the same thing:

```bash
carbonforge graphene --nx 6 --ny 6 --nitrogen pyridinic --nitrogen-count 2 \
                     --out out/pyridinic
carbonforge nitrogen-report structure.xyz
```

| Configuration | N coordination | How it is built |
|---|---|---|
| Graphitic (quaternary) | 3 | Substitutes a basal carbon |
| Pyridinic | 2 | Vacancy first, then N on the rim |
| Pyrrolic | 2 (+H) | Five-membered ring — see caveat below |
| Pyridinic N-oxide | 2 (+O) | Pyridinic plus O on the nitrogen |

They separate in N 1s XPS (≈398, 400, 401 and 402 eV respectively) and dope
the material differently, so "5 % N" without saying which says very little.

**The pyrrolic caveat:** a true pyrrolic site needs a *five-membered ring*,
and that reconstruction is driven by energy minimisation, not geometry.
`make_pyrrolic_like` builds the composition and neighbourhood but leaves
six-membered rings, labels the result `pyrrolic_precursor`, and says so in
the metadata. Relax it, then check with `ring_statistics` that a pentagon
actually formed.

Everything here produces **idealised, unrelaxed** geometries. Groups rotate
about their single bonds and interact with neighbours; relax before drawing
conclusions.

## Batch sweeps over any structure

```python
from carbonforge.workflows import batch_structure_sweep, write_dataset
from carbonforge.builders import build_nanoribbon
from carbonforge.functionalization import functionalize_random
from functools import partial

jobs = batch_structure_sweep(
    build_nanoribbon,
    {"width": [4, 6, 8], "edge": ["zigzag", "armchair"], "length": [3]},
    post_factory=lambda params, seed: [
        partial(functionalize_random, group_key="NH2", n_groups=2, seed=seed)
    ],
)
write_dataset(jobs, "out/sweep")
```

## Pseudopotentials

carbonforge writes pseudopotential *names* but cannot ship the files. This
tells you exactly which ones you need, why, and where to get them — then
checks your directory:

```bash
carbonforge pseudos structure.xyz --raman --spinorbit --dir ./pseudo
```

The family follows from what you are computing: Raman forces
norm-conserving, spin-orbit forces fully-relativistic, and asking for both
lands you in PseudoDojo's `nc-fr` tables. When an exact filename is missing
but another file for that element is present, it is offered as a possible
substitute — never silently used, since that is your call.

## Analysing results

carbonforge does not run anything — it writes inputs and reads outputs. Once
your job has finished:

```bash
carbonforge plot-bands    bands.dat    --labels G,M,K,G --out bands.png
carbonforge plot-spectrum dynmat.out   --kind raman --laser 532 --out raman.png
```

Or from Python:

```python
from carbonforge.results import read_qe_bands, read_dynmat
from carbonforge.results.bands   import plot_bands
from carbonforge.results.spectra import plot_spectrum

bands = read_qe_bands("bands.dat")
print(bands.band_gap(fermi=-4.2))       # None when metallic

spectrum = read_dynmat("dynmat.out")
print(spectrum.summary())               # warns about imaginary modes
plot_spectrum(spectrum, "raman", laser_wavelength_nm=532.0, temperature_k=300.0)
```

`spectrum.summary()` flags two things worth catching early: **imaginary
modes**, which mean the structure sits at a saddle point rather than a
minimum and invalidate the spectrum; and an **acoustic-mode count other than
three**, which usually means the acoustic sum rule was not applied.

Raman and IR columns are *activities*. Converting them to something
comparable with an experiment needs the Bose factor and the
`(ν_laser − ν)⁴` prefactor — both opt-in via `plot_spectrum`, so the axis
label always says which is shown.

Formats read: QE `bands.dat` and `bands.dat.gnu`, SIESTA `SystemLabel.bands`,
and the `dynmat.x` mode table.

## Density of states

A band structure says whether there is a gap. For a doped carbon the more
useful question is *which atoms* put states at the Fermi level — that is the
projected DOS, and it is what separates graphitic from pyridinic nitrogen.

```bash
carbonforge graphene --nx 5 --ny 5 --nitrogen graphitic --task dos --out out/dos
cd out/dos/qe && ./run_dos.sh
carbonforge plot-dos . --fermi <E_F from pw.scf.out> --window -8 4
```

which reports, per element:

```
En el nivel de Fermi (0.000 eV):
  C: 52.3 %
  N: 47.7 %
```

The workflow is scf → nscf → `dos.x` → `projwfc.x`. Two details are handled
rather than left as traps: the **nscf step uses a denser k-mesh** (2× by
default), because a mesh that converges the charge density is too coarse to
resolve a DOS curve; and the summed projection is reported against the total
so you can see how complete it is — projections onto atomic orbitals
typically recover 90-95 %, and a much lower figure means the per-element
fractions should not be trusted.

## Convergence

The shipped cutoffs are starting points, not converged values. To settle it:

```bash
carbonforge converge structure.xyz --parameter cutoff --out conv
cd conv && ./run_sweep.sh
carbonforge converge-report conv --tolerance 1.0 --out conv.png
```

The report compares each point with the **next** one — answering "can I stop
here?" — in meV per atom, so the tolerance means the same thing at any
system size.

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
from carbonforge.builders import build_nanocoil
from carbonforge.relax    import harmonic_pre_relax

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
from carbonforge.viz import save_structure_png
save_structure_png(atoms, "coil.png", view=(20, 45))
```

Matplotlib-based 3D scatter with bond lines. Intended for quick QA —
for publication rendering export to `.xyz`/`.cif` and use OVITO / VESTA.

## Typical workflow

1. **Build.** Pick a builder and parameters; obtain an `ase.Atoms` with
   correct `pbc` and vacuum padding.
2. **Customise.** Compose `dopants.*` and `defects.*` — they return new
   `Atoms` objects and accept a `seed` for reproducibility.
3. **Validate.** `carbonforge.validation.run_basic_checks(atoms)` returns
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

113 tests covering builders (including nanocoils), dopants, defects,
topology, validation, exporters, workflows, result parsers, convergence,
the bonus modules (relax / viz / ML dataset) and the GUI logic. The Tk layer
is smoke-tested against a stubbed Tk with a real matplotlib figure, so it
runs headless.

The output parsers are tested against **synthetic fixtures** matching the
documented QE and SIESTA formats, not against files from a real run — no DFT
installation was available during development. Treat your first real file as
a test of the parser too.

## Repository layout

```
carbonforge/
├── builders/      # structure generators (CNT, graphene, ribbon, nanocoil, foam)
├── dopants/       # substitutional chemistry
├── defects/       # vacancies, Stone-Wales, distortion
├── topology/      # networkx connectivity / rings
├── validation/    # structural checks
├── exports/       # QE + LAMMPS writers
├── relax/         # ASE optimizers + harmonic pre-relaxation
├── viz/           # matplotlib 3D viewer / PNG exporter
├── results/       # parse + plot band structures and vibrational spectra
├── workflows/     # batch generation, convergence sweeps, ML dataset
├── gui/           # Tkinter desktop app (params logic + widgets)
├── utils/         # constants, geometry, RNG
├── cli/           # command line
├── tests/         # pytest suite
└── examples/      # runnable example scripts
```

## License

MIT.
