# CLAUDE.md

This file gives instructions to Claude Code (or any assistant) working on this repository.

## Project scope
`carbonforge` builds nanocarbon structures (1D/2D/3D) and prepares complete first-principles calculations for them — band structure, Raman/IR spectra, spin-orbit coupling — targeting Quantum ESPRESSO, SIESTA and LAMMPS.

Note: a separate project of the user's has a similar name and is the *structure generator*; this one is the simulation-preparation suite. Keep the distinction in naming and docs.

The framework must remain **scientifically valid**: physical bond lengths, correct coordination, no impossible geometries, reproducible dopant / defect placement.

## Repository layout
```
carbonforge/
├── builders/      # 1D/2D/3D structure generators incl. nanocoils (ASE-compatible Atoms)
├── dopants/       # substitutional dopants (N, B, S, P, co-doping)
├── functionalization/  # attached groups + nitrogen lattice configurations
├── defects/       # vacancies, Stone-Wales, topological defects
├── topology/      # networkx-based connectivity / coordination analysis
├── validation/    # geometry checks + calculation-level physics checks
├── calculations/  # band paths, phonon/IR/Raman, spin-orbit setups
├── results/       # parse + plot finished runs (bands, spectra)
├── exports/       # Quantum ESPRESSO, SIESTA and LAMMPS writers
├── relax/         # ASE optimizer wrapper + calculator-free harmonic pre-relax
├── viz/           # matplotlib 3D viewer
├── gui/           # Tkinter desktop app (params logic + widgets)
├── workflows/     # batch generation, convergence sweeps, ML dataset
├── utils/         # constants, geometry helpers
├── cli/           # command line interface
├── tests/         # pytest unit tests
└── examples/      # runnable example scripts
```

## Golden rules for contributors / assistants
1. **Python >= 3.10**, type hints on all public functions.
2. **Docstrings** are mandatory. Each public function must document inputs, outputs, and (when relevant) the physical assumption.
3. Every builder must return an `ase.Atoms` with correct `pbc` and `cell`.
4. Every structure exported **must pass `validation.run_basic_checks`** first, and every calculation setup must pass `validation.calculations.check_full_setup`.
5. Random operations (dopants, defects, disordered foams) **must accept a `seed`** for reproducibility.
6. **No hardcoded paths**. Use `pathlib.Path` and user-supplied output directories.
7. **No monolithic scripts**: keep modules small and composable.
8. New features ship with a matching pytest test in `tests/`.

## Calculation guardrails (do not weaken these)
- Raman/IR require a band gap (`epsil`) — reject metallic systems.
- Raman requires norm-conserving pseudopotentials — PAW/USPP must be rejected.
- Spin-orbit requires `rel-` pseudopotentials; scalar ones give silent zero splitting.
- `vc-relax` on 1D/2D must set `cell_dofree`, or the vacuum collapses.
- Metals must not use `occupations='fixed'`.
- A builder's declared `pbc` must be the axis the atoms actually tile. The
  nanoribbon builder once declared y (its vacuum direction) and every ribbon
  export was silently wrong; `test_builders.py` now asserts this for ribbons.
- Attached functional groups eat into the vacuum padding: re-pad with
  `utils.geometry.ensure_vacuum` after functionalising.
- Attached groups and lattice nitrogen are different chemistry. Do not let
  `--group NH2` and `--nitrogen graphitic` blur together in docs or UI.
- `make_pyrrolic_like` is a precursor, not a pyrrolic site. Keep the name and
  the warning honest.
- `dynmat.x`: `filout` must never be `dynmat.out` — the runner script
  redirects stdout there and the two would clobber each other.
- Result parsers are validated against synthetic fixtures only; say so in
  user-facing docs rather than implying they are battle-tested.

## Scientific guardrails
- Carbon bond length: default 1.42 Å (sp2). Accept anything in `[1.20, 1.80]` Å as bonded; anything in `(0, 0.9]` Å is a hard error.
- Expected C coordination: **2 (edge)**, **3 (sp2 bulk)**. Coordination >= 5 or == 1 in the bulk is rejected by validation.
- Dopants replace carbon atoms: N, B (sp2 compatible, coord 3), S, P (larger, often require local relaxation). The module flags clustered dopants as warnings, not errors.
- 2D structures: vacuum along the non-periodic direction must be `>= 12 Å` by default.
- 1D structures (CNT): vacuum in the two transverse directions must be `>= 10 Å` beyond the tube radius.

## Quick test commands
```bash
pip install -e .[dev]
pytest carbonforge/tests -q
python -m carbonforge.cli.main cnt --n 6 --m 6 --length 10 --out out/cnt --format qe
```

## Where to add things
- New builder type → `builders/<name>.py` + export in `builders/__init__.py` + test in `tests/test_<name>.py`.
- New dopant chemistry → `dopants/<element>.py`, reuse `dopants.base.substitute_atoms`.
- New exporter → `exports/<backend>.py` implementing `write(atoms, outdir, **kwargs)`.

## What not to do
- Do **not** export structures that fail validation without explicit `force=True`.
- Do **not** introduce non-deterministic random state without a `seed` argument.
- Do **not** add dependencies outside the ones declared in `pyproject.toml` without justification.
