# CLAUDE.md — tbkit

Parametrisable tight binding for nanocarbons. The fourth package of the
workspace; see the root `CLAUDE.md` for the rule that matters most: tbkit
imports none of the other packages (structures come in as files).

## Layout
```
tbkit/
├── params.py         # TBModel, distance laws, built-in sets (π Hückel, Xu sp3 carbon)
├── basis.py          # orbital index map
├── slater_koster.py  # s/p two-centre blocks; local "pi" orbital
├── hamiltonian.py    # System: bonds, H(k), S(k)
├── kpoints.py        # Γ, meshes, band paths
├── solver.py         # eigenstates, Fermi level, Solution (spin-resolved)
├── analysis.py       # P, Mulliken/Löwdin, bond orders, DOS/PDOS, bands, cube files
├── scc.py            # self-consistent charges (finite only)
├── hubbard.py        # mean-field Hubbard, magnetisation vs energy/field/doping
├── skf.py            # DFTB .skf reader (simple format)
├── fit.py            # least-squares fitting (with a before/after report), GPAW log reader
├── repulsive.py      # pair / embedded repulsion, .skf spline
├── forces.py         # Mermin free energy, Hellmann-Feynman forces
├── calculator.py     # ASE calculator (relax, ase.vibrations)
├── tasks.py          # levels, dos, bands, hubbard, relax, phonons as functions
├── record.py         # reproducible records, `tbkit run simulation.json`, replay
├── parameters/       # built-in parameter sets (JSON, every number with unit and source)
└── cli.py            # `tbkit` console script
```

## Guardrails (do not weaken)
- Units: eV and Å everywhere. `.skf` values are converted from Hartree/Bohr on
  reading; nothing downstream sees atomic units.
- A π model uses the local `"pi"` orbital with no angular factor. A Cartesian
  `pz` with Slater-Koster angles loses the axial bonds of a nanotube (every
  atom ends with two bonds): that bug existed once and `test_zigzag_nanotubes`
  guards it.
- Parameters live in `parameters/*.json`, never inside functions. Every
  number has a unit and a source; every file its reference, system and
  validity (`test_reproducibility.py` enforces it). A fitted set is returned
  and saved as a new file, never written over a built-in one, and the fit
  reports each change (before -> after) and the metrics that moved.
- H = H† and S > 0 are checked on every build/diagonalisation, never assumed.
- Forces are derivatives of the Mermin free energy (band - TS + repulsion
  [+ SCC]); every force path is tested against finite differences of the
  energy. Laws that enter forces must be smooth: use `Tail` cutoffs.
- SCC forces need tightly converged charges (Anderson mixing, tol 1e-10 in
  the calculator); a loose tolerance shows up as a force error, not a crash.
- Validation against experiment states the model's error; do not retune a
  built-in set to hit a number.
- Published `.skf` sets are never bundled (their licences); tests generate
  synthetic files in the documented format.
- SCC refuses periodic systems (no Ewald). Do not approximate silently.
- Mean-field moments are an order parameter, not a correlated ground state;
  user-facing text must not imply otherwise. Lieb's theorem is the check.
- Energies and forces require `model.repulsive`; the π model has none and
  must refuse (calculator, forces, relax).
- Tests compare with closed-form results (graphene, Hückel, Lieb). A new
  feature needs such a check, not only a regression number.

## Test
```bash
OMP_NUM_THREADS=1 uv run python -m pytest tbkit/tests -q -n 4 --dist loadscope
```
