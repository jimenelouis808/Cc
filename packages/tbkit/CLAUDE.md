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
├── fit.py            # least-squares fitting, GPAW log reader
└── cli.py            # `tbkit` console script
```

## Guardrails (do not weaken)
- Units: eV and Å everywhere. `.skf` values are converted from Hartree/Bohr on
  reading; nothing downstream sees atomic units.
- A π model uses the local `"pi"` orbital with no angular factor. A Cartesian
  `pz` with Slater-Koster angles loses the axial bonds of a nanotube (every
  atom ends with two bonds): that bug existed once and `test_zigzag_nanotubes`
  guards it.
- Every built-in parameter carries its source in the docstring. No number
  without a reference.
- Published `.skf` sets are never bundled (their licences); tests generate
  synthetic files in the documented format.
- SCC refuses periodic systems (no Ewald). Do not approximate silently.
- Mean-field moments are an order parameter, not a correlated ground state;
  user-facing text must not imply otherwise. Lieb's theorem is the check.
- No total energies or forces: there is no repulsive term. Do not add
  relaxation on top of these models without one.
- Tests compare with closed-form results (graphene, Hückel, Lieb). A new
  feature needs such a check, not only a regression number.

## Test
```bash
OMP_NUM_THREADS=1 uv run python -m pytest tbkit/tests -q -n 4 --dist loadscope
```
