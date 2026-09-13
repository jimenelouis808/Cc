# nanocarbon

Three packages for nanocarbon and nanomaterials research, in one workspace.

| Package | What it does |
|---|---|
| `packages/ramancarbon` | Raman, XRD, XPS and electrochemistry: readers, fitting, analysis, a configurable plot engine, a Tkinter application |
| `packages/carbonforge` | Structures into DFT and MD: Quantum ESPRESSO, SIESTA, LAMMPS, force fields, EDLC cells, reading results back |
| `packages/nanocarbon_lab` | Structure generation: nanotubes, fullerenes, haeckelites, schwarzites, junctions, foams, TMDs, dopants and defects |

## The one rule

**The packages are independent and stay that way.** None imports another. Each
keeps its own `pyproject.toml`, its own version, its own tests and its own
console script, and each must remain installable and lintable on its own.

The workspace exists so they share a lockfile, a licence and CI — not so they
share code. Extracting a common core, splitting a package or merging two of
them are separate decisions, none of which has been taken. Until one is, code
does not move between packages.

## Working here

```bash
uv sync --all-packages --extra dev

# Fast suite for one package (~90 s for the largest):
cd packages/nanocarbon_lab
OMP_NUM_THREADS=1 uv run python -m pytest nanocarbon_lab/tests \
    -q -n 4 --dist loadscope -m "not slow"

# Everything, slow tests included:
OMP_NUM_THREADS=1 uv run python -m pytest nanocarbon_lab/tests -q -n 4 --dist loadscope
```

Two things about that command are not optional:

- **`OMP_NUM_THREADS=1`.** The arrays are small, and the linear-algebra
  library's own threading contends with the test workers rather than helping.
  Pinning it cuts the suite from five minutes to ninety seconds.
- **`--dist loadscope`.** The expensive fixtures are class-scoped. Distributing
  by test rebuilds them in every worker.

Tests marked `slow` are mesh-and-relax builds that take minutes each. They run
nightly in CI, not on every push.

## Things that will bite

- **`bpy` pins numpy 1.26.** It is declared as conflicting in the root
  `pyproject.toml` so it cannot dictate the numpy the other packages get. Do
  not remove that declaration: carbonforge's DOS reader needs numpy 2.
- **No `ruff format`.** The code was never formatted with it and 301 of 395
  files would change. Run `ruff check`, not `ruff format`.
- **`carbonforge` and `nanocarbon_lab` overlap.** Both build structures, dope
  them and validate them, from a common ancestor but with no shared git
  history. They have diverged. Reconciling them is a known, deferred piece of
  work — do not do it accidentally while fixing something else.
