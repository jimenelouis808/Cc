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
├── scc.py            # self-consistent charges (finite, or periodic through ewald.py)
├── ewald.py          # periodic γ: Ewald for 1/r and 1/r³, short-range rest
├── hubbard.py        # mean-field Hubbard, magnetisation vs energy/field/doping
├── skf.py            # DFTB .skf reader (simple format)
├── fit.py            # least-squares fitting (with a before/after report), GPAW log reader
├── repulsive.py      # pair / embedded repulsion, .skf spline
├── forces.py         # Mermin free energy, Hellmann-Feynman forces
├── calculator.py     # ASE calculator (relax, ase.vibrations)
├── tasks.py          # levels, dos, bands, hubbard, relax, phonons as functions
├── optics.py         # polarizability: sum over states, SCC linear response, ε∞
├── dipoles.py        # intra-atomic s-p dipoles: position operator, multipole screening
├── infrared.py       # IR: model dipole (SCC charges + intra-atomic dipoles), Born charges
├── modes.py          # mode analysis: participation, cylindrical character, symmetry, VDOS
├── graphene.py       # graphene G / 2D / 2D' by (double) resonance, phonons in the whole BZ
├── raman.py          # non-resonant Raman on the model's own Γ phonons
├── resonance.py      # resonant Raman: ∂α(ω_L + iη)/∂Q, excitation profiles
├── record.py         # reproducible records, `tbkit run simulation.json`, replay
├── qe.py             # Quantum ESPRESSO Γ modes (dynmat/matdyn) for Raman with QE phonons
├── references.py     # DFT reference sets (JSON): GPAW levels, energies, forces, frequencies
├── recipes/          # reproducible fits: xu_family (machinery), xu_chn, xu_chno; GPAW references
│                     #   frequency_scaling: one scale factor per set against GPAW
│                     #   crystal_validation + run_crystals.sh: the sets in crystals vs GPAW (resumable)
├── parameters/       # built-in parameter sets (JSON, every number with unit and source)
├── cli.py            # `tbkit` console script
└── gui/              # tbkit-gui (PySide6 + pyvista, extra `gui`): actions (no Qt), worker, viewer, app
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
- Resonant Raman (`resonance.py`) needs η > 0; its static, below-gap limit
  must equal the non-resonant result (tested). Resonance energies are the
  model's (small TB/KS gaps): never present them as optical energies.
- Raman is non-resonant: refuse gapless systems (any fractionally occupied
  state means no gap: `Solution.gap()`), open shells and lasers near the gap.
  Selection rules (diamond T2g, C60 2Ag+8Hg) are the first check of any
  change to optics or phonons. Finite polarizabilities default to the
  screened SCC linear response; the finite-field version is its test.
- In pair sums over states (χ, α), count each unordered pair once: with
  smearing a state can be both "occupied" and "empty" (the bug of a 20 %
  overcount that the linear-response = sum-over-states test caught).
- Published `.skf` sets are never bundled (their licences); tests generate
  synthetic files in the documented format.
- Periodic SCC sums γ over images (`ewald.py`, Elstner 1998): Ewald for C/r,
  a second Ewald for the Klopman-Ohno r⁻³ asymptote (its divergent G = 0
  constant dropped, stated), the r⁻⁵ rest in real space with a smooth taper.
  Checks: NaCl Madelung constant, a molecule in a large box equals the finite
  result, forces against finite differences (h-BN). Summing the r⁻³ tail in
  real space oscillated by 10⁻² eV: do not go back to it. Vacuum directions
  are a supercell. SCC linear-response α of crystals
  (`optics.polarizability_periodic_screened`, q → 0, charge local fields through
  the Ewald γ without G = 0): a molecule in a box must give the finite α with its
  Lorentz field, α/(1 - 4πα/3V), and pristine h-BN exactly the unscreened α
  (C3 sites carry no induced charge; both tested). Dipole channels are not
  screened in crystals yet. Raman of crystals uses it only with screening="scc".
- The sets were fitted on molecules; crystals are an extrapolation, measured in
  `recipes/crystal_validation.py` (13 crystals, GPAW with the model's k mesh
  and smearing; `validation/crystals_tb_vs_gpaw.json`) and stated in each set's
  `validity` through the recipe's `crystal_notes` (a test compares file and
  recipe: edit the recipe, never the JSON by hand). Scale errors by the set's
  own molecular error measured the same way, not by a pooled RMS. Strained
  GPAW cells keep the relaxed cell's grid (`gpaw_factory(grid_of=...)`):
  GPAW rounds gpts to multiples of 4 and a 2 % strain changed them (~1 eV).
  The GPAW stage is resumable (one file per point, BFGS trajectory and
  Hessian); relaunch `run_crystals.sh` after a restart, never start over.
- `TBModel.scc` says which ground state the parameters were made for: True
  for `.skf` sets, `xu_chn` and `xu_chno`, False for Xu and π. Calculators, tasks and
  the linear-response α follow it unless told otherwise; a set fitted with
  SCC must not be run without it (and vice versa) by default.
- `.skf` heteronuclear convention: `A-B.skf` Hsp0 = <s_A|H|p_B>, verified
  against DFTB+'s `getFullTable`. Do not flip it without a new reference.
- `xu_chn` keeps Xu's C-C untouched; H and N are fitted around it
  (`recipes/xu_chn.py`, from `recipes/chn_references.py` GPAW data whose
  SHA-256 is stored in the parameter file). Refit by rerunning the recipe
  and saving a new file; never hand-edit fitted numbers. Hubbard U are
  computed (GPAW atom, dε/dn), not fitted.
- `xu_chnob`/`xu_chnos`/`xu_chnop`/`xu_chnose` (`recipes/xu_bsp.py`) are `xu_chno` held
  fixed (`XuFamily.base`) plus one element: a structure without B, S or P
  must give exactly the `xu_chno` energy (tested). Do not refit `xu_chno`
  without refitting them. Every pair among a set's elements (H-H aside) must
  have hopping laws: a missing law is silently zero, so the family refuses
  to build without it.
- Spurious minima are found with `recipes/active_learning.py` (relax from GPAW,
  sample the path, GPAW single points; `--torsions X` adds rigid X-O-H scans) and
  fed back to the fit. `CentredAngleTerm`/`CentredTorsionTerm` exist and are
  tested, but no shipped set uses them: on Se the all-ligand angle term broke
  divalent Se (H2Se 64 -> 289 cm⁻¹) and neither fixed the X-OH torsion of
  seleninic/phosphonic acids, which stays a stated limit in `validity`. An O-X-O-only
  angle term (`xu_bsp --angular-ligands O`, zero in H2Se) was fitted active
  (~0.9 eV) and still left CH3SeO2H drifting 0.83 Å (0.87 without). Cause,
  measured: the minimal basis. Along the rigid OH scan the TB repulsion is flat
  and the electronic energy puts 240-300° 0.1 eV below the minimum; GPAW with
  a minimal sz basis on every atom does the same (-0.06 eV; dzp +0.03), while
  GPAW without d on Se alone does not. No repulsive or angle term can fix an
  electronic, basis-set error: the fix is polarisation functions (out of scope). The boronic case was the missing H-H repulsion, now
  `repulsive.HHContactTerm` (`recipes/hh_contact.py`: GPAW H2···H2 wall, not
  fitted; zero between hydrogens of the same atom, through a smooth bond
  weight, and in H2). It sits second in every xu_ch* repulsion and is a fixed
  base term of any refit (`XuFamily.hh_contact`). Never scale it to hit a
  geometry: ×3 already overshoots PhB(OH)2 (321° against GPAW's 337°).
- Imported QE modes: L = e/√m with e normalised from the file's
  displacements (or eigenvectors), per degenerate set a real basis of the
  subspace; never take the real part of a complex Γ mode without fixing its
  phase. The model then gives only α (no repulsion needed).
- Intra-atomic dipoles (`dipoles.py`): ⟨s|r|p⟩ magnitudes are computed
  (GPAW free atom), never fitted; their sign comes from the model's orbital
  convention (probe), so flipping every spσ must leave α unchanged (tested).
  The multipole kernel derives from the same Klopman-Ohno γ; the ground
  state stays as fitted and the dipole terms act on δM. The finite-field
  solver of that functional is the test of the linear response.
- With dipoles, α along σ bonds drops (hybrid centroids): that is physics,
  not a bug. Do not retune d or U to recover a number.
- `extra_polarizability` is the only optical number fitted (one per element, to
  GPAW FD tensors, `recipes/xu_chn_alpha.py`; xu_chno fits only O and keeps
  H, C, N from xu_chn); extra dipoles never interact with
  their own atom. C60 and diamond are validation, never fit targets.
- IR uses the same position operator as α (charges + intra-atomic dipoles,
  λ = 1): do not switch the default to "charges only" to improve static
  dipoles without re-running the GPAW comparison. Born charges must sum to
  the total charge (tested).
- Frequency scale factors (`frequency_scale` in each set, recipe
  `recipes/frequency_scaling.py`, against GPAW, not experiment) are reported
  beside the raw frequencies (`frequencies_scaled_cm1`); never apply them
  silently to phonons, Raman or IR, and rerun the recipe after any refit.
- Mode identification (RBM, G) is by symmetry and character, never by
  frequency window alone: zone folding puts other A modes nearby (M-point
  modes in zigzag tubes). DFT force constants get the acoustic sum rule on
  loading; never report their raw acoustic "frequencies".
- The acute-angle term (`repulsive.AcuteAngleTerm`, xu_chno) must stay exactly
  zero for angles ≥ θ0 = 80°: graphene, diamond, nanotubes, fullerenes and
  aromatics keep Xu's results (tested). It exists for three-membered rings
  (epoxide, oxirane, aziridine, cyclopropane); never raise θ0 to fix
  something else. Its coefficients are only pinned down by the ring-opening
  scans (`chno_references.RING_SCANS`, ring C-C to 1.95 Å): random
  distortions sample only the minimum, and without the scans the fit left
  the rings with no barrier (epoxide C-C +0.57 Å). Check the relaxed
  coronene epoxide after any refit.
- Repulsion polynomials are only determined where the training data are:
  check V(r) and its slope in that range, not the coefficients, and state
  the range in `validity`.
- GUI: calculations live in `gui/actions.py` (no Qt, tested like the rest);
  `app.py` only lays out and draws. The worker pauses Python's cyclic GC while
  a job runs and collects in the GUI thread: the collector running in the
  worker destroyed Qt objects of the GUI thread and crashed the window at
  random places. Do not remove that; results come back through queued slots.
  PySide6/pyvista stay optional (the `gui` extra); GUI tests skip without them.
  Help texts (`actions.HELP` per control label, `actions.PANEL_HELP` per tab)
  are applied as tooltips and info boxes; a test fails if a new button, box
  or form row in `app.py` has no text there.
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
