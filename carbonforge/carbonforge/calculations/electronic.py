"""Electronic-structure settings: spin, van der Waals, functionals, Hubbard U.

These are the knobs that decide *what physics the calculation contains*, as
opposed to what quantity it reports. Three of them are routinely needed for
carbon nanostructures and are easy to omit by accident:

**Spin polarisation.** Zigzag graphene nanoribbons have spin-polarised edge
states, antiferromagnetically coupled across the ribbon — one of the
best-established results in the field (Son, Cohen & Louie, *Nature* 444, 347,
2006). A non-spin-polarised calculation of a ZGNR converges happily to a
metallic, non-magnetic state that is **not** the ground state, and reports a
band structure that is simply wrong. :func:`setup_antiferromagnetic_edges`
prepares the two-sublattice initial guess that finds the right one.

**Van der Waals.** Semilocal functionals have no dispersion, so PBE gets
interlayer binding in graphite, molecular physisorption and stacking wrong —
it barely binds them at all. Grimme's D3 correction is the standard remedy
and costs essentially nothing.

**Hybrid functionals.** PBE underestimates band gaps by roughly half. For a
semiconducting nanoribbon or nanotube whose gap is the point of the study,
HSE06 is the usual answer, at 10-100x the cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from ase import Atoms

SpinMode = Literal["none", "collinear", "noncollinear"]

#: van der Waals schemes accepted by Quantum ESPRESSO's ``vdw_corr``.
VDW_SCHEMES: dict[str, str] = {
    "none": "Sin corrección: sin dispersión. Mal para apilamiento y fisisorción.",
    "grimme-d2": "Grimme D2: barata y antigua; D3 la reemplaza.",
    "grimme-d3": "Grimme D3: el estándar actual. Coste despreciable.",
    "ts": "Tkatchenko-Scheffler: depende de la densidad, algo más cara.",
    "xdm": "Exchange-hole dipole moment.",
}

#: Exchange-correlation functionals, with their cost relative to PBE.
FUNCTIONALS: dict[str, tuple[str, str]] = {
    "pbe": ("PBE", "GGA estándar. Subestima gaps ~50 %."),
    "pbesol": ("PBEsol", "PBE reparametrizado para sólidos; mejores geometrías."),
    "revpbe": ("revPBE", "Variante de PBE, común con correcciones vdW."),
    "hse": ("HSE06", "Híbrido apantallado. Gaps mucho mejores, 10-100x el coste."),
    "b3lyp": ("B3LYP", "Híbrido de química cuántica; poco usado en sólidos."),
    "vdw-df2": ("vdW-DF2", "Funcional no local con dispersión incorporada."),
}

#: Functionals that are hybrids, and therefore expensive and needing an EXX grid.
HYBRID_FUNCTIONALS: tuple[str, ...] = ("hse", "b3lyp")


@dataclass
class ElectronicSpec:
    """What physics the electronic-structure calculation includes.

    Attributes
    ----------
    spin
        ``"none"`` (non-polarised), ``"collinear"`` (``nspin=2``) or
        ``"noncollinear"``. Use collinear for magnetic edge states; the
        non-collinear case is handled by
        :mod:`~carbonforge.calculations.spinorbit`.
    starting_magnetization
        Per-species initial moment, in units of the valence charge, as
        ``{"C": 0.0, "C1": 0.5, "C2": -0.5}``. Only the *sign pattern* really
        matters: it decides which magnetic state the SCF falls into.
    vdw_correction
        Key from :data:`VDW_SCHEMES`.
    functional
        Key from :data:`FUNCTIONALS`. ``None`` keeps whatever the
        pseudopotentials were generated with, normally PBE.
    exx_fraction, screening_parameter
        Hybrid-functional parameters. The defaults are the HSE06 values.
    exx_grid_factor
        The EXX q-grid is the k-mesh divided by this factor. 2 is the usual
        compromise; 1 is exact and very expensive.
    hubbard_u
        Per-species Hubbard U in eV, for DFT+U. Rarely needed for pure
        carbon, but relevant once transition metals are present.
    tot_magnetization
        Constrain the total moment (in Bohr magnetons per cell). ``None``
        lets it relax freely, which is normally what you want.
    """

    spin: SpinMode = "none"
    starting_magnetization: dict[str, float] = field(default_factory=dict)
    vdw_correction: str = "none"
    functional: Optional[str] = None
    exx_fraction: float = 0.25
    screening_parameter: float = 0.106
    exx_grid_factor: int = 2
    hubbard_u: dict[str, float] = field(default_factory=dict)
    tot_magnetization: Optional[float] = None

    def __post_init__(self) -> None:
        if self.vdw_correction not in VDW_SCHEMES:
            raise ValueError(
                f"Corrección vdW desconocida: '{self.vdw_correction}'. "
                f"Opciones: {', '.join(sorted(VDW_SCHEMES))}."
            )
        if self.functional is not None and self.functional not in FUNCTIONALS:
            raise ValueError(
                f"Funcional desconocido: '{self.functional}'. "
                f"Opciones: {', '.join(sorted(FUNCTIONALS))}."
            )
        if self.exx_grid_factor < 1:
            raise ValueError("exx_grid_factor debe ser >= 1.")
        if not 0.0 <= self.exx_fraction <= 1.0:
            raise ValueError("exx_fraction debe estar entre 0 y 1.")
        if self.spin == "none" and self.starting_magnetization:
            raise ValueError(
                "Se dieron magnetizaciones iniciales pero spin='none'. "
                "Usa spin='collinear' o quita las magnetizaciones."
            )

    @property
    def is_hybrid(self) -> bool:
        """Whether the chosen functional is a hybrid, and so expensive."""
        return self.functional in HYBRID_FUNCTIONALS

    @property
    def is_spin_polarized(self) -> bool:
        return self.spin in ("collinear", "noncollinear")

    def cost_multiplier(self) -> float:
        """Rough cost relative to a plain non-polarised PBE run.

        Spin polarisation roughly doubles the work; a hybrid is one to two
        orders of magnitude worse. These are order-of-magnitude guides for
        deciding what will fit in a queue, not benchmarks.
        """
        factor = 1.0
        if self.spin == "collinear":
            factor *= 2.0
        elif self.spin == "noncollinear":
            factor *= 4.0
        if self.is_hybrid:
            factor *= 30.0
        if self.vdw_correction in ("ts", "xdm"):
            factor *= 1.2
        return factor

    def describe(self) -> str:
        """Human-readable summary of the physics included."""
        lines = []
        spin_labels = {
            "none": "sin polarizar (nspin=1)",
            "collinear": "polarizado colineal (nspin=2)",
            "noncollinear": "no colineal",
        }
        lines.append(f"Espín: {spin_labels[self.spin]}")
        if self.starting_magnetization:
            pattern = ", ".join(
                f"{k}={v:+.2f}" for k, v in sorted(self.starting_magnetization.items())
            )
            lines.append(f"  magnetización inicial: {pattern}")

        name, note = FUNCTIONALS.get(
            self.functional or "pbe", ("PBE", "por defecto del pseudopotencial")
        )
        lines.append(f"Funcional: {name} — {note}")
        lines.append(f"vdW: {VDW_SCHEMES[self.vdw_correction]}")
        if self.hubbard_u:
            values = ", ".join(f"{k}={v} eV" for k, v in sorted(self.hubbard_u.items()))
            lines.append(f"Hubbard U: {values}")

        multiplier = self.cost_multiplier()
        lines.append(f"\nCoste aproximado: {multiplier:.0f}x respecto a PBE sin espín")
        if multiplier >= 20:
            lines.append(
                "  ⚠️  Esto es caro. Prueba primero con una celda pequeña para "
                "estimar el tiempo real antes de lanzar el sistema completo."
            )
        return "\n".join(lines)


def qe_system_fields(
    spec: ElectronicSpec,
    species_order: list[str],
) -> dict[str, object]:
    """Return the ``&SYSTEM`` entries implementing this setup in QE.

    Parameters
    ----------
    spec
        The electronic settings.
    species_order
        Species in the order they appear in ``ATOMIC_SPECIES``, because QE
        indexes ``starting_magnetization`` and ``Hubbard_U`` by position.
    """
    fields: dict[str, object] = {}

    if spec.spin == "collinear":
        fields["nspin"] = 2
        for symbol, moment in spec.starting_magnetization.items():
            if symbol in species_order:
                index = species_order.index(symbol) + 1
                fields[f"starting_magnetization({index})"] = moment
        if spec.tot_magnetization is not None:
            fields["tot_magnetization"] = spec.tot_magnetization

    if spec.vdw_correction != "none":
        fields["vdw_corr"] = spec.vdw_correction

    if spec.functional is not None:
        fields["input_dft"] = spec.functional
        if spec.is_hybrid:
            fields["exx_fraction"] = spec.exx_fraction
            if spec.functional == "hse":
                fields["screening_parameter"] = spec.screening_parameter

    for symbol, value in spec.hubbard_u.items():
        if symbol in species_order:
            index = species_order.index(symbol) + 1
            fields[f"Hubbard_U({index})"] = value
    if spec.hubbard_u:
        fields["lda_plus_u"] = True

    return fields


def exx_grid(
    kmesh: tuple[int, int, int],
    factor: int,
) -> tuple[int, int, int]:
    """Return the EXX q-grid for a hybrid run, from the k-mesh.

    The Fock operator is evaluated on this coarser grid; making it equal to
    the k-mesh is exact but usually unaffordable.
    """
    return tuple(max(1, value // factor) for value in kmesh)  # type: ignore[return-value]


def setup_antiferromagnetic_edges(
    atoms: Atoms,
    moment: float = 0.5,
) -> tuple[Atoms, ElectronicSpec]:
    """Prepare a ribbon for its antiferromagnetic edge-state ground state.

    Zigzag graphene nanoribbons carry spin-polarised states localised on each
    edge, and the ground state has the two edges coupled **antiferromagnetically**
    — opposite spin on opposite sides. An SCF started from a uniform guess
    will not find it: it settles into the non-magnetic state, or at best a
    ferromagnetic one, and reports the wrong band structure and gap.

    Finding it requires breaking the symmetry in the initial guess, and in
    Quantum ESPRESSO that means the two edges must be *different species*, so
    they can be given opposite ``starting_magnetization``.

    ASE will not accept invented symbols like ``C_up``, so the split is
    carried as **ASE tags**: 1 for one edge, 2 for the other, 0 for the rest.
    The Quantum ESPRESSO writer expands tagged atoms into separate species
    (``C1``, ``C2``) sharing the carbon pseudopotential — the same element,
    labelled apart only so their initial moments can differ.

    Parameters
    ----------
    atoms
        A ribbon or flake with two distinguishable edges.
    moment
        Initial moment magnitude, in units of the valence charge. 0.5 is a
        strong enough guess to break the symmetry without over-constraining.

    Returns
    -------
    (ase.Atoms, ElectronicSpec)
        A relabelled copy, and the matching spin setup.

    Raises
    ------
    ValueError
        If fewer than two edge atoms are found, or if they cannot be split
        into two spatially separated groups.

    Notes
    -----
    The split is geometric: edge carbons are divided by which side of the
    ribbon's centre they sit on, along the widest non-periodic axis. That is
    right for a straight ribbon; for an irregular flake, inspect the result
    before trusting it.
    """
    from ..functionalization.sites import find_sites

    edges = find_sites(atoms, kind="edge", element="C")
    if len(edges) < 2:
        raise ValueError(
            f"Solo se encontraron {len(edges)} carbonos de borde. La "
            "configuración antiferromagnética necesita dos bordes; ¿es una "
            "lámina periódica sin bordes?"
        )

    positions = np.array([site.origin for site in edges])
    pbc = atoms.get_pbc()
    # Split along the widest direction that is not the periodic one: for a
    # ribbon that is its width.
    candidates = [axis for axis in range(3) if not pbc[axis]]
    if not candidates:
        raise ValueError(
            "La estructura es periódica en las tres direcciones: no hay "
            "bordes que separar."
        )
    axis = max(candidates, key=lambda a: float(np.ptp(positions[:, a])))
    spread = float(np.ptp(positions[:, axis]))
    if spread < 1.0:
        raise ValueError(
            "Los átomos de borde no se separan en dos grupos distinguibles "
            f"(dispersión {spread:.2f} Å). Esto está pensado para cintas."
        )

    midpoint = positions[:, axis].mean()
    out = atoms.copy()
    out.info = {**atoms.info}
    tags = list(out.get_tags())
    n_up = n_down = 0
    for site, position in zip(edges, positions):
        if position[axis] >= midpoint:
            tags[site.index] = 1
            n_up += 1
        else:
            tags[site.index] = 2
            n_down += 1

    if n_up == 0 or n_down == 0:
        raise ValueError(
            "Todos los átomos de borde cayeron del mismo lado; no se pudo "
            "separar en dos subredes."
        )

    out.set_tags(tags)
    out.info["afm_edges"] = {
        "axis": axis,
        "n_up": n_up,
        "n_down": n_down,
        "moment": moment,
    }

    # Species names match what the QE writer generates for tagged atoms.
    spec = ElectronicSpec(
        spin="collinear",
        starting_magnetization={"C1": moment, "C2": -moment, "C": 0.0},
    )
    return out, spec


def tagged_species(atoms: Atoms) -> tuple[list[str], dict[str, str]]:
    """Expand ASE tags into distinct species labels for Quantum ESPRESSO.

    An atom with tag ``t > 0`` becomes ``<symbol><t>`` — carbon tagged 1
    becomes ``C1``. Untagged atoms keep their plain symbol. This is how two
    sublattices of the same element get independent initial magnetisations,
    which is what an antiferromagnetic guess needs.

    Returns
    -------
    (labels, pseudo_map)
        ``labels`` is the per-atom species label. ``pseudo_map`` maps each
        generated label back to its true element, so the writer can give
        ``C1`` and ``C2`` the same pseudopotential file.
    """
    symbols = atoms.get_chemical_symbols()
    tags = atoms.get_tags()
    labels: list[str] = []
    pseudo_map: dict[str, str] = {}
    for symbol, tag in zip(symbols, tags):
        label = f"{symbol}{tag}" if tag > 0 else symbol
        labels.append(label)
        pseudo_map[label] = symbol
    return labels, pseudo_map
