"""The ``Crystal`` object: a lattice, a symmetry group and a list of atoms.

Everything the diffraction side computes comes from here, and nothing is
tabulated. A peak position is ``2 arcsin(λ / 2d_hkl)`` with ``d_hkl`` from
the metric tensor; a peak intensity is a structure factor summed over the
atoms the symmetry generates. That means a refined lattice parameter moves
the peaks the way it physically would, and a wrong occupancy changes the
intensities the way it physically would — neither of which a table of
"reference peak positions" can do.

The distinction that matters when reading the numbers this produces:
**peak positions depend only on the lattice**, and lattice constants are
known to five or six figures for any well-studied phase. Atom coordinates
are less certain, and they affect only intensities. So phase identification
by peak position is robust even when a structure's coordinates are
approximate, while a Rietveld intensity fit is not. The reference library
says which of its entries have coordinates worth trusting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from .symmetry import ClosedGroup, SymmetryOperation, close_group, orbit, parse_xyz

#: Avogadro's number, for the density.
AVOGADRO = 6.02214076e23

#: How close an operation has to come to map a site onto itself, in
#: fractional units, for :meth:`Crystal.site_freedoms`.
#:
#: Loose on purpose. A CIF writes 1/3 as 0.3333, so the three-fold at
#: (0.3333, 0.6667, 1/4) misses the site by 1e-4 and a tolerance tighter
#: than the file's own rounding drops it from the stabiliser. The cost of
#: that is not a missing warning: the site then looks free along the
#: basal direction the three-fold forbids, and graphite's second carbon
#: gets a refinable coordinate that pulls the honeycomb apart. Too loose
#: in the other direction would have to confuse two distinct sites a
#: thousandth of a cell apart, which no real structure has.
SITE_SYMMETRY_TOLERANCE = 1e-3

#: Standard atomic weights, u. Only what the reference library and ordinary
#: nanomaterial chemistry need; unknown elements fall back to 0 and the
#: density is refused rather than guessed.
ATOMIC_WEIGHT = {
    "H": 1.008, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305, "Al": 26.982,
    "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "K": 39.098,
    "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546,
    "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971,
    "Br": 79.904, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Ru": 101.07, "Rh": 102.906, "Pd": 106.42,
    "Ag": 107.868, "Cd": 112.414, "In": 114.818, "Sn": 118.710,
    "Sb": 121.760, "Te": 127.60, "I": 126.904, "Cs": 132.905, "Ba": 137.327,
    "La": 138.905, "Ce": 140.116, "Hf": 178.49, "Ta": 180.948, "W": 183.84,
    "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084,
    "Au": 196.967, "Hg": 200.592, "Tl": 204.38, "Pb": 207.2, "Bi": 208.980,
}


class StructureError(ValueError):
    """Raised when a structure is geometrically or chemically impossible."""


@dataclass(frozen=True)
class Lattice:
    """A unit cell, in ångströms and degrees."""

    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    def __post_init__(self) -> None:
        for name in ("a", "b", "c"):
            if getattr(self, name) <= 0.0:
                raise StructureError(f"el parámetro de celda {name} debe ser > 0")
        for name in ("alpha", "beta", "gamma"):
            angle = getattr(self, name)
            if not 0.0 < angle < 180.0:
                raise StructureError(
                    f"el ángulo {name} = {angle}° está fuera de (0, 180)"
                )
        if self.volume <= 0.0:
            raise StructureError(
                "los ángulos dados no definen una celda real (volumen ≤ 0). "
                "Comprueba alpha, beta y gamma"
            )

    @property
    def _cosines(self) -> tuple[float, float, float]:
        return (
            math.cos(math.radians(self.alpha)),
            math.cos(math.radians(self.beta)),
            math.cos(math.radians(self.gamma)),
        )

    @property
    def volume(self) -> float:
        """Cell volume in Å³."""
        ca, cb, cg = self._cosines
        factor = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
        if factor <= 0.0:
            return -1.0
        return self.a * self.b * self.c * math.sqrt(factor)

    @property
    def reciprocal_metric(self) -> np.ndarray:
        """The reciprocal metric tensor ``G*``.

        ``1/d²_hkl = h G* hᵀ``. Computing it once and contracting is both
        faster and less error-prone than the per-system closed forms, which
        is why there is no ``if tetragonal`` anywhere in this file.
        """
        ca, cb, cg = self._cosines
        direct = np.array(
            [
                [self.a * self.a, self.a * self.b * cg, self.a * self.c * cb],
                [self.a * self.b * cg, self.b * self.b, self.b * self.c * ca],
                [self.a * self.c * cb, self.b * self.c * ca, self.c * self.c],
            ]
        )
        return np.linalg.inv(direct)

    def d_spacing(self, hkl: Sequence[Sequence[float]] | Sequence[float]) -> np.ndarray:
        """d-spacing in Å for one or many ``(h, k, l)``."""
        indices = np.atleast_2d(np.asarray(hkl, dtype=float))
        inverse_squared = np.einsum(
            "ij,jk,ik->i", indices, self.reciprocal_metric, indices
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            spacing = 1.0 / np.sqrt(inverse_squared)
        spacing[~np.isfinite(spacing)] = np.inf
        return spacing

    def two_theta(self, hkl, wavelength: float) -> np.ndarray:
        """Bragg angle 2θ in degrees, ``inf`` where the reflection cannot occur."""
        spacing = self.d_spacing(hkl)
        with np.errstate(divide="ignore", invalid="ignore"):
            sine = wavelength / (2.0 * spacing)
        angles = np.full(sine.shape, np.inf)
        usable = np.isfinite(sine) & (np.abs(sine) <= 1.0)
        angles[usable] = 2.0 * np.degrees(np.arcsin(sine[usable]))
        return angles

    def scaled(self, factors: Sequence[float]) -> "Lattice":
        """A copy with ``a, b, c`` multiplied by ``factors``.

        Used by the refinement, which varies the cell in relative terms so
        that one step size is sensible for a 2.5 Å axis and a 14 Å one.
        """
        fa, fb, fc = factors
        return Lattice(
            self.a * fa, self.b * fb, self.c * fc, self.alpha, self.beta, self.gamma
        )

    def describe(self) -> str:
        return (
            f"a={self.a:.5f} b={self.b:.5f} c={self.c:.5f} Å, "
            f"α={self.alpha:g}° β={self.beta:g}° γ={self.gamma:g}°, "
            f"V={self.volume:.3f} Å³"
        )


@dataclass
class Site:
    """One crystallographic site of the asymmetric unit."""

    element: str
    fract: tuple[float, float, float]
    occupancy: float = 1.0
    u_iso: float = 0.005
    """Isotropic displacement parameter U in Å². ``B = 8π²U``."""
    label: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.occupancy <= 1.0 + 1e-9:
            raise StructureError(
                f"ocupación {self.occupancy} en el sitio {self.label or self.element}: "
                "tiene que estar entre 0 y 1"
            )
        if self.u_iso < 0.0:
            raise StructureError(
                f"U_iso negativo ({self.u_iso}) en {self.label or self.element}"
            )

    @property
    def b_iso(self) -> float:
        return 8.0 * math.pi * math.pi * self.u_iso


@dataclass
class Crystal:
    """A crystal structure: lattice, symmetry and asymmetric unit."""

    name: str
    lattice: Lattice
    sites: list[Site]
    operations: list[SymmetryOperation] = field(default_factory=list)
    space_group: str = "P1"
    formula: str = ""
    source: str = ""
    confidence: str = "unknown"
    """How much the *coordinates* are to be trusted. Positions of peaks
    depend only on the lattice and are safe regardless; intensities do not."""
    notes: str = ""
    stacking: str = "ordered"
    """``"turbostratic"`` for a layered carbon whose sheets are rotated at
    random about the stacking axis. That is not a detail: random rotation
    destroys three-dimensional coherence, so every ``hkl`` with ``l ≠ 0``
    AND ``(h, k) ≠ (0, 0)`` simply does not exist. A CVD carbon is this,
    not graphite, and asking it for graphite's 101 and 112 is asking for
    reflections the material cannot produce. See
    :func:`~ramancarbon.xrd.powder.reflections`."""

    def __post_init__(self) -> None:
        if not self.sites:
            raise StructureError(f"{self.name}: la estructura no tiene átomos")
        if not self.operations:
            self.operations = ClosedGroup([parse_xyz("x,y,z")])
        elif not isinstance(self.operations, ClosedGroup):
            self.operations = close_group(self.operations)

    # -- derived quantities ------------------------------------------

    @property
    def order(self) -> int:
        """Number of symmetry operations in the closed group."""
        return len(self.operations)

    def lattice_constraint(self) -> tuple[str, str, str]:
        """Which cell axes symmetry forces to be equal.

        Returns a label per axis; equal labels must refine together. The
        answer is read off the symmetry *operations*, not off a space-group
        name or off the numerical values of a, b and c — a name can be
        written in a dozen settings and equal numbers can be a
        coincidence, whereas an operation that maps **a** onto **b** is
        proof.

        Without this a hexagonal cell refines a and b independently and
        comes back with a = 2.46359, b = 2.46451: two numbers that differ
        by four times their own standard error and that symmetry says are
        one number.
        """
        cached = getattr(self, "_constraint_cache", None)
        if cached is not None:
            return cached

        cyclic = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        four_fold = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        three_fold = np.array([[0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        for operation in self.operations:
            rotation = operation.R
            if np.allclose(rotation, cyclic) or np.allclose(rotation, cyclic.T):
                return self._remember_constraint(("a", "a", "a"))
        for operation in self.operations:
            rotation = operation.R
            if (
                np.allclose(rotation, four_fold)
                or np.allclose(rotation, four_fold.T)
                or np.allclose(rotation, three_fold)
                or np.allclose(rotation, three_fold.T)
            ):
                return self._remember_constraint(("a", "a", "c"))
        return self._remember_constraint(("a", "b", "c"))

    def _remember_constraint(self, value: tuple[str, str, str]):
        """Cache the constraint on the instance.

        Same reason as the ``expanded()`` cache: a refinement builds a new
        ``Crystal`` per trial cell and asks each one for its constraint
        once per residual evaluation, and reading it means comparing every
        symmetry operation against three matrices. On one single-phase
        refinement that was 165 000 calls to ``allclose`` and half the
        total time. The answer depends only on the operations, which never
        change once the group is closed.
        """
        object.__setattr__(self, "_constraint_cache", value)
        return value

    def expanded(self) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
        """Every atom in the conventional cell.

        Returns ``(fractional coordinates, element symbols, occupancies,
        U_iso)``. Symmetry images that coincide are merged, so an atom on a
        special position contributes once per distinct image and not once
        per operation — without which an atom at the origin of a cubic
        group would scatter 24 times too strongly.
        """
        cached = getattr(self, "_expanded_cache", None)
        if cached is not None:
            return cached
        coordinates: list[np.ndarray] = []
        elements: list[str] = []
        occupancies: list[float] = []
        displacements: list[float] = []
        for site in self.sites:
            images = orbit(site.fract, self.operations)
            for image in images:
                coordinates.append(image)
                elements.append(site.element)
                occupancies.append(site.occupancy)
                displacements.append(site.u_iso)
        result = (
            np.asarray(coordinates, dtype=float),
            elements,
            np.asarray(occupancies, dtype=float),
            np.asarray(displacements, dtype=float),
        )
        # Cached because a refinement recomputes reflections on every
        # residual evaluation while the atoms never move: expanding the
        # 56 atoms of magnetite through 192 operations each time made the
        # refinement several times slower than the fit itself.
        object.__setattr__(self, "_expanded_cache", result)
        return result

    @property
    def atoms_per_cell(self) -> int:
        return len(self.expanded()[0])

    def cell_composition(self) -> dict[str, float]:
        """Element → number of atoms in the cell, weighted by occupancy."""
        _, elements, occupancies, _ = self.expanded()
        counts: dict[str, float] = {}
        for element, occupancy in zip(elements, occupancies):
            counts[element] = counts.get(element, 0.0) + float(occupancy)
        return counts

    def cell_formula(self) -> str:
        """Contents of the unit cell as a formula string."""
        counts = self.cell_composition()
        parts = []
        for element, number in sorted(counts.items()):
            rounded = round(number, 3)
            parts.append(
                element if abs(rounded - 1.0) < 1e-6 else f"{element}{rounded:g}"
            )
        return "".join(parts)

    @property
    def density(self) -> Optional[float]:
        """Crystallographic density in g/cm³, or ``None`` if an element is
        not in the weight table."""
        counts = self.cell_composition()
        mass = 0.0
        for element, number in counts.items():
            weight = ATOMIC_WEIGHT.get(element)
            if weight is None:
                return None
            mass += weight * number
        return mass / (AVOGADRO * self.lattice.volume * 1e-24)

    def site_freedoms(self) -> tuple[np.ndarray, ...]:
        """Per site, the displacement directions symmetry actually allows.

        Returns one ``(3, n)`` array per site whose columns are a basis of
        the fractional displacements invariant under that site's
        stabiliser — the operations mapping the site onto itself. ``n`` is
        0 for an atom on a fully fixed special position and 3 for one in a
        general position.

        These are the only coordinates a refinement may touch, and the
        reason is that they are the only ones that keep the structure in
        its space group. A displacement outside this subspace does not
        move one atom; it moves the whole symmetry orbit incoherently, so
        the result is not the same phase with a slightly different
        geometry but a structure with a lower symmetry than the one
        declared. Confining the shift to the basis makes the space group
        an invariant of the refinement rather than something to check
        afterwards.

        The condition is ``(R - I) d = 0`` for every stabiliser operation,
        so the basis is the null space of those stacked matrices. It is
        read off the operations, exactly as :meth:`lattice_constraint`
        is, and not off a Wyckoff letter — a letter depends on the
        setting and on the origin choice, an operation that maps the site
        onto itself does not.

        Not computed through the averaged projector ``(1/|S|) Σ R``: that
        matrix is a projector but it is not symmetric in a fractional
        basis, and symmetrising it to use ``eigh`` invents an eigenvector
        of eigenvalue 1 for the hexagonal 2c site, which is in fact
        completely fixed.
        """
        cached = getattr(self, "_freedom_cache", None)
        if cached is not None:
            return cached
        identity = np.eye(3)
        bases: list[np.ndarray] = []
        for site in self.sites:
            fract = np.asarray(site.fract, dtype=float)
            stabiliser = [
                operation.R for operation in self.operations
                if np.all(np.abs(
                    np.mod(operation.R @ fract + operation.t - fract + 0.5, 1.0)
                    - 0.5) < SITE_SYMMETRY_TOLERANCE)
            ]
            if not stabiliser:  # pragma: no cover - the identity is always in
                bases.append(identity.copy())
                continue
            stack = np.vstack([rotation - identity for rotation in stabiliser])
            _, singular, right = np.linalg.svd(stack)
            singular = np.concatenate([singular, np.zeros(3 - singular.size)])
            null = right[singular < 1e-8]
            bases.append(null.T if null.size else np.zeros((3, 0)))
        result = tuple(bases)
        object.__setattr__(self, "_freedom_cache", result)
        return result

    @property
    def free_coordinates(self) -> int:
        """How many positional parameters symmetry leaves refinable."""
        return sum(basis.shape[1] for basis in self.site_freedoms())

    def displaced(self, shifts: Sequence[float]) -> "Crystal":
        """A copy with the free coordinates moved by ``shifts``.

        ``shifts`` is one number per free coordinate, in the order
        :meth:`site_freedoms` enumerates them — site by site, and within
        a site column by column. Each one is a displacement along its
        basis direction in fractional units.

        The expanded atom list is deliberately **not** carried over: this
        is the one copy that does move the atoms.
        """
        bases = self.site_freedoms()
        values = np.asarray(shifts, dtype=float)
        if values.size != self.free_coordinates:
            raise StructureError(
                f"{self.name}: {values.size} desplazamientos para "
                f"{self.free_coordinates} coordenadas libres"
            )
        moved: list[Site] = []
        cursor = 0
        for site, basis in zip(self.sites, bases):
            count = basis.shape[1]
            if count == 0:
                moved.append(site)
                continue
            delta = basis @ values[cursor:cursor + count]
            cursor += count
            moved.append(
                Site(
                    element=site.element,
                    fract=tuple(float(v) for v in np.asarray(site.fract) + delta),
                    occupancy=site.occupancy,
                    u_iso=site.u_iso,
                    label=site.label,
                )
            )
        clone = replace(self, sites=moved)
        constraint = getattr(self, "_constraint_cache", None)
        if constraint is not None:
            object.__setattr__(clone, "_constraint_cache", constraint)
        # The freedoms are a property of the symmetry and of which
        # Wyckoff orbit each site sits on, and a shift along an allowed
        # direction keeps the site on its own orbit. So they carry over,
        # which also means the enumeration order cannot drift between a
        # structure and its displaced copy.
        object.__setattr__(clone, "_freedom_cache", bases)
        return clone

    def with_lattice(self, lattice: Lattice) -> "Crystal":
        """A copy with a different cell, sharing the symmetry and atoms.

        The expanded atom list is carried over rather than recomputed:
        changing the cell moves nothing in fractional coordinates.

        Built with :func:`dataclasses.replace` rather than by naming the
        fields, and that is not a tidying-up. The hand-written version
        listed nine fields and ``stacking`` was added later, so every
        copy made here silently came back ``"ordered"`` — and
        ``PhaseModel.current_crystal`` calls this as soon as the cell
        stage frees a lattice parameter. A turbostratic carbon therefore
        turned into graphite one stage into every refinement: measured on
        the bundled 3.44 Å carbon, 5 reflections became 11 and it
        acquired a 101 at 44.37° with 15 % relative intensity — an
        invented second-strongest line, sitting on top of α-Fe's 110 at
        44.67°, which is the strongest peak of the real pattern this was
        found on. ``replace`` carries whatever fields the class has.
        """
        clone = replace(self, lattice=lattice, sites=list(self.sites))
        cached = getattr(self, "_expanded_cache", None)
        if cached is not None:
            object.__setattr__(clone, "_expanded_cache", cached)
        constraint = getattr(self, "_constraint_cache", None)
        if constraint is not None:
            # The symmetry does not change when only the cell does, so the
            # constraint carries over. Without this the cache is empty on
            # every trial cell, which is every residual evaluation.
            object.__setattr__(clone, "_constraint_cache", constraint)
        return clone

    def describe(self) -> str:
        density = self.density
        lines = [
            f"{self.name}  [{self.formula or self.cell_formula()}]",
            f"  grupo espacial : {self.space_group} ({self.order} operaciones)",
            f"  celda          : {self.lattice.describe()}",
            f"  contenido      : {self.cell_formula()} "
            f"({self.atoms_per_cell} átomos por celda)",
        ]
        if density is not None:
            lines.append(f"  densidad       : {density:.3f} g/cm³")
        lines.append(f"  confianza      : {self.confidence}")
        if self.source:
            lines.append(f"  fuente         : {self.source}")
        return "\n".join(lines)


__all__ = [
    "ATOMIC_WEIGHT",
    "AVOGADRO",
    "SITE_SYMMETRY_TOLERANCE",
    "Crystal",
    "Lattice",
    "Site",
    "StructureError",
]
