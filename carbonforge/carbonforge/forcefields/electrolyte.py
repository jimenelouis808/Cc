"""Building electrolyte boxes to fill an EDLC cell.

Three electrolytes are provided, covering the three families used in
supercapacitor work:

* **Aqueous** — SPC/E water plus alkali-halide ions. Cheap, well
  characterised, and the reference case.
* **Ionic liquid** — a coarse-grained one-site-per-ion model. Crude, but it
  is what makes the microsecond timescales an ionic-liquid double layer
  needs affordable, and it reproduces capacitance trends.
* **Vacuum** — no electrolyte, for testing the electrode alone.

Molecules are placed on a jittered lattice with a hard-sphere rejection
check. That is a starting configuration, not an equilibrated liquid: the
generated LAMMPS script equilibrates before measuring anything, which is
where the real structure comes from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from ase import Atoms

from ..utils.rng import make_rng
from .parameters import SPCE_GEOMETRY

ElectrolyteKind = Literal["aqueous", "ionic_liquid", "vacuum"]

#: Bulk densities used to work out how many molecules fit a given volume.
DENSITIES = {
    "water": 1.0,          # g/cm3
    "ionic_liquid": 1.36,  # g/cm3, typical for BMIM-PF6
}

_MOLAR_MASS = {"water": 18.015, "BMIM": 139.22, "PF6": 144.96}
#: Avogadro's number scaled for g/cm3 -> molecules/A^3.
_N_PER_A3 = 0.0006022142


@dataclass
class ElectrolyteBox:
    """A filled electrolyte region, ready to be stacked with electrodes.

    Attributes
    ----------
    atoms
        The molecules, positioned within the box.
    types
        Force-field type per atom, parallel to ``atoms``.
    molecule_ids
        Which molecule each atom belongs to, 1-based. LAMMPS needs this for
        rigid-body and SHAKE constraints.
    composition
        How many of each species were placed.
    """

    atoms: Atoms
    types: list[str] = field(default_factory=list)
    molecule_ids: list[int] = field(default_factory=list)
    composition: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.atoms)

    def summary(self) -> str:
        lines = [f"{len(self.atoms)} átomos de electrolito:"]
        for species, count in sorted(self.composition.items()):
            lines.append(f"  {species}: {count}")
        return "\n".join(lines)


def _water_molecule(centre: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return SPC/E water coordinates (O, H, H) at a random orientation."""
    d = SPCE_GEOMETRY["oh_distance"]
    half_angle = math.radians(SPCE_GEOMETRY["hoh_angle"] / 2.0)

    # Build in a local frame, then rotate randomly.
    local = np.array([
        [0.0, 0.0, 0.0],
        [d * math.sin(half_angle), 0.0, d * math.cos(half_angle)],
        [-d * math.sin(half_angle), 0.0, d * math.cos(half_angle)],
    ])

    # Uniform random rotation (Shoemake).
    u1, u2, u3 = rng.random(3)
    q = np.array([
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    ])
    x, y, z, w = q
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return centre + local @ rotation.T


def _candidate_sites(
    box: tuple[float, float, float],
    z_range: tuple[float, float],
    n_needed: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Jittered lattice points covering the electrolyte region.

    A lattice keeps molecules apart without the rejection loop having to work
    hard; the jitter stops the starting configuration being a crystal, which
    would take longer to melt than to place.
    """
    lx, ly = box[0], box[1]
    lz = z_range[1] - z_range[0]
    volume = lx * ly * lz
    if volume <= 0:
        raise ValueError("La región de electrolito tiene volumen nulo.")

    # Grid dense enough to hold what is asked for. Truncating each axis
    # independently loses up to one row per dimension, which compounds, so
    # ask for 30 % more sites than needed and round up.
    spacing = (volume / max(n_needed * 1.3, 1)) ** (1 / 3)
    nx = max(1, int(math.ceil(lx / spacing)))
    ny = max(1, int(math.ceil(ly / spacing)))
    nz = max(1, int(math.ceil(lz / spacing)))

    points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                points.append([
                    (i + 0.5) * lx / nx,
                    (j + 0.5) * ly / ny,
                    z_range[0] + (k + 0.5) * lz / nz,
                ])
    sites = np.array(points)
    if len(sites) == 0:
        raise ValueError("No caben moléculas en la región indicada.")

    # Jitter, but only lightly: at liquid density the lattice spacing is
    # barely 3 Å, and a molecule's hydrogens already reach 1 Å out, so a
    # large displacement guarantees overlaps that the rejection check below
    # would then have to undo.
    jitter = 0.12 * min(lx / nx, ly / ny, lz / nz)
    sites = sites + rng.uniform(-jitter, jitter, size=sites.shape)
    rng.shuffle(sites)
    return sites


def _fits(
    candidate: np.ndarray,
    placed: np.ndarray,
    box: tuple[float, float, float],
    min_distance: float,
) -> bool:
    """Whether a molecule's atoms clear everything already placed.

    Checked under the minimum-image convention in x and y, since the cell is
    periodic in the plane and a molecule near the edge is a neighbour of one
    at the opposite face.
    """
    if len(placed) == 0:
        return True
    delta = candidate[:, None, :] - placed[None, :, :]
    for axis in (0, 1):
        length = box[axis]
        delta[:, :, axis] -= length * np.round(delta[:, :, axis] / length)
    return bool(np.linalg.norm(delta, axis=-1).min() >= min_distance)


def _next_free_site(
    sites: np.ndarray,
    cursor: int,
    placed: np.ndarray,
    box: tuple[float, float, float],
    min_distance: float,
    n_atoms: int,
    builder=None,
    orientation_tries: int = 12,
) -> tuple[np.ndarray, int]:
    """Advance through candidate sites until a molecule fits.

    For a molecule with extent — water, whose hydrogens reach 1 Å out — most
    rejections are about *orientation*, not position: the site is fine and
    the molecule is simply turned the wrong way. So each site gets several
    random orientations before being abandoned. Skipping straight to the next
    site instead makes it impossible to reach liquid density, since at 30 Å³
    per molecule the sites are only ~3 Å apart and almost every first attempt
    clashes.

    Returns the accepted coordinates and the new cursor. Raises when the pool
    is exhausted, which genuinely means the region is too small — not
    something to paper over by letting atoms overlap.
    """
    tries = orientation_tries if builder is not None else 1
    while cursor < len(sites):
        centre = sites[cursor]
        cursor += 1
        for _ in range(tries):
            coords = builder(centre) if builder else centre.reshape(1, 3)
            if _fits(coords, placed, box, min_distance):
                return coords, cursor
    raise ValueError(
        "No caben más moléculas sin solaparse. La región es demasiado "
        "pequeña para la densidad pedida: amplía la separación entre "
        "electrodos, o baja la densidad o la molaridad."
    )


def build_aqueous(
    box: tuple[float, float, float],
    z_range: tuple[float, float],
    salt: str = "NaCl",
    molarity: float = 1.0,
    density: float = DENSITIES["water"],
    seed: Optional[int] = None,
    min_distance: float = 1.5,
) -> ElectrolyteBox:
    """Fill a region with SPC/E water and dissolved salt.

    Parameters
    ----------
    box
        Cell dimensions ``(lx, ly, lz)`` in Å.
    z_range
        The ``(z_min, z_max)`` slab the electrolyte occupies, between the
        electrodes.
    salt
        ``"NaCl"``, ``"KCl"`` or ``"LiCl"``. Charge neutrality is enforced by
        placing equal numbers of cations and anions.
    molarity
        Salt concentration in mol/L.
    density
        Water density in g/cm³, used to work out the molecule count.
    seed
        RNG seed, so the same call always gives the same starting box.
    min_distance
        Closest approach allowed between atoms of different molecules, in Å.
        1.5 Å sits just below a hydrogen bond, which is what lets liquid
        density be reached at all, while still keeping the configuration far
        from the overlaps that make a first MD step diverge. The generated
        script minimises before any dynamics, which relaxes the rest.

    Returns
    -------
    ElectrolyteBox

    Raises
    ------
    ValueError
        If the region cannot hold the requested amount without overlaps.
    """
    cation = {"NaCl": "Na", "KCl": "K", "LiCl": "Li"}.get(salt)
    if cation is None:
        raise ValueError(
            f"Sal desconocida: '{salt}'. Opciones: NaCl, KCl, LiCl."
        )

    rng = make_rng(seed)
    volume = box[0] * box[1] * (z_range[1] - z_range[0])  # Å³
    volume_litres = volume * 1e-27

    n_water = int(density * _N_PER_A3 * volume / _MOLAR_MASS["water"] * 1000)
    n_water = max(n_water, 1)
    # mol/L * L * N_A
    n_ion_pairs = int(molarity * volume_litres * 6.02214e23)

    n_molecules = n_water + 2 * n_ion_pairs
    # Generous headroom: the rejection check discards sites, so the pool has
    # to be larger than the number of molecules wanted.
    # One lattice site per molecule, with a little slack. Inflating the pool
    # would place the sites closer than the molecules themselves are, which
    # only guarantees clashes.
    sites = _candidate_sites(box, z_range, int(n_molecules * 1.15), rng)

    symbols: list[str] = []
    positions: list[np.ndarray] = []
    types: list[str] = []
    molecule_ids: list[int] = []
    molecule = 0
    cursor = 0
    placed = np.zeros((0, 3))

    # Ions first: they are single sites and easier to fit once the box is
    # crowded with water.
    for species, count in ((cation, n_ion_pairs), ("Cl", n_ion_pairs)):
        for _ in range(count):
            site, cursor = _next_free_site(
                sites, cursor, placed, box, min_distance, n_atoms=1,
            )
            molecule += 1
            symbols.append(species)
            positions.append(site[0])
            types.append(species)
            molecule_ids.append(molecule)
            placed = np.vstack([placed, site])

    for _ in range(n_water):
        def make(centre):
            return _water_molecule(centre, rng)

        coords, cursor = _next_free_site(
            sites, cursor, placed, box, min_distance, n_atoms=3, builder=make,
        )
        molecule += 1
        symbols += ["O", "H", "H"]
        positions += [coords[0], coords[1], coords[2]]
        types += ["OW", "HW", "HW"]
        molecule_ids += [molecule] * 3
        placed = np.vstack([placed, coords])

    atoms = Atoms(
        symbols=symbols,
        positions=np.array(positions) if positions else np.zeros((0, 3)),
        cell=np.diag(box),
        pbc=(True, True, False),
    )
    return ElectrolyteBox(
        atoms=atoms,
        types=types,
        molecule_ids=molecule_ids,
        composition={
            "H2O": n_water, cation: n_ion_pairs, "Cl": n_ion_pairs,
        },
    )


def build_ionic_liquid(
    box: tuple[float, float, float],
    z_range: tuple[float, float],
    density: float = DENSITIES["ionic_liquid"],
    seed: Optional[int] = None,
) -> ElectrolyteBox:
    """Fill a region with a coarse-grained BMIM-PF6 ionic liquid.

    One site per ion. That is a severe simplification — it throws away the
    cation's shape and its ability to lie flat against a surface — but an
    all-atom ionic liquid needs microseconds to equilibrate at an electrode,
    and the coarse-grained model reproduces capacitance trends at a
    thousandth of the cost.
    """
    rng = make_rng(seed)
    volume = box[0] * box[1] * (z_range[1] - z_range[0])
    pair_mass = _MOLAR_MASS["BMIM"] + _MOLAR_MASS["PF6"]
    n_pairs = max(1, int(density * _N_PER_A3 * volume / pair_mass * 1000))

    sites = _candidate_sites(box, z_range, int(2 * n_pairs * 1.15), rng)

    symbols: list[str] = []
    positions: list[np.ndarray] = []
    types: list[str] = []
    molecule_ids: list[int] = []
    cursor = 0
    placed = np.zeros((0, 3))
    # Coarse-grained ions are large (sigma ~5 Å) so they need more room than
    # water, but not as much as their sigma suggests: at 1.36 g/cm3 the ions
    # sit only ~5.6 Å apart, so demanding much more than 4 Å between centres
    # rejects almost every lattice site. Minimisation absorbs the rest.
    ion_min_distance = 4.0

    for index in range(n_pairs):
        for species in ("BMIM", "PF6"):
            site, cursor = _next_free_site(
                sites, cursor, placed, box, ion_min_distance, n_atoms=1,
            )
            # Single sites; the element is only for visualisation.
            symbols.append("N" if species == "BMIM" else "P")
            positions.append(site[0])
            types.append(species)
            molecule_ids.append(index + 1)
            placed = np.vstack([placed, site])

    atoms = Atoms(
        symbols=symbols,
        positions=np.array(positions),
        cell=np.diag(box),
        pbc=(True, True, False),
    )
    return ElectrolyteBox(
        atoms=atoms,
        types=types,
        molecule_ids=molecule_ids,
        composition={"BMIM": n_pairs, "PF6": n_pairs},
    )


def build_electrolyte(
    kind: ElectrolyteKind,
    box: tuple[float, float, float],
    z_range: tuple[float, float],
    **kwargs,
) -> ElectrolyteBox:
    """Dispatch to the requested electrolyte builder."""
    if kind == "aqueous":
        return build_aqueous(box, z_range, **kwargs)
    if kind == "ionic_liquid":
        return build_ionic_liquid(box, z_range, **kwargs)
    if kind == "vacuum":
        empty = Atoms(cell=np.diag(box), pbc=(True, True, False))
        return ElectrolyteBox(atoms=empty)
    raise ValueError(
        f"Electrolito desconocido: '{kind}'. "
        "Opciones: aqueous, ionic_liquid, vacuum."
    )
