"""Nanoparticles cut out of an imported crystal, and coated with carbon.

The one part of this framework whose input is somebody else's file. Every
other builder here generates a structure from indices and lengths; this
one starts from a ``.cif`` -- a real measured crystal, from a database or
from the user's own refinement -- and cuts a particle out of it.

Four modules, in the order a build uses them:

* :mod:`~nanocarbon_lab.nanoparticle.crystal` reads the file, says what
  is in it, and turns Miller indices into Cartesian facet normals
  through the reciprocal lattice.
* :mod:`~nanocarbon_lab.nanoparticle.shapes` is the catalogue of shapes
  and the algebra that composes them: signed distances, so a shape
  nobody anticipated is a union or a difference rather than new code.
* :mod:`~nanocarbon_lab.nanoparticle.carve` does the cut, and measures
  the four things the cut decides and the caller did not: the centring,
  what is left dangling, the stoichiometry, and whether a facet landed
  on a lattice plane.
* :mod:`~nanocarbon_lab.nanoparticle.coat` wraps the result in
  graphitic carbon -- pure, doped or functionalised -- through the same
  implicit route the junctions and schwarzites use.
* :mod:`~nanocarbon_lab.nanoparticle.quality` turns those measurements
  into the one-line verdict the rest of this framework prints. It is
  separate from the sp2 one because a gold particle's nearest contact is
  2.88 Å and has no business being compared to a bond length.

**This is deliberately not under** ``builders/``. A builder there returns
nanocarbon; what comes out of here is gold, or magnetite, or iron
carbide, with carbon on the outside at most. The same reasoning keeps
:mod:`nanocarbon_lab.tmd` separate: the carbon machinery does not
transfer, because there are no rings to count and the sp2 windows mean
nothing for a metal. What *is* reused is everything that is about
surfaces rather than about carbon -- the signed-distance fields, the
marching cubes, the isotropic remesher and the valence force field -- and
that reuse is the reason the coating is only a few hundred lines.
"""

from .carve import CarveReport, carve_nanoparticle, coordination
from .crystal import (
    CrystalReport,
    family_normals_for,
    interplanar_spacing,
    lattice_point_group,
    miller_family,
    miller_normal,
    read_crystal,
    repeats_for,
    supercell_for,
)
from .quality import particle_quality
from .shapes import (
    SHAPE_GROUPS,
    SHAPES,
    Shape,
    difference,
    hollow,
    hollowed,
    intersection,
    make_shape,
    offset,
    smooth_union,
    union,
    wulff,
)

__all__ = [
    "SHAPES",
    "SHAPE_GROUPS",
    "CarveReport",
    "CrystalReport",
    "Shape",
    "carve_nanoparticle",
    "coordination",
    "difference",
    "family_normals_for",
    "hollow",
    "hollowed",
    "interplanar_spacing",
    "intersection",
    "lattice_point_group",
    "make_shape",
    "miller_family",
    "miller_normal",
    "offset",
    "particle_quality",
    "read_crystal",
    "repeats_for",
    "smooth_union",
    "supercell_for",
    "union",
    "wulff",
]
