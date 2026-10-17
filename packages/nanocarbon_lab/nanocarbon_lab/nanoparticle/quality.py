"""Judging a carved particle, which is not judged the way carbon is.

:func:`~nanocarbon_lab.validation.quality.sp2_quality` asks whether every
bond is between 1.30 and 1.55 Å and every angle between 100 and 135
degrees. On a gold particle that question has no answer: the nearest
contact is 2.88 Å, there are no rings to count and no angle anything is
trying to be. So this module exists for the same reason
:mod:`nanocarbon_lab.tmd.quality` does -- a different material is a
different set of questions -- and it asks the four that a cut can get
wrong:

* **Did the cut move anything?** It must not. The carver only selects
  atoms, so the shortest contact in the particle has to be the shortest
  contact in the crystal. If it is shorter, the imported cell was
  already broken (partial occupancies expanded onto one site is the
  usual cause) and every particle cut from it carries that.
* **Is anything barely attached?** Measured against each element's own
  bulk coordination, not a threshold: an atom with one neighbour where
  the bulk has twelve is a leftover of the cut.
* **Is the composition the compound's?** For a single element this is
  vacuous. For anything else it is the question that decides whether the
  cluster is charged.
* **Is there enough of it to be a particle?** A cut whose surface is
  most of its volume is a real small particle; a cut with no interior at
  all is a shell of the shape rather than a particle of the material,
  and the two are worth telling apart. Unless it was *asked* to be
  hollow -- a nanocage is all surface because that is what a nanocage is,
  and judging one by the solid thresholds is the mistake
  :func:`~nanocarbon_lab.validation.quality.sp2_quality` made when its
  hexagonal window called every sound haeckelite broken. The carver
  records which was asked for, so this does not have to guess.

**There is no bond-length window here, and there should not be.** The
geometry of a carved particle is the crystal's own, exactly, because
nothing has been relaxed. Checking it against a window would be checking
the ``.cif``, which :func:`~nanocarbon_lab.nanoparticle.crystal.read_crystal`
already did. What a window *would* be good for is a particle after
relaxation, and nothing here relaxes a metal -- the valence force field
in this framework is a carbon one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from .carve import CarveReport

Verdict = Literal["clean", "strained", "broken"]

#: Shortest contact, in Å, below which the particle is broken whatever
#: the material: no pair of atoms in any crystal sits this close.
HARD_CONTACT = 0.8

#: Relative stoichiometry deviation that turns a compound particle from
#: clean to strained, and the one that turns it broken. The first is
#: roughly what an arbitrary cut of a few hundred atoms cannot avoid;
#: the second is a composition nobody would call the compound.
STOICHIOMETRY_STRAINED = 0.02
STOICHIOMETRY_BROKEN = 0.15

#: Surface fraction above which the particle has essentially no
#: interior. Not an error -- it is what a 1 nm particle is -- but it is
#: the boundary past which no bulk property of the material applies.
SURFACE_STRAINED = 0.60
SURFACE_BROKEN = 0.90

#: Mean coordination as a fraction of the bulk value, below which the
#: cut has produced something more like a cluster of fragments than a
#: particle.
COORDINATION_BROKEN = 0.45


def particle_quality(report: CarveReport) -> tuple[Verdict, str]:
    """Classify a carved particle as clean, strained or broken.

    Parameters
    ----------
    report
        The :class:`~nanocarbon_lab.nanoparticle.carve.CarveReport` from
        the cut. Everything judged here was measured there; this function
        adds no measurement of its own, it applies the thresholds -- so
        the numbers in the explanation and the numbers in the report can
        never disagree.

    Returns
    -------
    (verdict, explanation)
        ``"clean"`` -- a particle of the material: stoichiometric to
        within what a cut can manage, nothing barely attached, and an
        interior.
        ``"strained"`` -- intact and usable, with something about it
        that will show up in any calculation: an off-stoichiometric
        compound, or a particle small enough to be all surface.
        ``"broken"`` -- overlapping atoms, a composition that is not the
        compound's, or so little coordination left that it is fragments.
        The explanation names which, with the number.

    Notes
    -----
    A "strained" particle here is usually not a problem with the build at
    all -- it is the physics of being small. The verdict exists to make
    that visible before somebody quotes a bulk comparison, not to
    suggest the cut be redone.
    """
    problems: list[str] = []
    notes: list[str] = []

    if report.min_distance < HARD_CONTACT:
        problems.append(
            f"shortest contact {report.min_distance:.3f} A is below "
            f"{HARD_CONTACT} A, so two atoms are on top of each other. The "
            "cut does not move atoms, so this came in with the crystal")

    if len(report.composition) > 1:
        deviation = report.stoichiometry_deviation
        if deviation > STOICHIOMETRY_BROKEN:
            problems.append(
                f"composition is {100 * deviation:.0f} % off the crystal's "
                "ratios, which is not this compound any more")
        elif deviation > STOICHIOMETRY_STRAINED:
            notes.append(
                f"composition {100 * deviation:.1f} % off stoichiometry, so "
                "an ionic core is charged")

    bulk = _mean_bulk(report.bulk_coordination, report.composition)
    if bulk > 0:
        fraction = report.mean_coordination / bulk
        if fraction < COORDINATION_BROKEN:
            problems.append(
                f"mean coordination {report.mean_coordination:.1f} is only "
                f"{100 * fraction:.0f} % of the bulk's {bulk:.1f}, which is "
                "fragments rather than a particle")

    if report.hollow:
        # A nanocage is almost all surface because that is what it is.
        # Judging it by the solid particle's thresholds is the mistake
        # `sp2_quality` made when its hexagonal window called every sound
        # haeckelite broken: the window was right and the family was
        # wrong.
        notes.append(
            f"hollow by design, {100 * report.surface_fraction:.0f} % "
            "surface")
    elif report.surface_fraction > SURFACE_BROKEN:
        problems.append(
            f"{100 * report.surface_fraction:.0f} % of the atoms are on the "
            "surface, so there is no interior at all")
    elif report.surface_fraction > SURFACE_STRAINED:
        notes.append(
            f"{100 * report.surface_fraction:.0f} % of the atoms are on the "
            "surface, so nothing here will behave like the bulk")

    if report.atoms_on_the_cut:
        notes.append(
            f"{report.atoms_on_the_cut} atoms sit exactly on the cutting "
            "surface, so the atom count is sensitive to the last digit of "
            "the size")

    if problems:
        return "broken", "; ".join(problems)
    if notes:
        return "strained", "; ".join(notes)
    return "clean", (
        f"{report.n_atoms} atoms, stoichiometric, "
        f"{100 * report.surface_fraction:.0f} % surface, shortest contact "
        f"{report.min_distance:.3f} A")


def _mean_bulk(bulk_coordination: Mapping[str, float],
               composition: Mapping[str, int]) -> float:
    """Composition-weighted bulk coordination.

    Weighted rather than averaged over elements: in rutile the metal has
    six neighbours and the oxygen three, and the unweighted mean of those
    is not the coordination of anything.
    """
    total = sum(composition.values())
    if not total:
        return 0.0
    return sum(bulk_coordination.get(element, 0.0) * count
               for element, count in composition.items()) / total


__all__ = [
    "COORDINATION_BROKEN",
    "HARD_CONTACT",
    "STOICHIOMETRY_BROKEN",
    "STOICHIOMETRY_STRAINED",
    "SURFACE_BROKEN",
    "SURFACE_STRAINED",
    "Verdict",
    "particle_quality",
]
