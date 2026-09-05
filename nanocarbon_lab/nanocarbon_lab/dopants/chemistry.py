"""Which heteroatoms may replace a carbon, and how many of them.

The host is **always carbon**. Doping is an edit applied to a finished
nanocarbon, never a different material, and every builder returns pure
carbon unless a dopant is asked for explicitly.

What this module adds is the part a bare element list cannot say: the
elements differ enormously in how much substitution they tolerate, and a
single `DOPANT_ELEMENTS` tuple invited 10% Fe-doped graphene as readily
as 10% N-doped graphene. One of those is a real material and the other
is not.

Three site types, and they are a statement about where the atom ends up
rather than about how it is placed:

``planar``
    Fits the sp2 lattice with the sheet staying flat. Only N and B --
    both within 0.15 Å of carbon's covalent radius, and both
    isoelectronic with it to within one electron. These are the only
    dopants that reach tens of per cent in real samples, where they
    order into stoichiometric phases (BC3 at 25%, C3N4 beyond).

``puckered``
    Substitutes, but pulls its site out of the plane because the atom is
    too big for a 1.42 Å lattice: P and S are ~40% larger than carbon,
    Se and Ge more. Real concentrations are a few per cent, and the
    local geometry is sp3-like, so **these structures need a relaxation
    before they mean anything** -- the builder places them on the ideal
    lattice site and cannot know how far out they will move.

``vacancy``
    Not a lattice substitution at all in reality. The 3d metals sit in a
    mono- or divacancy, usually with nitrogen co-doped around them
    (the M-N4 motif of single-atom catalysis), and are isolated by
    construction -- two of them adjacent is a different, much less stable
    object. Substituting one for a carbon on a perfect lattice is a
    *starting geometry* for that, not the motif itself.

``max_fraction`` is the ceiling above which a warning fires. They are
literature-scale numbers, not hard limits: doping past them is allowed,
because metastable and computational structures are a legitimate subject,
but silently is not.

Nothing here is a hard error except an unknown element. The module's
posture throughout the package is that chemistry the user may have meant
gets a warning and chemistry that cannot be represented gets an
exception.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..utils.constants import COVALENT_RADII

#: How a dopant sits once substituted. See the module docstring.
SiteType = str


@dataclass(frozen=True)
class DopantChemistry:
    """What is known about one substitutional dopant for sp2 carbon.

    Attributes
    ----------
    symbol
        Element symbol.
    site
        ``"planar"``, ``"puckered"`` or ``"vacancy"``.
    max_fraction
        Substitution fraction above which the placement stops describing
        a real material and a warning is issued.
    note
        One line on what the dopant actually does, shown by the CLI and
        the GUI so the choice is not made blind.
    """

    symbol: str
    site: SiteType
    max_fraction: float
    note: str

    @property
    def radius(self) -> float:
        """Covalent radius (Å)."""
        return COVALENT_RADII[self.symbol]

    @property
    def size_mismatch(self) -> float:
        """Fractional radius difference against carbon's 0.76 Å.

        The single number that predicts whether a dopant stays in the
        plane: N and B are within 0.15 Å and planar, everything past
        ~30% puckers.
        """
        return (self.radius - COVALENT_RADII["C"]) / COVALENT_RADII["C"]


#: Every heteroatom the package will substitute for carbon.
#:
#: Ordered by how much the sp2 lattice tolerates them, which is also
#: roughly the order of how often they appear in the literature. The
#: halogens are deliberately absent: F and Cl bond *to* a carbon sheet
#: rather than replacing a carbon in it, so fluorographene is an
#: adsorption problem and not a substitution one.
DOPANT_CHEMISTRY: dict[str, DopantChemistry] = {
    "N": DopantChemistry(
        "N", "planar", 0.20,
        "The canonical dopant: smaller than carbon, n-type, and the only "
        "one with three distinct well-known motifs (graphitic, pyridinic, "
        "pyrrolic). Substitution here gives the graphitic one."),
    "B": DopantChemistry(
        "B", "planar", 0.20,
        "p-type counterpart to N; isoelectronic with C-. Orders into BC3 "
        "at 25%, so high fractions describe a compound rather than a "
        "doped sheet."),
    "P": DopantChemistry(
        "P", "puckered", 0.05,
        "41% larger than carbon, so it puckers out of the plane and turns "
        "its site sp3. Widely used to raise the ORR activity of carbons; "
        "relax before trusting the geometry."),
    "S": DopantChemistry(
        "S", "puckered", 0.05,
        "Thiophene-like, strongly preferring edges and defects to the "
        "basal plane. In-plane substitution strains the lattice locally."),
    "Se": DopantChemistry(
        "Se", "puckered", 0.03,
        "Heavier chalcogen, larger mismatch than S and correspondingly "
        "rarer; studied for the same catalytic reasons as S doping."),
    "O": DopantChemistry(
        "O", "puckered", 0.03,
        "Substitutional O in the basal plane is uncommon -- oxygen on "
        "carbon overwhelmingly means epoxide, carbonyl or hydroxyl at a "
        "defect or edge. Use this for the substitutional case knowingly."),
    "Si": DopantChemistry(
        "Si", "puckered", 0.03,
        "Well characterised atom-by-atom in STEM, in both 3- and "
        "4-coordinate sites. Isoelectronic with carbon but 46% larger."),
    "Ge": DopantChemistry(
        "Ge", "puckered", 0.02,
        "The next isoelectronic step past Si, studied as a single-atom "
        "site; too large for anything but isolated substitution."),
    "Al": DopantChemistry(
        "Al", "puckered", 0.02,
        "p-type and strongly Lewis-acidic, of current interest as an "
        "isolated main-group catalytic site."),
    "Mn": DopantChemistry(
        "Mn", "vacancy", 0.01,
        "Single-atom catalysis: sits in a vacancy, usually with four "
        "nitrogens around it (Mn-N4), not on a pristine lattice site."),
    "Fe": DopantChemistry(
        "Fe", "vacancy", 0.01,
        "The most studied M-N4 single-atom site of all, for oxygen "
        "reduction. Needs a vacancy and N co-doping to be the real motif."),
    "Co": DopantChemistry(
        "Co", "vacancy", 0.01,
        "Co-N4, the other workhorse single-atom site; hydrogen evolution "
        "and CO2 reduction."),
    "Ni": DopantChemistry(
        "Ni", "vacancy", 0.01,
        "Ni-N4, the standout for CO2-to-CO selectivity."),
    "Cu": DopantChemistry(
        "Cu", "vacancy", 0.01,
        "Cu-N4, studied for CO2 reduction past two electrons."),
    "Zn": DopantChemistry(
        "Zn", "vacancy", 0.01,
        "d10 and redox-inert, so it is the usual control against which "
        "the catalytic 3d metals are compared."),
}

#: The symbols themselves, in table order.
DOPANT_ELEMENTS: tuple[str, ...] = tuple(DOPANT_CHEMISTRY)

#: Dopants that keep the sheet flat, for callers that want the safe set.
PLANAR_DOPANTS: tuple[str, ...] = tuple(
    symbol for symbol, chem in DOPANT_CHEMISTRY.items() if chem.site == "planar"
)


def get_chemistry(element: str) -> DopantChemistry:
    """Look up a dopant's chemistry.

    Raises
    ------
    ValueError
        Naming the supported elements. This is the one hard error in the
        module: an element with no entry has no radius, no coordination
        ceiling and no idea of what fraction is sensible, so validation
        and export would both misjudge it downstream.
    """
    try:
        return DOPANT_CHEMISTRY[element]
    except KeyError:
        raise ValueError(
            f"Unsupported dopant {element!r}. Supported: "
            f"{', '.join(DOPANT_ELEMENTS)}."
        ) from None


def describe(element: str) -> str:
    """A one-line human summary of a dopant, for the CLI and GUI."""
    chem = get_chemistry(element)
    return (f"{chem.symbol}: {chem.site}, sensible up to "
            f"{chem.max_fraction:.0%}, radius {chem.radius:.2f} Å "
            f"({chem.size_mismatch:+.0%} vs C). {chem.note}")


__all__ = [
    "DOPANT_CHEMISTRY",
    "DOPANT_ELEMENTS",
    "PLANAR_DOPANTS",
    "DopantChemistry",
    "SiteType",
    "describe",
    "get_chemistry",
]
