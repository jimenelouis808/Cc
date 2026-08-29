"""Rule lexicons for classifying records by dopant, defect, method and use.

Design notes that matter scientifically
---------------------------------------
* **Single-letter dopant abbreviations are case-sensitive.** ``N-doped`` is
  nitrogen; ``n-doped`` almost always means n-type electronic doping. Patterns
  marked ``cs=True`` are compiled without :data:`re.IGNORECASE`.
* **``P-doped`` carries a mandatory guard.** ``p-doped``/``p-type`` collision is
  the single worst false positive in this literature, so the phosphorus
  abbreviation only fires when ``phosphor*`` also appears in the record.
* **Doping mode is a separate facet from dopant identity.** Substitutional
  lattice incorporation and charge-transfer doping with HNO3 are physically
  different phenomena that the word "doping" hides. Splitting them is one of the
  distinguishing contributions of this review (see ``docs/PROTOCOL.md`` §3).

Each entry is ``(pattern, guard_or_None, case_sensitive)``. Extend freely, but
re-run ``pytest`` afterwards and re-validate a sample by hand — a lexicon change
silently changes every downstream figure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Rule", "FACETS", "HOST_MATERIALS", "SUPPORT_VERBS", "compile_facets"]


@dataclass(frozen=True, slots=True)
class Rule:
    """One compiled classification rule.

    Attributes
    ----------
    label:
        The value assigned when the rule fires.
    pattern:
        Compiled regex searched over title + abstract + keywords.
    guard:
        Optional second regex that must *also* match somewhere in the record for
        the rule to fire. Used to disambiguate single-letter abbreviations.
    """

    label: str
    pattern: re.Pattern[str]
    guard: re.Pattern[str] | None = None

    def matches(self, text: str) -> bool:
        """True when the pattern fires and any guard is satisfied."""
        if not self.pattern.search(text):
            return False
        return self.guard is None or bool(self.guard.search(text))


# (pattern, guard, case_sensitive)
_Spec = tuple[str, str | None, bool]

FACETS: dict[str, dict[str, list[_Spec]]] = {
    # ------------------------------------------------------------------ dopant
    "dopant": {
        "nitrogen": [
            (r"\bnitrogen[\s\-]?dop(?:ed|ing|ant)", None, False),
            (r"\bdoped with nitrogen\b", None, False),
            (r"\bN[\-\s]doped\b", None, True),
            (r"\bN[\-\s]?dop(?:ing|ant)\b", None, True),
            (r"\bNCNTs?\b|\bN[\-]CNTs?\b|\bN[\-]MWCNTs?\b|\bCN[x]\b", None, True),
            (r"\bpyridinic\b|\bpyrrolic\b", None, False),
            (r"\bgraphitic nitrogen\b|\bquaternary nitrogen\b|\bgraphitic N\b", None, False),
            (r"\bnitrogen (?:incorporation|substitution|functionali)", None, False),
        ],
        "boron": [
            (r"\bboron[\s\-]?dop(?:ed|ing|ant)", None, False),
            (r"\bdoped with boron\b", None, False),
            (r"\bB[\-\s]doped\b", r"boron|\bBC[23]\b", True),
            (r"\bboron (?:incorporation|substitution)", None, False),
        ],
        "phosphorus": [
            (r"\bphosphor(?:us|ous)[\s\-]?dop(?:ed|ing|ant)", None, False),
            (r"\bdoped with phosphor(?:us|ous)\b", None, False),
            # Mandatory guard: "P-doped" is p-type far more often than phosphorus.
            (r"\bP[\-\s]doped\b", r"phosphor", True),
        ],
        "sulfur": [
            (r"\bsul[fp]h?ur[\s\-]?dop(?:ed|ing|ant)", None, False),
            (r"\bdoped with sul[fp]h?ur\b", None, False),
            (r"\bS[\-\s]doped\b", r"sul[fp]h?ur|thiophen", True),
        ],
        "fluorine": [
            (r"\bfluorin(?:e|ated)[\s\-]?dop(?:ed|ing|ant)", None, False),
            (r"\bF[\-\s]doped\b", r"fluorin", True),
        ],
        "halogen_other": [
            (r"\b(?:chlorine|bromine|iodine)[\s\-]?dop(?:ed|ing|ant)", None, False),
            (r"\b(?:Cl|Br|I)[\-\s]doped\b", r"chlorin|bromin|iodin", True),
        ],
        "silicon": [
            (r"\bsilicon[\s\-]?dop(?:ed|ing|ant)", None, False),
            (r"\bSi[\-\s]doped\b", None, True),
        ],
        "oxygen": [
            (r"\boxygen[\s\-]?dop(?:ed|ing|ant)", None, False),
            (r"\bO[\-\s]doped\b", r"oxygen", True),
        ],
        "selenium": [
            (r"\bselenium[\s\-]?dop(?:ed|ing|ant)", None, False),
            (r"\bSe[\-\s]doped\b", r"selen", True),
        ],
        "transition_metal": [
            (r"\b(?:Fe|Co|Ni|Mn|Cu|Zn|Mo|W|Pt|Pd|Ru|Ir|Ag|Au)[\-\s]doped\b", None, True),
            (r"\btransition[\s\-]metal[\s\-]?dop", None, False),
            (r"\bsingle[\s\-]atom (?:catalyst|site)", None, False),
            (r"\b(?:Fe|Co|Ni)[\-\s]N[\-\s]?[4x]\b", None, True),
        ],
        "codoped": [
            (r"\bco[\s\-]?dop(?:ed|ing|ant)", None, False),
            (r"\bdual[\s\-]dop(?:ed|ing)", None, False),
            (r"\b(?:ternary|tri)[\s\-]dop(?:ed|ing)", None, False),
            (r"\b[NBSPF],\s?[NBSPF][\s\-]", None, True),
        ],
    },
    # ------------------------------------------------------------------ defect
    "defect": {
        "vacancy": [
            (r"\bvacanc(?:y|ies)\b", None, False),
            (r"\b(?:mono|di|single|double|multi)[\s\-]?vacanc", None, False),
        ],
        "stone_wales": [
            (r"\bstone[\s\-]?wales\b", None, False),
            (r"\bSW defect", None, True),
            (r"\b5[\-\s]?7[\-\s]?7[\-\s]?5\b", None, False),
            (r"\bpentagon[\s\-]heptagon\b|\b5\-7 (?:pair|defect)", None, False),
        ],
        "topological": [
            (r"\btopological defect", None, False),
            (r"\bpentagon(?:al)?\b|\bheptagon(?:al)?\b|\boctagon(?:al)?\b", None, False),
            (r"\bnon[\s\-]hexagonal\b", None, False),
            (r"\bring (?:defect|statistic)", None, False),
        ],
        "grain_boundary": [(r"\bgrain boundar", None, False)],
        "edge": [
            (r"\bedge (?:state|site|defect|atom|carbon)", None, False),
            (r"\b(?:armchair|zigzag) edge", None, False),
            (r"\bopen[\s\-](?:end|tip)(?:ed)?\b", None, False),
        ],
        "sp3": [
            (r"\bsp3?[\s\-]?(?:hybrid|defect|carbon site)", None, False),
            (r"\bsp\^?3\b", None, False),
        ],
        "adatom_interstitial": [
            (r"\badatom", None, False),
            (r"\binterstitial", None, False),
        ],
        "dangling_bond": [(r"\bdangling bond", None, False)],
        "disorder_amorphous": [
            (r"\bamorphous carbon\b", None, False),
            (r"\b(?:structural|lattice) disorder\b", None, False),
            (r"\bdisordered carbon\b", None, False),
        ],
        "irradiation_induced": [
            (r"\b(?:ion|electron|proton|neutron|gamma)[\s\-]irradiat", None, False),
            (r"\bion implantation\b", None, False),
        ],
        "defect_engineering": [(r"\bdefect[\s\-]engineer", None, False)],
        "defect_free": [(r"\bdefect[\s\-]free\b|\bpristine\b", None, False)],
    },
    # ------------------------------------------------------------- doping mode
    "doping_mode": {
        "substitutional": [
            (r"\bsubstitutional", None, False),
            (r"\bin[\s\-]situ dop", None, False),
            (r"\b(?:incorporat|substitut)\w* (?:into|in) the (?:carbon )?lattice", None, False),
            (r"\bgraphitic nitrogen\b|\bpyridinic\b|\bpyrrolic\b", None, False),
            (r"\blattice (?:substitution|incorporation)", None, False),
        ],
        "charge_transfer": [
            (r"\bcharge[\s\-]transfer dop", None, False),
            (r"\b[np][\s\-]type dop", None, False),
            (r"\b(?:HNO3|nitric acid|SOCl2|AuCl3|FeCl3|H2SO4) dop", None, False),
            (r"\bmolecular dop", None, False),
            (r"\bintercalat", None, False),
        ],
        "functionalization": [
            (r"\b(?:covalent|non[\s\-]covalent|surface|side[\s\-]?wall) functionali", None, False),
            (r"\bgraft(?:ed|ing)\b", None, False),
            (r"\bacid treatment\b|\bcarboxyl(?:ated|ation)\b", None, False),
        ],
    },
    # ----------------------------------------------------------------- method
    "method_theory": {
        "dft": [
            (r"\bdensity[\s\-]functional theor", None, False),
            (r"\bDFT\b", None, True),
            (r"\bfirst[\s\-]principle", None, False),
            (r"\bab[\s\-]initio\b", None, False),
            (r"\b(?:VASP|SIESTA|Quantum ESPRESSO|CASTEP|CP2K|GPAW|ABINIT)\b", None, True),
            (r"\bformation energ", None, False),
            (r"\b(?:PBE|GGA|LDA|HSE06|B3LYP)\b", None, True),
        ],
        "molecular_dynamics": [
            (r"\bmolecular dynamic", None, False),
            (r"\b(?:ReaxFF|AIREBO|LAMMPS|Tersoff|Brenner potential)\b", None, True),
            (r"\bclassical simulation", None, False),
        ],
        "tight_binding": [(r"\btight[\s\-]binding\b", None, False)],
        "transport_theory": [
            (r"\bnon[\s\-]?equilibrium green", None, False),
            (r"\bNEGF\b", None, True),
            (r"\bLandauer\b", None, False),
            (r"\bBoltzmann transport", None, False),
        ],
        "many_body": [
            (r"\bmany[\s\-]body perturbation", None, False),
            (r"\bBethe[\s\-]Salpeter\b", None, False),
            (r"\bGW approximation\b", None, False),
        ],
        "monte_carlo": [(r"\b(?:kinetic )?monte[\s\-]carlo\b", None, False)],
        "machine_learning": [
            (r"\bmachine[\s\-]learn(?:ing|ed) (?:potential|interatomic)", None, False),
            (r"\bneural network potential", None, False),
            (r"\bmachine learning\b", None, False),
            (r"\bhigh[\s\-]throughput screening\b", None, False),
        ],
    },
    "method_experiment": {
        "synthesis_cvd": [
            (r"\bchemical vap[o]?ur? deposition\b", None, False),
            (r"\bCVD\b", None, True),
            (r"\b(?:floating catalyst|aerosol[\s\-]assisted|PECVD)\b", None, False),
        ],
        "synthesis_other": [
            (r"\barc[\s\-]discharge\b", None, False),
            (r"\blaser ablation\b", None, False),
            (r"\belectrospinn?ing\b", None, False),
            (r"\bpyrolysis\b|\bcarboniz", None, False),
            # Weak but broad signal: most experimental abstracts say one of these
            # somewhere, and without it short abstracts fall into "unclear".
            (r"\b(?:were|was|is|are) (?:synthesi|prepar|fabricat|grown|produced|obtained)", None, False),
            (r"\bwe (?:synthesi|prepar|fabricat|report the synthesis)", None, False),
        ],
        "raman": [
            (r"\bRaman (?:spectroscop|spectra|scattering|analysis)", None, False),
            (r"\b(?:I_?D\s*/\s*I_?G|ID/IG)\b", None, True),
            (r"\bD[\s\-]band\b|\bG[\s\-]band\b|\b2D band\b", None, True),
        ],
        "xps": [
            (r"\bX[\s\-]ray photoelectron\b", None, False),
            (r"\bXPS\b", None, True),
            (r"\bN 1s\b|\bC 1s\b|\bB 1s\b", None, True),
        ],
        "electron_microscopy": [
            (r"\btransmission electron microscop", None, False),
            (r"\b(?:HRTEM|TEM|STEM|HAADF|SEM)\b", None, True),
            (r"\belectron energy[\s\-]loss\b|\bEELS\b", None, False),
        ],
        "scanning_probe": [
            (r"\bscanning tunn?el(?:l)?ing (?:microscop|spectroscop)", None, False),
            (r"\bSTM\b|\bSTS\b|\bAFM\b", None, True),
        ],
        "xray_absorption": [
            (r"\bXANES\b|\bNEXAFS\b|\bEXAFS\b|\bXAS\b", None, True),
            (r"\bnear[\s\-]edge X[\s\-]ray absorption\b", None, False),
            (r"\bsynchrotron\b", None, False),
        ],
        "spin_resonance": [
            (r"\belectron (?:paramagnetic|spin) resonance\b", None, False),
            (r"\bEPR\b|\bESR\b", None, True),
        ],
        "electrochemistry": [
            (r"\bcyclic voltammetr", None, False),
            (r"\brotating (?:disk|ring[\s\-]disk) electrode\b|\bRDE\b|\bRRDE\b", None, False),
            (r"\belectrochemical impedance\b|\bEIS\b", None, False),
            (r"\bTafel slope\b", None, False),
        ],
    },
    # ------------------------------------------------------------- morphology
    "morphology": {
        "forest_vacnt": [
            (r"\b(?:nanotube|CNT) forest", None, False),
            (r"\bvertically[\s\-]aligned\b", None, False),
            (r"\bVACNTs?\b|\bVA[\s\-]?CNTs?\b", None, True),
        ],
        "fiber_yarn": [
            (r"\b(?:nanotube|CNT)[\s\-](?:fib(?:er|re)|yarn|thread|filament)", None, False),
            (r"\bwet[\s\-]spun\b|\bdry[\s\-]spun\b|\bspinning of\b", None, False),
            (r"\bmacroscopic fib(?:er|re)", None, False),
        ],
        "film_buckypaper": [
            (r"\bbucky[\s\-]?paper\b", None, False),
            (r"\bfree[\s\-]standing (?:film|membrane|paper)\b", None, False),
            (r"\b(?:nanotube|CNT) (?:film|membrane|mat)\b", None, False),
        ],
        "sponge_aerogel_foam": [
            (r"\b(?:nanotube|CNT|carbon) (?:sponge|aerogel|foam|monolith)", None, False),
            (r"\baerogel\b|\bsponge\b|\bxerogel\b", None, False),
            (r"\b(?:3D|three[\s\-]dimensional) (?:network|architecture|framework|scaffold)", None, False),
        ],
        "array_bundle": [
            (r"\b(?:nanotube|CNT) (?:array|bundle|rope)", None, False),
            (r"\baligned (?:nanotube|CNT)", None, False),
        ],
        "network_junction": [
            (r"\b(?:nanotube|CNT) (?:network|junction)", None, False),
            (r"\bY[\s\-]junction\b|\bbranched nanotube", None, False),
            (r"\bpercolat", None, False),
        ],
        "composite": [
            (r"\b(?:polymer|epoxy|ceramic|cement|metal matrix) (?:composite|matrix|nanocomposite)", None, False),
            (r"\bnanocomposite\b", None, False),
        ],
    },
    # ------------------------------------------------------------ application
    "application": {
        "orr_fuelcell": [
            (r"\boxygen reduction( reaction)?\b", None, False),
            (r"\bORR\b", None, True),
            (r"\bfuel cell", None, False),
            (r"\bmetal[\s\-]free (?:electro)?catalys", None, False),
        ],
        "her_oer_water": [
            (r"\bhydrogen evolution\b|\boxygen evolution\b", None, False),
            (r"\bHER\b|\bOER\b", None, True),
            (r"\bwater splitting\b|\bwater electrolysis\b", None, False),
        ],
        "co2_n2_reduction": [
            (r"\bCO2 (?:reduction|electroreduction)\b|\bCO2RR\b", None, True),
            (r"\bnitrogen reduction reaction\b|\bNRR\b", None, True),
        ],
        "supercapacitor": [
            (r"\bsupercapacitor", None, False),
            (r"\belectrochemical (?:double[\s\-]layer )?capacitor", None, False),
            (r"\bpseudocapacit", None, False),
        ],
        "battery": [
            (r"\blithium[\s\-]ion\b|\bsodium[\s\-]ion\b|\bpotassium[\s\-]ion\b", None, False),
            (r"\blithium[\s\-]sulfur\b|\bLi[\s\-]S batter", None, False),
            (r"\b(?:metal[\s\-]air|zinc[\s\-]air) batter", None, False),
            (r"\banode material\b|\bcathode material\b", None, False),
        ],
        "sensor": [
            (r"\b(?:gas|chemical|bio|electrochemical) sens", None, False),
            (r"\bbiosensor\b|\bchemiresistor\b", None, False),
            (r"\bdetection of\b", None, False),
        ],
        "field_emission": [(r"\bfield emission\b|\bemitter\b", None, False)],
        "electronic_device": [
            (r"\bfield[\s\-]effect transistor\b|\bFET\b", None, True),
            (r"\btransistor\b|\binterconnect\b|\blogic device\b", None, False),
        ],
        "thermal_thermoelectric": [
            (r"\bthermoelectric", None, False),
            (r"\bthermal conductivity\b|\bphonon transport\b|\bSeebeck\b", None, False),
        ],
        "mechanical_composite": [
            (r"\b(?:mechanical|tensile) (?:propert|strength|reinforcement)", None, False),
            (r"\bYoung'?s modulus\b|\bfracture\b|\bstiffness\b", None, False),
        ],
        "adsorption_separation": [
            (r"\badsorption\b|\badsorbent\b", None, False),
            (r"\bCO2 capture\b|\bgas separation\b", None, False),
            (r"\bwater (?:treatment|purification|desalination)\b", None, False),
            (r"\bcapacitive deionization\b", None, False),
        ],
        "hydrogen_storage": [(r"\bhydrogen storage\b|\bhydrogen uptake\b", None, False)],
        "photocatalysis": [(r"\bphotocataly", None, False)],
        "emi_microwave": [
            (r"\bEMI shielding\b|\belectromagnetic interference\b", None, False),
            (r"\bmicrowave absor", None, False),
        ],
        "biomedical": [
            (r"\bdrug delivery\b|\bbioimaging\b|\btissue (?:engineering|scaffold)\b", None, False),
            (r"\bcytotoxic|\bbiocompatib", None, False),
        ],
    },
    # ------------------------------------------------------------- host object
    "host": {
        "carbon_1d": [
            (r"\bcarbon nano(?:tube|fib(?:er|re)|coil|filament)", None, False),
            (r"\b(?:SW|MW|DW)C?NTs?\b", None, True),
        ],
        "graphene_2d": [
            (r"\bgraphene\b|\bgraphene oxide\b|\brGO\b", None, False),
            (r"\bgraphene nanoribbon", None, False),
        ],
    },
}

#: Non-carbon materials whose presence suggests the dopant may sit in *their*
#: lattice rather than in the nanocarbon — the dominant false positive of this
#: topic (see ``docs/PROTOCOL.md`` §3).
HOST_MATERIALS = (
    r"\bTiO2\b|\bZnO\b|\bFe2O3\b|\bFe3O4\b|\bCo3O4\b|\bCeO2\b|\bSnO2\b|\bMnO2\b"
    r"|\bNiO\b|\bCu2O\b|\bWO3\b|\bMoS2\b|\bWS2\b|\bg\-C3N4\b|\bBiVO4\b|\bZnS\b"
    r"|\bCdS\b|\bLiFePO4\b|\bperovskite\b|\btitania\b|\bzirconia\b|\bsilica\b"
)

#: Verbs that indicate the nanocarbon is acting as a support, not as the host.
SUPPORT_VERBS = (
    r"\bsupported on\b|\bdecorated (?:with|on)\b|\banchored (?:on|to)\b"
    r"|\bimmobili[sz]ed on\b|\bdeposited on\b|\bloaded on\b|\bgrown on\b"
)


def compile_facets() -> dict[str, list[Rule]]:
    """Compile :data:`FACETS` into ``{facet: [Rule, ...]}``.

    Called once at import time by :mod:`nanocarbon_biblio.classify`; exposed
    separately so tests can compile a modified lexicon without monkeypatching.
    """
    compiled: dict[str, list[Rule]] = {}
    for facet, labels in FACETS.items():
        rules: list[Rule] = []
        for label, specs in labels.items():
            for pattern, guard, case_sensitive in specs:
                flags = 0 if case_sensitive else re.IGNORECASE
                rules.append(
                    Rule(
                        label=label,
                        pattern=re.compile(pattern, flags),
                        guard=re.compile(guard, re.IGNORECASE) if guard else None,
                    )
                )
        compiled[facet] = rules
    return compiled
