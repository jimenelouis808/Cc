"""Synthetic demo corpus, so the GUI can be explored before touching Scopus.

Generates realistic Scopus CSV and WoS tagged exports covering 1991–2025, with
the awkward cases the pipeline exists to handle deliberately mixed in:

* the same paper in both databases with small title differences,
* records with no DOI,
* ``p-doped silicon`` false positives,
* dopants sitting in a non-carbon host ("N-doped TiO2 supported on MWCNTs"),
* co-doping phrased as an element list ("nitrogen and sulfur co-doped").

Topic mixes shift with time on purpose — theory-heavy in the 1990s, ORR
electrocatalysis appearing after 2008, 3D assemblies after 2010, single-atom
sites after 2016, machine-learned potentials after 2019 — so the temporal
figures show something worth looking at rather than noise.

**This is fabricated data.** Titles, authors and DOIs are invented. It exists to
exercise the pipeline, never to stand in for a real corpus: no number produced
from it belongs in a manuscript.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

__all__ = ["generate_demo_corpus", "DemoConfig"]

_DOPANTS = [
    ("Nitrogen-doped", "nitrogen", 0.34),
    ("Boron-doped", "boron", 0.11),
    ("Phosphorus-doped", "phosphorus", 0.06),
    ("Sulfur-doped", "sulfur", 0.07),
    ("Fluorine-doped", "fluorine", 0.04),
    ("Silicon-doped", "silicon", 0.03),
    ("Selenium-doped", "selenium", 0.02),
    ("Nitrogen and sulfur co-doped", "codoped", 0.07),
    ("Boron and nitrogen co-doped", "codoped", 0.05),
    ("Iron and nitrogen co-doped", "codoped", 0.03),
    ("Undoped", "none", 0.18),
]

_STRUCTURES_EARLY = [
    "multi-walled carbon nanotubes", "single-walled carbon nanotubes",
    "carbon nanofibers", "double-walled carbon nanotubes",
    "vapor grown carbon fibers", "carbon nanocoils",
]
_STRUCTURES_3D = [
    "carbon nanotube sponges", "vertically aligned carbon nanotube forests",
    "carbon nanotube fibers", "buckypaper films", "carbon nanotube aerogels",
    "three-dimensional carbon nanotube networks", "carbon nanotube yarns",
]

_DEFECTS = [
    ("with Stone-Wales defects", "Stone-Wales defects and pentagon-heptagon pairs were identified."),
    ("containing monovacancies", "Monovacancies and divacancies were quantified."),
    ("with divacancies", "Divacancy formation energies are reported."),
    ("with grain boundaries", "Grain boundaries in the graphitic walls were resolved."),
    ("with edge sites", "Armchair edge and zigzag edge sites dominate the response."),
    ("with dangling bonds", "Dangling bonds and sp3 carbon sites were detected."),
    ("under ion irradiation", "Ion irradiation was used to introduce point defects in a controlled way."),
    ("", ""),
]

_APPLICATIONS_ALWAYS = [
    ("for field emission", "field emission current density"),
    ("for polymer composites", "mechanical reinforcement and tensile strength of the nanocomposite"),
    ("for gas sensing", "chemical sensing response towards NO2"),
    ("for hydrogen storage", "hydrogen storage capacity"),
    ("for thermal management", "thermal conductivity and phonon transport"),
    ("", ""),
]
_APPLICATIONS_MODERN = [
    ("for the oxygen reduction reaction", "oxygen reduction reaction activity in alkaline media, measured by RRDE"),
    ("for hydrogen evolution", "hydrogen evolution reaction overpotential and Tafel slope"),
    ("for supercapacitors", "supercapacitor specific capacitance"),
    ("for lithium-ion batteries", "lithium-ion battery anode capacity"),
    ("for sodium-ion batteries", "sodium-ion storage performance"),
    ("for CO2 capture", "CO2 capture and adsorption isotherms"),
    ("for electromagnetic shielding", "EMI shielding effectiveness and microwave absorption"),
    ("for capacitive deionization", "capacitive deionization of brackish water"),
]

_THEORY_SENTENCES = [
    "Density functional theory calculations with VASP were used to obtain the formation energy and the band structure.",
    "First-principles calculations within the PBE functional describe the charge transfer and the density of states.",
    "Ab initio molecular dynamics simulations reveal the migration barrier of the defect.",
    "Tight-binding and non-equilibrium Green function calculations give the transport coefficients.",
    "Classical molecular dynamics with the AIREBO potential in LAMMPS were used to study mechanical failure.",
    "A machine-learning interatomic potential trained on DFT data was used for large-scale simulations.",
]
_EXPERIMENT_SENTENCES = [
    "Samples were grown by chemical vapor deposition and characterised by XPS and Raman spectroscopy.",
    "The material was synthesised by arc discharge and analysed by HRTEM and electron energy loss spectroscopy.",
    "Carbon nanofibers were prepared by electrospinning followed by carbonization; the N 1s XPS region was deconvoluted.",
    "Scanning tunneling microscopy and XANES were used to resolve the local electronic structure.",
    "Cyclic voltammetry and rotating ring-disk electrode measurements were performed in 0.1 M KOH.",
    "The ID/IG ratio from Raman spectroscopy tracks the defect density across the annealing series.",
]
_NITROGEN_SENTENCES = [
    "Pyridinic, pyrrolic and graphitic nitrogen configurations were distinguished.",
    "The quaternary nitrogen content correlates with the measured activity.",
]

_JOURNALS = [
    ("Carbon", 0.14), ("Journal of Physical Chemistry C", 0.09), ("Nanoscale", 0.08),
    ("Physical Review B", 0.08), ("ACS Nano", 0.06), ("Journal of Materials Chemistry A", 0.06),
    ("Applied Physics Letters", 0.05), ("Chemistry of Materials", 0.05),
    ("Advanced Functional Materials", 0.05), ("Electrochimica Acta", 0.05),
    ("Nanotechnology", 0.05), ("Diamond and Related Materials", 0.04),
    ("Applied Surface Science", 0.04), ("Journal of Power Sources", 0.04),
    ("Small", 0.03), ("Chemical Physics Letters", 0.03), ("Nano Letters", 0.03),
    ("ACS Applied Materials & Interfaces", 0.03),
]

_SURNAMES = [
    "Terrones", "Ajayan", "Endo", "Dai", "Ruoff", "Zhang", "Wang", "Li", "Chen", "Liu",
    "Kim", "Park", "Lee", "Sato", "Tanaka", "Muller", "Schmidt", "Rossi", "Ferrari",
    "Garcia", "Martinez", "Silva", "Ivanov", "Petrov", "Dubois", "Laurent", "Novak",
    "Okoro", "Adeyemi", "Nguyen", "Tran", "Gupta", "Sharma", "Patel", "Ahmed", "Hassan",
]
_INITIALS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "P", "R", "S", "T", "Y"]

_COUNTRIES = [
    ("Peoples R China", "Tsinghua Univ, Beijing", 0.26),
    ("USA", "Penn State Univ, University Pk, PA", 0.16),
    ("Japan", "Shinshu Univ, Nagano", 0.08),
    ("Germany", "Max Planck Inst, Stuttgart", 0.07),
    ("South Korea", "KAIST, Daejeon", 0.06),
    ("India", "Indian Inst Technol, Chennai", 0.06),
    ("Spain", "Univ Alicante, Alicante", 0.05),
    ("France", "CNRS, Toulouse", 0.05),
    ("Mexico", "IPN, Mexico City", 0.04),
    ("United Kingdom", "Univ Cambridge, Cambridge", 0.04),
    ("Brazil", "Univ Estadual Campinas, Campinas", 0.03),
    ("Italy", "Politecn Torino, Turin", 0.03),
    ("Russia", "Russian Acad Sci, Moscow", 0.03),
    ("Singapore", "Nanyang Technol Univ, Singapore", 0.02),
    ("Canada", "Univ Toronto, Toronto", 0.02),
]

_SEMINAL_REFS = [
    "IIJIMA S, 1991, NATURE, V354, P56",
    "STONE AJ, 1986, CHEM PHYS LETT, V128, P501",
    "GONG KP, 2009, SCIENCE, V323, P760",
    "NOVOSELOV KS, 2004, SCIENCE, V306, P666",
    "TERRONES M, 1999, NATURE, V388, P52",
    "KROTO HW, 1985, NATURE, V318, P162",
]


@dataclass(slots=True)
class DemoConfig:
    """Parameters of the synthetic corpus.

    Attributes
    ----------
    n_works:
        Number of distinct works. Roughly 55 % land in both databases, so the
        record count across the two files is about 1.55x this.
    seed:
        Fixed so two runs produce byte-identical files.
    """

    n_works: int = 1200
    seed: int = 20260830
    first_year: int = 1991
    last_year: int = 2025
    p_both: float = 0.55
    p_scopus_only: float = 0.25
    p_host_ambiguous: float = 0.04
    p_ptype_noise: float = 0.02
    p_no_doi: float = 0.06


def _weighted(rng: random.Random, options: list[tuple]) -> tuple:
    """Pick one option by its trailing weight."""
    weights = [option[-1] for option in options]
    return rng.choices(options, weights=weights, k=1)[0]


def _year(rng: random.Random, cfg: DemoConfig) -> int:
    """Draw a year from a growth curve with a plateau in the graphene years.

    Publication counts on CNT doping grow steeply through the 2000s, flatten
    around 2010-2014 as the community migrates to graphene, then recover as
    macroscopic assemblies and single-atom sites take over.
    """
    years = list(range(cfg.first_year, cfg.last_year + 1))
    weights = []
    for year in years:
        base = 1.35 ** min(year - cfg.first_year, 18)
        if 2010 <= year <= 2014:
            base *= 0.72          # the graphene shock
        if year >= 2016:
            base *= 1.25          # 3D assemblies and single-atom catalysis
        if year >= 2023:
            base *= 0.9           # indexing lag near the cut-off
        weights.append(base)
    return rng.choices(years, weights=weights, k=1)[0]


def _build_work(rng: random.Random, cfg: DemoConfig, index: int) -> dict:
    """Assemble one synthetic work with internally consistent metadata."""
    year = _year(rng, cfg)

    # Deliberate noise: p-type doping papers that must NOT be read as phosphorus.
    if rng.random() < cfg.p_ptype_noise:
        title = f"Transport in p-doped silicon nanowires contacted by carbon nanotubes (study {index})"
        abstract = ("We report p-doped and n-type silicon nanowires contacted by carbon nanotubes. "
                    "Defect scattering at the contact is analysed. "
                    + rng.choice(_EXPERIMENT_SENTENCES))
    # Deliberate noise: the dopant lives in a non-carbon host.
    elif rng.random() < cfg.p_host_ambiguous:
        oxide = rng.choice(["TiO2", "ZnO", "CeO2", "Fe3O4", "MoS2"])
        title = (f"Nitrogen-doped {oxide} nanoparticles supported on multi-walled carbon nanotubes "
                 f"for photocatalysis (system {index})")
        abstract = (f"Nitrogen-doped {oxide} was deposited on multi-walled carbon nanotubes and "
                    "anchored on the support for photocatalytic degradation. "
                    + rng.choice(_EXPERIMENT_SENTENCES))
    else:
        dopant_phrase, dopant_kind, _ = _weighted(rng, _DOPANTS)
        structures = _STRUCTURES_EARLY + (_STRUCTURES_3D if year >= 2010 else [])
        structure = rng.choice(structures)
        defect_phrase, defect_sentence = rng.choice(_DEFECTS)
        pool = list(_APPLICATIONS_ALWAYS)
        if year >= 2008:
            pool += _APPLICATIONS_MODERN
        app_phrase, app_sentence = rng.choice(pool)

        parts = [dopant_phrase, structure, defect_phrase, app_phrase]
        title = " ".join(p for p in parts if p).strip()
        title = title[0].upper() + title[1:] + f" (series {index})"

        # Theory share falls over time as the field industrialises.
        p_theory = max(0.18, 0.62 - 0.014 * (year - cfg.first_year))
        roll = rng.random()
        sentences: list[str] = []
        if roll < p_theory * 0.55:
            sentences.append(rng.choice(_THEORY_SENTENCES))
        elif roll < p_theory:
            sentences.append(rng.choice(_THEORY_SENTENCES))
            sentences.append(rng.choice(_EXPERIMENT_SENTENCES))
        else:
            sentences.append(rng.choice(_EXPERIMENT_SENTENCES))
        if year >= 2019 and rng.random() < 0.12:
            sentences.append("A machine-learning interatomic potential was fitted to the dataset.")
        if year >= 2016 and rng.random() < 0.10:
            sentences.append("Single-atom catalyst sites of the Fe-N4 type were identified.")
        if dopant_kind == "nitrogen" and rng.random() < 0.6:
            sentences.append(rng.choice(_NITROGEN_SENTENCES))
        if defect_sentence:
            sentences.append(defect_sentence)
        if app_sentence:
            sentences.append(f"We evaluate the {app_sentence}.")
        abstract = f"{dopant_phrase} {structure} were investigated. " + " ".join(sentences)

    journal, _ = _weighted(rng, _JOURNALS)
    country, affiliation, _ = _weighted(rng, _COUNTRIES)
    n_authors = rng.randint(1, 6)
    authors = [f"{rng.choice(_SURNAMES)}, {rng.choice(_INITIALS)}." for _ in range(n_authors)]

    age = max(0, 2026 - year)
    citations = int(rng.lognormvariate(0.6 + 0.09 * min(age, 22), 1.25))

    keywords = [w for w in title.replace(",", "").split() if len(w) > 4][:4]
    refs = rng.sample(_SEMINAL_REFS, k=rng.randint(1, 3))
    refs += [f"{rng.choice(_SURNAMES).upper()} {rng.choice(_INITIALS)}, "
             f"{rng.randint(1985, max(1986, year))}, CARBON, V{rng.randint(30, 200)}, "
             f"P{rng.randint(1, 999)}" for _ in range(rng.randint(2, 14))]

    doi = "" if rng.random() < cfg.p_no_doi else f"10.{rng.randint(1000, 1099)}/demo.{year}.{index:05d}"
    return {
        "index": index, "year": year, "title": title, "abstract": abstract,
        "journal": journal, "authors": authors, "citations": citations,
        "keywords": keywords, "refs": refs, "doi": doi,
        "country": country, "affiliation": affiliation,
    }


def _scopus_title(work: dict) -> str:
    """Scopus sentence-cases titles; WoS keeps the submitted casing."""
    return work["title"]


def _wos_title(work: dict, rng: random.Random) -> str:
    """Introduce the small variations that make cross-database dedup non-trivial."""
    title = work["title"]
    if rng.random() < 0.3:
        title = title.replace("-", " ")
    if rng.random() < 0.2:
        title = title.replace("carbon nanotubes", "carbon nano-tubes")
    return title


def _write_scopus(works: list[dict], path: Path) -> None:
    """Write a Scopus CSV export with the fields the loader expects."""
    columns = [
        "Authors", "Title", "Year", "Source title", "Cited by", "DOI",
        "Affiliations", "Document Type", "EID", "Abstract",
        "Author Keywords", "Index Keywords", "References",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for work in works:
            writer.writerow({
                "Authors": "; ".join(work["authors"]),
                "Title": _scopus_title(work),
                "Year": work["year"],
                "Source title": work["journal"],
                "Cited by": work["citations"],
                "DOI": work["doi"],
                "Affiliations": f"{work['affiliation']}, {work['country']}",
                "Document Type": "Article",
                "EID": f"2-s2.0-{90000000 + work['index']}",
                "Abstract": work["abstract"],
                "Author Keywords": "; ".join(work["keywords"]),
                "Index Keywords": "CARBON NANOTUBES; DOPING",
                "References": "; ".join(work["refs"]),
            })


def _write_wos(works: list[dict], path: Path, rng: random.Random) -> None:
    """Write a WoS tagged plain-text export, continuation lines and all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["FN Clarivate Analytics Web of Science", "VR 1.0"]
    for work in works:
        title = _wos_title(work, rng)
        lines.append("PT J")
        authors = [f"{a.split(',')[0]}, {a.split(',')[1].strip().replace('.', '')}"
                   for a in work["authors"]]
        lines.append(f"AU {authors[0]}")
        lines.extend(f"   {a}" for a in authors[1:])
        # Fold the title at ~65 characters, the way WoS does.
        words, current = title.split(), ""
        title_lines: list[str] = []
        for word in words:
            if len(current) + len(word) + 1 > 65:
                title_lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        title_lines.append(current)
        lines.append(f"TI {title_lines[0]}")
        lines.extend(f"   {line}" for line in title_lines[1:])
        lines.append(f"SO {work['journal'].upper()}")
        lines.append(f"DE {'; '.join(work['keywords'])}")
        lines.append("ID CARBON NANOTUBES; DOPING; DEFECTS")
        lines.append(f"AB {work['abstract']}")
        lines.append(f"C1 {work['affiliation']}, {work['country']}")
        lines.append(f"CR {work['refs'][0]}")
        lines.extend(f"   {ref}" for ref in work["refs"][1:])
        lines.append(f"NR {len(work['refs'])}")
        lines.append(f"TC {work['citations']}")
        lines.append(f"PY {work['year']}")
        if work["doi"]:
            lines.append(f"DI {work['doi']}")
        lines.append(f"UT WOS:{900000000000000 + work['index']}")
        lines.append("ER")
        lines.append("")
    lines.append("EF")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_demo_corpus(outdir: str | Path, config: DemoConfig | None = None) -> dict[str, object]:
    """Write a synthetic Scopus CSV and WoS plain-text export under ``outdir``.

    Returns a summary dict with the file paths and the true overlap counts, so
    a test can assert that the pipeline recovers them.

    The data is fabricated. It is for exercising the pipeline and learning the
    GUI, never for analysis.
    """
    cfg = config or DemoConfig()
    rng = random.Random(cfg.seed)
    outdir = Path(outdir)

    works = [_build_work(rng, cfg, i) for i in range(cfg.n_works)]
    in_scopus: list[dict] = []
    in_wos: list[dict] = []
    both = scopus_only = wos_only = 0
    for work in works:
        roll = rng.random()
        if roll < cfg.p_both:
            in_scopus.append(work); in_wos.append(work); both += 1
        elif roll < cfg.p_both + cfg.p_scopus_only:
            in_scopus.append(work); scopus_only += 1
        else:
            in_wos.append(work); wos_only += 1

    scopus_path = outdir / "scopus_demo_1991-2025.csv"
    wos_path = outdir / "wos_demo_1991-2025.txt"
    _write_scopus(in_scopus, scopus_path)
    _write_wos(in_wos, wos_path, rng)

    return {
        "scopus_file": str(scopus_path),
        "wos_file": str(wos_path),
        "n_works": len(works),
        "n_scopus_records": len(in_scopus),
        "n_wos_records": len(in_wos),
        "true_overlap": {"both": both, "scopus_only": scopus_only, "wos_only": wos_only},
        "warning": "Synthetic data. For exercising the pipeline only, never for analysis.",
    }
