"""Shared fixtures: minimal but realistic Scopus and WoS exports.

The fixtures deliberately include the awkward cases this pipeline exists to
handle: the same paper in both databases with slightly different titles, a
record with no DOI, a ``p-doped`` false positive, and a paper where the dopant
sits in a non-carbon host.
"""

from __future__ import annotations

from pathlib import Path

import pytest

WOS_TEXT = """FN Clarivate Analytics Web of Science
VR 1.0
PT J
AU Terrones, M
   Ajayan, PM
AF Terrones, Mauricio
TI Nitrogen-doped carbon nanotubes: synthesis and
   electronic structure
SO CARBON
DE N-doped CNT; pyridinic nitrogen; CVD
ID DENSITY-FUNCTIONAL THEORY; DEFECTS
AB Nitrogen-doped multi-walled carbon nanotubes were grown by chemical vapor deposition and characterised by XPS and Raman spectroscopy. Density functional theory calculations of the formation energy of pyridinic sites are reported.
C1 Penn State Univ, University Pk, PA USA
CR IIJIMA S, 1991, NATURE, V354, P56
   STONE AJ, 1986, CHEM PHYS LETT, V128, P501
NR 2
TC 145
PY 2005
DI 10.1016/j.carbon.2005.01.001
UT WOS:000228000100001
ER

PT J
AU Zhang, L
TI Boron-doped carbon nanofibers for the oxygen reduction reaction
SO NANOSCALE
DE B-doped; ORR
AB Boron-doped carbon nanofibers were prepared by electrospinning and tested for the oxygen reduction reaction in alkaline media.
CR GONG KP, 2009, SCIENCE, V323, P760
NR 1
TC 30
PY 2015
DI 10.1039/C5NR00001A
UT WOS:000350000100002
ER

PT J
AU Ivanov, A
TI Vacancy formation in single-walled carbon nanotube sponges
SO PHYS REV B
DE vacancy; aerogel
AB Monovacancies and divacancies in carbon nanotube sponges are studied with first-principles calculations and molecular dynamics.
CR IIJIMA S, 1991, NATURE, V354, P56
NR 1
TC 5
PY 2018
UT WOS:000400000100003
ER

EF
"""

SCOPUS_CSV = '''Authors,Title,Year,Source title,Cited by,DOI,Document Type,EID,Abstract,Author Keywords,Index Keywords,References
"Terrones, M.; Ajayan, P.M.","Nitrogen-doped carbon nanotubes: Synthesis and electronic structure",2005,Carbon,150,10.1016/j.carbon.2005.01.001,Article,2-s2.0-111,"Nitrogen-doped multi-walled carbon nanotubes were grown by chemical vapor deposition and studied by XPS.","N-doped CNT; pyridinic","DEFECTS","Iijima S., Nature, 354, (1991); Stone A.J., Chem Phys Lett, 128, (1986)"
"Wang, Y.","Stone-Wales defects in single-walled carbon nanotubes: a first-principles study",2012,Physical Review B,88,10.1103/PhysRevB.85.111111,Article,2-s2.0-222,"We use density functional theory to study Stone-Wales defects and monovacancies in SWCNTs.","Stone-Wales; DFT","","Iijima S., Nature, 354, (1991)"
"Silva, R.","N-doped TiO2 supported on multi-walled carbon nanotubes for photocatalysis",2019,Applied Catalysis B,12,10.1016/j.apcatb.2019.02.002,Article,2-s2.0-333,"Nitrogen-doped TiO2 nanoparticles were deposited on multi-walled carbon nanotubes and were prepared by a sol-gel route for photocatalytic degradation.","TiO2; photocatalysis","","Iijima S., Nature, 354, (1991)"
"Okoro, C.","p-doped silicon nanowire transistors with carbon nanotube contacts",2016,Nano Letters,7,,Article,2-s2.0-444,"We report p-doped and n-type silicon nanowires contacted by carbon nanotubes; defect scattering is analysed.","silicon; transistor","","Iijima S., Nature, 354, (1991)"
'''


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    """A directory holding one WoS and one Scopus export."""
    directory = tmp_path / "raw"
    directory.mkdir()
    (directory / "wos_chunk1.txt").write_text(WOS_TEXT, encoding="utf-8")
    (directory / "scopus_chunk1.csv").write_text(SCOPUS_CSV, encoding="utf-8")
    return directory
