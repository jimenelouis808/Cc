"""Getting structures and pseudopotentials in from elsewhere.

* :mod:`~carbonforge.io.importer` reads a structure from any of ASE's
  formats and **diagnoses what is missing** — a cell an XYZ file never had,
  atoms a CIF duplicated, vacuum too thin to be safe.
* :mod:`~carbonforge.io.autofix` repairs the unambiguous ones and explains
  what it declined to touch.
* :mod:`~carbonforge.io.upf` reads what a pseudopotential file says about
  itself, instead of guessing from its name.
* :mod:`~carbonforge.io.catalog` scans a directory of them and matches it
  against what a calculation needs.
"""

from .autofix import (
    FixRecord,
    FixResult,
    add_missing_cell,
    autofix,
    remove_duplicate_atoms,
    suggest_periodicity,
)
from .catalog import (
    DOWNLOAD_SOURCES,
    MatchResult,
    PseudoCatalog,
    download_instructions,
    match_requirements,
    scan_directory,
)
from .importer import (
    IMPORT_FORMATS,
    ImportIssue,
    ImportResult,
    diagnose,
    import_structure,
)
from .upf import PseudoInfo, read_upf_header

__all__ = [
    # structures
    "import_structure",
    "ImportResult",
    "ImportIssue",
    "IMPORT_FORMATS",
    "diagnose",
    # repairs
    "autofix",
    "FixResult",
    "FixRecord",
    "remove_duplicate_atoms",
    "add_missing_cell",
    "suggest_periodicity",
    # pseudopotentials
    "read_upf_header",
    "PseudoInfo",
    "scan_directory",
    "PseudoCatalog",
    "match_requirements",
    "MatchResult",
    "download_instructions",
    "DOWNLOAD_SOURCES",
]
