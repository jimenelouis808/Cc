"""Script-level core of vibspec: no GUI imports, runnable in batch on a cluster."""

from .calcspec import CalcSpec
from .checks import (
    FMAX_LIMIT,
    FMAX_RECOMMENDED,
    SpinAdvice,
    check_ready_for_vibrations,
    check_structure,
    check_vibration_settings,
    electron_count,
    suggest_spin,
    undercoordinated_atoms,
    unformed_pyrrolic_nitrogens,
    vacuum_per_side,
)
from .engines import gpaw_available
from .presets import PRESETS, Preset, apply_preset, describe_presets, get_preset
from .record import CalcRecord, find_records, index_records
from .sites import (
    EdgeSite,
    edge_sites,
    interior_carbons,
    pick_edge_site,
    pick_interior_carbon,
    strip_hydrogen,
    zigzag_runs,
)
from .workflow import VibspecError, collect, prepare, run

__all__ = [
    # workflow
    "CalcSpec",
    "CalcRecord",
    "VibspecError",
    "collect",
    "find_records",
    "gpaw_available",
    "index_records",
    "prepare",
    "run",
    # presets
    "PRESETS",
    "Preset",
    "apply_preset",
    "describe_presets",
    "get_preset",
    # sites
    "EdgeSite",
    "edge_sites",
    "interior_carbons",
    "pick_edge_site",
    "pick_interior_carbon",
    "strip_hydrogen",
    "zigzag_runs",
    # checks
    "FMAX_LIMIT",
    "FMAX_RECOMMENDED",
    "SpinAdvice",
    "check_ready_for_vibrations",
    "check_structure",
    "check_vibration_settings",
    "electron_count",
    "suggest_spin",
    "undercoordinated_atoms",
    "unformed_pyrrolic_nitrogens",
    "vacuum_per_side",
]
