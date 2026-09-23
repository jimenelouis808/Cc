"""Script-level core of vibspec: no GUI imports, runnable in batch on a cluster."""

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
from .presets import PRESETS, Preset, apply_preset, describe_presets, get_preset
from .sites import (
    EdgeSite,
    edge_sites,
    interior_carbons,
    pick_edge_site,
    pick_interior_carbon,
    strip_hydrogen,
    zigzag_runs,
)

__all__ = [
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
