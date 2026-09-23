"""Script-level core of vibspec: no GUI imports, runnable in batch on a cluster."""

from .analysis import (
    BandMatch,
    ExperimentalSpectrum,
    computed_curve,
    export_csv,
    find_bands,
    fit_scale_factor,
    match_bands,
    match_table,
    normalise,
    prepare_experiment,
    read_ftir,
    rubberband_baseline,
    search_scale_factor,
    to_absorbance,
)
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
from .plot import draw_ir_comparison, plot_ir_comparison
from .workflow import VibspecError, collect, prepare, run

__all__ = [
    # analysis and plotting
    "BandMatch",
    "ExperimentalSpectrum",
    "computed_curve",
    "draw_ir_comparison",
    "export_csv",
    "find_bands",
    "fit_scale_factor",
    "match_bands",
    "match_table",
    "normalise",
    "plot_ir_comparison",
    "prepare_experiment",
    "read_ftir",
    "rubberband_baseline",
    "search_scale_factor",
    "to_absorbance",
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
