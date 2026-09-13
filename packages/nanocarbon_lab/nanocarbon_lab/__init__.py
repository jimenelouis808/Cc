"""nanocarbon_lab — modular framework for nanocarbon structure generation,
validation and export to Quantum ESPRESSO / LAMMPS."""

from .utils.constants import CC_BOND, MAX_CC_DISTANCE, MIN_CC_DISTANCE

__version__ = "0.1.0"
__all__ = ["CC_BOND", "MAX_CC_DISTANCE", "MIN_CC_DISTANCE", "__version__"]
