"""carbonforge — modular framework for nanocarbon structure generation,
validation and export to Quantum ESPRESSO / LAMMPS."""

from .utils.constants import CC_BOND, MIN_CC_DISTANCE, MAX_CC_DISTANCE

__version__ = "0.1.0"
__all__ = ["CC_BOND", "MIN_CC_DISTANCE", "MAX_CC_DISTANCE", "__version__"]
