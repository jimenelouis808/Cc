"""Calculator factories. GPAW is imported only here, and only when called.

carbonforge installs and works without GPAW -- on Windows, for building,
checking and analysing -- so nothing at module level may import it. A
factory has the signature ``factory(spec, txt, spinpol) -> ase Calculator``; tests pass
their own to run the whole workflow without any DFT code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

from .calcspec import CalcSpec

CalculatorFactory = Callable[[CalcSpec, Path, bool], Any]


def gpaw_available() -> bool:
    """Whether GPAW can be imported in this environment."""
    return importlib.util.find_spec("gpaw") is not None


def gpaw_version() -> str | None:
    """GPAW's version, or ``None`` when it is not installed."""
    if not gpaw_available():
        return None
    import gpaw

    return str(gpaw.__version__)


def gpaw_parameters(spec: CalcSpec, spinpol: bool) -> dict[str, Any]:
    """Keyword arguments for ``gpaw.GPAW`` implied by ``spec``.

    Point-group symmetry is switched off: every finite displacement breaks
    it, and letting GPAW symmetrise the density of a displaced structure
    would erase the very force the displacement is meant to measure.
    """
    params: dict[str, Any] = {
        "mode": {"name": "pw", "ecut": spec.ecut} if spec.mode == "pw" else spec.mode,
        "xc": spec.xc,
        "spinpol": spinpol,
        "charge": spec.charge,
        "convergence": dict(spec.convergence),
        "symmetry": "off",
    }
    if spec.mode != "pw":
        params["h"] = spec.h
    if spec.mode == "lcao":
        params["basis"] = spec.basis
    return params


def make_gpaw_calculator(spec: CalcSpec, txt: Path, spinpol: bool = False) -> Any:
    """Build a GPAW calculator for ``spec``, logging to ``txt``."""
    if not gpaw_available():
        raise ImportError(
            "GPAW no está instalado en este entorno. En Ubuntu: instala "
            "build-essential, libxc-dev, libopenblas-dev y luego "
            "`uv pip install gpaw` y `gpaw install-data`. Ver INSTALACION.md."
        )
    from gpaw import GPAW

    return GPAW(txt=str(txt), **gpaw_parameters(spec, spinpol))
