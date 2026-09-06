"""Non-collinear magnetism and spin-orbit coupling.

Enabling spin-orbit coupling (SOC) in Quantum ESPRESSO means switching to a
non-collinear, spinor calculation::

    noncolin  = .true.
    lspinorb  = .true.

and supplying **fully-relativistic** pseudopotentials (``rel-`` in their
name). A scalar-relativistic pseudopotential silently contains no spin-orbit
information, so the run completes and returns a band structure with no
splitting whatsoever — a failure mode that looks exactly like "SOC is
negligible here".

A word on magnitudes, because it decides whether this is worth the cost.
Spin-orbit strength scales steeply with atomic number (roughly Z⁴ for
valence states). Carbon is Z=6, and the intrinsic SOC gap in graphene is of
order **10⁻² meV** — far below the accuracy of a routine DFT calculation and
utterly invisible at room temperature (k_BT ≈ 25 meV). The dopants this
package supports (N, B, S, P) are all light too; sulfur and phosphorus reach
only a few meV.

SOC in nanocarbons becomes physically interesting mainly through *proximity
or adatom effects*: heavy adatoms (Au, Bi, Pb), a heavy substrate, or
transition-metal decoration can enhance it by orders of magnitude. Running
SOC on pristine carbon roughly doubles the cost for an effect you will
struggle to resolve, so :mod:`carbonforge.validation.calculations` warns
about it rather than letting it pass unremarked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

# Elements heavy enough for spin-orbit coupling to matter in practice.
# Threshold is a judgement call: Z >= 30 is roughly where SOC splittings
# reach the tens-of-meV range that DFT can resolve reliably.
_HEAVY_Z_THRESHOLD = 30

_ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ge": 32,
    "Se": 34, "Mo": 42, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47,
    "Sn": 50, "Te": 52, "W": 74, "Re": 75, "Os": 76, "Ir": 77,
    "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83,
}


@dataclass
class SpinOrbitSpec:
    """Settings for a non-collinear / spin-orbit calculation.

    Attributes
    ----------
    enabled
        Turn spin-orbit coupling on. Implies ``noncolin``.
    noncolin
        Non-collinear magnetism. Required by SOC, but also usable alone for
        non-collinear magnetic order without spin-orbit terms.
    starting_magnetization
        Per-species initial magnetization in units of the valence charge,
        as ``{"C": 0.0, "N": 0.1}``. Only needed for magnetic systems.
    angle1, angle2
        Per-species polar and azimuthal angles (degrees) of the starting
        magnetization, for non-collinear arrangements.
    """

    enabled: bool = True
    noncolin: bool = True
    starting_magnetization: dict[str, float] = field(default_factory=dict)
    angle1: dict[str, float] = field(default_factory=dict)
    angle2: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # lspinorb without noncolin is rejected by QE; fix it rather than
        # letting the user discover it from a cryptic crash.
        if self.enabled:
            self.noncolin = True


def soc_setup(**overrides) -> SpinOrbitSpec:
    """Build a spin-orbit setup with sensible defaults."""
    return SpinOrbitSpec(**overrides)


def heaviest_element(symbols: Sequence[str]) -> tuple[str, int]:
    """Return the heaviest element present and its atomic number.

    Unknown symbols are treated as Z=0 so they never masquerade as heavy.
    """
    if not symbols:
        return ("", 0)
    best = max(symbols, key=lambda s: _ATOMIC_NUMBERS.get(s, 0))
    return (best, _ATOMIC_NUMBERS.get(best, 0))


def soc_is_physically_relevant(symbols: Sequence[str]) -> bool:
    """Whether SOC is likely to produce a resolvable effect.

    True only when an element of Z >= 30 is present. For all-light systems
    the splitting sits below DFT's practical resolution.
    """
    _, z = heaviest_element(symbols)
    return z >= _HEAVY_Z_THRESHOLD


def qe_system_fields(spec: SpinOrbitSpec) -> dict[str, object]:
    """Return the ``&SYSTEM`` entries implementing this setup in QE."""
    if not (spec.enabled or spec.noncolin):
        return {}
    fields: dict[str, object] = {"noncolin": True}
    if spec.enabled:
        fields["lspinorb"] = True
    return fields


def relativistic_pseudo_name(symbol: str, base: str) -> str:
    """Suggest a fully-relativistic pseudopotential filename.

    QE's PSLibrary marks these with ``rel-``, e.g.
    ``C.pbe-n-kjpaw_psl.1.0.0.UPF`` → ``C.rel-pbe-n-kjpaw_psl.1.0.0.UPF``.
    This only rewrites the name; the file still has to be downloaded.
    """
    prefix = f"{symbol}."
    if base.startswith(prefix) and "rel-" not in base:
        return prefix + "rel-" + base[len(prefix):]
    return base
