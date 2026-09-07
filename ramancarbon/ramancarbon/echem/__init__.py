"""Electrochemistry: cyclic voltammetry, galvanostatic cycling, impedance.

The measurements this package handles are easy to take and easy to
misreport, and most of what is in here exists to keep the second from
happening. Three things in particular:

**A material with redox peaks has a capacity, not a capacitance.** Dividing
a charge by a voltage window to quote "F/g" for a battery-like electrode is
the most criticised practice in the energy-storage literature, and it can
inflate the headline number several-fold. :mod:`ramancarbon.echem.evaluate`
classifies the mechanism from the shape of the curves and the b-value, and
refuses to express a battery-like electrode in farads.

**Two conventions for capacitance from a voltammogram differ by exactly
two.** Integrating the whole closed loop and integrating one sweep are both
common, both defensible, and are almost never stated. Both are reported
here, each with its name.

**A Tafel slope fitted over less than a decade of current is not a Tafel
slope.** Nor is one fitted through a mass-transport-limited region, or
through data that were not iR-corrected. All three are checked.

Modules
-------
``curve``     the data objects and the electrode description
``io``        readers for the usual potentiostat exports
``cv``        voltammetry: capacitance, peaks, b-value, Trasatti, ECSA
``gcd``       galvanostatic cycling: capacity, IR drop, energy and power
``eis``       impedance: circuits, complex non-linear fitting, Kramers–Kronig
``evaluate``  what kind of electrode this is, and HER/OER figures of merit
``report``    orchestration and the written report
"""

from __future__ import annotations

__all__ = [
    "ChargeDischarge",
    "Electrode",
    "Impedance",
    "Voltammogram",
    "analyse_cv",
    "analyse_eis",
    "analyse_gcd",
]


def __getattr__(name: str):  # pragma: no cover - lazy re-export
    if name in ("Voltammogram", "ChargeDischarge", "Impedance", "Electrode"):
        from . import curve

        return getattr(curve, name)
    if name == "analyse_cv":
        from .cv import analyse_cv

        return analyse_cv
    if name == "analyse_gcd":
        from .gcd import analyse_gcd

        return analyse_gcd
    if name == "analyse_eis":
        from .eis import analyse_eis

        return analyse_eis
    raise AttributeError(name)
