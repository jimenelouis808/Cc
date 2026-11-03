"""XPS shifts by the equivalent core (``tbkit.xps``).

Exact checks: symmetry-equivalent sites have the same energy; a reference shifted from
itself is zero; translating the molecule changes nothing. Physics checks (ordering, not
magnitude, which is GPAW's job in recipes/xps_validation): carbon bonded to more oxygen
binds its 1s more strongly (CH4 < CH3OH < H2CO < CO2). Refusals: a model without SCC or
without the Z+1 element.
"""

from __future__ import annotations

import pytest
from ase.build import molecule

from tbkit.params import load_parameters, xu_carbon
from tbkit.xps import core_levels, ionisation_energy, spectrum


@pytest.fixture(scope="module")
def chno():
    return load_parameters("xu_chno")


def test_equivalent_sites_and_reference(chno):
    ethane = molecule("C2H6")
    levels = core_levels(ethane, chno, [0, 1], reference=(ethane, 0))
    assert levels[0].shift_eV == pytest.approx(0.0, abs=1e-8)
    assert levels[1].shift_eV == pytest.approx(0.0, abs=1e-6)


def test_translation_invariance(chno):
    a = molecule("CH3OH")
    b = a.copy()
    b.translate([3.0, -1.0, 2.0])
    assert ionisation_energy(a, chno, 0) == pytest.approx(ionisation_energy(b, chno, 0),
                                                          abs=1e-6)


def test_oxidation_orders_the_carbon_1s(chno):
    ref = (molecule("CH4"), 0)
    shifts = [core_levels(molecule(n), chno, [i], reference=ref)[0].shift_eV
              for n, i in (("CH3OH", 0), ("H2CO", 1), ("CO2", 0))]
    assert 0 < shifts[0] < shifts[1] < shifts[2]


def test_refusals():
    with pytest.raises(ValueError, match="SCC|autoconsistentes"):
        ionisation_energy(molecule("CH4"), xu_carbon(), 0)
    with pytest.raises(ValueError, match="equivalente"):
        ionisation_energy(molecule("H2O"), load_parameters("xu_chno"), 0)   # O -> F


def test_spectrum_peaks_at_the_shifts(chno):
    levels = core_levels(molecule("CH3CN"), chno, [0, 1], reference=(molecule("CH4"), 0))
    grid, intensity = spectrum(levels, fwhm=0.2)
    top = grid[intensity.argmax()]
    assert min(abs(top - lv.shift_eV) for lv in levels) < 0.05


def test_calibrated_scale_is_shipped_only_where_it_holds():
    from tbkit.params import read_parameter_file
    from tbkit.xps import xps_scale

    lam = xps_scale("xu_chno", "C")
    assert 0.3 < lam < 1.0
    scales = read_parameter_file("xps_scale")["scales"]
    assert "N" not in scales.get("xu_chno", {})            # N 1s: no scale fixes it
    ref = (molecule("CH4"), 0)
    raw = core_levels(molecule("CO2"), load_parameters("xu_chno"), [0], reference=ref)[0]
    scaled = core_levels(molecule("CO2"), load_parameters("xu_chno"), [0], reference=ref,
                         scale=lam)[0]
    assert scaled.shift_eV == pytest.approx(lam * raw.shift_eV)
