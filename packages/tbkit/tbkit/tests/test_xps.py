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
        ionisation_energy(molecule("H2O"), load_parameters("xu_chno"), 0, method="z+1")


def test_spectrum_peaks_at_the_shifts(chno):
    levels = core_levels(molecule("CH3CN"), chno, [0, 1], reference=(molecule("CH4"), 0))
    grid, intensity = spectrum(levels, fwhm=0.2)
    top = grid[intensity.argmax()]
    assert min(abs(top - lv.shift_eV) for lv in levels) < 0.05


def test_hole_parts_add_up_and_translation_invariant(chno):
    from tbkit.xps import ionisation_parts

    a = molecule("H2CO")
    b = a.copy()
    b.translate([1.0, 2.0, -3.0])
    initial, relaxation = ionisation_parts(a, chno, 1)
    assert initial + relaxation == pytest.approx(ionisation_energy(a, chno, 1), abs=1e-7)
    assert ionisation_energy(b, chno, 1) == pytest.approx(initial + relaxation, abs=1e-6)
    assert relaxation < 0                    # screening lowers the cost of the hole


def test_hole_needs_no_z_plus_one_element(chno):
    """O 1s with the hole works on a set without F (z+1 refuses)."""
    ref = (molecule("H2O"), 0)
    co2 = core_levels(molecule("CO2"), chno, [1], reference=ref)[0]
    assert co2.shift_eV is not None
    with pytest.raises(ValueError, match="equivalente"):
        core_levels(molecule("CO2"), chno, [1], reference=ref, method="z+1")
