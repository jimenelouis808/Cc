"""Tests for the output parsers and spectrum construction.

The fixtures below are synthetic files written to match the documented
Quantum ESPRESSO and SIESTA formats. They are not captured from a real run —
no QE or SIESTA installation is available here — so these tests pin the
parsers' behaviour and their error handling, not agreement with a real code's
output.
"""

from __future__ import annotations

import numpy as np
import pytest

from carbonforge.results.bands import (
    BandStructure,
    attach_path_labels,
    read_qe_bands,
    read_qe_bands_gnu,
    read_siesta_bands,
)
from carbonforge.results.spectra import (
    VibrationalMode,
    VibrationalSpectrum,
    broaden,
    read_dynmat,
)

# --- fixtures ---------------------------------------------------------------

QE_BANDS = """ &plot nbnd=   4, nks=   3 /
           0.000000  0.000000  0.000000
   -20.100   -8.500    2.300    6.100
           0.000000  0.000000  0.250000
   -19.800   -8.100    2.700    6.400
           0.000000  0.000000  0.500000
   -19.500   -7.700    3.100    6.900
"""

QE_GNU = """  0.0000  -20.100
  0.2500  -19.800
  0.5000  -19.500

  0.0000   -8.500
  0.2500   -8.100
  0.5000   -7.700

  0.0000    2.300
  0.2500    2.700
  0.5000    3.100
"""

SIESTA_BANDS = """  -4.230000
   0.000000   0.500000
 -21.000000   8.000000
   3   1   3
   0.000000  -20.100   -8.500    2.300
   0.250000  -19.800   -8.100    2.700
   0.500000  -19.500   -7.700    3.100
   2
   0.000000  G
   0.500000  X
"""

DYNMAT_FULL = """     Polarizability (A^3 units)
     IR activities are in (D/A)^2/amu units
     Raman activities are in A^4/amu units

# mode   [cm-1]    [THz]      IR          Raman   depol.fact
    1       0.00    0.0000    0.0000      0.0000    0.0000
    2       0.00    0.0000    0.0000      0.0000    0.0000
    3       0.00    0.0000    0.0000      0.0000    0.0000
    4     865.31   25.9414    0.0132     18.4471    0.7500
    5    1583.22   47.4640    0.0000    142.3300    0.0100

  end of the table
"""

DYNMAT_PHONON_ONLY = """# mode   [cm-1]    [THz]
    1       0.00    0.0000
    2     865.31   25.9414
    3    1583.22   47.4640
"""

DYNMAT_UNSTABLE = """# mode   [cm-1]    [THz]      IR          Raman   depol.fact
    1    -120.50   -3.6127    0.0000      0.0000    0.0000
    2       0.00    0.0000    0.0000      0.0000    0.0000
    3    1583.22   47.4640    0.0000    142.3300    0.0100
"""

# Six modes but only one near zero: the acoustic sum rule was clearly not
# applied, which is what the summary should point out.
DYNMAT_BAD_ACOUSTIC = """# mode   [cm-1]    [THz]      IR          Raman   depol.fact
    1       0.00    0.0000    0.0000      0.0000    0.0000
    2      42.10    1.2621    0.0010      0.5000    0.7500
    3      55.80    1.6728    0.0020      0.6000    0.7500
    4     865.31   25.9414    0.0132     18.4471    0.7500
    5    1583.22   47.4640    0.0000    142.3300    0.0100
    6    1620.40   48.5787    0.0050     90.1000    0.2000
"""


@pytest.fixture()
def qe_bands_file(tmp_path):
    path = tmp_path / "bands.dat"
    path.write_text(QE_BANDS)
    return path


@pytest.fixture()
def dynmat_file(tmp_path):
    path = tmp_path / "dynmat.out"
    path.write_text(DYNMAT_FULL)
    return path


# --- band parsing -----------------------------------------------------------


class TestQEBands:
    def test_shape(self, qe_bands_file):
        bands = read_qe_bands(qe_bands_file)
        assert bands.n_kpoints == 3
        assert bands.n_bands == 4

    def test_energies_and_kpoints(self, qe_bands_file):
        bands = read_qe_bands(qe_bands_file)
        assert bands.energies[0, 0] == pytest.approx(-20.100)
        assert bands.energies[2, 3] == pytest.approx(6.900)
        assert bands.kpoints[2, 2] == pytest.approx(0.5)

    def test_distances_are_cumulative(self, qe_bands_file):
        bands = read_qe_bands(qe_bands_file)
        assert bands.distances[0] == 0.0
        assert np.all(np.diff(bands.distances) > 0)
        assert bands.distances[-1] == pytest.approx(0.5)

    def test_rejects_wrong_file(self, tmp_path):
        path = tmp_path / "nope.dat"
        path.write_text("esto no es un archivo de bandas\n")
        with pytest.raises(ValueError, match="filband"):
            read_qe_bands(path)

    def test_detects_truncated_file(self, tmp_path):
        path = tmp_path / "short.dat"
        path.write_text(" &plot nbnd=   4, nks=   3 /\n 0.0 0.0 0.0\n 1.0 2.0\n")
        with pytest.raises(ValueError, match="se esperaban"):
            read_qe_bands(path)

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.dat"
        path.write_text("")
        with pytest.raises(ValueError, match="vacío"):
            read_qe_bands(path)


class TestQEBandsGnu:
    def test_shape(self, tmp_path):
        path = tmp_path / "bands.dat.gnu"
        path.write_text(QE_GNU)
        bands = read_qe_bands_gnu(path)
        assert bands.n_bands == 3
        assert bands.n_kpoints == 3

    def test_values(self, tmp_path):
        path = tmp_path / "bands.dat.gnu"
        path.write_text(QE_GNU)
        bands = read_qe_bands_gnu(path)
        assert bands.energies[0, 0] == pytest.approx(-20.100)
        assert bands.energies[-1, -1] == pytest.approx(3.100)

    def test_uneven_blocks_rejected(self, tmp_path):
        path = tmp_path / "bad.gnu"
        path.write_text("0.0 1.0\n0.5 2.0\n\n0.0 3.0\n")
        with pytest.raises(ValueError, match="truncado"):
            read_qe_bands_gnu(path)


class TestSiestaBands:
    def test_parses_fermi_and_specials(self, tmp_path):
        path = tmp_path / "carbon.bands"
        path.write_text(SIESTA_BANDS)
        bands = read_siesta_bands(path)
        assert bands.fermi_energy == pytest.approx(-4.23)
        assert bands.special_points == [(0.0, "G"), (0.5, "X")]

    def test_shape_and_values(self, tmp_path):
        path = tmp_path / "carbon.bands"
        path.write_text(SIESTA_BANDS)
        bands = read_siesta_bands(path)
        assert bands.n_kpoints == 3 and bands.n_bands == 3
        assert bands.energies[1, 1] == pytest.approx(-8.100)

    def test_spin_polarised_rejected(self, tmp_path):
        path = tmp_path / "spin.bands"
        path.write_text(SIESTA_BANDS.replace("   3   1   3", "   3   2   3"))
        with pytest.raises(ValueError, match="nspin"):
            read_siesta_bands(path)


class TestBandStructureHelpers:
    def test_shift_uses_fermi(self, tmp_path):
        path = tmp_path / "carbon.bands"
        path.write_text(SIESTA_BANDS)
        shifted = read_siesta_bands(path).shifted()
        assert shifted.energies[0, 0] == pytest.approx(-20.100 + 4.23)

    def test_shift_without_reference_raises(self, qe_bands_file):
        with pytest.raises(ValueError, match="referencia"):
            read_qe_bands(qe_bands_file).shifted()

    def test_gap_detection(self, qe_bands_file):
        bands = read_qe_bands(qe_bands_file)
        # Fermi at 0 eV: valence max -7.7, conduction min 2.3.
        gap = bands.band_gap(fermi=0.0)
        assert gap == pytest.approx(2.300 - (-7.700))

    def test_metallic_returns_none(self):
        # One band crosses the reference energy.
        bands = BandStructure(
            distances=np.array([0.0, 1.0]),
            energies=np.array([[-1.0], [1.0]]),
        )
        assert bands.band_gap(fermi=0.0) is None

    def test_attach_labels(self, qe_bands_file):
        bands = attach_path_labels(read_qe_bands(qe_bands_file), ["G", "X"])
        assert [n for _, n in bands.special_points] == ["G", "X"]
        assert bands.special_points[0][0] == pytest.approx(0.0)
        assert bands.special_points[-1][0] == pytest.approx(0.5)

    def test_attach_labels_needs_two(self, qe_bands_file):
        with pytest.raises(ValueError):
            attach_path_labels(read_qe_bands(qe_bands_file), ["G"])


# --- spectra ----------------------------------------------------------------


class TestReadDynmat:
    def test_full_table(self, dynmat_file):
        spectrum = read_dynmat(dynmat_file)
        assert len(spectrum) == 5
        assert spectrum.has_ir and spectrum.has_raman

    def test_values(self, dynmat_file):
        spectrum = read_dynmat(dynmat_file)
        mode = spectrum.modes[4]
        assert mode.frequency_cm1 == pytest.approx(1583.22)
        assert mode.raman_activity == pytest.approx(142.33)
        assert mode.depolarisation == pytest.approx(0.01)

    def test_phonon_only_has_no_activities(self, tmp_path):
        path = tmp_path / "dynmat.out"
        path.write_text(DYNMAT_PHONON_ONLY)
        spectrum = read_dynmat(path)
        assert not spectrum.has_ir and not spectrum.has_raman
        assert len(spectrum) == 3

    def test_requesting_missing_activity_raises(self, tmp_path):
        path = tmp_path / "dynmat.out"
        path.write_text(DYNMAT_PHONON_ONLY)
        spectrum = read_dynmat(path)
        with pytest.raises(ValueError, match="lraman"):
            spectrum.activities("raman")

    def test_bad_kind_raises(self, dynmat_file):
        with pytest.raises(ValueError, match="'ir' o 'raman'"):
            read_dynmat(dynmat_file).activities("xrd")

    def test_no_table_raises(self, tmp_path):
        path = tmp_path / "dynmat.out"
        path.write_text("Error: something went wrong\n")
        with pytest.raises(ValueError, match="tabla de modos"):
            read_dynmat(path)

    def test_acoustic_and_optical_split(self, dynmat_file):
        spectrum = read_dynmat(dynmat_file)
        assert len(spectrum.optical_modes()) == 2

    def test_imaginary_modes_flagged(self, tmp_path):
        path = tmp_path / "dynmat.out"
        path.write_text(DYNMAT_UNSTABLE)
        spectrum = read_dynmat(path)
        assert len(spectrum.imaginary_modes) == 1
        assert "imaginario" in spectrum.summary()

    def test_summary_warns_on_wrong_acoustic_count(self, tmp_path):
        path = tmp_path / "dynmat.out"
        path.write_text(DYNMAT_BAD_ACOUSTIC)  # 6 modes, only 1 near zero
        assert "acústicos" in read_dynmat(path).summary()

    def test_small_systems_skip_the_acoustic_warning(self, tmp_path):
        """Three modes are all acoustic by definition; warning would be noise."""
        path = tmp_path / "dynmat.out"
        path.write_text(DYNMAT_PHONON_ONLY)
        assert "acústicos" not in read_dynmat(path).summary()

    def test_clean_summary_has_no_warning(self, dynmat_file):
        summary = read_dynmat(dynmat_file).summary()
        assert "⚠️" not in summary


class TestBroaden:
    def test_peak_lands_on_the_mode(self):
        grid, intensity = broaden([1583.0], [1.0], width_cm1=5.0)
        assert grid[int(np.argmax(intensity))] == pytest.approx(1583.0, abs=2.0)

    def test_acoustic_modes_excluded(self):
        """A mode at zero must not create a spurious peak."""
        _, intensity = broaden([0.0, 1583.0], [5.0, 1.0])
        # Only one peak should exist; total weight comes from the optical mode.
        grid, only_optical = broaden([1583.0], [1.0])
        assert intensity.max() == pytest.approx(only_optical.max(), rel=1e-6)

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="misma longitud"):
            broaden([100.0, 200.0], [1.0])

    def test_negative_width_rejected(self):
        with pytest.raises(ValueError):
            broaden([100.0], [1.0], width_cm1=-1.0)

    def test_wider_lines_are_lower_and_broader(self):
        _, narrow = broaden([1000.0], [1.0], width_cm1=2.0)
        _, wide = broaden([1000.0], [1.0], width_cm1=20.0)
        assert narrow.max() > wide.max()

    def test_bose_factor_increases_low_frequency_weight(self):
        """Thermal population boosts low-wavenumber Stokes lines."""
        _, raw = broaden([200.0], [1.0])
        _, warm = broaden([200.0], [1.0], temperature_k=300.0)
        assert warm.max() > raw.max()

    def test_laser_prefactor_applied(self):
        _, raw = broaden([1583.0], [1.0])
        _, scaled = broaden([1583.0], [1.0], laser_wavelength_nm=532.0)
        # (nu_laser - nu)^4 / nu_laser^4 < 1, so intensity drops.
        assert scaled.max() < raw.max()

    def test_mode_above_laser_rejected(self):
        with pytest.raises(ValueError, match="láser"):
            broaden([30000.0], [1.0], laser_wavelength_nm=532.0)

    def test_empty_input_gives_flat_spectrum(self):
        grid, intensity = broaden([], [])
        assert grid.size > 0
        assert np.allclose(intensity, 0.0)
