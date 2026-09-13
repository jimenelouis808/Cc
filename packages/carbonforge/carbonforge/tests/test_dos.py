"""Tests for the density-of-states setup, parsing and analysis.

As with the other output parsers, the fixtures are synthetic files matching
the documented Quantum ESPRESSO formats. No QE installation was available,
so these pin the parsers' behaviour rather than agreement with a real run.
"""

from __future__ import annotations

import numpy as np
import pytest

from carbonforge.builders import build_graphene_supercell
from carbonforge.calculations.dos import (
    DOSSpec,
    format_dos_input,
    format_dos_runner,
    format_projwfc_input,
)
from carbonforge.exports.qe import write_qe_dos
from carbonforge.functionalization import make_graphitic_n
from carbonforge.results.dos import (
    DensityOfStates,
    ProjectedDOS,
    read_dos,
    read_pdos,
)


def _gaussian(energies, centre, width, height):
    return height * np.exp(-((energies - centre) ** 2) / (2 * width ** 2))


@pytest.fixture()
def dos_file(tmp_path):
    """A dos.dat with a gap centred on E_F = 0."""
    energies = np.linspace(-10.0, 10.0, 401)
    dos = _gaussian(energies, -3.0, 0.8, 2.0) + _gaussian(energies, 3.0, 0.8, 2.0)
    integrated = np.cumsum(dos) * (energies[1] - energies[0])
    path = tmp_path / "dos.dat"
    lines = ["#  E (eV)   dos(E)   Int dos(E)  EFermi =    0.000 eV"]
    lines += [
        f"  {e:.4f}  {d:.6f}  {i:.6f}"
        for e, d, i in zip(energies, dos, integrated)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture()
def pdos_dir(tmp_path):
    """A projwfc.x output directory: two C atoms and one N."""
    energies = np.linspace(-10.0, 10.0, 201)
    directory = tmp_path / "pdos_run"
    directory.mkdir()

    curves = {
        "pdos.pdos_atm#1(C)_wfc#1(s)": _gaussian(energies, -6.0, 1.0, 0.5),
        "pdos.pdos_atm#1(C)_wfc#2(p)": _gaussian(energies, -2.0, 1.5, 1.0),
        "pdos.pdos_atm#2(C)_wfc#1(s)": _gaussian(energies, -6.0, 1.0, 0.5),
        "pdos.pdos_atm#2(C)_wfc#2(p)": _gaussian(energies, -2.0, 1.5, 1.0),
        # The nitrogen contributes right at the Fermi level, which is the
        # feature an N-doping study is looking for.
        "pdos.pdos_atm#3(N)_wfc#1(s)": _gaussian(energies, -7.0, 1.0, 0.3),
        "pdos.pdos_atm#3(N)_wfc#2(p)": _gaussian(energies, 0.0, 0.6, 1.5),
    }
    for name, curve in curves.items():
        (directory / name).write_text(
            "\n".join(
                f"  {e:.4f}  {c:.6f}  {c:.6f}" for e, c in zip(energies, curve)
            )
            + "\n"
        )

    total = sum(curves.values()) * 1.05  # projection is slightly incomplete
    (directory / "pdos.pdos_tot").write_text(
        "#  E (eV)  dos(E)  pdos(E)  EFermi =    0.000 eV\n"
        + "\n".join(
            f"  {e:.4f}  {t:.6f}  {t:.6f}" for e, t in zip(energies, total)
        )
        + "\n"
    )
    return directory


class TestDOSSpec:
    def test_defaults_are_sane(self):
        spec = DOSSpec()
        assert spec.kmesh_factor >= 2
        assert spec.projected

    def test_degauss_reported_in_ev(self):
        assert DOSSpec(degauss=0.01).degauss_ev == pytest.approx(0.136, abs=0.01)

    def test_rejects_bad_values(self):
        with pytest.raises(ValueError):
            DOSSpec(delta_e=0)
        with pytest.raises(ValueError):
            DOSSpec(degauss=-1)
        with pytest.raises(ValueError):
            DOSSpec(kmesh_factor=0)
        with pytest.raises(ValueError):
            DOSSpec(ngauss=42)
        with pytest.raises(ValueError):
            DOSSpec(energy_min=5.0, energy_max=-5.0)

    def test_warns_when_step_exceeds_broadening(self):
        """A grid coarser than the smearing undersamples the curve."""
        spec = DOSSpec(delta_e=1.0, degauss=0.001)
        assert "submuestreando" in spec.describe()

    def test_input_cards(self):
        spec = DOSSpec(energy_min=-20.0, energy_max=20.0)
        dos_text = format_dos_input(spec)
        assert "&DOS" in dos_text and "Emin = -20.0" in dos_text
        proj_text = format_projwfc_input(spec)
        assert "&PROJWFC" in proj_text and "filpdos" in proj_text

    def test_runner_includes_projwfc_only_when_asked(self):
        assert "projwfc.x" in format_dos_runner(DOSSpec(projected=True))
        assert "projwfc.x" not in format_dos_runner(DOSSpec(projected=False))


class TestDOSExport:
    def test_writes_all_steps(self, tmp_path):
        sheet = make_graphitic_n(build_graphene_supercell(4, 4), n_sites=1, seed=0)
        written = write_qe_dos(sheet, tmp_path)
        assert set(written) == {"scf", "nscf", "dos", "projwfc", "script"}
        for path in written.values():
            assert path.exists() and path.stat().st_size > 0

    def test_nscf_mesh_is_denser_than_scf(self, tmp_path):
        """A mesh that converges the density is too coarse for a DOS."""
        written = write_qe_dos(build_graphene_supercell(4, 4), tmp_path)

        def first_mesh(path):
            lines = path.read_text().splitlines()
            index = next(i for i, ln in enumerate(lines) if "K_POINTS" in ln)
            return [int(v) for v in lines[index + 1].split()[:3]]

        scf = first_mesh(written["scf"])
        nscf = first_mesh(written["nscf"])
        assert any(n > s for n, s in zip(nscf, scf))

    def test_nscf_requests_empty_states(self, tmp_path):
        written = write_qe_dos(build_graphene_supercell(3, 3), tmp_path)
        assert "nbnd" in written["nscf"].read_text()

    def test_unprojected_skips_projwfc(self, tmp_path):
        written = write_qe_dos(
            build_graphene_supercell(3, 3), tmp_path, spec=DOSSpec(projected=False)
        )
        assert "projwfc" not in written


class TestReadDOS:
    def test_shape_and_fermi(self, dos_file):
        dos = read_dos(dos_file)
        assert dos.energies.size == 401
        assert dos.fermi_energy == pytest.approx(0.0)
        assert dos.integrated is not None

    def test_dos_at_fermi_is_low_in_a_gap(self, dos_file):
        assert read_dos(dos_file).at_fermi() < 0.01

    def test_gap_estimate(self, dos_file):
        """Compared against the value the fixture's own parameters imply.

        The fixture places Gaussians of height 2.0 and sigma 0.8 at ±3.0 eV.
        One exceeds a threshold t where |x ∓ 3| < sigma·sqrt(-2·ln(t/height)),
        so the empty window between them is 2·(3 − that half-width). Deriving
        it here rather than hardcoding a range keeps the test meaningful if
        the fixture ever changes.
        """
        height, sigma, centre, threshold = 2.0, 0.8, 3.0, 0.05
        half_width = sigma * np.sqrt(-2.0 * np.log(threshold / height))
        expected = 2.0 * (centre - half_width)

        gap = read_dos(dos_file).gap_estimate(threshold=threshold)
        assert gap is not None
        # The grid is 0.05 eV, so agreement to a couple of steps is exact
        # enough.
        assert gap == pytest.approx(expected, abs=0.15)

    def test_metal_has_no_gap(self, tmp_path):
        energies = np.linspace(-5, 5, 101)
        dos = np.full_like(energies, 1.0)
        path = tmp_path / "dos.dat"
        path.write_text(
            "#  E (eV)  dos(E)  EFermi =    0.000 eV\n"
            + "\n".join(f"{e} {d}" for e, d in zip(energies, dos))
        )
        assert read_dos(path).gap_estimate() is None

    def test_missing_fermi_raises(self, tmp_path):
        path = tmp_path / "dos.dat"
        path.write_text("# E dos\n-1.0 0.5\n0.0 0.6\n1.0 0.7\n")
        with pytest.raises(ValueError, match="nivel de Fermi"):
            read_dos(path).at_fermi()

    def test_empty_file_rejected(self, tmp_path):
        path = tmp_path / "dos.dat"
        path.write_text("# solo comentarios\n")
        with pytest.raises(ValueError, match="numéricos"):
            read_dos(path)


class TestReadPDOS:
    def test_groups_by_element(self, pdos_dir):
        pdos = read_pdos(pdos_dir)
        assert pdos.elements == ["C", "N"]

    def test_orbital_breakdown(self, pdos_dir):
        orbitals = read_pdos(pdos_dir).by_orbital("C")
        assert set(orbitals) == {"s", "p"}

    def test_per_atom_curves_kept(self, pdos_dir):
        pdos = read_pdos(pdos_dir)
        assert (3, "N", "p") in pdos.atom_contributions

    def test_carbon_sums_over_both_atoms(self, pdos_dir):
        """Two carbons must add up, not overwrite each other."""
        pdos = read_pdos(pdos_dir)
        carbon_p = pdos.by_orbital("C")["p"]
        single = pdos.atom_contributions[(1, "C", "p")]
        np.testing.assert_allclose(carbon_p, 2 * single)

    def test_nitrogen_dominates_at_the_fermi_level(self, pdos_dir):
        """The question an N-doping study actually asks."""
        pdos = read_pdos(pdos_dir)
        fraction = pdos.element_fraction_at(0.0, "N")
        assert fraction > 0.5

    def test_carbon_dominates_deep_below(self, pdos_dir):
        assert read_pdos(pdos_dir).element_fraction_at(-2.0, "C") > 0.8

    def test_unknown_element_rejected(self, pdos_dir):
        with pytest.raises(ValueError, match="no aparece"):
            read_pdos(pdos_dir).element_fraction_at(0.0, "B")

    def test_projection_completeness_reported(self, pdos_dir):
        completeness = read_pdos(pdos_dir).projection_completeness()
        # The fixture makes the total 5 % larger than the sum of projections.
        assert completeness == pytest.approx(1 / 1.05, rel=0.02)

    def test_summary_names_the_contributors(self, pdos_dir):
        text = read_pdos(pdos_dir).summary(fermi=0.0)
        assert "N:" in text and "C:" in text
        assert "Completitud" in text

    def test_summary_without_fermi_says_so(self, pdos_dir):
        pdos = read_pdos(pdos_dir)
        pdos.fermi_energy = None
        assert "Sin nivel de Fermi" in pdos.summary()

    def test_low_completeness_warns(self, pdos_dir):
        pdos = read_pdos(pdos_dir)
        pdos.total = pdos.total * 3.0  # projection now covers only a third
        assert "⚠️" in pdos.summary(fermi=0.0)

    def test_missing_directory_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="no es un directorio"):
            read_pdos(tmp_path / "nope")

    def test_no_projection_files_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="projwfc"):
            read_pdos(tmp_path)


class TestPlotting:
    def test_dos_figure_renders(self, dos_file, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        from carbonforge.results.dos import plot_dos

        figure = plot_dos(read_dos(dos_file))
        out = tmp_path / "dos.png"
        figure.savefig(out)
        assert out.stat().st_size > 0

    def test_pdos_figure_renders_per_element(self, pdos_dir, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        from carbonforge.results.dos import plot_dos

        figure = plot_dos(read_pdos(pdos_dir), reference=0.0)
        out = tmp_path / "pdos.png"
        figure.savefig(out)
        assert out.stat().st_size > 0
