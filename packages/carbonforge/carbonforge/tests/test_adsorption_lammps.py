"""The LAMMPS route writes the in-house model, and reads back like it."""

import json

import numpy as np
import pytest
from ase import Atoms

from carbonforge.adsorption import lammps as L
from carbonforge.adsorption.potentials import H2_C, feynman_hibbs


def _box(edge=30.0):
    return Atoms("C2", positions=[[1, 1, 1], [5, 5, 5]], cell=[edge] * 3,
                 pbc=True)


def _section(text, name):
    lines = text.split(name + "\n", 1)[1].split("\n\n", 2)[1].splitlines()
    return np.array([[float(x) for x in line.split()] for line in lines])


def test_the_table_is_the_in_house_potential(tmp_path):
    folder = L.write_lammps_gcmc(_box(), tmp_path, 77.0, 1.0,
                                 equilibration=300, production=600)
    rows = _section((folder / "pair.table").read_text(), "HC")
    r, energy = rows[:, 1], rows[:, 2]
    expected = feynman_hibbs(r, H2_C, 77.0) * L.KCAL_PER_KELVIN
    assert np.allclose(energy, expected, rtol=1e-8, atol=1e-12)
    assert r[0] == L.TABLE_INNER and r[-1] == pytest.approx(12.0)
    assert L.TABLE_INNER < L.OVERLAP_CUTOFF


def test_the_reservoir_is_the_virial_fugacity(tmp_path):
    from carbonforge.adsorption.bulk import fit_virial

    folder = L.write_lammps_gcmc(_box(), tmp_path, 77.0, 10.0)
    request = json.loads((folder / "request.json").read_text())
    gas = fit_virial(77.0)
    assert request["fugacity_coefficient"] == pytest.approx(
        gas.fugacity(10.0) / 10.0e6)
    script = (folder / "in.gcmc").read_text()
    assert f"fugacity_coeff {request['fugacity_coefficient']:.8f}" in script
    assert f"pressure {10.0 * L.ATM_PER_MPA:.8f}" in script


def test_a_cell_the_in_house_run_refuses_is_refused(tmp_path):
    with pytest.raises(ValueError, match="twice"):
        L.write_lammps_gcmc(_box(20.0), tmp_path, 77.0, 1.0)


def test_a_count_trace_reads_back_as_a_result(tmp_path):
    folder = L.write_lammps_gcmc(_box(), tmp_path, 293.0, 5.0)
    np.savetxt(folder / "counts.dat", np.full(200, 50.0))
    result = L.read_lammps_gcmc(folder)
    assert result.mean_count == 50.0 and result.count_error == 0.0
    volume = 30.0 ** 3 * 1e-24
    assert result.absolute_density_g_cm3 == pytest.approx(
        50 * 2.01588 / 6.02214076e23 / volume)
