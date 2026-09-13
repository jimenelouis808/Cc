"""Tests for parameter sweeps over any mode.

The old sweep covered chirality and length of a nanotube, which was the
whole program when it was written; there are nineteen modes now and none
of the others could be swept. Building this on :mod:`nanocarbon_lab.jobs`
rather than adding a ``batch_X_sweep`` per mode means a new mode gets a
sweep for free -- and the test that matters most is the one asserting
exactly that.

The rest is about a sweep being the one place where a small mistake is
expensive: a mistyped parameter name must be caught before the first
build, not hours in, and every structure needs its own seed or the
dataset repeats one defect pattern throughout.
"""

from __future__ import annotations

import pytest

from nanocarbon_lab.jobs import MODES, parameter_names
from nanocarbon_lab.workflows import describe_sweep, expand, sweep_jobs, sweep_name


class TestExpansion:
    def test_it_takes_the_cartesian_product(self):
        jobs = expand("capped tube", {"freq": 2},
                      {"n_body_rings": [4, 6], "bond": [1.40, 1.42, 1.44]})
        assert len(jobs) == 6

    def test_the_first_parameter_varies_slowest(self):
        """So a sorted directory listing reads as a table."""
        jobs = expand("capped tube", {},
                      {"freq": [2, 3], "n_body_rings": [4, 6]})
        assert [(j.params["freq"], j.params["n_body_rings"]) for j in jobs] == [
            (2, 4), (2, 6), (3, 4), (3, 6)
        ]

    def test_fixed_parameters_reach_every_job(self):
        jobs = expand("capped tube", {"bond": 1.40}, {"freq": [2, 3]})
        assert all(job.params["bond"] == 1.40 for job in jobs)

    def test_nothing_to_vary_is_a_single_job(self):
        """A useful degenerate case, not an error."""
        assert len(expand("fullerene", {"family": "C60", "freq": 1})) == 1

    def test_an_empty_value_list_is_rejected(self):
        """It would silently collapse the product to zero jobs, which
        looks like a sweep that ran and found no work."""
        with pytest.raises(ValueError, match="no values to sweep"):
            expand("capped tube", {}, {"freq": []})

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            expand("nanotube-ish", {}, {"freq": [2]})


class TestParameterNameChecking:
    """The expensive mistake this exists to prevent."""

    def test_a_mistyped_varied_parameter_is_caught_before_building(self):
        with pytest.raises(ValueError, match="has no parameter"):
            expand("capped tube", {}, {"rings": [4, 6]})

    def test_a_mistyped_fixed_parameter_is_caught_too(self):
        with pytest.raises(ValueError, match="has no parameter"):
            expand("capped tube", {"radius": 6.0}, {"freq": [2]})

    def test_the_message_lists_what_is_accepted(self):
        """Naming the real parameters is the difference between a useful
        error and another guess."""
        with pytest.raises(ValueError) as excinfo:
            expand("capped tube", {}, {"rings": [4]})
        assert "n_body_rings" in str(excinfo.value)

    def test_parameter_names_come_from_the_real_signature(self):
        names = parameter_names("network")
        assert "tube_radius" in names and "cell" in names
        assert "rings" not in names


class TestNaming:
    def test_only_the_varying_parameters_appear(self):
        """Repeating the fixed ones in every name adds length, not
        information."""
        jobs = expand("capped tube", {"bond": 1.42}, {"freq": [3]})
        name = sweep_name("capped tube", jobs[0], {"freq": [3]})
        assert name == "capped_tube__freq3"
        assert "bond" not in name

    def test_names_are_filename_safe(self):
        jobs = expand("coil (relaxed)", {}, {"coil_radius": [34.5]})
        name = sweep_name("coil (relaxed)", jobs[0], {"coil_radius": [34.5]})
        assert "(" not in name and " " not in name and "/" not in name

    def test_every_name_in_a_sweep_is_distinct(self):
        """Colliding names would silently overwrite each other's output."""
        summary = describe_sweep("capped tube", {},
                                 {"freq": [2, 3], "n_body_rings": [4, 6]})
        assert len(set(summary["names"])) == summary["n_structures"]


class TestCostBeforeRunning:
    def test_it_counts_and_sizes_without_building(self):
        summary = describe_sweep("capped tube", {"freq": 3},
                                 {"n_body_rings": [4, 8]})
        assert summary["n_structures"] == 2
        assert summary["atoms_min"] < summary["atoms_max"]
        assert summary["atoms_total"] == pytest.approx(
            summary["atoms_min"] + summary["atoms_max"])

    def test_it_names_the_family(self):
        assert describe_sweep("TMD layers", {}, {})["family"] == "dichalcogenide"

    def test_the_product_grows_as_the_product(self):
        """Three parameters at four values each is 64 structures, and
        that arithmetic is exactly what a user needs before waiting."""
        summary = describe_sweep(
            "capped tube", {},
            {"freq": [2, 3, 4, 5], "n_body_rings": [4, 6, 8, 10],
             "bond": [1.40, 1.42, 1.44, 1.46]})
        assert summary["n_structures"] == 64


class TestSweepJobs:
    def test_each_structure_gets_its_own_seed(self):
        """A shared seed would put the identical defect pattern in every
        structure of the dataset -- the one thing it must not have."""
        jobs = sweep_jobs("capped tube", {"freq": 2, "n_body_rings": 4},
                          {"bond": [1.42, 1.43]}, dopant="N",
                          dopant_conc=0.05, seed=0)
        placements = [
            tuple(i for i, s in enumerate(job.builder().get_chemical_symbols())
                  if s == "N")
            for job in jobs
        ]
        assert placements[0] != placements[1]

    def test_the_doping_goes_through_the_shared_policy(self):
        jobs = sweep_jobs("capped tube", {"freq": 2, "n_body_rings": 4},
                          {}, dopant="N", dopant_conc=0.05,
                          dopant_site="pentagon", seed=0)
        atoms = jobs[0].builder()
        assert atoms.info["doping_mode"] == "ring5"

    def test_the_export_format_reaches_every_job(self):
        jobs = sweep_jobs("fullerene", {"family": "C60", "freq": 1},
                          {}, export="both")
        assert all(job.export == "both" for job in jobs)


class TestEveryModeIsSweepable:
    """The property the design is for: one mapping, so a mode added to
    ``jobs.py`` is sweepable without touching this module."""

    @pytest.mark.parametrize("mode", list(MODES))
    def test_a_mode_can_be_described_without_building(self, mode):
        summary = describe_sweep(mode, {}, {})
        assert summary["n_structures"] == 1
        assert summary["atoms_total"] > 0

    @pytest.mark.parametrize("mode", list(MODES))
    def test_a_mode_exposes_its_parameters(self, mode):
        assert parameter_names(mode)


@pytest.mark.slow
class TestWritesADataset:
    def test_it_writes_features_and_a_manifest(self, tmp_path):
        from nanocarbon_lab.workflows import write_ml_dataset

        jobs = sweep_jobs("fullerene", {"family": "C60"}, {"freq": [1]})
        manifest = write_ml_dataset(jobs, tmp_path)
        assert manifest.exists()
        assert (tmp_path / "features.csv").exists()
        assert (tmp_path / "structures").is_dir()

    def test_it_writes_simulation_inputs(self, tmp_path):
        from nanocarbon_lab.workflows import write_dataset

        jobs = sweep_jobs("fullerene", {"family": "C60"}, {"freq": [1]},
                          export="qe")
        metadata = write_dataset(jobs, tmp_path)
        assert metadata.exists()
        # Directories are named by the sweep name, so the varied value is
        # visible in a listing rather than only inside the metadata.
        assert (tmp_path / "fullerene__freq1" / "qe" / "pw.in").exists()
