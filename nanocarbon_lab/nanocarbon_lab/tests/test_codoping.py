"""Several heteroatoms at once, each with its own fraction and a correlation.

Two properties carry this module, and both were wrong or missing in the
sequential version it replaces: every fraction is of the *same* denominator,
and the placement is correlated on purpose. The second is only worth
anything if its effect is measurable, so most of these tests check the
recorded bond counts rather than the request.
"""

from __future__ import annotations

import warnings

import pytest

from nanocarbon_lab.builders import build_capped_cnt, build_graphene_supercell
from nanocarbon_lab.dopants.codoping import (
    AFFINITIES,
    _exact_counts,
    codope,
    describe_codoping,
)
from nanocarbon_lab.jobs import Job, apply_codoping, parse_codope_spec, to_cli
from nanocarbon_lab.validation import run_basic_checks


def sheet_400():
    """A 400-carbon sheet: big enough that 5% is a round 20 sites."""
    return build_graphene_supercell(nx=10, ny=10)


class TestExactCounts:
    """Largest remainder, because independent rounding does not add up."""

    def test_four_equal_tenths_do_not_round_to_forty(self):
        """The case that motivates the helper: 0.1 of 95 is 9.5 four times
        over, and rounding each on its own gives 10 each -- 40 sites for a
        requested 38."""
        counts = _exact_counts([0.1] * 4, 95)
        assert sum(counts) == 38
        assert all(count in (9, 10) for count in counts)

    def test_the_parts_sum_to_the_rounded_whole(self):
        for fractions, n_sites in (([0.05, 0.05], 400), ([0.04, 0.03, 0.02], 400),
                                   ([0.033, 0.033, 0.033], 91), ([0.5], 7)):
            counts = _exact_counts(fractions, n_sites)
            assert sum(counts) == round(sum(f * n_sites for f in fractions))
            assert all(count >= 0 for count in counts)


class TestFractionsMeanWhatTheySay:
    def test_two_species_at_the_same_fraction_get_the_same_count(self):
        """The bug this module exists to fix. Doped sequentially, the second
        species' fraction was of the carbons the first one left, so 5% and
        5% came out 5.00% and 4.75% -- silently, since `info` recorded the
        request.
        """
        out = codope(sheet_400(), [("N", 0.05), ("B", 0.05)], seed=1)
        counts = out.info["codoping"]["counts"]
        assert counts["N"] == counts["B"] == 20

    def test_every_fraction_is_of_the_original_carbon_count(self):
        host = sheet_400()
        out = codope(host, [("N", 0.04), ("B", 0.03), ("P", 0.02)], seed=2)
        record = out.info["codoping"]
        assert record["host_carbons"] == len(host)
        for element, asked in (("N", 0.04), ("B", 0.03), ("P", 0.02)):
            assert record["achieved"][element] == pytest.approx(asked, abs=1e-9)

    def test_the_achieved_fraction_is_recorded_not_the_requested_one(self):
        """720 carbons at 3% is 21.6 sites, so the achieved fraction cannot
        be 3% and the record must not claim it is."""
        tube = build_capped_cnt(n_body_rings=8, freq=3)
        out = codope(tube, [("N", 0.03), ("B", 0.03)], seed=3)
        record = out.info["codoping"]
        assert record["requested"]["N"] == 0.03
        assert record["achieved"]["N"] != 0.03
        # `achieved` is rounded to six decimals in the record, so compare
        # at that precision rather than at pytest's default relative one.
        assert record["achieved"]["N"] == pytest.approx(
            record["counts"]["N"] / record["host_carbons"], abs=1e-6)

    def test_fractions_summing_above_one_are_refused(self):
        with pytest.raises(ValueError, match="more than every carbon"):
            codope(sheet_400(), [("N", 0.6), ("B", 0.6)])

    def test_a_repeated_element_is_refused(self):
        """Two entries for one element would make "its own fraction"
        ambiguous, and silently summing them is a guess."""
        with pytest.raises(ValueError, match="once"):
            codope(sheet_400(), [("N", 0.02), ("N", 0.03)])

    def test_an_unknown_element_is_refused(self):
        with pytest.raises(ValueError):
            codope(sheet_400(), [("Xx", 0.02)])

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_a_nonsense_fraction_is_refused(self, fraction):
        with pytest.raises(ValueError):
            codope(sheet_400(), [("N", fraction)])


class TestAffinity:
    """The placement rule has to show up in the structure, not the docstring."""

    SPEC = [("N", 0.05), ("B", 0.05)]

    def test_avoid_leaves_no_two_dopants_bonded(self):
        out = codope(sheet_400(), self.SPEC, affinity="avoid", seed=1)
        assert out.info["codoping"]["dopant_bonds"] == 0

    def test_seek_puts_most_dopants_in_unlike_pairs(self):
        """The B-N domain case: a pair is isoelectronic with a C-C pair, so
        real co-doped samples grow domains rather than a solid solution."""
        out = codope(sheet_400(), self.SPEC, affinity="seek", seed=1)
        record = out.info["codoping"]
        total = sum(record["counts"].values())
        assert record["dopant_bonds"] >= total // 2
        # Nearly every one of them should join *different* species.
        assert record["unlike_bonds"] >= 0.9 * record["dopant_bonds"]

    def test_the_three_rules_are_ordered_as_they_claim(self):
        """seek > random > avoid in dopant-dopant bonds, on one host and one
        seed, which is the only comparison that means anything."""
        host = sheet_400()
        bonds = {}
        for affinity in AFFINITIES:
            out = codope(host, self.SPEC, affinity=affinity, seed=7)
            bonds[affinity] = out.info["codoping"]["dopant_bonds"]
        assert bonds["seek"] > bonds["random"] > bonds["avoid"]
        assert bonds["avoid"] == 0

    def test_seek_with_one_species_pairs_it_with_itself(self):
        """There is nothing to alternate with, and saying so beats pretending
        the rule did something it could not."""
        out = codope(sheet_400(), [("N", 0.05)], affinity="seek", seed=1)
        record = out.info["codoping"]
        assert record["dopant_bonds"] > 0
        assert record["unlike_bonds"] == 0

    def test_an_unknown_affinity_lists_what_exists(self):
        with pytest.raises(ValueError, match="Unknown affinity"):
            codope(sheet_400(), self.SPEC, affinity="repel")

    def test_avoid_reports_what_it_could_not_place(self):
        """An independent set of the requested size need not exist on a
        trivalent lattice. Filling the remainder in anyway would give up the
        one property that was asked for, so it warns and records instead.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = codope(sheet_400(), [("N", 0.30), ("B", 0.30)],
                         affinity="avoid", seed=1)
        record = out.info["codoping"]
        assert record["dopant_bonds"] == 0, "the rule must hold for what it did place"
        if record["unplaced"]:
            assert any("could not" in str(w.message) or "satisfies" in str(w.message)
                       for w in caught)

    def test_a_fraction_over_its_ceiling_warns_but_builds(self):
        with pytest.warns(UserWarning, match="realistic"):
            out = codope(sheet_400(), [("Fe", 0.05), ("N", 0.02)], seed=1)
        assert out.info["codoping"]["counts"]["Fe"] == 20


class TestReproducibility:
    def test_one_seed_gives_one_structure(self):
        host = sheet_400()
        first = codope(host, [("N", 0.05), ("B", 0.05)], affinity="seek", seed=5)
        again = codope(host, [("N", 0.05), ("B", 0.05)], affinity="seek", seed=5)
        assert (first.get_chemical_symbols() == again.get_chemical_symbols())

    def test_different_seeds_give_different_structures(self):
        host = sheet_400()
        first = codope(host, [("N", 0.05), ("B", 0.05)], seed=5)
        other = codope(host, [("N", 0.05), ("B", 0.05)], seed=6)
        assert first.get_chemical_symbols() != other.get_chemical_symbols()


class TestItStaysAStructure:
    @pytest.mark.parametrize("affinity", AFFINITIES)
    def test_the_host_is_not_mutated(self, affinity):
        host = sheet_400()
        before = host.get_chemical_symbols()
        codope(host, [("N", 0.05), ("B", 0.05)], affinity=affinity, seed=1)
        assert host.get_chemical_symbols() == before

    @pytest.mark.parametrize("affinity", AFFINITIES)
    def test_the_atom_count_is_unchanged(self, affinity):
        """Substitution, not addition: a dopant replaces a carbon."""
        host = sheet_400()
        out = codope(host, [("N", 0.05), ("B", 0.05)], affinity=affinity, seed=1)
        assert len(out) == len(host)

    @pytest.mark.parametrize("affinity", AFFINITIES)
    def test_it_still_passes_validation(self, affinity):
        out = codope(sheet_400(), [("N", 0.03), ("B", 0.03)],
                     affinity=affinity, seed=1)
        assert run_basic_checks(out).ok, run_basic_checks(out).summary()

    def test_it_runs_on_a_curved_host_with_a_recorded_bond_graph(self):
        """The neighbour table comes from `info["bonds"]` when there is one,
        for the reason the rest of the framework does: on a curved shell a
        distance cutoff sweeps up neighbours that are close but not bonded.
        """
        tube = build_capped_cnt(n_body_rings=8, freq=3)
        out = codope(tube, [("N", 0.03), ("B", 0.03)], affinity="avoid", seed=3)
        assert out.info["codoping"]["dopant_bonds"] == 0
        assert len(out) == len(tube)


class TestItSurvivesADeletion:
    """A census must be recomputed, never carried -- the `ring_counts` rule."""

    def test_the_counts_follow_the_survivors(self):
        from nanocarbon_lab.defects import introduce_vacancies

        tube = build_capped_cnt(n_body_rings=8, freq=3)
        doped = codope(tube, [("N", 0.05), ("B", 0.05)],
                       affinity="avoid", seed=1)
        before = dict(doped.info["codoping"]["counts"])
        cut = introduce_vacancies(doped, n_defects=8, seed=2)
        record = cut.info["codoping"]
        symbols = cut.get_chemical_symbols()
        for element, count in record["counts"].items():
            assert count == symbols.count(element), element
            assert count <= before[element]

    def test_the_affinity_s_evidence_is_remeasured_not_remembered(self):
        """An `avoid` record still claiming 0 dopant bonds after a vacancy
        opened one would assert a property the structure had lost."""
        from nanocarbon_lab.defects import introduce_vacancies

        tube = build_capped_cnt(n_body_rings=8, freq=3)
        doped = codope(tube, [("N", 0.05), ("B", 0.05)],
                       affinity="seek", seed=1)
        cut = introduce_vacancies(doped, n_defects=6, seed=4)
        record = cut.info["codoping"]
        element_of = {}
        for entry in cut.info.get("dopants", []):
            for index in entry["indices"]:
                element_of[int(index)] = entry["element"]
        expected = sum(1 for a, b in cut.info["bonds"]
                       if int(a) in element_of and int(b) in element_of)
        assert record["dopant_bonds"] == expected

    def test_the_denominator_is_the_host_it_was_doped_into(self):
        """The fraction asked for was of the host, not of what a later
        vacancy left, so the denominator must not drift."""
        from nanocarbon_lab.defects import introduce_vacancies

        tube = build_capped_cnt(n_body_rings=8, freq=3)
        doped = codope(tube, [("N", 0.05), ("B", 0.05)], seed=1)
        host = doped.info["codoping"]["host_carbons"]
        cut = introduce_vacancies(doped, n_defects=8, seed=2)
        assert cut.info["codoping"]["host_carbons"] == host


class TestDescribe:
    def test_it_reports_the_achieved_amount_and_the_bond_counts(self):
        out = codope(sheet_400(), [("N", 0.05), ("B", 0.05)],
                     affinity="seek", seed=1)
        text = describe_codoping(out)
        assert "seek" in text
        assert "N 5" in text and "B 5" in text
        assert "unlike" in text

    def test_it_is_quiet_on_an_undoped_structure(self):
        assert describe_codoping(sheet_400()) == "not co-doped."


class TestThroughJobs:
    """The GUI, the CLI and a sweep share one placement policy."""

    def test_the_spec_parses_as_the_cli_spells_it(self):
        assert parse_codope_spec("N:0.05,B:0.05") == [("N", 0.05), ("B", 0.05)]
        assert parse_codope_spec(" N:0.04 , B:0.03 ") == [("N", 0.04), ("B", 0.03)]
        assert parse_codope_spec("") == []

    @pytest.mark.parametrize("text", ["N", "N:", "N:0.05:0.1", "N:lots"])
    def test_a_malformed_spec_says_how_to_write_it(self, text):
        with pytest.raises(ValueError):
            parse_codope_spec(text)

    def test_a_job_applies_it(self):
        job = Job(mode="capped tube", codope="N:0.05,B:0.05",
                  codope_affinity="avoid", seed=1)
        out = apply_codoping(sheet_400(), job)
        assert out.info["codoping"]["dopant_bonds"] == 0
        assert out.info["codoping"]["affinity"] == "avoid"

    def test_a_job_refuses_doping_twice_over(self):
        """Both substitute carbons, so running one after the other would make
        the second fraction a fraction of what the first left -- which is the
        very confusion this module removes."""
        with pytest.raises(ValueError, match="not both"):
            Job(mode="capped tube", dopant="N", dopant_conc=0.05,
                codope="B:0.05")

    def test_the_command_line_round_trips(self):
        job = Job(mode="capped tube", params={"n_body_rings": 6, "freq": 2},
                  codope="N:0.05,B:0.05", codope_affinity="seek")
        command = to_cli(job, "out/x")
        assert "--codope N:0.05,B:0.05" in command
        assert "--codope-affinity seek" in command

    def test_a_random_affinity_is_left_off_the_command_line(self):
        """It is the default, and a command line that repeats every default
        is harder to read than one that does not."""
        job = Job(mode="capped tube", params={"n_body_rings": 6, "freq": 2},
                  codope="N:0.05")
        assert "--codope-affinity" not in to_cli(job, "out/x")
