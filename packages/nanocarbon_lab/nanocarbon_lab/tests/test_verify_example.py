"""The script that checks every exact structure must itself be checked.

`examples/verify_exact_structures.py` is what someone runs to satisfy
themselves that the laws hold, so a silent breakage there is worse than
one in a builder: it would report "All laws met" about a table it never
built. These tests run it.
"""

from __future__ import annotations

import pytest

from nanocarbon_lab.examples.verify_exact_structures import CASES, main


class TestTheTableIsTheModeList:
    def test_every_case_names_a_real_mode(self):
        from nanocarbon_lab.jobs import MODES

        for _, mode, _, _, _ in CASES:
            assert mode in MODES

    def test_every_case_passes_parameters_the_builder_takes(self):
        from nanocarbon_lab.jobs import parameter_names

        for label, mode, params, _, _ in CASES:
            known = set(parameter_names(mode))
            assert set(params) <= known, label

    def test_the_knee_routes_are_all_represented(self):
        modes = {mode for _, mode, _, _, _ in CASES}
        assert modes == {
            "toroid (knees)", "coil (knees)", "coil (knees, periodic)",
            "junction (knees)", "supernetwork (knees)", "schwarzite (knees)",
        }

    def test_each_expected_budget_is_the_law_not_a_number(self):
        from nanocarbon_lab.builders.knee import node_budget

        wanted = {("junction (knees)", "y"): node_budget(3),
                  ("junction (knees)", "x"): node_budget(4),
                  ("junction (knees)", "tetrahedral"): node_budget(4)}
        for _, mode, params, budget, _ in CASES:
            key = (mode, params.get("kind"))
            if key in wanted:
                assert budget == wanted[key]


class TestItRunsAndAgrees:
    @pytest.mark.slow
    def test_the_fast_set_meets_every_law(self, capsys):
        # Returns 0 only when every measured sum(6-n) equals the budget
        # its skeleton fixed before anything was meshed.
        assert main(["--no-relax"]) == 0
        printed = capsys.readouterr().out
        assert "All laws met." in printed
        assert "WRONG" not in printed

    @pytest.mark.slow
    def test_it_writes_what_it_was_asked_to(self, tmp_path, capsys):
        assert main(["--no-relax", "--write", str(tmp_path)]) == 0
        written = sorted(p.name for p in tmp_path.glob("*.xyz"))
        assert len(written) == sum(1 for *_, slow in CASES if not slow)
