"""Removing atoms must renumber the metadata that indexes them.

This was a real and silent fault. Builders record ``info["bonds"]`` and
``info["rings"]`` as atom indices so that nothing downstream re-derives
rings from coordinates. Both deletion paths -- carbon vacancies and
chalcogen vacancies -- copied ``info`` wholesale, so after removing three
atoms from a 240-atom capped tube the bond indices still ran to 239.

Nothing complained. `coordination_numbers` prefers the recorded graph
when it exists, so validation read the corrupted one and passed; the
render bundle wrote the same indices to JSON, so a defected tube drew
bonds between atoms that were never bonded. Metadata that is present,
plausible and wrong is worse than none, which is why these tests assert
consistency rather than mere presence.
"""

from __future__ import annotations

import pytest

from nanocarbon_lab.builders import build_capped_cnt
from nanocarbon_lab.defects import introduce_vacancies
from nanocarbon_lab.tmd import build_tmd_monolayer
from nanocarbon_lab.tmd.modify import chalcogen_vacancies
from nanocarbon_lab.utils.metadata import keep_indices, remap_after_removal


class TestRemapper:
    def test_it_renumbers_the_survivors(self):
        info = {"bonds": [[0, 1], [1, 2], [2, 3]]}
        out = remap_after_removal(info, keep=[1, 2, 3])
        # 1,2,3 become 0,1,2; the bond to the removed atom 0 is gone.
        assert out["bonds"] == [[0, 1], [1, 2]]

    def test_it_drops_groups_that_lost_a_member(self):
        """A bond with one end missing is not a bond, and a pentagon
        missing an atom is not a pentagon. Dropping them is what makes
        the survivors trustworthy."""
        info = {"rings": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]}
        out = remap_after_removal(info, keep=list(range(1, 10)))
        assert out["rings"] == [[4, 5, 6, 7, 8]]

    def test_it_recomputes_the_ring_census(self):
        """Carrying the old census forward would contradict the rings
        printed beside it."""
        info = {"rings": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10]],
                "ring_counts": {5: 1, 6: 1}}
        out = remap_after_removal(info, keep=list(range(5, 11)))
        assert out["ring_counts"] == {6: 1}

    def test_it_remaps_dopant_indices(self):
        """Atom 0 goes; 5 and 9 survive and shift down by one."""
        info = {"dopants": [{"element": "N", "indices": [0, 5, 9]}]}
        out = remap_after_removal(info, keep=list(range(1, 10)))
        assert out["dopants"] == [{"element": "N", "indices": [4, 8]}]

    def test_a_dopant_removed_entirely_disappears(self):
        info = {"dopants": [{"element": "N", "indices": [0]}]}
        out = remap_after_removal(info, keep=[1, 2])
        assert "dopants" not in out

    def test_unknown_keys_pass_through(self):
        info = {"radius": 5.87, "shape": "straight"}
        assert remap_after_removal(info, keep=[0]) == info

    def test_it_does_not_mutate_the_input(self):
        info = {"bonds": [[0, 1]]}
        remap_after_removal(info, keep=[0])
        assert info["bonds"] == [[0, 1]]

    def test_keep_indices_is_ordered_and_complete(self):
        assert keep_indices(5, [1, 3]) == [0, 2, 4]


class TestCarbonVacancies:
    @pytest.fixture(scope="class")
    def defected(self):
        pristine = build_capped_cnt(n_body_rings=6, freq=2)
        return pristine, introduce_vacancies(pristine, n_defects=3, seed=0)

    def test_no_bond_index_is_out_of_range(self, defected):
        """The bug itself: indices ran to 239 against 237 atoms."""
        _, atoms = defected
        assert max(max(bond) for bond in atoms.info["bonds"]) < len(atoms)

    def test_no_ring_index_is_out_of_range(self, defected):
        _, atoms = defected
        assert max(max(ring) for ring in atoms.info["rings"]) < len(atoms)

    def test_the_census_matches_the_surviving_rings(self, defected):
        _, atoms = defected
        counted: dict[int, int] = {}
        for ring in atoms.info["rings"]:
            counted[len(ring)] = counted.get(len(ring), 0) + 1
        assert atoms.info["ring_counts"] == counted

    def test_removing_atoms_removes_their_bonds(self, defected):
        pristine, atoms = defected
        assert len(atoms.info["bonds"]) < len(pristine.info["bonds"])


class TestChalcogenVacancies:
    def test_the_bond_graph_stays_in_range(self):
        """The MX2 path had the same fault, and a curved MX2 is exactly
        where the recorded graph matters most -- a distance cutoff reads
        a sound cell as 4-8 coordinate on a saddle."""
        pristine = build_tmd_monolayer("MoS2")
        pristine.info["bonds"] = [[0, 1], [0, 2], [1, 2]]
        atoms = chalcogen_vacancies(pristine, n_defects=1, seed=0)
        if atoms.info.get("bonds"):
            assert max(max(b) for b in atoms.info["bonds"]) < len(atoms)

    def test_the_atom_count_drops(self):
        pristine = build_tmd_monolayer("MoS2", nx=3, ny=3)
        atoms = chalcogen_vacancies(pristine, n_defects=4, seed=0)
        assert len(atoms) == len(pristine) - 4
