"""Doping a structure that has no pentagons.

The site selector offered ``"pentagon"`` and nothing else, which is the
right site on a fullerene or a capped tube and a site that **does not
exist** on a saddle: a schwarzite, a junction or a knee supernetwork is
hexagons plus heptagons, and its chemistry is on the heptagons.
``dopants.rings`` always took the ring size; only the policy layer
hardcoded five.
"""

from __future__ import annotations

import warnings

import pytest

from nanocarbon_lab.jobs import (
    DOPANT_RING_SIZES,
    DOPANT_SITES,
    Job,
    build,
)


@pytest.fixture(scope="module")
def saddle():
    """A Y junction: hexagons and six heptagons, and no pentagon at all."""
    return Job("junction (knees)", {"circumference": 10, "arm_rows": 7})


@pytest.fixture(scope="module")
def both():
    """A knee toroid, which carries pentagons AND heptagons."""
    return Job("toroid (knees)", {})


class TestTheSiteListCoversEveryDisclination:
    def test_the_three_ring_sites_are_offered(self):
        for site in ("pentagon", "heptagon", "octagon"):
            assert site in DOPANT_SITES
            assert site in DOPANT_RING_SIZES

    def test_each_names_its_own_ring_size(self):
        assert DOPANT_RING_SIZES == {"pentagon": 5, "heptagon": 7,
                                     "octagon": 8}


class TestASaddleIsDopedOnItsHeptagons:
    def test_asking_for_pentagons_says_what_rings_there_are(self, saddle):
        job = Job(saddle.mode, dict(saddle.params), dopant="N",
                  dopant_conc=0.05, dopant_site="pentagon", seed=1)
        with pytest.raises(ValueError) as excinfo:
            build(job)
        message = str(excinfo.value)
        assert "no 5-membered rings" in message
        # And it names what IS there, so the next attempt is obvious.
        assert "7-" in message

    def test_heptagon_placement_puts_nitrogen_on_them(self, saddle):
        job = Job(saddle.mode, dict(saddle.params), dopant="N",
                  dopant_conc=0.20, dopant_site="heptagon", seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = build(job)
        symbols = atoms.get_chemical_symbols()
        assert symbols.count("N") > 0
        # Every nitrogen is on a heptagon, which is the whole claim.
        heptagons = {x for ring in atoms.info["rings"] if len(ring) == 7
                     for x in ring}
        assert all(i in heptagons for i, s in enumerate(symbols) if s == "N")

    def test_the_fraction_is_of_that_ring_size_not_the_structure(self,
                                                                 saddle):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = build(Job(saddle.mode, dict(saddle.params), dopant="N",
                              dopant_conc=0.20, dopant_site="heptagon",
                              seed=1))
        sites = {x for ring in atoms.info["rings"] if len(ring) == 7
                 for x in ring}
        placed = atoms.get_chemical_symbols().count("N")
        # Against the heptagon sites, not against 272 atoms.
        assert placed == max(1, round(0.20 * len(sites)))


class TestAStructureWithBothIsDopedOnEither:
    @pytest.mark.parametrize("site,size", [("pentagon", 5), ("heptagon", 7)])
    def test_each_lands_on_its_own_rings(self, both, site, size):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = build(Job(both.mode, dict(both.params), dopant="N",
                              dopant_conc=0.25, dopant_site=site, seed=2))
        symbols = atoms.get_chemical_symbols()
        assert symbols.count("N") > 0
        chosen = {x for ring in atoms.info["rings"] if len(ring) == size
                  for x in ring}
        assert all(i in chosen for i, s in enumerate(symbols) if s == "N")


class TestTheKneeStructuresTakeTheOtherTwoAxesToo:
    """Doping replaces an atom, grafting adds one; both must reach these."""

    def test_codoping_places_both_species(self, saddle):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = build(Job(saddle.mode, dict(saddle.params),
                              codope="N:0.03,B:0.03",
                              codope_affinity="seek", seed=3))
        symbols = atoms.get_chemical_symbols()
        assert symbols.count("N") > 0 and symbols.count("B") > 0
        assert atoms.info["codoping"]["dopant_bonds"] > 0

    @pytest.mark.parametrize("group", ["hydroxyl", "carboxyl"])
    def test_grafting_adds_atoms(self, saddle, group):
        bare = build(Job(saddle.mode, dict(saddle.params)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = build(Job(saddle.mode, dict(saddle.params), graft=group,
                              graft_coverage=0.1, seed=4))
        assert len(atoms) > len(bare)
        assert atoms.info["grafted_atoms"]
