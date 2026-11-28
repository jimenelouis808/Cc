"""A preset must build, and build the structure its entry describes.

Three of the six knee modes had **no branch** in the application's job
builder, so they fell through to the capped-tube arguments and the
builder refused a keyword it had never heard of: the toroid and both
coils failed outright from the menu while building perfectly from the
library. The other three took their shape from a lookup keyed on
``self.var_mode`` -- which is the *Blender representation* box, holding
``"ballstick"`` -- so the lookup never matched and every kind kept
whatever the boxes happened to contain. The diamond junction refused
(circumference 20 does not close a tetrahedral node), and the gyroid
quietly built 5584 atoms where its entry documents 1744.

None of that could be caught, because it lived in a module that needs a
display to import. The mapping is now :func:`nanocarbon_lab.presets.
knee_params`, a pure function of the parameter values, and these are the
tests that were impossible before.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

from nanocarbon_lab.builders.knee import (
    adsorption_knee_shape,
    default_knee_shape,
)
from nanocarbon_lab.jobs import MODES, Job, build, builder_for
from nanocarbon_lab.presets import (
    KNEE_MODES,
    KNEE_NODE_MODES,
    PRESETS,
    knee_params,
)

#: What each knee preset is *for*: the entry in the catalogue claims a
#: census, and a preset that builds something else is not a preset. The
#: numbers are measured, and they are the ones the entries quote.
KNEE_CENSUS = {
    "Carbon toroid (12 knees, exact)": (1032, {5: 12, 6: 492, 7: 12}, 0),
    "Nanocoil (knees, D/d 3.7)": (798, {5: 30, 6: 331, 7: 30}, None),
    "Nanocoil (knees, periodic, DFT-ready)": (672, {5: 12, 6: 312, 7: 12}, 0),
    "Nanocoil (knees, periodic, smallest)": (204, {5: 12, 6: 78, 7: 12}, 0),
    "Schwarz P (knees, no pentagons)": (968, {6: 456, 7: 24}, -24),
    "Schwarz D (knees, no pentagons)": (1440, {6: 608, 7: 96}, -96),
    "Gyroid (knees, no pentagons)": (1744, {6: 816, 7: 48}, -48),
    "Hexagonal D, lonsdaleite (knees, no spheres)": (
        1440, {6: 608, 7: 96}, -96),
    "I-WP, bcc net (knees, no spheres)": (
        3624, {5: 48, 6: 1632, 7: 120}, -72),
    "Super-graphene (knees, tubes on a honeycomb)": (920, {6: 432, 7: 24}, -24),
    "Super-square (knees, tubes on a square net)": (180, {5: 4, 6: 68, 7: 16},
                                                    -12),
    # The two shaped for uptake per gram rather than for bond spread.
    "Schwarz P (knees, for adsorption)": (488, {6: 216, 7: 24}, -24),
    "Super-square (knees, for adsorption)": (372, {5: 4, 6: 164, 7: 16}, -12),
    "Y junction (knees, six heptagons)": (536, {6: 240, 7: 6}, -6),
    "X junction (knees, four poles + sixteen crotches)": (
        780, {5: 4, 6: 328, 7: 16}, -12),
    "Diamond junction (knees, twelve heptagons)": (756, {6: 328, 7: 12}, -12),
}


def _knee_presets() -> list[str]:
    return [name for name, preset in PRESETS.items()
            if preset.get("mode_kind") in KNEE_MODES]


class TestTheCatalogueIsWellFormed:
    def test_every_preset_names_a_real_mode(self):
        for name, preset in PRESETS.items():
            assert preset.get("mode_kind") in MODES, name

    def test_every_knee_mode_is_a_mode(self):
        for mode in KNEE_MODES:
            assert mode in MODES, mode

    def test_every_knee_preset_is_accounted_for(self):
        # If a knee preset is added without a measured census, this fails
        # rather than letting it ship unchecked.
        assert sorted(_knee_presets()) == sorted(KNEE_CENSUS)


class TestTheMappingFitsTheBuilder:
    """The bug was arguments a builder had never heard of, so this asks
    the builder itself rather than trusting the mapping."""

    @pytest.mark.parametrize("mode", KNEE_MODES)
    def test_every_knee_mode_maps_to_keywords_its_builder_accepts(self, mode):
        params = knee_params(mode, {})
        accepted = set(inspect.signature(builder_for(mode)).parameters)
        assert set(params) <= accepted, (mode, set(params) - accepted)

    @pytest.mark.parametrize("name", _knee_presets())
    def test_every_knee_preset_maps_to_keywords_its_builder_accepts(self,
                                                                    name):
        mode = PRESETS[name]["mode_kind"]
        params = knee_params(mode, PRESETS[name])
        accepted = set(inspect.signature(builder_for(mode)).parameters)
        assert set(params) <= accepted, (name, set(params) - accepted)

    def test_a_mode_that_is_not_a_knee_mode_is_refused(self):
        with pytest.raises(KeyError):
            knee_params("nanoribbon", {})


class TestTheShapeIsTheKindsOwn:
    """A node preset states its shape, and a TABLE states the same one.
    They are written in two places because the user edits the boxes;
    they must not be allowed to drift.

    There are two tables, because there are two objectives: the
    bond-spread default and the adsorption shape, which differ for
    Schwarz P and super-square and agree for everything else. A preset
    must match one of them exactly -- a shape in neither is a number
    nobody measured."""

    @pytest.mark.parametrize("name", [
        n for n in _knee_presets()
        if PRESETS[n]["mode_kind"] in KNEE_NODE_MODES])
    def test_the_preset_shape_is_a_tables(self, name):
        preset = PRESETS[name]
        mode = preset["mode_kind"]
        params = knee_params(mode, preset)
        kind = params.get("kind", params.get("net"))
        shape = (params["circumference"], params["arm_rows"])
        tables = (default_knee_shape(mode, kind),
                  adsorption_knee_shape(mode, kind))
        assert shape in tables, (name, shape, tables)

    @pytest.mark.parametrize("name", [
        n for n in _knee_presets() if "for adsorption" in n])
    def test_an_adsorption_preset_takes_the_adsorption_shape(self, name):
        """And not the default one, or it would not be a second entry."""
        preset = PRESETS[name]
        mode = preset["mode_kind"]
        params = knee_params(mode, preset)
        kind = params.get("kind", params.get("net"))
        shape = (params["circumference"], params["arm_rows"])
        assert adsorption_knee_shape(mode, kind) == shape, name
        assert default_knee_shape(mode, kind) != shape, (
            f"{name}: this shape is the bond-spread default, so the "
            "preset is a duplicate rather than a second objective")

    def test_a_kind_with_no_adsorption_shape_says_so(self):
        """Absence is the claim that the default is already best.

        Schwarz D and super-graphene measured best for uptake at their
        own bond-spread shapes, and the gyroid has no shape in the band
        at all. A table entry repeating a default would be a second
        place for it to drift.
        """
        assert adsorption_knee_shape("schwarzite (knees)", "diamond") is None
        assert adsorption_knee_shape("schwarzite (knees)", "gyroid") is None
        assert adsorption_knee_shape("supernetwork (knees)",
                                     "super-graphene") is None

    def test_a_kind_without_its_own_shape_is_a_different_structure(self):
        # Why the above matters: the shapes are not interchangeable.
        assert default_knee_shape("junction (knees)", "y") != \
            default_knee_shape("junction (knees)", "tetrahedral")
        assert default_knee_shape("toroid (knees)", "y") is None


class TestEveryKneePresetBuildsWhatItClaims:
    @pytest.mark.parametrize("name", _knee_presets())
    def test_it(self, name):
        atoms_expected, census, deficit = KNEE_CENSUS[name]
        mode = PRESETS[name]["mode_kind"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = build(Job(mode, knee_params(mode, PRESETS[name])))
        assert len(atoms) == atoms_expected, name
        assert atoms.info["ring_counts"] == census, name
        assert atoms.info.get("ring_deficit") == deficit, name
