"""Tests for the camera, the scene effects and the glare pass.

All three are arithmetic, and all three live in modules that import no
``bpy`` precisely so they can be tested here -- in an ordinary
interpreter, with no 3D application installed. What is *not* tested here
is whether the result looks good; that is a judgement, and it was made
by rendering.

Three findings are pinned because each was measured and each is easy to
undo by accident:

1. **Framing by the bounding sphere is wrong for anything that is not a
   ball.** A row of structures 77 Å long and 12 Å deep has a bounding
   sphere of radius 39 Å, so the camera backed off to fit a 78 Å ball
   and the subject filled about a quarter of the frame.
2. **The best angle is measurable.** The silhouette area of the
   projection is the criterion, and it correctly refuses to look down
   the length of a row or at the end of a tube.
3. **Bloom needs a ceiling as well as a floor.** An emissive filament is
   many times brighter than white in linear light, so without a maximum
   its glow swamps the frame.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

BLENDER_DIR = Path(__file__).resolve().parent.parent / "blender"


def _load(name: str):
    sys.path.insert(0, str(BLENDER_DIR))
    spec = importlib.util.spec_from_file_location(name, BLENDER_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def styles():
    return _load("styles")


@pytest.fixture(scope="module")
def bundle_mod():
    """The pure bundle reader: colour keys, depth bands, seam bonds."""
    return _load("bundle")


@pytest.fixture(scope="module")
def camera():
    return _load("camera")


@pytest.fixture(scope="module")
def effects():
    return _load("effects")


@pytest.fixture(scope="module")
def row():
    """Two blobs side by side along x: a composition, in miniature."""
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal(centre, 3.0, size=(300, 3))
                      for centre in ([-30, 0, 0], [30, 0, 0])])


@pytest.fixture(scope="module")
def tube():
    """A long thin tube along z."""
    t = np.linspace(0.0, 60.0, 400)
    return np.column_stack([3 * np.cos(t), 3 * np.sin(t),
                            np.linspace(-20.0, 20.0, 400)])


class TestBasis:
    def test_it_is_orthonormal_and_right_handed(self, camera):
        basis = camera.basis_for((1.0, -1.4, 0.6))
        assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(basis) == pytest.approx(1.0)

    def test_a_vertical_view_does_not_degenerate(self, camera):
        """The cross product with +Z is zero looking straight down, and
        without a swapped world up every projected coordinate comes back
        nan -- silently, since nothing downstream checks."""
        basis = camera.basis_for((0.0, 0.0, 1.0))
        assert np.all(np.isfinite(basis))
        assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-12)

    def test_a_zero_direction_is_refused(self, camera):
        with pytest.raises(ValueError):
            camera.basis_for((0.0, 0.0, 0.0))


class TestHullArea:
    def test_a_unit_square(self, camera):
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert camera.convex_hull_area(square) == pytest.approx(1.0)

    def test_interior_points_do_not_change_it(self, camera):
        rng = np.random.default_rng(1)
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        inside = rng.uniform(0.1, 0.9, size=(200, 2))
        assert camera.convex_hull_area(np.vstack([square, inside])) == \
            pytest.approx(1.0)

    def test_degenerate_inputs_are_zero_not_an_error(self, camera):
        """A straight chain really does have no silhouette area, and the
        view chooser must be able to ask about one without raising."""
        assert camera.convex_hull_area(np.zeros((2, 2))) == 0.0
        line = np.column_stack([np.arange(10.0), np.zeros(10)])
        assert camera.convex_hull_area(line) == pytest.approx(0.0, abs=1e-9)


class TestBestView:
    def test_it_refuses_to_look_down_a_row(self, camera, row):
        """The failure the fixed direction had: a row laid along x was
        photographed from (1, -1.4, 0.6), which is mostly along the row
        itself, so the structures hid behind one another."""
        direction = camera.best_view(row)
        assert abs(float(direction[0])) < 0.5

    def test_it_refuses_to_look_at_the_end_of_a_tube(self, camera, tube):
        direction = camera.best_view(tube)
        assert abs(float(direction[2])) < 0.5

    def test_it_picks_a_three_quarter_view_of_a_sphere(self, camera):
        """Every direction shows the same silhouette, so the tie-break is
        what decides -- and a dead-on elevation reads as a drawing."""
        rng = np.random.default_rng(2)
        points = rng.normal(0.0, 1.0, size=(400, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        direction = camera.best_view(points * 5.0)
        assert float(np.min(np.abs(direction))) > 0.2

    def test_it_never_looks_from_below(self, camera, row):
        """A camera underneath is legitimate and almost never what a
        figure wants, so it is reachable by name and not by search."""
        assert float(camera.best_view(row)[2]) > 0.0
        assert all(vector[2] > 0 for vector in camera.view_candidates())

    def test_the_named_views_are_not_in_the_search(self, camera):
        """An exact elevation is an exact optimum, which no tie-break
        bonus can pull away from."""
        candidates = camera.view_candidates()
        for vector in camera.NAMED_VIEWS.values():
            unit = np.asarray(vector, dtype=float)
            unit /= np.linalg.norm(unit)
            assert not np.any(np.all(np.isclose(candidates, unit), axis=1))

    def test_a_name_is_honoured_exactly(self, camera):
        assert np.allclose(camera.named_direction("top"), [0, 0, 1])
        with pytest.raises(ValueError):
            camera.named_direction("sideways")


class TestFraming:
    @staticmethod
    def _coverage(camera, points, direction, distance, lens, sensor,
                  resolution):
        """What fraction of the frame width the subject spans."""
        half, _ = camera.projected_extent(points, direction)
        long_side, short_side = max(resolution), min(resolution)
        half_long = math.atan(0.5 * sensor / lens)
        half_h = (half_long if resolution[0] >= resolution[1]
                  else math.atan(math.tan(half_long) * short_side / long_side))
        return half[0] / ((distance - half[2]) * math.tan(half_h))

    def test_the_subject_fills_the_frame_it_was_asked_to(self, camera, row):
        direction = camera.best_view(row)
        _, _, distance = camera.frame_camera(row, direction, 50.0, 36.0,
                                             (400, 300), fill=0.86)
        coverage = self._coverage(camera, row, direction, distance, 50.0,
                                  36.0, (400, 300))
        assert coverage == pytest.approx(0.86, abs=0.02)

    def test_it_beats_bounding_sphere_framing_on_a_row(self, camera, row):
        """The measurement that motivated the change: 0.56 of the frame
        width against the 0.86 asked for."""
        direction = camera.best_view(row)
        centre = row.mean(axis=0)
        sphere_radius = float(np.linalg.norm(row - centre, axis=1).max())
        half_long = math.atan(0.5 * 36.0 / 50.0)
        half_short = math.atan(math.tan(half_long) * 300 / 400)
        old = sphere_radius / math.sin(half_short) / 0.82
        _, _, new = camera.frame_camera(row, direction, 50.0, 36.0,
                                        (400, 300), fill=0.86)
        assert new < old
        assert self._coverage(camera, row, direction, old, 50.0, 36.0,
                              (400, 300)) < 0.7

    def test_a_longer_lens_stands_further_back(self, camera, row):
        direction = camera.named_direction("iso")
        near = camera.frame_camera(row, direction, 50.0, 36.0, (400, 400))[2]
        far = camera.frame_camera(row, direction, 90.0, 36.0, (400, 400))[2]
        assert far > near * 1.5

    def test_the_depth_of_the_subject_is_allowed_for(self, camera):
        """A camera placed as though the structure sat entirely at its
        centre plane frames the far half correctly and cuts the near
        half off."""
        deep = np.array([[0, 0, -20.0], [0, 0, 20.0], [5, 5, 0], [-5, -5, 0]])
        direction = np.array([0.0, 0.0, 1.0])
        _, _, distance = camera.frame_camera(deep, direction, 50.0, 36.0,
                                             (400, 400))
        half, _ = camera.projected_extent(deep, direction)
        assert distance > half[2]

    def test_atom_radii_push_the_camera_back(self, camera, row):
        """An atom is drawn as a ball, so framing to its centre cuts half
        of the outermost one off."""
        direction = camera.named_direction("iso")
        bare = camera.frame_camera(row, direction, 50.0, 36.0, (400, 400))[2]
        grown = camera.frame_camera(row, direction, 50.0, 36.0, (400, 400),
                                    radii=np.full(len(row), 2.0))[2]
        assert grown > bare


class TestEffectSpecs:
    def test_every_effect_has_defaults_for_every_parameter(self, effects):
        for name in effects.EFFECTS:
            spec = effects.EffectSpec(kind=name)
            for key in effects.EFFECT_DEFAULTS[name]:
                assert spec.get(key) is not None

    def test_an_unknown_effect_names_the_ones_that_exist(self, effects):
        with pytest.raises(ValueError, match="laser"):
            effects.EffectSpec(kind="sparkles")

    def test_an_unknown_parameter_is_refused_not_ignored(self, effects):
        """A typo that is silently dropped is a setting that does
        nothing, and the render still takes minutes to tell you so."""
        with pytest.raises(ValueError, match="stength"):
            effects.EffectSpec(kind="laser", params={"stength": 5.0})

    def test_the_spec_string_parses(self, effects):
        spec = effects.parse_effect("laser:strength=80,color=cyan")
        assert spec.kind == "laser"
        assert spec.get("strength") == 80.0
        assert spec.colour() == effects.NAMED_COLOURS["cyan"]

    def test_a_bare_name_takes_every_default(self, effects):
        spec = effects.parse_effect("arc")
        assert spec.params == {}
        assert spec.get("count") == effects.EFFECT_DEFAULTS["arc"]["count"]

    def test_a_missing_equals_is_refused(self, effects):
        with pytest.raises(ValueError):
            effects.parse_effect("laser:strength")

    def test_colours_parse_three_ways(self, effects):
        assert effects.parse_colour("cyan") == effects.NAMED_COLOURS["cyan"]
        assert effects.parse_colour("#ff0000") == pytest.approx((1.0, 0.0, 0.0))
        assert effects.parse_colour("0.5/0.25/0") == pytest.approx(
            (0.5, 0.25, 0.0))
        with pytest.raises(ValueError):
            effects.parse_colour("puce")

    def test_a_colour_cannot_use_commas(self, effects):
        """They already separate one parameter from the next, so a colour
        written 1,0,0 would read as three parameters with no values."""
        with pytest.raises(ValueError):
            effects.parse_effect("laser:color=1,0,0")


class TestArcGeometry:
    def test_the_ends_do_not_move(self, effects):
        """An arc whose ends wander is an arc that has come off the thing
        it is supposed to be striking."""
        rng = np.random.default_rng(3)
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([10.0, 0.0, 0.0])
        path = effects.arc_polyline(start, end, 24, 0.5, 5.0, rng)
        assert np.allclose(path[0], start, atol=1e-9)
        assert np.allclose(path[-1], end, atol=1e-9)

    def test_it_wanders_in_between(self, effects):
        rng = np.random.default_rng(4)
        path = effects.arc_polyline(np.zeros(3), np.array([10.0, 0, 0]),
                                    24, 0.4, 5.0, rng)
        assert np.abs(path[:, 1:]).max() > 0.2

    def test_zero_jitter_is_a_straight_line(self, effects):
        rng = np.random.default_rng(5)
        path = effects.arc_polyline(np.zeros(3), np.array([10.0, 0, 0]),
                                    24, 0.0, 5.0, rng)
        assert np.abs(path[:, 1:]).max() == pytest.approx(0.0, abs=1e-12)

    def test_sphere_points_are_on_the_sphere(self, effects):
        rng = np.random.default_rng(6)
        points = effects.sphere_points(40, 7.0, rng)
        assert np.allclose(np.linalg.norm(points, axis=1), 7.0, atol=1e-9)

    def test_angles_give_a_unit_vector(self, effects):
        vector = effects.direction_from_angles(135.0, 22.0)
        assert np.linalg.norm(vector) == pytest.approx(1.0)
        assert vector[2] > 0


class TestSurfaceAlong:
    """Where a beam should stop. An isotropic radius is only right for
    something round, and the structures here are mostly not."""

    def test_a_long_thin_tube_is_not_a_sphere(self, effects, tube):
        """The failure this exists for: a capped nanotube's framed radius
        is its half-LENGTH while the tube is a few Angstrom across, so a
        beam stopping at a fraction of that radius stopped in mid-air
        beside the subject -- a glowing spot next to the structure."""
        centre = tube.mean(axis=0)
        across = effects.surface_along(tube, centre, (1.0, 0.0, 0.0))
        along = effects.surface_along(tube, centre, (0.0, 0.0, 1.0))
        assert across < 5.0
        assert along > 15.0

    def test_it_is_the_reach_in_that_direction(self, effects):
        points = np.array([[0, 0, 0], [10.0, 0, 0], [-10.0, 0, 0]])
        assert effects.surface_along(points, (0, 0, 0), (1, 0, 0),
                                     quantile=1.0) == pytest.approx(10.0)

    def test_one_stray_atom_does_not_move_the_impact(self, effects):
        """A grafted group or a lone cap atom should not push the beam's
        landing point off the body, which is why it is a quantile and
        not a maximum."""
        body = np.column_stack([np.linspace(-5.0, 5.0, 200),
                                np.zeros(200), np.zeros(200)])
        stray = np.array([[60.0, 0.0, 0.0]])
        with_stray = np.vstack([body, stray])
        assert effects.surface_along(with_stray, (0, 0, 0), (1, 0, 0)) < 10.0

    def test_no_points_is_none_not_a_crash(self, effects):
        """So the caller can fall back to the radius."""
        assert effects.surface_along(None, (0, 0, 0), (1, 0, 0)) is None
        assert effects.surface_along(np.zeros((0, 3)), (0, 0, 0),
                                     (1, 0, 0)) is None


class TestGlare:
    def test_it_leaves_a_dark_image_alone(self, effects):
        frame = np.full((16, 16, 4), 0.1)
        frame[:, :, 3] = 1.0
        assert np.allclose(effects.glare_image(frame, threshold=0.75), frame)

    def test_a_bright_spot_spreads(self, effects):
        """The point of bloom: light where there was none, around the
        source and not on it.

        On a 101 px frame, because the blur radius is a fraction of the
        image's short side -- the same glow has to be the same *relative*
        size on a thumbnail and on a 6000 px cover, so on a 33 px test
        image it is correctly about one pixel wide and there is nothing
        to measure.
        """
        frame = np.zeros((101, 101, 4))
        frame[:, :, 3] = 1.0
        frame[50, 50, :3] = 4.0
        out = effects.glare_image(frame, strength=1.0, size=1.0,
                                  threshold=0.5)
        assert out[50, 42, 0] > 1e-5
        assert out[50, 50, 0] > frame[50, 50, 0]
        # ...and not everywhere: a glow that reaches the frame edge is a
        # wash, not a bloom.
        assert out[50, 0, 0] == pytest.approx(0.0, abs=1e-9)

    def test_the_ceiling_bounds_the_glow(self, effects):
        """Without one an emissive filament -- many times brighter than
        white in linear light -- washes the whole frame out."""
        frame = np.zeros((101, 101, 4))
        frame[:, :, 3] = 1.0
        frame[50, 50, :3] = 50.0
        capped = effects.glare_image(frame, strength=1.0, size=1.0,
                                     maximum=3.0)
        loose = effects.glare_image(frame, strength=1.0, size=1.0,
                                    maximum=1000.0)
        assert capped[50, 42, 0] < loose[50, 42, 0]

    def test_it_keeps_the_hue_of_what_is_glowing(self, effects):
        """A white bloom around a green laser is a lens flare from a
        different photograph."""
        frame = np.zeros((101, 101, 4))
        frame[:, :, 3] = 1.0
        frame[50, 50] = (0.0, 5.0, 0.0, 1.0)
        out = effects.glare_image(frame, strength=1.0, size=1.0,
                                  threshold=0.5)
        glow = out[50, 42, :3]
        assert glow[1] > 10 * max(glow[0], glow[2], 1e-12)

    def test_the_alpha_channel_is_untouched(self, effects):
        """Or a transparent background loses its cut-out to the bloom."""
        frame = np.zeros((20, 20, 4))
        frame[:, :, :3] = 2.0
        frame[:, :, 3] = 0.25
        out = effects.glare_image(frame, strength=1.0, threshold=0.1)
        assert np.allclose(out[:, :, 3], 0.25)

    def test_a_blur_conserves_what_it_spreads(self, effects):
        """A box blur is a mean, so the total survives -- which is what
        makes the bloom add light rather than invent it."""
        plane = np.zeros((41, 41))
        plane[20, 20] = 1.0
        blurred = effects._gaussian_blur(plane, 4.0)
        assert blurred.sum() == pytest.approx(1.0, rel=1e-6)
        assert blurred[20, 20] < 0.5


class TestFinishes:
    """A finish is a third axis beside the style and the colour mode:
    "element colours, in glass, on the cosmos backdrop" has to be one
    render rather than a style nobody wrote."""

    def test_the_colour_survives_and_the_character_does_not(self, styles):
        base = styles.MaterialSpec(base_color=(0.2, 0.6, 0.9, 1.0),
                                   metallic=1.0, roughness=0.9)
        glass = styles.apply_finish(base, "glass")
        assert glass.transmission == 1.0
        assert glass.metallic == 0.0
        # Still recognisably blue: a glass version of an element must
        # still say which element it is.
        assert glass.base_color[2] > glass.base_color[0]

    def test_style_is_a_no_op(self, styles):
        base = styles.get_style("gold_nanotech").hexagon
        assert styles.apply_finish(base, "style") is base

    def test_glass_lifts_a_dark_colour_or_it_reads_as_soot(self, styles):
        """Transmitted light is multiplied by the base colour at every
        bounce, so carbon's grey through a cage's near and far walls
        comes out nearly black."""
        dark = styles.MaterialSpec(base_color=(0.1, 0.1, 0.1, 1.0))
        glass = styles.apply_finish(dark, "glass")
        assert min(glass.base_color[:3]) > 0.5

    def test_every_finish_produces_a_valid_material(self, styles):
        base = styles.MaterialSpec(base_color=(0.4, 0.5, 0.6, 1.0))
        for name in styles.FINISHES:
            spec = styles.apply_finish(base, name)
            assert spec.shader in styles.SHADERS
            assert 0.0 <= spec.roughness <= 1.0
            assert 0.0 <= spec.metallic <= 1.0

    def test_gold_sets_its_own_colour(self, styles):
        """"Gold" is a material, not a hue to apply to someone else's
        colour, and a blue gold is a contradiction."""
        blue = styles.MaterialSpec(base_color=(0.0, 0.0, 1.0, 1.0))
        gold = styles.apply_finish(blue, "gold")
        assert gold.base_color[0] > gold.base_color[2]

    def test_an_unknown_finish_names_the_ones_that_exist(self, styles):
        with pytest.raises(ValueError, match="glass"):
            styles.apply_finish(styles.MaterialSpec(base_color=(1, 1, 1, 1)),
                                "velvet")

    def test_it_composes_with_every_colour_mode(self, styles):
        """The point of the axis being separate."""
        style = styles.get_style("cosmos")
        for mode, keys in (("ring", [6, 5]), ("element", ["C", "N"]),
                           ("structure", [0, 1]), ("uniform", [0])):
            specs = styles.materials_for(style, mode, keys, finish="glass")
            assert len(specs) == len(keys)
            assert all(spec.transmission == 1.0 for spec in specs)

    def test_describe_covers_every_finish(self, styles):
        """A finish with no description is one nobody can choose."""
        text = styles.describe_finishes()
        for name in styles.FINISHES:
            assert name in text


class TestRepresentation:
    """Ball-and-stick or space-filling, and the mixed case -- which is
    the one that matters: a carbon-coated metal particle is a solid core
    inside an open cage, and drawing it either way alone is wrong about
    half of it."""

    def test_spacefill_uses_real_van_der_waals_radii(self, styles):
        """Not display radii scaled for taste: a space-filling model is
        *defined* by atoms touching at their vdW contact."""
        assert styles.vdw_radius("C") == pytest.approx(1.70, abs=0.01)
        assert styles.vdw_radius("Au") == pytest.approx(2.14, abs=0.01)
        drawn = styles.model_radius("C", "spacefill")
        assert drawn == pytest.approx(1.70 * styles.SPACEFILL_SCALE, rel=1e-6)

    def test_spacefill_is_much_bigger_than_ballstick(self, styles):
        assert (styles.model_radius("C", "spacefill")
                > 4 * styles.model_radius("C", "ballstick"))

    def test_auto_splits_metals_from_the_rest(self, styles):
        """The whole point. Gold solid, carbon open, in one picture."""
        assert styles.model_radius("Au", "auto") == \
            pytest.approx(styles.model_radius("Au", "spacefill"))
        assert styles.model_radius("C", "auto") == \
            pytest.approx(styles.model_radius("C", "ballstick"))

    def test_auto_draws_no_stick_into_a_metal(self, styles):
        """It would start inside a sphere of its own radius and come out
        the far side -- and between two metals it is an invention: a
        gold cluster's atoms are packed, not bonded in pairs."""
        assert styles.bond_is_drawn("auto", "C", "C") is True
        assert styles.bond_is_drawn("auto", "C", "Au") is False
        assert styles.bond_is_drawn("auto", "Au", "Au") is False

    def test_ballstick_draws_every_bond(self, styles):
        for pair in (("C", "C"), ("C", "Au"), ("Au", "Au")):
            assert styles.bond_is_drawn("ballstick", *pair) is True

    def test_spacefill_draws_none(self, styles):
        assert styles.draws_bonds("spacefill") is False
        assert styles.bond_is_drawn("spacefill", "C", "C") is False

    def test_licorice_drops_the_spheres_not_the_sticks(self, styles):
        assert styles.model_radius("C", "licorice") == 0.0
        assert styles.draws_bonds("licorice") is True

    def test_the_non_metals_of_this_package_stay_ball_and_stick(self,
                                                                styles):
        """Carbon, the dopants and the graft elements are what the
        lattice is made of, and the lattice is the subject."""
        for symbol in ("C", "N", "B", "O", "H", "S", "P", "Se"):
            assert not styles.is_spacefill_element(symbol)
        for symbol in ("Au", "Ag", "Pt", "Fe", "Ni", "Cu", "Mo", "W"):
            assert styles.is_spacefill_element(symbol)

    def test_an_unknown_model_names_the_real_ones(self, styles):
        with pytest.raises(ValueError, match="spacefill"):
            styles.model_radius("C", "sausage")

    def test_a_style_can_prefer_a_model(self, styles):
        """A model-kit look is ball-and-stick by definition and a
        supported-catalyst look is mixed."""
        assert styles.get_style("clusters_on_sheet").default_model == "auto"
        assert styles.get_style("model_kit").default_model == "ballstick"
        for name in styles.list_styles():
            assert styles.get_style(name).default_model in styles.MODELS

    def test_an_unknown_element_still_gets_a_radius(self, styles):
        assert styles.vdw_radius("Uup") == styles.DEFAULT_VDW_RADIUS
        assert styles.model_radius("Uup", "spacefill") > 0


class TestPeriodicSeams:
    """The worst-looking defect the pipeline had on a periodic
    structure, and it was not a rendering fault: a minimum-image bond
    list records the pair and throws away which image the far end was
    in, so a bond across a cell face is drawn as a line as long as the
    cell."""

    @pytest.fixture
    def periodic(self, bundle_mod):
        """A small cubic cell with one bond that leaves it."""
        positions = [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (9.0, 0.0, 0.0)]
        cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
        # (0, 2) is 9.0 A apart in raw coordinates and 1.0 A through the
        # face: the one the old code drew as a 9 A girder.
        bonds = [(0, 1), (0, 2)]
        return bundle_mod, positions, bonds, cell

    def test_an_interior_bond_survives(self, periodic):
        mod, positions, bonds, cell = periodic
        interior, _ = mod.split_seam_bonds(positions, bonds, cell)
        assert [0, 1] in interior

    def test_the_crossing_bond_is_taken_out(self, periodic):
        mod, positions, bonds, cell = periodic
        interior, stubs = mod.split_seam_bonds(positions, bonds, cell)
        assert [0, 2] not in interior
        assert len(stubs) == 2          # one at each end

    def test_the_stub_points_the_way_the_structure_continues(self, periodic):
        mod, positions, bonds, cell = periodic
        _, stubs = mod.split_seam_bonds(positions, bonds, cell)
        lengths = [float(np.linalg.norm(stub[1:])) for stub in stubs]
        # Half of the TRUE 1.0 A bond, not half of the 9 A raw one.
        assert all(abs(length - 0.5) < 1e-6 for length in lengths)

    def test_drop_leaves_nothing_behind(self, periodic):
        mod, positions, bonds, cell = periodic
        interior, stubs = mod.split_seam_bonds(positions, bonds, cell,
                                               "drop")
        assert interior == [[0, 1]] and stubs == []

    def test_keep_reproduces_the_defect(self, periodic):
        """Kept only so the bug can be reproduced on purpose."""
        mod, positions, bonds, cell = periodic
        interior, stubs = mod.split_seam_bonds(positions, bonds, cell,
                                               "keep")
        assert len(interior) == 2 and stubs == []

    def test_an_aperiodic_structure_loses_nothing(self, bundle_mod):
        positions = [(0.0, 0, 0), (1.4, 0, 0), (2.8, 0, 0)]
        bonds = [(0, 1), (1, 2)]
        interior, stubs = bundle_mod.split_seam_bonds(positions, bonds, None)
        assert len(interior) == 2 and stubs == []

    def test_a_cell_with_no_live_axis_is_left_alone(self, bundle_mod):
        """An all-zero cell is what a finite structure carries."""
        positions = [(0.0, 0, 0), (1.4, 0, 0)]
        interior, stubs = bundle_mod.split_seam_bonds(
            positions, [(0, 1)], [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert len(interior) == 1 and stubs == []

    def test_it_works_on_a_sheared_cell(self, bundle_mod):
        """Rounding the fractional coordinate is right for an
        orthogonal cell and can be off by one for a sheared one, which
        is why the shift is a candidate scan."""
        cell = [[10.0, 0, 0], [4.0, 9.0, 0], [0, 0, 10.0]]
        positions = [(0.0, 0.0, 0.0), (3.6, 8.5, 0.0)]
        interior, stubs = bundle_mod.split_seam_bonds(positions, [(0, 1)],
                                                      cell)
        # The short vector goes through the sheared face, so this is a
        # seam bond and the stub must be short.
        assert interior == []
        assert float(np.linalg.norm(stubs[0][1:])) < 1.5


class TestMeshDetail:
    """The faceting. An icosphere at subdivision 2 is EIGHTY triangles,
    which is what the pipeline drew for every atom at every size, and it
    is visibly faceted the moment an atom is more than a few per cent of
    the frame."""

    def test_the_levels_really_do_rise(self, styles):
        order = ["low", "normal", "high", "ultra"]
        subdivs = [styles.DETAIL_LEVELS[name]["sphere_subdiv"]
                   for name in order]
        segments = [styles.DETAIL_LEVELS[name]["bond_segments"]
                    for name in order]
        assert subdivs == sorted(subdivs) and len(set(subdivs)) == 4
        assert segments == sorted(segments)

    def test_the_old_fixed_value_is_now_the_floor(self, styles):
        """2 was the only level there was. It is the cheapest one now,
        for structures where an atom is a few pixels across."""
        assert styles.DETAIL_LEVELS["low"]["sphere_subdiv"] == 2
        assert styles.DETAIL_LEVELS["normal"]["sphere_subdiv"] > 2

    def test_a_small_molecule_gets_the_most(self, styles):
        """It fills the frame with big spheres; every triangle shows."""
        assert styles.detail_for(60)["sphere_subdiv"] >= 4

    def test_a_superstructure_gets_less(self, styles):
        """2000 atoms at subdivision 5 is ten million triangles before a
        single bond, and nobody can see them."""
        small = styles.detail_for(60)["sphere_subdiv"]
        large = styles.detail_for(1500)["sphere_subdiv"]
        huge = styles.detail_for(9000)["sphere_subdiv"]
        assert small > large > huge

    def test_it_is_monotone_in_the_atom_count(self, styles):
        previous = 99
        for count in (10, 100, 500, 1000, 2500, 5000, 20000):
            level = styles.detail_for(count)["sphere_subdiv"]
            assert level <= previous
            previous = level

    def test_an_explicit_level_wins(self, styles):
        assert styles.detail_for(60, "low")["sphere_subdiv"] == 2
        assert styles.detail_for(50000, "ultra")["sphere_subdiv"] == 5

    def test_an_unknown_level_names_the_real_ones(self, styles):
        with pytest.raises(ValueError, match="ultra"):
            styles.detail_for(100, "insane")


class TestDepthColouring:
    """The one colour mode that says anything on an all-carbon net,
    where element is a single hue and the rings are nearly all
    hexagons."""

    def test_it_is_a_real_colour_mode(self, styles):
        assert "depth" in styles.COLOUR_MODES
        assert "depth" in styles.SPLIT_BONDS
        assert styles.COLOUR_HELP["depth"]

    def test_the_ramp_runs_from_one_end_to_the_other(self, styles):
        low = styles.depth_colour(0.0)
        high = styles.depth_colour(1.0)
        assert low != high
        assert sum(high) > sum(low)       # ends light, as viridis does

    def test_the_ramp_is_monotone_in_brightness(self, styles):
        """So it survives being printed in black and white, which a
        rainbow does not: a figure must not lose its ordering."""
        greys = [sum(styles.depth_colour(i / 20.0)) for i in range(21)]
        assert greys == sorted(greys)

    def test_it_clamps_rather_than_raising(self, styles):
        assert styles.depth_colour(-5.0) == styles.depth_colour(0.0)
        assert styles.depth_colour(5.0) == styles.depth_colour(1.0)

    def test_the_bands_follow_the_structures_own_long_axis(self,
                                                           bundle_mod):
        """A sheet lying in xy and a tube standing in z should both get
        a gradient across the whole thing; picking z for both would
        give the sheet one colour."""
        flat = [(x, 0.0, 0.0) for x in np.linspace(0, 20, 40)]
        bands = bundle_mod.depth_bands(flat)
        assert min(bands) == 0
        assert max(bands) > 0

    def test_a_diagonal_structure_is_still_graded(self, bundle_mod):
        """Measured by principal axis, not by bounding box."""
        diagonal = [(t, t, t) for t in np.linspace(0, 20, 40)]
        bands = bundle_mod.depth_bands(diagonal)
        assert max(bands) - min(bands) > 10

    def test_materials_cover_every_band(self, styles):
        keys = list(range(styles.DEPTH_BANDS))
        specs = styles.materials_for(styles.get_style("lattice_dark"),
                                     "depth", keys)
        assert len(specs) == styles.DEPTH_BANDS
        assert specs[0].base_color != specs[-1].base_color


class TestSurfaceModel:
    def test_it_is_one_of_the_models(self, styles):
        assert "surface" in styles.MODELS
        assert styles.MODEL_HELP["surface"]

    def test_it_draws_neither_spheres_nor_sticks(self, styles):
        assert styles.model_radius("C", "surface") == 0.0
        assert styles.draws_bonds("surface") is False

    def test_the_style_built_on_it_uses_it(self, styles):
        assert styles.get_style("membrane").default_model == "surface"


class TestElaborateStyles:
    def test_they_use_the_new_machinery_rather_than_restyling(self,
                                                              styles):
        assert styles.get_style("porous_depth").default_colour_by == "depth"
        assert styles.get_style("membrane").default_model == "surface"
        assert styles.get_style("catalyst_stage").default_model == "auto"

    def test_the_catalyst_stage_is_built_for_a_mixed_phase(self, styles):
        """auto, so a metal phase comes out solid against the open
        support -- the picture a catalysis abstract wants."""
        style = styles.get_style("catalyst_stage")
        assert style.default_model == "auto"
        assert style.background.floor_metallic > 0

    def test_the_transmissive_one_pays_for_it_in_samples(self, styles):
        assert styles.get_style("membrane").samples >= 512


class TestSuperstructureStyles:
    """900-2000 atoms is a different problem: the subject is a lattice
    rather than an object, so the atoms have to shrink, the depth of
    field has to open and the colour has to come from curvature."""

    NAMES = ("lattice_dark", "lattice_depth", "lattice_blueprint",
             "lattice_studio")

    def test_they_shrink_the_atoms(self, styles):
        """Anything that fattens an atom closes the holes the eye reads
        depth through."""
        for name in self.NAMES:
            style = styles.get_style(name)
            if style.default_model != "licorice":
                assert style.atom_scale < 1.0

    def test_they_do_not_blur_two_thirds_of_the_structure(self, styles):
        """A lattice photographed at f/2 is a blur with a sharp band
        through it, which reads as a mistake rather than a choice."""
        for name in self.NAMES:
            stop = styles.get_style(name).camera_dof_fstop
            assert stop is None or stop >= 5.0

    def test_none_of_them_colours_by_element(self, styles):
        """An all-carbon net coloured by element is one grey mass."""
        for name in self.NAMES:
            assert styles.get_style(name).default_colour_by != "element"

    def test_the_blueprint_drops_the_spheres(self, styles):
        """At this size the atoms are what hide the topology, and the
        topology is the result."""
        assert styles.get_style("lattice_blueprint").default_model == \
            "licorice"

    def test_the_hazy_one_actually_carries_haze(self, styles):
        """Atmospheric perspective is the only depth cue that still
        works when a structure is mostly its own repetition."""
        style = styles.get_style("lattice_depth")
        assert any(spec.kind == "fog" for spec in style.effects)


class TestStyleSummary:
    """The window prints what a preset decides. A name plus a sentence
    of mood does not say which of a dozen choices it made."""

    def test_it_names_the_choices_that_change_the_picture(self, styles):
        text = styles.style_summary(styles.get_style("lattice_blueprint"))
        for field in ("backdrop", "colour by", "model", "camera", "render"):
            assert field in text
        assert "licorice" in text

    def test_it_lists_the_effects_a_style_brings(self, styles):
        assert "grid" in styles.style_summary(
            styles.get_style("lattice_blueprint"))
        assert "none" in styles.style_summary(
            styles.get_style("matter_section"))

    def test_every_style_can_describe_itself(self, styles):
        for name in styles.list_styles():
            text = styles.style_summary(styles.get_style(name))
            assert len(text) > 120
            assert "\n" in text

    def test_every_choice_has_a_help_line(self, styles):
        """A control whose options are not explained is a control the
        user has to guess at."""
        for model in styles.MODELS:
            assert styles.MODEL_HELP[model]
        for mode in styles.COLOUR_MODES:
            assert styles.COLOUR_HELP[mode]
        for kind in styles.BACKGROUNDS:
            assert styles.BACKGROUND_HELP[kind]


class TestNewEffects:
    def test_dust_grid_and_wave_exist_with_defaults(self, effects):
        for name in ("dust", "grid", "wave"):
            assert name in effects.EFFECTS
            spec = effects.EffectSpec(kind=name)
            assert spec.get("color") is not None

    def test_dust_is_one_merged_mesh(self):
        """Two hundred separate objects is two hundred outliner entries
        for somebody to scroll past after --save-blend."""
        source = (BLENDER_DIR / "effects.py").read_text()
        start = source.index("def _build_dust")
        body = source[start:source.index("def _build_grid")]
        assert body.count("bpy.data.objects.new") == 1
        assert "create_icosphere" in body

    def test_the_wave_is_a_torus_not_a_curve(self):
        """A wavefront has thickness; a curve's round bevel reads as a
        wire."""
        source = (BLENDER_DIR / "effects.py").read_text()
        body = source[source.index("def _build_wave"):
                      source.index("def _build_bokeh")]
        assert "create_cone" in body
        assert '"thickness"' in body

    def test_every_effect_has_a_builder(self, effects):
        source = (BLENDER_DIR / "effects.py").read_text()
        for name in effects.EFFECTS:
            assert f'"{name}": _build_' in source or name == "glare"

    def test_none_of_them_lights_the_scene_by_default(self, effects):
        """An emissive object is a light source in Cycles, and a field
        of motes bright enough to light anything is a field nobody would
        believe."""
        for name in ("dust", "grid", "wave", "bokeh", "arc", "resonance"):
            assert effects.EFFECT_DEFAULTS[name]["light"] == 0.0


class TestBondMaterials:
    """The bonds are decided separately from the atoms. They were not:
    whenever the colour belonged to the atom the bonds simply reused the
    atoms' material list, so glass atoms meant glass bonds and there was
    no way to say otherwise."""

    def test_auto_keeps_the_old_behaviour(self, styles):
        style = styles.get_style("paper_figure")
        _, split = styles.bond_materials(style, "element", ["C", "O"])
        assert split is True
        _, split = styles.bond_materials(style, "ring", [6, 5])
        assert split is False

    def test_atoms_gives_one_material_per_atom_slot(self, styles):
        style = styles.get_style("paper_figure")
        specs, split = styles.bond_materials(style, "element",
                                             ["C", "N", "O"], "atoms")
        assert split is True and len(specs) == 3

    def test_style_and_neutral_give_exactly_one(self, styles):
        style = styles.get_style("paper_figure")
        for mode in ("style", "neutral"):
            specs, split = styles.bond_materials(style, "element",
                                                 ["C", "O"], mode)
            assert split is False and len(specs) == 1

    def test_neutral_is_grey_whatever_the_style(self, styles):
        for name in ("cosmos", "paper_figure", "neon_cyber"):
            specs, _ = styles.bond_materials(styles.get_style(name),
                                             "element", ["C"], "neutral")
            red, green, blue = specs[0].base_color[:3]
            assert abs(red - green) < 0.05 and abs(green - blue) < 0.05

    def test_atoms_falls_back_where_there_are_no_halves(self, styles):
        """Ring colouring has no per-atom colour to halve: an atom on a
        pentagon and a hexagon has two, and picking one would be
        arbitrary."""
        specs, split = styles.bond_materials(styles.get_style("cosmos"),
                                             "ring", [6, 5], "atoms")
        assert split is False and len(specs) == 1

    def test_the_bonds_can_take_their_own_finish(self, styles):
        """Glass atoms on metal bonds is two decisions, not one."""
        style = styles.get_style("paper_figure")
        specs, _ = styles.bond_materials(style, "element", ["C"],
                                         "style", finish="metal")
        assert specs[0].metallic == 1.0

    def test_an_unknown_bond_colour_is_refused(self, styles):
        with pytest.raises(ValueError, match="neutral"):
            styles.bond_materials(styles.get_style("cosmos"), "element",
                                  ["C"], "rainbow")


class TestRimLight:
    """Emission only where the surface turns away from the camera. It is
    the one cover-render trick a *lamp* cannot do: a rim light draws a
    bright edge on one side, while this draws one on every silhouette
    the viewer can see, including the far wall of a cage seen through
    the near one."""

    def test_it_is_off_by_default(self, styles):
        assert styles.MaterialSpec(base_color=(1, 1, 1, 1)).rim == 0.0

    def test_the_dark_cover_styles_use_it(self, styles):
        assert styles.get_style("nature_nano").hexagon.rim > 0

    def test_the_figure_styles_do_not(self, styles):
        """A rim glow on a manuscript figure is decoration on something
        a reviewer has to measure."""
        for name in ("matter_section", "paper_figure"):
            assert styles.get_style(name).hexagon.rim == 0.0

    def test_the_render_script_wires_it_by_fresnel(self):
        source = (BLENDER_DIR / "render_cnt.py").read_text()
        assert "ShaderNodeLayerWeight" in source
        # Added, not mixed: a rim is light arriving on top of the
        # material, and mixing would dim the body to pay for the edge.
        assert "ShaderNodeAddShader" in source

    def test_a_finish_does_not_silently_drop_it(self, styles):
        base = styles.MaterialSpec(base_color=(0.3, 0.3, 0.3, 1.0), rim=2.0)
        assert styles.apply_finish(base, "metal").rim == 2.0


class TestStrokeShape:
    """A filament of constant width is the single thing that makes an
    effect read as a drawn line rather than as light."""

    def test_the_ends_profile_is_thin_at_both_ends(self, effects):
        profile = effects.taper_profile(21, "ends")
        assert profile[0] < 0.3 and profile[-1] < 0.3
        assert profile[10] > 0.9

    def test_the_taper_profile_only_thins_one_end(self, effects):
        profile = effects.taper_profile(21, "taper")
        assert profile[0] > 0.9 and profile[-1] < 0.2

    def test_none_is_flat(self, effects):
        assert np.allclose(effects.taper_profile(9, "none"), 1.0)

    def test_no_profile_ever_reaches_zero(self, effects):
        """A stroke of zero radius leaves a visible gap in the curve."""
        for kind in ("none", "ends", "taper", "beam"):
            assert effects.taper_profile(33, kind).min() > 0.0


class TestFractalArcs:
    """Midpoint displacement, not a random walk. A walk's displacement
    is the sum of its steps, so it drifts slowly and smoothly and the
    result is a meander -- drawn as a filament it reads as a wire
    somebody bent, which is exactly what the arcs looked like."""

    def test_the_ends_are_pinned(self, effects):
        rng = np.random.default_rng(7)
        offsets = effects.fractal_offsets(65, rng)
        assert np.allclose(offsets[0], 0.0)
        assert np.allclose(offsets[-1], 0.0)

    def test_it_is_normalised_so_jitter_means_one_thing(self, effects):
        """Otherwise a finer arc is also a wilder one."""
        rng = np.random.default_rng(8)
        for count in (17, 33, 65, 129):
            assert np.abs(effects.fractal_offsets(count, rng)).max() == \
                pytest.approx(1.0)

    def test_it_has_detail_at_more_than_one_scale(self, effects):
        """The property a random walk lacks: compare the roughness of
        the whole path with that of a small window of it. A walk is
        smooth locally; a fractal is not."""
        rng = np.random.default_rng(9)
        offsets = effects.fractal_offsets(257, rng)[:, 0]
        fine = np.abs(np.diff(offsets)).mean()
        coarse = np.abs(np.diff(offsets[::16])).mean()
        # Detail survives at the fine scale rather than averaging out.
        assert fine > coarse / 16.0

    def test_roughness_controls_how_wild_it_is(self, effects):
        rng = np.random.default_rng(10)
        calm = np.abs(np.diff(
            effects.fractal_offsets(129, rng, roughness=0.35)[:, 0])).mean()
        rng = np.random.default_rng(10)
        wild = np.abs(np.diff(
            effects.fractal_offsets(129, rng, roughness=0.85)[:, 0])).mean()
        assert wild > calm

    def test_an_arc_still_meets_its_anchors(self, effects):
        rng = np.random.default_rng(11)
        start, end = np.zeros(3), np.array([10.0, 0.0, 0.0])
        path = effects.arc_polyline(start, end, 64, 0.4, 5.0, rng)
        assert np.allclose(path[0], start, atol=1e-9)
        assert np.allclose(path[-1], end, atol=1e-9)


class TestHaloIsAVolume:
    def test_it_is_built_as_a_volume_not_a_surface(self):
        """A surface shell -- emissive, or Fresnel-driven at its limb --
        always keeps the sphere's silhouette, so it reads as a disc or a
        ring drawn around the structure however it is tuned. Both were
        tried; neither stopped being a hard edge."""
        source = (BLENDER_DIR / "effects.py").read_text()
        assert "ShaderNodeVolumePrincipled" in source
        assert 'out.inputs["Volume"]' in source

    def test_its_brightness_is_divided_by_the_path_length(self):
        """Volume emission accumulates along the ray, so a shell twice
        as wide glows twice as brightly for the same setting. The first
        try came back a white frame with the structure a ghost in it."""
        source = (BLENDER_DIR / "effects.py").read_text()
        assert "2.0 * reach" in source

    def test_halo_has_a_density(self, effects):
        assert "density" in effects.EFFECT_DEFAULTS["halo"]


class TestQualityPresets:
    """Lives in ``styles`` rather than in the render script so it can be
    imported rather than scraped: the first version of this test parsed
    the dict out of the source text and broke on its own brace."""

    def test_resolution_and_samples_rise_together(self, styles):
        """One decision, not two: a 6000 px frame at 64 samples is a big
        noisy picture and a 600 px one at 1024 is a slow thumbnail."""
        presets = styles.QUALITY_PRESETS
        order = ["draft", "preview", "print", "cover"]
        sides = [presets[name]["long_side"] for name in order]
        samples = [presets[name]["samples"] for name in order]
        assert sides == sorted(sides)
        assert samples == sorted(samples)

    def test_print_is_300_dpi_across_a_journal_column(self, styles):
        """8.5 cm at 300 dpi is about 1000 px; the preset leaves room to
        crop, so it must be comfortably above that and below a page."""
        presets = styles.QUALITY_PRESETS
        assert 1600 <= presets["print"]["long_side"] <= 3000
        assert presets["cover"]["long_side"] >= 4000

    def test_the_script_keeps_the_aspect_ratio(self):
        """A cover asked for at 2000x2400 must stay 5:6, not go square."""
        source = (BLENDER_DIR / "render_cnt.py").read_text()
        assert "max(width, height)" in source


class TestEngine:
    def test_every_style_asks_for_cycles(self, styles):
        """Cycles is the default throughout, and --engine is the
        override. This pins which way round that is -- the question
        "is this EEVEE?" has a definite answer."""
        assert {styles.get_style(n).engine
                for n in styles.list_styles()} == {"CYCLES"}

    def test_the_render_script_maps_both_engine_names(self):
        """EEVEE's identifier changed across Blender versions
        (BLENDER_EEVEE_NEXT through 4.x), so both are tried."""
        source = (BLENDER_DIR / "render_cnt.py").read_text()
        assert "BLENDER_EEVEE_NEXT" in source
        assert "BLENDER_EEVEE" in source
        assert '"CYCLES"' in source

    def test_eevee_is_ruled_out_before_it_can_abort(self):
        """EEVEE is a rasteriser: with no GPU context it does not raise,
        it kills the process with SIGABRT after one line about libEGL,
        which cannot be caught from Python once it starts. So it is
        checked for beforehand -- and specifically for EGL, since libGL
        is present on headless boxes and finding it proves nothing."""
        source = (BLENDER_DIR / "render_cnt.py").read_text()
        assert "_can_rasterise" in source
        assert 'find_library("EGL")' in source


class TestBokeh:
    def test_it_is_a_real_effect_with_parameters(self, effects):
        assert "bokeh" in effects.EFFECTS
        spec = effects.parse_effect("bokeh:count=12,color=cyan")
        assert spec.get("count") == 12.0

    def test_it_is_geometry_behind_the_subject(self):
        """Painted discs stay the same size and shape wherever the
        camera goes; real spheres behind the focal plane grow, shift and
        overlap when the lens or the aperture changes."""
        source = (BLENDER_DIR / "effects.py").read_text()
        assert "_build_bokeh" in source
        assert '"depth"' in source

    def test_the_style_that_uses_it_opens_the_aperture(self, styles):
        """Bokeh with everything in focus is a field of dots."""
        style = styles.get_style("neon_violet")
        assert any(spec.kind == "bokeh" for spec in style.effects)
        assert style.camera_dof_fstop is not None
        assert style.camera_dof_fstop < 2.0


class TestReferenceStyles:
    """The five worked back from photographs and stock renders. Each
    test is the one claim its description makes."""

    def test_the_model_kit_is_glossy_plastic_on_matte_rods(self, styles):
        style = styles.get_style("model_kit")
        assert style.hexagon.clearcoat > 0.8      # injection-moulded gloss
        assert style.hexagon.metallic == 0.0
        assert style.bond.roughness > 0.4         # matte white rod
        assert style.camera_dof_fstop < 2.0       # blurred bench behind

    def test_clear_glass_transmits(self, styles):
        style = styles.get_style("clear_glass")
        assert style.hexagon.transmission == 1.0
        # Glass is the slowest thing here: every sphere refracts twice.
        assert style.samples >= 512

    def test_neon_violet_rims_in_magenta(self, styles):
        rim = styles.get_style("neon_violet").hexagon
        assert rim.rim > 1.0
        assert rim.rim_color[0] > rim.rim_color[1]

    def test_teal_lab_keeps_colour_rather_than_reflection(self, styles):
        """Everything glossy and nothing metallic, or the enamel turns
        into a mirror and the colour goes."""
        style = styles.get_style("teal_lab")
        assert style.hexagon.metallic == 0.0
        assert style.hexagon.clearcoat > 0.8

    def test_the_glossy_floors_reflect(self, styles):
        for name in ("clusters_on_sheet", "neon_violet"):
            background = styles.get_style(name).background
            assert background.has_floor
            assert background.floor_metallic > 0.0
            assert background.floor_roughness < 0.2


class TestStylesCarryEffects:
    def test_the_cover_styles_bring_their_own(self, styles):
        """A style that is *about* a laser should arrive with the laser
        rather than needing it typed in beside the style name."""
        assert styles.get_style("laser_probe").effects
        assert any(spec.kind == "laser"
                   for spec in styles.get_style("laser_probe").effects)
        assert any(spec.kind == "arc"
                   for spec in styles.get_style("electric_storm").effects)

    def test_every_style_effect_is_a_real_effect(self, styles, effects):
        for name in styles.list_styles():
            for spec in styles.get_style(name).effects:
                assert spec.kind in effects.EFFECTS

    def test_effects_are_a_tuple_on_every_style(self, styles):
        """A mutable default on a frozen dataclass is shared between
        every instance, and a style that accumulated effects across
        renders would be very hard to diagnose."""
        for name in styles.list_styles():
            assert isinstance(styles.get_style(name).effects, tuple)

    def test_the_figure_styles_bring_none(self, styles):
        """A manuscript figure with a laser in it is not a figure."""
        assert styles.get_style("paper_figure").effects == ()
        assert styles.get_style("electron_micrograph").effects == ()


class TestSkinRadius:
    """The surface model's skin is a style property, not a constant.

    At the old hardcoded 1.6 A a skin is wider than a graphene
    hexagon's circumradius (1.42), so adjacent struts merge and the
    pores weld shut -- which is right for a solid wide shot and wrong
    for a style whose subject is the holes.
    """

    def test_every_style_has_a_skin_radius(self, styles):
        for name in styles.list_styles():
            style = styles.get_style(name)
            assert style.skin_radius > 0.0, name

    def test_default_is_the_old_constant(self, styles):
        """Existing styles must render exactly as they did."""
        assert styles.Style.skin_radius == 1.6

    def test_membrane_keeps_its_pores_open(self, styles):
        """Its description promises holes through it, so measure that.

        A graphene hexagon's circumradius is 1.42 A, and a strut skinned
        at radius r closes the ring when r approaches that. Half of it
        leaves a clear pore.
        """
        membrane = styles.get_style("membrane")
        assert membrane.default_model == "surface"
        assert membrane.skin_radius < 1.42 / 2


class TestIridescentStyles:
    """Depth ramps are per style, and the trade is stated.

    The viridis default is monotone in brightness so a greyscale print
    keeps its ordering. A hue-cycling ramp is the better cover and the
    worse data, which is why it is opt-in per style rather than a
    global change to the depth mode.
    """

    IRIDESCENT = ("oil_film", "aurora_net", "plasma_depth")

    def test_the_three_styles_exist_and_colour_by_depth(self, styles):
        for name in self.IRIDESCENT:
            style = styles.get_style(name)
            assert style.default_colour_by == "depth", name
            assert style.depth_ramp, name

    def test_default_ramp_is_unchanged(self, styles):
        """A style that sets no ramp must render exactly as before."""
        assert styles.Style.depth_ramp == ()
        assert styles.depth_colour(0.5) == styles.depth_colour(
            0.5, styles.DEPTH_RAMP)

    def test_an_override_actually_changes_the_colour(self, styles):
        assert (styles.depth_colour(0.5, styles.IRIDESCENT_RAMP)
                != styles.depth_colour(0.5, styles.DEPTH_RAMP))

    def test_ramps_stay_inside_the_unit_cube(self, styles):
        for ramp in (styles.IRIDESCENT_RAMP, styles.AURORA_RAMP,
                     styles.PLASMA_RAMP):
            for entry in ramp:
                assert len(entry) == 3
                assert all(0.0 <= channel <= 1.0 for channel in entry)

    def test_plasma_is_the_one_that_survives_greyscale(self, styles):
        """Its description claims that, so measure it.

        Rec. 709 luma, which is what a greyscale conversion uses.
        """
        def luma(rgb):
            return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

        greys = [luma(entry) for entry in styles.PLASMA_RAMP]
        assert greys == sorted(greys)

    def test_the_cycling_ramps_do_not_claim_to(self, styles):
        """The honest half: these are covers, not measurements."""
        def luma(rgb):
            return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

        for ramp in (styles.IRIDESCENT_RAMP, styles.AURORA_RAMP):
            greys = [luma(entry) for entry in ramp]
            assert greys != sorted(greys)

    def test_the_gradient_is_not_drowned_in_haze(self, styles):
        """porous_depth read as a grey wash at fog 0.0035.

        A fog volume scatters the key light across the whole frame, so
        past a trace of it the backdrop lifts and the gradient loses the
        contrast it is made of.
        """
        fogs = [spec for spec in styles.get_style("porous_depth").effects
                if spec.kind == "fog"]
        assert fogs and fogs[0].get("strength") <= 0.001

    def test_a_space_background_actually_has_stars(self, styles):
        """star_density is a Voronoi SCALE, not a fraction.

        aurora_net was written with 0.22, read as "22 per cent". At that
        scale one Voronoi cell covers the whole sky and the field comes
        out starless -- while the style's description promises far
        stars. The floor here is well below any working value (the two
        shipped space styles use 260 and 300) and well above a fraction.
        """
        for name in styles.list_styles():
            background = styles.get_style(name).background
            if background.kind not in ("space", "nebula"):
                continue
            assert background.star_density >= 10.0, (
                f"{name}: star_density {background.star_density} is a "
                "fraction, not a Voronoi scale")


class TestDepthAxis:
    """A round subject has no longest axis, so `depth` needs a choice.

    Measured on the structures this package builds: a C240 cage's first
    two singular values are equal to four decimals (1.0000) and a coated
    spherical nanoparticle's to two (1.0123), against 1.73 for a sheet,
    3.01 for a row of three and 6.29 for a long net. Below the cage
    threshold the principal axis is decided by rounding, so the gradient
    runs along a direction a tiny perturbation rotates.
    """

    def test_a_sphere_is_flagged_and_a_sheet_is_not(self, bundle_mod):
        rng = np.random.default_rng(0)
        shell = rng.normal(size=(400, 3))
        shell /= np.linalg.norm(shell, axis=1)[:, None]
        assert bundle_mod.depth_axis_quality(shell) < \
            bundle_mod.DEGENERATE_AXIS

        sheet = rng.uniform(size=(400, 3)) * np.array([40.0, 8.0, 0.4])
        assert bundle_mod.depth_axis_quality(sheet) >= \
            bundle_mod.DEGENERATE_AXIS

    def test_the_default_path_is_unchanged(self, bundle_mod):
        """None must mean exactly what no argument meant before."""
        rng = np.random.default_rng(1)
        points = rng.uniform(size=(120, 3)) * np.array([30.0, 6.0, 2.0])
        assert (bundle_mod.depth_bands(points)
                == bundle_mod.depth_bands(points, None))

    def test_an_explicit_axis_grades_along_it(self, bundle_mod):
        """A sheet in xy graded along z must come out one band."""
        rng = np.random.default_rng(2)
        sheet = rng.uniform(size=(200, 3)) * np.array([40.0, 40.0, 0.0])
        along_z = bundle_mod.depth_bands(sheet, (0.0, 0.0, 1.0))
        assert len(set(along_z)) == 1
        along_x = bundle_mod.depth_bands(sheet, (1.0, 0.0, 0.0))
        assert len(set(along_x)) > 1

    def test_the_zero_vector_is_refused(self, bundle_mod):
        points = np.zeros((4, 3))
        points[:, 0] = [0.0, 1.0, 2.0, 3.0]
        with pytest.raises(ValueError):
            bundle_mod.depth_bands(points, (0.0, 0.0, 0.0))

    def test_the_flag_offers_the_remedy(self):
        """The note must name the option that fixes it."""
        source = (BLENDER_DIR / "render_cnt.py").read_text()
        assert "--depth-axis" in source
        assert '"view"' in source or "'view'" in source
