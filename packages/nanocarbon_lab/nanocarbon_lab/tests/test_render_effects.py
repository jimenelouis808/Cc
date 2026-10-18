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
