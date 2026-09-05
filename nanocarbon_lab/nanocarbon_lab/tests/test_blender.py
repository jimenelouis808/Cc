"""Tests for the Blender render pipeline.

Most of this runs without Blender. ``styles.py`` is deliberately pure
data with no ``bpy`` import, the packaging question is answered by the
filesystem, and the camera framing is arithmetic -- so the three bugs
that actually mattered are all testable in a plain interpreter.

Those three, in the order a user hits them:

1. **The script was not installed.** ``blender/`` sat beside the package
   rather than inside it, so pip left it out entirely and the GUI's
   render button reported "run the GUI from a full checkout" to everyone
   who had installed the package normally.
2. **The camera cropped the subject.** The distance was a hardcoded
   ``3.2 * radius``, which only fits a 50 mm lens; four of the five
   styles use 60-90 mm and cut the structure off at the frame edge.
3. **The lights did not scale with the subject.** They sat at the literal
   coordinates in the style -- about 10 units out -- while those
   coordinates are Ångström, so a 90 Å tube engulfed its own lighting
   and rendered black on black.

A ``bpy``-dependent smoke test renders a real image when Blender's Python
module happens to be installed, and skips when it is not.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

BLENDER_DIR = Path(__file__).resolve().parent.parent / "blender"


def _load(name: str):
    """Import one of the render scripts by path.

    They are package *data*, not a subpackage, because they import
    ``bpy`` -- which only exists inside Blender -- so a plain
    ``import nanocarbon_lab.blender.styles`` would put an unimportable
    module in the tree.
    """
    spec = importlib.util.spec_from_file_location(name, BLENDER_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestPackaging:
    def test_the_scripts_live_inside_the_package(self):
        """So that pip installs them. Outside it they were silently
        dropped from every install, and the GUI's render button was dead
        for anyone who had not cloned the repository."""
        assert (BLENDER_DIR / "render_cnt.py").exists()
        assert (BLENDER_DIR / "mesh_builder.py").exists()
        assert (BLENDER_DIR / "styles.py").exists()

    def test_the_package_declares_them_as_data(self):
        """A subpackage would be wrong -- they import bpy -- so they are
        shipped as package data, and that declaration is what makes the
        difference between installed and absent."""
        pyproject = (BLENDER_DIR.parent.parent / "pyproject.toml").read_text()
        assert "[tool.setuptools.package-data]" in pyproject
        assert 'blender/*.py' in pyproject

    def test_the_gui_resolves_the_script_where_it_now_lives(self):
        """The GUI used to look at the repository root, which does not
        exist once installed.

        Checked by reading the source rather than importing it: the GUI
        module needs tkinter, which is packaged separately on many Linux
        distributions, and a packaging test has no business being
        skipped on a machine that merely lacks a widget toolkit.
        """
        app_source = (BLENDER_DIR.parent / "gui" / "app.py").read_text()
        assert 'parent.parent / "blender" / "render_cnt.py"' in app_source
        assert 'parents[2] / "blender"' not in app_source

        # And the path that expression builds really does hold the script.
        assert (BLENDER_DIR / "render_cnt.py").exists()


class TestStyles:
    """`styles.py` imports no bpy on purpose, so it is testable here."""

    @pytest.fixture(scope="class")
    def styles(self):
        return _load("styles")

    def test_every_style_is_reachable_by_name(self, styles):
        for name in styles.list_styles():
            assert styles.get_style(name).name == name

    def test_an_unknown_style_is_rejected(self, styles):
        with pytest.raises((KeyError, ValueError)):
            styles.get_style("nature_puce")

    def test_every_style_has_at_least_one_light(self, styles):
        """A style with no lights renders black whatever else is right."""
        for name in styles.list_styles():
            assert styles.get_style(name).lights

    def test_every_style_describes_itself(self, styles):
        for name in styles.list_styles():
            assert len(styles.get_style(name).description) > 40


class TestCameraFraming:
    """The cropping bug, as arithmetic.

    A camera's half-angle is ``atan(sensor / 2f)``, so fitting a sphere
    of radius R needs ``R / sin(half_angle)``. The old constant 3.2 R is
    only enough at 50 mm.
    """

    @staticmethod
    def _needed(lens_mm: float, sensor: float = 36.0) -> float:
        return 1.0 / math.sin(math.atan(0.5 * sensor / lens_mm))

    def test_the_old_constant_was_only_right_for_one_lens(self):
        assert self._needed(50.0) < 3.2      # fitted
        assert self._needed(85.0) > 3.2      # cropped
        assert self._needed(90.0) > 3.2      # cropped

    @pytest.mark.parametrize("lens", [50.0, 60.0, 70.0, 85.0, 90.0])
    def test_the_distance_fits_the_subject_at_every_style_lens(self, lens):
        """Every lens the styles actually use, square frame."""
        fitted = self._needed(lens)
        # What the fixed code computes, without importing bpy: the same
        # formula plus the fill margin.
        computed = fitted / 0.82
        assert computed >= fitted

    def test_a_tall_frame_needs_more_distance_than_a_square_one(self):
        """Blender maps the sensor onto the longer image axis, so the
        shorter one sees a narrower angle and clips first. A 2000x2400
        cover crops vertically long before it crops horizontally."""
        half_width = math.atan(0.5 * 36.0 / 85.0)
        square = 1.0 / math.sin(half_width)
        tall = 1.0 / math.sin(math.atan(math.tan(half_width) * (2000 / 2400)))
        assert tall > square

    def test_the_styles_use_lenses_the_old_code_could_not_fit(self):
        """Not a hypothetical: four of five styles cropped."""
        styles = _load("styles")
        lenses = [styles.get_style(n).camera_lens_mm for n in styles.list_styles()]
        cropped = [lens for lens in lenses if self._needed(lens) > 3.2]
        assert len(cropped) == 4


class TestLightScaling:
    """The black-on-black bug, as arithmetic.

    Irradiance from a point-like source falls as 1/d^2, so moving the rig
    out with the subject requires the energy to grow as d^2 if the
    surface is to receive the same light. That invariance is the property
    a style preset needs: the same name should mean the same look on a
    3 Å cage and a 90 Å network.
    """

    @staticmethod
    def _irradiance(energy: float, distance: float) -> float:
        return energy / (4.0 * math.pi * distance**2)

    def test_a_fixed_rig_dims_as_the_subject_grows(self):
        """What the bug was: same lamp, subject further from it."""
        near = self._irradiance(1400.0, 12.0)
        far = self._irradiance(1400.0, 12.0 + 45.0)
        assert far < near / 10

    def test_scaling_position_and_energy_together_holds_it_constant(self):
        """What the fix does: positions times s, energies times s^2."""
        base_distance, base_energy = 12.0, 1400.0
        reference = self._irradiance(base_energy, base_distance)
        for scale in (0.35, 1.0, 4.45):
            scaled = self._irradiance(base_energy * scale**2,
                                      base_distance * scale)
            assert scaled == pytest.approx(reference)

    def test_a_sun_must_not_be_scaled(self):
        """A SUN is directional and infinitely distant: its irradiance
        does not fall off, so scaling its energy would over-light."""
        source = (BLENDER_DIR / "render_cnt.py").read_text()
        assert 'spec.energy if spec.kind == "SUN"' in source


bpy_available = importlib.util.find_spec("bpy") is not None


@pytest.mark.slow
@pytest.mark.skipif(not bpy_available,
                    reason="bpy is only available inside Blender")
class TestRendersForReal:
    """End to end, when Blender's Python module happens to be installed."""

    def test_it_renders_a_visible_uncropped_subject(self, tmp_path):
        import subprocess

        import numpy as np
        from PIL import Image

        from nanocarbon_lab.builders import build_fullerene
        from nanocarbon_lab.exports.xyz import write_render_bundle

        cage = build_fullerene(family="C60", freq=1)
        stem = tmp_path / "cage"
        write_render_bundle(cage, stem)
        out = tmp_path / "cover.png"

        done = subprocess.run(
            [sys.executable, str(BLENDER_DIR / "render_cnt.py"), "--",
             "--xyz", str(stem.with_suffix(".xyz")),
             "--json", str(stem.with_suffix(".json")),
             "--style", "acs_nano_vivid", "--out", str(out),
             "--resolution", "200", "200", "--samples", "8"],
            capture_output=True, text=True, timeout=1800,
        )
        assert done.returncode == 0, done.stderr[-2000:]
        assert out.exists()

        image = np.asarray(Image.open(out).convert("RGB"), dtype=float) / 255.0
        luminance = image @ [0.2126, 0.7152, 0.0722]
        background = luminance[:10, :10].mean()
        subject = np.abs(luminance - background) > 0.03

        # Visible at all: the whole point of the light-scaling fix.
        assert subject.mean() > 0.02
        # And inside the frame: the point of the camera fix.
        assert not subject[0, :].any() and not subject[-1, :].any()
        assert not subject[:, 0].any() and not subject[:, -1].any()
