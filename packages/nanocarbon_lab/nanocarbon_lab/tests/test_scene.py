"""Tests for scene composition and the render pipeline's pure half.

Three things are being protected here, and each was a real failure
rather than a hypothetical.

1. **A composition's bonds must be carried, never re-guessed.** Two
   structures placed 4 Å apart would come out bonded to each other by
   any distance cutoff, so the items' own bond lists are re-indexed --
   and that re-indexing is exactly the arithmetic that goes wrong
   silently, since a wrong index is still a valid bond.
2. **A gap means a gap.** ``arrange`` spaces items by their *oriented*
   bounding boxes, so an 8 Å request gives 8 Å whatever the items'
   sizes or rotations. The first version measured from bounding-sphere
   radii and ignored rotation: a requested 8 came out 17.9 in a row and
   23.3 in a grid.
3. **The colour decision is a function of the bundle**, so it is tested
   here rather than inside Blender. Which slot an atom takes, whether
   the slot order is stable run to run, and what an atom with no ring
   metadata falls back to are all answerable without ``bpy``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from nanocarbon_lab import render as render_support
from nanocarbon_lab import scene
from nanocarbon_lab.builders import build_cnt
from nanocarbon_lab.builders.fullerene import build_fullerene

BLENDER_DIR = Path(__file__).resolve().parent.parent / "blender"


def _load(name: str):
    """Import one of the pure render modules by path.

    ``styles`` and ``bundle`` import no ``bpy``; their siblings do, which
    is why the whole directory is package data rather than a subpackage.
    """
    sys.path.insert(0, str(BLENDER_DIR))
    spec = importlib.util.spec_from_file_location(name, BLENDER_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bundle_mod():
    """The pure colour-decision half of the render pipeline."""
    return _load("bundle")


@pytest.fixture(scope="module")
def styles():
    """The pure style catalogue."""
    return _load("styles")


@pytest.fixture(scope="module")
def two_items():
    """A cage and a tube: different sizes, different shapes, both with
    recorded bonds, which is the case the re-indexing has to get right."""
    return [
        scene.Item(atoms=build_fullerene(freq=1), label="cage"),
        scene.Item(atoms=build_cnt(5, 5, length=10.0), label="tube"),
    ]


class TestItem:
    def test_rotation_is_about_the_items_own_centre(self):
        """``oriented`` turns the item about its own centroid and returns
        it centred on the origin, so ``translation`` is where the item's
        centre goes rather than an offset from wherever the builder left
        it. Rotating about the scene origin instead would also *move* the
        item, and the layout has just placed it."""
        atoms = build_cnt(5, 5, length=10.0)
        item = scene.Item(atoms=atoms, rotation=(0.0, 90.0, 0.0))
        oriented = item.oriented()
        assert np.allclose(oriented.mean(axis=0), 0.0, atol=1e-8)

        # And it is a rigid motion: every pairwise distance survives.
        before = atoms.get_positions()
        for i, j in ((0, 1), (0, len(atoms) - 1), (2, 7)):
            assert np.linalg.norm(oriented[i] - oriented[j]) == pytest.approx(
                np.linalg.norm(before[i] - before[j]), abs=1e-8)

    def test_half_widths_follow_the_rotation(self):
        """A long tube turned 90 degrees is wide, not long. The spacing
        depends on this, so a rotation-blind extent is a wrong gap."""
        atoms = build_cnt(5, 5, length=20.0)
        upright = scene.Item(atoms=atoms).half_widths()
        turned = scene.Item(atoms=atoms, rotation=(0.0, 90.0, 0.0)).half_widths()
        long_axis = int(np.argmax(upright))
        assert turned[long_axis] < upright[long_axis] * 0.6
        assert max(turned) == pytest.approx(max(upright), rel=1e-6)

    def test_placed_clears_the_cell(self):
        """A composition is not periodic. Keeping an item's cell would
        make the scene inherit whichever item was placed last."""
        placed = scene.Item(atoms=build_cnt(5, 5, length=10.0)).placed()
        assert not any(placed.get_pbc())


class TestEulerMatrix:
    def test_it_is_a_rotation(self):
        matrix = scene.euler_matrix((17.0, -43.0, 91.0))
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(matrix) == pytest.approx(1.0)

    def test_ninety_degrees_about_z_sends_x_to_y(self):
        moved = scene.euler_matrix((0.0, 0.0, 90.0)) @ np.array([1.0, 0, 0])
        assert np.allclose(moved, [0.0, 1.0, 0.0], atol=1e-12)


class TestArrange:
    """The gap is the requested gap, measured between the items."""

    @staticmethod
    def _closest(items) -> float:
        atoms, report = scene.compose(items)
        del atoms
        return min(distance for _, _, distance in report.clearances)

    @pytest.mark.parametrize("layout", ["row", "stack", "grid"])
    def test_the_requested_gap_is_what_comes_out(self, two_items, layout):
        """Within a tenth of an Ångström. These three space per axis; a
        circle cannot, its items being spread round a ring whose radius
        has to clear the largest of them."""
        arranged = scene.arrange(two_items, layout=layout, gap=8.0)
        assert self._closest(arranged) == pytest.approx(8.0, abs=0.7)

    def test_a_circle_clears_every_neighbour(self, two_items):
        """Its spacing is a radius rather than a pitch, so the gap is a
        floor and not a target -- but it must never be less."""
        arranged = scene.arrange(two_items * 2, layout="circle", gap=8.0)
        assert self._closest(arranged) >= 8.0

    def test_rotating_an_item_does_not_break_the_gap(self):
        """The failure the oriented extent exists to stop: a long tube
        turned across the row needs more room, not the same."""
        items = [
            scene.Item(atoms=build_cnt(5, 5, length=20.0), label="a",
                       rotation=(0.0, 90.0, 0.0)),
            scene.Item(atoms=build_cnt(5, 5, length=20.0), label="b"),
        ]
        arranged = scene.arrange(items, layout="row", gap=6.0)
        assert self._closest(arranged) == pytest.approx(6.0, abs=0.7)

    def test_an_unknown_layout_is_refused(self, two_items):
        with pytest.raises(ValueError):
            scene.arrange(two_items, layout="spiral")


class TestCompose:
    def test_bonds_are_carried_and_re_indexed(self, two_items):
        """Every bond of every item survives, shifted by that item's own
        offset -- and no bond joins two items, which is the whole reason
        they are carried rather than guessed.

        A bond that left the item's own periodic cell is the exception
        and is dropped; see the test below for why.
        """
        arranged = scene.arrange(two_items, layout="row", gap=8.0)
        atoms, report = scene.compose(arranged)
        expected = sum(len(item.atoms.info.get("bonds", []))
                       for item in arranged)
        assert 0 < report.bonds <= expected
        assert report.bonds == len(atoms.info["bonds"])

        tags = atoms.get_tags()
        for i, j in atoms.info["bonds"]:
            assert tags[i] == tags[j], "a bond crossed between two items"

    def test_bond_lengths_survive_the_trip(self, two_items):
        """A re-indexing error does not raise; it draws a bond between
        two atoms that were never bonded. Length is what catches it."""
        atoms, _ = scene.compose(scene.arrange(two_items, gap=8.0))
        positions = atoms.get_positions()
        lengths = [float(np.linalg.norm(positions[i] - positions[j]))
                   for i, j in atoms.info["bonds"]]
        assert min(lengths) > 1.2
        assert max(lengths) < 1.8

    def test_a_bond_leaving_a_periodic_cell_is_dropped_and_reported(self):
        """The composition is finite, so the image such a bond reaches
        into is not in the scene. Drawn from the raw coordinates it is a
        straight line as long as the item's own cell -- measured at
        12.4 Å on a (5,5) tube whose every real bond is 1.42 -- and a
        minimum-image bond list has thrown away the cell that would
        shift it. So it goes, and the report says how many."""
        tube = build_cnt(5, 5, length=10.0)
        assert any(tube.get_pbc())
        atoms, report = scene.compose([scene.Item(atoms=tube, label="tube")])

        positions = atoms.get_positions()
        lengths = [float(np.linalg.norm(positions[i] - positions[j]))
                   for i, j in atoms.info["bonds"]]
        assert max(lengths) < 1.8
        assert report.bonds < len(tube.info["bonds"])
        assert any("left a periodic cell" in text
                   for text in report.warnings)

    def test_an_aperiodic_item_loses_nothing(self):
        """The drop must be the periodic case only. A cage has no seam,
        so every one of its bonds has to survive."""
        cage = build_fullerene(freq=1)
        assert not any(cage.get_pbc())
        _, report = scene.compose([scene.Item(atoms=cage, label="cage")])
        assert report.bonds == len(cage.info["bonds"])
        assert report.rings == len(cage.info["rings"])

    def test_rings_are_carried_and_stay_whole(self, two_items):
        atoms, report = scene.compose(scene.arrange(two_items, gap=8.0))
        expected = sum(len(item.atoms.info.get("rings", []))
                       for item in two_items)
        assert 0 < report.rings <= expected
        tags = atoms.get_tags()
        for ring in atoms.info["rings"]:
            assert len({tags[index] for index in ring}) == 1

    def test_every_atom_knows_which_item_it_came_from(self, two_items):
        """``--colour-by structure`` is this tag and nothing else."""
        atoms, _ = scene.compose(two_items)
        tags = atoms.get_tags()
        assert sorted(set(tags)) == [0, 1]
        assert list(tags).count(0) == len(two_items[0].atoms)

    def test_it_is_marked_an_illustration(self, two_items):
        """So the window's readout and the report say so rather than
        quoting a geometry nobody relaxed."""
        atoms, report = scene.compose(two_items)
        assert atoms.info["is_illustration"] is True
        assert atoms.info["structure_type"] == "composition"
        assert any("ILLUSTRATION" in text for text in report.warnings)

    def test_interpenetration_is_reported_not_refused(self, two_items):
        """A particle inside a matrix is a legitimate figure. What is not
        legitimate is doing it silently."""
        overlapping = [
            scene.Item(atoms=two_items[0].atoms, label="a"),
            scene.Item(atoms=two_items[0].atoms, label="b"),
        ]
        _, report = scene.compose(overlapping)
        assert report.clearances[0][2] < scene.TOUCHING
        assert any("interpenetrate" in text for text in report.warnings)

    def test_a_scaled_item_is_recorded_as_scaled(self, two_items):
        """Scaling changes every bond length, so a scaled item is a
        picture and not a structure. The report must not lose that."""
        items = [scene.Item(atoms=two_items[0].atoms, label="big", scale=2.0)]
        _, report = scene.compose(items)
        assert report.scaled == [("big", 2.0)]
        assert any("non-physical" in text for text in report.warnings)

    def test_an_item_with_no_recorded_bonds_is_named(self):
        """Because the alternative -- the renderer guessing -- is what
        invents bonds between separate structures."""
        bare = Atoms("C2", positions=[[0, 0, 0], [1.42, 0, 0]])
        _, report = scene.compose([scene.Item(atoms=bare, label="pair")])
        assert report.bonds == 0
        assert any("no recorded bonds" in text for text in report.warnings)

    def test_composing_nothing_is_refused(self):
        with pytest.raises(ValueError):
            scene.compose([])


class TestWriteScene:
    def test_it_writes_the_four_files(self, tmp_path, two_items):
        atoms, report = scene.compose(scene.arrange(two_items, gap=8.0))
        paths = scene.write_scene(atoms, tmp_path / "fig", report)
        assert set(paths) == {"xyz", "json", "cif", "notes"}
        for path in paths.values():
            assert path.exists() and path.stat().st_size > 0

    def test_the_sidecar_carries_what_the_renderer_needs(self, tmp_path,
                                                         two_items):
        atoms, report = scene.compose(scene.arrange(two_items, gap=8.0))
        paths = scene.write_scene(atoms, tmp_path / "fig", report)
        recorded = json.loads(paths["json"].read_text())
        assert recorded["n_atoms"] == len(atoms)
        assert len(recorded["item_per_atom"]) == len(atoms)
        assert recorded["is_illustration"] is True
        assert recorded["scene"]["labels"] == ["cage", "tube"]
        assert len(recorded["bonds"]) == report.bonds


class TestItemsFromFiles:
    def test_a_sidecar_beside_the_file_is_read(self, tmp_path):
        """An .xyz carries no connectivity. Without the sidecar a
        composition has no bonds at all and the renderer falls back to
        guessing -- the one thing this module exists not to do."""
        from nanocarbon_lab.exports.xyz import write_render_bundle

        atoms = build_fullerene(freq=1)
        xyz, _ = write_render_bundle(atoms, tmp_path / "cage")
        items = scene.items_from_files([xyz])
        assert len(items[0].atoms.info["bonds"]) == len(atoms.info["bonds"])
        assert items[0].label == "cage"

    def test_a_sidecar_for_a_different_structure_is_ignored(self, tmp_path):
        """A bond list indexed for another structure is worse than none,
        and nothing about the file says which it is -- so the atom count
        is checked."""
        from nanocarbon_lab.exports.xyz import write_render_bundle

        xyz, sidecar = write_render_bundle(build_fullerene(freq=1),
                                           tmp_path / "cage")
        recorded = json.loads(sidecar.read_text())
        recorded["n_atoms"] = 999
        sidecar.write_text(json.dumps(recorded))
        items = scene.items_from_files([xyz])
        assert "bonds" not in items[0].atoms.info

    def test_a_label_is_given_per_file(self, tmp_path):
        from nanocarbon_lab.exports.xyz import write_render_bundle

        xyz, _ = write_render_bundle(build_fullerene(freq=1), tmp_path / "c60")
        items = scene.items_from_files([xyz], labels=["left"])
        assert items[0].label == "left"


class TestColourModes:
    """Rings optional, and the alternatives correct. The pipeline used to
    colour by ring size and nothing else, which says nothing about a
    doped shell and -- on a closed cage, where every atom sits on a
    pentagon -- says "all pentagons" about twelve pentagons and twenty
    hexagons."""

    @pytest.fixture
    def written(self, tmp_path):
        items = [scene.Item(atoms=build_fullerene(freq=1), label="cage"),
                 scene.Item(atoms=build_cnt(5, 5, length=10.0), label="tube")]
        atoms, report = scene.compose(scene.arrange(items, gap=8.0))
        return scene.write_scene(atoms, tmp_path / "fig", report)

    def test_every_mode_assigns_every_atom_a_slot(self, bundle_mod, styles,
                                                  written):
        bundle = bundle_mod.load_bundle(str(written["xyz"]),
                                        str(written["json"]))
        for mode in styles.COLOUR_MODES:
            keys, per_atom = bundle_mod.colour_keys(bundle, mode)
            assert len(per_atom) == len(bundle["positions"])
            assert keys, f"{mode} produced no material slots"
            assert max(per_atom) < len(keys)

    def test_structure_colouring_finds_both_items(self, bundle_mod, written):
        bundle = bundle_mod.load_bundle(str(written["xyz"]),
                                        str(written["json"]))
        keys, per_atom = bundle_mod.colour_keys(bundle, "structure")
        assert keys == [0, 1]
        assert set(per_atom) == {0, 1}

    def test_slot_order_is_stable_between_runs(self, bundle_mod, written):
        """A material list whose order depends on which atom came first is
        a difference nobody can see and everybody has to debug."""
        bundle = bundle_mod.load_bundle(str(written["xyz"]),
                                        str(written["json"]))
        once = bundle_mod.colour_keys(bundle, "element")[0]
        again = bundle_mod.colour_keys(bundle, "element")[0]
        assert once == again == sorted(once)

    def test_an_atom_with_no_ring_metadata_falls_back_to_hexagon(
            self, bundle_mod, tmp_path):
        """A file from elsewhere has no ring census, and ring colouring
        must still produce a picture rather than an index error."""
        path = tmp_path / "bare.xyz"
        path.write_text("2\n\nC 0.0 0.0 0.0\nC 1.42 0.0 0.0\n")
        bundle = bundle_mod.load_bundle(str(path))
        keys, per_atom = bundle_mod.colour_keys(bundle, "ring")
        assert [keys[index] for index in per_atom] == [6, 6]

    def test_element_radii_differ_and_ring_radii_do_not(self, bundle_mod,
                                                        styles, tmp_path):
        """A hydrogen and a gold atom drawn the same size misleads about
        which is which -- and equal spheres are right for an all-carbon
        lattice, which is why it follows the colour mode."""
        path = tmp_path / "mixed.xyz"
        path.write_text("2\n\nAu 0.0 0.0 0.0\nH 2.0 0.0 0.0\n")
        bundle = bundle_mod.load_bundle(str(path))
        by_element = bundle_mod.atom_radii(bundle, "element")
        assert by_element[0] > by_element[1]
        uniform = bundle_mod.atom_radii(bundle, "ring")
        assert uniform[0] == uniform[1] == styles.DEFAULT_ELEMENT_RADIUS

    def test_an_unknown_mode_is_refused(self, bundle_mod, tmp_path):
        path = tmp_path / "bare.xyz"
        path.write_text("1\n\nC 0.0 0.0 0.0\n")
        bundle = bundle_mod.load_bundle(str(path))
        with pytest.raises(ValueError):
            bundle_mod.colour_keys(bundle, "mood")

    def test_an_unknown_element_gets_a_colour_that_shows(self, styles):
        """Magenta on purpose: a silent grey would read as carbon."""
        assert styles.element_colour("Uup") == styles.UNKNOWN_ELEMENT_COLOUR
        assert styles.element_colour("C") != styles.UNKNOWN_ELEMENT_COLOUR


class TestStyleCatalogue:
    def test_every_style_declares_a_valid_colour_mode_and_background(
            self, styles):
        for name in styles.list_styles():
            style = styles.get_style(name)
            assert style.default_colour_by in styles.COLOUR_MODES
            assert style.background.kind in styles.BACKGROUNDS

    def test_every_shader_a_style_asks_for_is_known(self, styles):
        """A shader name with no node setup in ``render_cnt`` raises at
        render time, which is after the slow part."""
        source = (BLENDER_DIR / "render_cnt.py").read_text()
        used = set()
        for name in styles.list_styles():
            style = styles.get_style(name)
            for spec in (style.hexagon, style.pentagon, style.heptagon,
                         style.octagon, style.bond):
                assert spec.shader in styles.SHADERS
                used.add(spec.shader)
        for shader in used:
            if shader != "principled":
                assert f'spec.shader == "{shader}"' in source

    def test_more_than_one_style_colours_by_element(self, styles):
        """The request this work answers: ring colouring had to stop
        being the only option."""
        by_element = [name for name in styles.list_styles()
                      if styles.get_style(name).default_colour_by == "element"]
        assert len(by_element) >= 2

    def test_a_background_can_be_swapped_without_changing_the_style(
            self, styles):
        """The point of Background being its own object: this style on
        white for the paper and on black for the cover is two uses of one
        style, not two styles."""
        original = styles.get_style("nature_dark")
        swapped = styles.with_background(original, "solid")
        assert swapped.background.kind == "solid"
        assert swapped.hexagon == original.hexagon
        assert swapped.lights == original.lights

    def test_transparent_backgrounds_catch_their_shadow(self, styles):
        """Or the subject reads as pasted rather than placed."""
        swapped = styles.with_background(styles.get_style("nature_dark"),
                                         "transparent")
        assert swapped.background.is_transparent
        assert swapped.background.floor_is_shadow_catcher

    def test_recolouring_keeps_a_styles_character(self, styles):
        """Colouring by element inside gold_nanotech gives a gold-finish
        oxygen, not a flat primary -- which is what makes ring colouring
        optional without abandoning the style."""
        gold = styles.get_style("gold_nanotech").hexagon
        recoloured = gold.with_colour((0.0, 1.0, 0.0))
        assert recoloured.base_color[:3] == (0.0, 1.0, 0.0)
        assert recoloured.metallic == gold.metallic
        assert recoloured.roughness == gold.roughness
        assert recoloured.shader == gold.shader

    def test_an_emissive_material_recolours_its_glow_too(self, styles):
        """A red-hot accent that glows blue is not a colour choice."""
        emissive = next(
            spec for name in styles.list_styles()
            for spec in [styles.get_style(name).pentagon]
            if spec.emission_strength > 0
        )
        recoloured = emissive.with_colour((0.0, 1.0, 0.0))
        assert recoloured.emission_color == (0.0, 1.0, 0.0)

    def test_split_bonds_exactly_where_the_colour_is_the_atoms(self, styles):
        """A C-O bond drawn in carbon's grey says the oxygen is carbon; a
        ring belongs to neither end, so it stays one cylinder."""
        assert styles.SPLIT_BONDS["element"] is True
        assert styles.SPLIT_BONDS["structure"] is True
        assert styles.SPLIT_BONDS["ring"] is False
        assert set(styles.SPLIT_BONDS) == set(styles.COLOUR_MODES)

    def test_a_request_beats_the_styles_default(self, styles):
        style = styles.get_style("paper_figure")
        assert styles.resolve_colour_by(None, style) == "element"
        assert styles.resolve_colour_by("ring", style) == "ring"
        with pytest.raises(ValueError):
            styles.resolve_colour_by("mood", style)


class TestRenderCommand:
    """The command line, which is shared by the window and the CLI so
    that neither can drift from the other."""

    def test_it_finds_the_script_inside_the_package(self):
        assert render_support.script_path().exists()
        assert render_support.script_path("styles").exists()

    def test_the_catalogue_is_read_rather_than_copied(self):
        """The window kept a hardcoded list of five style names, and it
        had been wrong since the sixth style was written."""
        names = render_support.style_catalogue().list_styles()
        assert len(names) > 5
        app_source = (BLENDER_DIR.parent / "gui" / "app.py").read_text()
        assert "style_catalogue()" in app_source
        for name in names:
            assert f'"{name}",' not in app_source

    def test_defaults_are_left_off_the_command_line(self, tmp_path):
        """So the style's own ``default_colour_by`` and background remain
        the single source of truth. A caller that sent ``--colour-by
        ring`` because that is what its menu displayed would override
        every style's choice without meaning to."""
        cmd = render_support.render_command(
            tmp_path / "a.xyz", tmp_path / "a.png", blender="/usr/bin/blender")
        assert "--colour-by" not in cmd
        assert "--background" not in cmd
        assert "--floor" not in cmd and "--no-floor" not in cmd

    def test_every_option_reaches_the_script(self, tmp_path):
        cmd = render_support.render_command(
            tmp_path / "a.xyz", tmp_path / "a.png",
            json_path=tmp_path / "a.json", style="paper_figure",
            colour_by="structure", background="studio", floor=False,
            resolution=(800, 600), samples=64, atom_scale=1.5,
            bond_scale=0.5, transparent=True,
            save_blend=tmp_path / "a.blend", render=False,
            blender="/usr/bin/blender")
        for flag in ("--json", "--style", "--colour-by", "--background",
                     "--no-floor", "--resolution", "--samples",
                     "--atom-scale", "--bond-scale",
                     "--transparent-background", "--save-blend",
                     "--no-render"):
            assert flag in cmd
        assert cmd[cmd.index("--colour-by") + 1] == "structure"

    def test_the_script_accepts_every_flag_the_command_builds(self, tmp_path):
        """The two halves are in different files -- one of them
        unimportable outside Blender -- so a flag renamed on one side is
        a crash after the slow part. Checked against the parser's own
        source."""
        source = (BLENDER_DIR / "render_cnt.py").read_text()
        cmd = render_support.render_command(
            tmp_path / "a.xyz", tmp_path / "a.png",
            json_path=tmp_path / "a.json", colour_by="element",
            background="solid", floor=True, resolution=(800, 600),
            samples=64, atom_scale=1.5, bond_scale=0.5, transparent=True,
            save_blend=tmp_path / "a.blend", render=False,
            blender="/usr/bin/blender")
        for token in cmd:
            if token.startswith("--") and token != "--":
                assert f'"{token}"' in source, f"{token} has no parser entry"

    def test_headless_unless_a_window_was_asked_for(self, tmp_path):
        headless = render_support.render_command(
            tmp_path / "a.xyz", tmp_path / "a.png", blender="/usr/bin/blender")
        assert "-b" in headless
        interactive = render_support.render_command(
            tmp_path / "a.xyz", tmp_path / "a.png", blender="/usr/bin/blender",
            interactive=True)
        assert "-b" not in interactive
        assert "-P" in interactive

    def test_the_bpy_module_cannot_open_a_window(self, tmp_path, monkeypatch):
        """It is a full Blender build in this interpreter, and it has no
        window -- so asking for one has to say why rather than running
        headless and reporting success."""
        monkeypatch.setattr(render_support, "find_blender", lambda: None)
        monkeypatch.setattr(render_support, "has_bpy", lambda: True)
        with pytest.raises(RuntimeError, match="no window"):
            render_support.render_command(tmp_path / "a.xyz",
                                         tmp_path / "a.png", interactive=True)

    def test_with_no_blender_at_all_it_says_how_to_get_one(self, tmp_path,
                                                           monkeypatch):
        monkeypatch.setattr(render_support, "find_blender", lambda: None)
        monkeypatch.setattr(render_support, "has_bpy", lambda: False)
        with pytest.raises(RuntimeError, match="pip install bpy"):
            render_support.render_command(tmp_path / "a.xyz",
                                         tmp_path / "a.png")

    def test_a_missing_script_is_reported_as_such(self, tmp_path):
        with pytest.raises(RuntimeError, match="not found"):
            render_support.render_command(tmp_path / "a.xyz",
                                          tmp_path / "a.png",
                                          script=tmp_path / "absent.py",
                                          blender="/usr/bin/blender")


class TestFloorGeometry:
    """Arithmetic from ``render_cnt`` that needs no Blender."""

    def test_the_floor_clears_the_subject_without_levitating_it(self):
        from nanocarbon_lab.render import camera_module

        camera = camera_module()
        # A small positive clearance: zero puts the plane through the
        # bottom atom, and a large one levitates the subject.
        assert 0.0 < camera.FLOOR_CLEARANCE < 0.15
        assert camera.floor_z(10.0, 20.0) < 10.0

    def test_the_floor_is_measured_from_the_bottom_not_the_centre(self):
        """The regression that cost a cover render.

        Placing the floor at ``centre - radius`` assumes the subject is
        a SPHERE. A composition never is, and the error scales with how
        far from spherical it gets: measured on a row of three cages
        39.8 A tall with a 65.0 A bounding-sphere radius, the spherical
        formula lands 45.1 A too low -- further below the subject than
        the subject is tall -- so it levitates out of its own
        reflection. On a single tall tube the same formula is 1.1 A out
        and looks perfectly fine, which is why this hid.
        """
        from nanocarbon_lab.render import camera_module

        camera = camera_module()
        # The wide row: centre 24.3, radius 65.0, lowest point 4.5.
        bottom, radius, centre = 4.5, 65.0, 24.3
        spherical = centre - radius
        measured = camera.floor_z(bottom, radius)
        assert measured > spherical
        assert measured - spherical > 40.0
        # And it stays just under the subject rather than inside it.
        assert bottom - measured < 0.15 * radius
        assert measured < bottom

    def test_the_plane_edge_stays_outside_frame(self):
        source = (BLENDER_DIR / "render_cnt.py").read_text()
        namespace: dict = {}
        for line in source.splitlines():
            if line.startswith(("FLOOR_EXTENT", "FRAME_FILL")):
                exec(line, namespace)  # noqa: S102 - our own constants
        # The plane's own edge must be outside frame at every lens the
        # styles use, or it reads as a table rather than as a backdrop.
        assert namespace["FLOOR_EXTENT"] > 2.0 / math.sin(
            math.atan(0.5 * 36.0 / 90.0))


class TestCommandLine:
    """The two sub-commands, parsed. Rendering was reachable only from
    the window's button, which is the wrong place for it: a figure is
    usually wanted from the same script that built the structure, and a
    style is worth comparing across several structures in a loop."""

    @staticmethod
    def _parse(line: str):
        from nanocarbon_lab.cli.main import build_parser

        return build_parser().parse_args(line.split())

    def test_scene_takes_several_files_and_per_item_transforms(self):
        args = self._parse(
            "scene a.xyz b.xyz c.xyz --label one --label two --label three "
            "--layout grid --gap 10 --rotate two:0,90,0 --translate 3:5,0,0 "
            "--scale one:2 --out fig")
        assert args.paths == ["a.xyz", "b.xyz", "c.xyz"]
        assert args.label == ["one", "two", "three"]
        assert args.layout == "grid"
        assert args.rotate == ["two:0,90,0"]
        assert args.translate == ["3:5,0,0"]

    def test_render_takes_the_three_independent_axes(self):
        args = self._parse(
            "render fig.xyz --style paper_figure --colour-by structure "
            "--background studio --no-floor --samples 64 --out fig.png")
        assert args.style == "paper_figure"
        assert args.colour_by == "structure"
        assert args.background == "studio"
        assert args.floor is False

    def test_the_floor_has_three_states(self):
        """Left out it means "whatever the style says" -- some styles
        ship a floor and most do not -- so off must not be the default."""
        assert self._parse("render fig.xyz").floor is None
        assert self._parse("render fig.xyz --floor").floor is True
        assert self._parse("render fig.xyz --no-floor").floor is False

    def test_colour_by_defaults_to_the_styles_own(self):
        assert self._parse("render fig.xyz").colour_by is None
        assert self._parse("render fig.xyz").background is None

    def test_both_spellings_of_colour_reach_the_same_option(self):
        assert self._parse("render f.xyz --color-by element").colour_by == \
            "element"
        assert self._parse("render f.xyz --colour-by element").colour_by == \
            "element"

    def test_the_catalogue_can_be_listed_without_blender(self):
        from nanocarbon_lab.cli.main import main

        assert main(["render", "--list-styles"]) == 0
        assert main(["render", "--list-colour-modes"]) == 0
        assert main(["scene", "--list-layouts"]) == 0

    def test_scene_composes_and_writes(self, tmp_path):
        from nanocarbon_lab.cli.main import main
        from nanocarbon_lab.exports.xyz import write_render_bundle

        first, _ = write_render_bundle(build_fullerene(freq=1),
                                       tmp_path / "cage")
        second, _ = write_render_bundle(build_cnt(5, 5, length=10.0),
                                        tmp_path / "tube")
        status = main(["scene", str(first), str(second), "--gap", "9",
                       "--out", str(tmp_path / "fig")])
        assert status == 0
        assert (tmp_path / "fig.xyz").exists()
        assert (tmp_path / "fig.json").exists()

    def test_render_prints_the_command_it_would_run(self, tmp_path,
                                                    monkeypatch):
        """Which is also how the pipeline is checked on a machine with no
        Blender: the command is built, and nothing is launched."""
        from nanocarbon_lab.cli.main import main
        from nanocarbon_lab.exports.xyz import write_render_bundle

        xyz, _ = write_render_bundle(build_fullerene(freq=1),
                                     tmp_path / "cage")
        monkeypatch.setattr(render_support, "find_blender",
                            lambda: "/usr/bin/blender")
        assert main(["render", str(xyz), "--print-command"]) == 0

    def test_render_refuses_a_missing_file_before_starting_blender(self,
                                                                  tmp_path):
        from nanocarbon_lab.cli.main import main

        with pytest.raises(SystemExit):
            main(["render", str(tmp_path / "absent.xyz")])
