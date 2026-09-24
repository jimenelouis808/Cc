"""Checks on the widget module that do not need a display.

``test_gui.py`` exercises the real widget tree, and skips itself wherever
Tkinter or an X display is missing -- which is most machines that are not
CI. That leaves the largest single module in the package with no coverage
at all in an ordinary checkout, and the two bugs it hides best are exactly
the two a linter cannot see: a ``command=`` naming a method that does not
exist, and a ``self.x`` read but never assigned. Both are attribute
lookups, resolved at click time, in a window nobody opened.

So these read the source rather than importing it, and check it against
itself.
"""

from __future__ import annotations

import ast
import pathlib
import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "gui" / "app.py"

#: Attributes the widget class gets from Tk or from its own construction
#: helpers rather than from a plain assignment.
INHERITED = {"root", "tk", "ttk"}


def source() -> str:
    return SOURCE.read_text(encoding="utf-8")


def test_the_widget_module_parses():
    ast.parse(source())


def test_every_self_attribute_used_is_assigned_somewhere():
    tree = ast.parse(source())
    for node in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        assigned: set[str] = set()
        read: set[str] = set()
        for child in ast.walk(node):
            if not (isinstance(child, ast.Attribute)
                    and isinstance(child.value, ast.Name)
                    and child.value.id == "self"):
                continue
            if isinstance(child.ctx, (ast.Store, ast.Del)):
                assigned.add(child.attr)
            else:
                read.add(child.attr)
        defined = {n.name for n in ast.walk(node)
                   if isinstance(n, ast.FunctionDef)}
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(
                    statement.target, ast.Name):
                defined.add(statement.target.id)
            elif isinstance(statement, ast.Assign):
                defined |= {t.id for t in statement.targets
                            if isinstance(t, ast.Name)}
        missing = read - assigned - defined - INHERITED
        assert not missing, f"{node.name}: read but never assigned: {sorted(missing)}"


def test_every_button_callback_exists():
    text = source()
    tree = ast.parse(text)
    methods = {n.name for c in ast.walk(tree) if isinstance(c, ast.ClassDef)
               for n in ast.walk(c) if isinstance(n, ast.FunctionDef)}
    # ``command=self._canvas.yview`` and friends hand off to a widget, not
    # to a method of ours.
    referenced = {name for name in re.findall(r"command=self\.([\w.]+)", text)
                  if "." not in name}
    missing = referenced - methods
    assert not missing, f"commands with no method: {sorted(missing)}"


def test_the_preview_draws_bonds_through_the_shift_helper():
    """The regression this guards is visual and therefore silent.

    A bond list built under the minimum image convention names the pair
    and not the cell the far atom is in. Drawing it from raw coordinates
    puts a line straight across the structure for every bond that leaves
    through a face -- 131 of them on a gyroid cell, up to 46 Å long. It
    looks like a rendering artefact and it is a missing subtraction.
    """
    text = source()
    assert "_bond_shifts" in text
    assert "var_show_wrapped" in text
    # The old form, which is the bug.
    assert "(pos[a] + shift, pos[b] + shift)" not in text


def test_the_supercell_goes_through_the_helper_that_carries_the_bonds():
    """``Atoms.repeat`` copies ``info`` verbatim, so a supercell built with
    it keeps one cell's bond list, ring list and ring census while having
    eight times the atoms. Nothing about the atom count shows it."""
    text = source()
    assert "supercell(self.atoms, counts)" in text
    assert "self.atoms.repeat(counts)" not in text


def test_the_subdivision_slider_cannot_offer_a_frequency_that_cannot_build():
    """The control offered freq=1, which no capped tube can use.

    Picking it cost a minute and a half of building and produced a message
    about a sweep tearing a wall. A control that offers a value the
    builder refuses is not a control.
    """
    text = source()
    assert "MIN_CAP_FREQ" in text, "the floor is hard-coded again"
    # Not [^)]* -- the label itself contains "(diameter)".
    call = re.search(
        r'self\._param\(\s*box,\s*"Subdivision freq.*?'
        r'command=self\._update_radius_hint\)', text, re.S)
    assert call, "the subdivision control moved"
    snippet = call.group(0)
    assert "MIN_CAP_FREQ, 8" in snippet, (
        f"slider floor is not the builder's: {snippet}")
    assert "hard_lo=MIN_CAP_FREQ" in text, (
        "the entry box still accepts a typed 1, which the slider no longer "
        f"offers: {snippet}")


def test_a_failed_build_is_reported_as_a_failure():
    """The elapsed line is what people read when the progress bar stops,
    and it used to say "took 1:31" whether the structure arrived or the
    builder refused it -- so a failure looked like a success that had not
    refreshed the preview."""
    text = source()
    assert 'failed=kind != "done"' in text, (
        "the poll no longer tells _finish_build that the build failed")
    assert "FAILED after" in text, "the elapsed line no longer marks failures"
    assert "_mark_preview_stale" in text, (
        "nothing labels the previous structure that stays on screen")


def _source() -> str:
    """The GUI module as text. It imports tkinter, which most checkouts
    do not have, so it is read rather than imported."""
    import nanocarbon_lab
    return (pathlib.Path(nanocarbon_lab.__file__).parent
            / "gui" / "app.py").read_text()


def _presets(source: str) -> dict:
    """The PRESETS literal, parsed rather than imported."""
    start = source.index("PRESETS: dict[str, dict[str, object]] = {")
    start = source.index("{", start)
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return ast.literal_eval(source[start:end])


class TestThePresetsPointAtRealThings:
    """A preset is a dict of variable names applied in a loop, so a typo
    in a key is silent: the preset applies, that one field keeps its old
    value, and the structure built is not the one named. The same goes
    for a mode: the window would switch to a panel that does not exist.

    Both are resolved at click time, so nothing else here sees them.
    """

    def test_every_preset_key_is_a_registered_variable(self):
        source = _source()
        registered = set(re.findall(r'self\._var\(\s*"([^"]+)"', source))
        assert registered, "no variables found -- the pattern has drifted"
        for name, preset in _presets(source).items():
            for key in preset:
                if key == "mode_kind":
                    continue
                assert key in registered, (
                    f"preset {name!r} sets {key!r}, which no _var registers")

    def test_every_preset_names_a_real_mode(self):
        from nanocarbon_lab.jobs import HETERO_MODES, MODES, TMD_MODES

        known = set(MODES) | set(TMD_MODES) | set(HETERO_MODES)
        for name, preset in _presets(_source()).items():
            mode = preset.get("mode_kind")
            assert mode in known, f"preset {name!r} names unknown mode {mode!r}"

    def test_no_curved_surface_preset_anneals(self):
        """On a curved surface the 5-7 pairs ARE how the net covers its
        curvature, so annealing them away leaves the survivors carrying
        all of it. Measured on the Y junction, 80 sweeps widen the bond
        spread from 0.0136 to 0.0175 Å. Every preset here carried 80
        until this test existed.
        """
        curved = {"junction", "schwarzite", "network", "supernetwork",
                  "coil (relaxed)", "coil (periodic, DFT)"}
        for name, preset in _presets(_source()).items():
            if preset.get("mode_kind") in curved:
                assert preset.get("anneal", 0) == 0, (
                    f"preset {name!r} anneals a curved surface")


class TestEverySuperlatticeIsReachableFromAPreset:
    """The Net box writes ``sn_graph`` and nothing else.

    ``_update_sn_hint`` is its only listener, so picking a net there
    leaves the cell, the tube radius and the blend at whatever the LAST
    preset put in them. That is not a theory: super-fcc had no preset,
    so the only way to it was the Net box, and choosing it after
    "Super-diamond (tubes at 109.47°)" built it at scale 60 with a 5 Å
    tube -- 32_000 Å² of wall, still going after ninety minutes, where
    its own preset asks for 12_800 and finishes in seven.

    A periodic net without a preset is therefore a net that can only be
    reached with someone else's numbers.
    """

    def _supernetwork_presets(self) -> dict:
        return {name: preset for name, preset in _presets(_source()).items()
                if preset.get("mode_kind") == "supernetwork"}

    def test_every_periodic_net_has_a_preset_of_its_own(self):
        from nanocarbon_lab.builders.supernetwork import SUPERLATTICES

        named = {preset["sn_graph"]
                 for preset in self._supernetwork_presets().values()}
        missing = sorted(set(SUPERLATTICES) - named)
        assert not missing, (
            f"{missing} can only be reached from the Net box, which keeps "
            "the previous preset's cell and tube radius")

    def test_no_preset_asks_for_a_build_of_tens_of_minutes(self):
        from nanocarbon_lab.builders.supernetwork import (
            CAGE_SLOW_AREA,
            SLOW_AREA,
            named_graph,
        )

        for name, preset in self._supernetwork_presets().items():
            scale = float(preset["sn_scale"])
            graph = named_graph(str(preset["sn_graph"]), scale)
            area = graph.wall_area(scale, float(preset["sn_radius"]))
            # A cage of the same area is an order of magnitude cheaper,
            # so the hypercube's 21_802 Å² at 88 s is not the hour that
            # a periodic cell of that size would be.
            limit = SLOW_AREA if graph.periodic else CAGE_SLOW_AREA
            assert area < limit, (
                f"preset {name!r} asks for {area:,.0f} Å² of wall")

    def test_every_preset_leaves_room_for_a_tube_between_the_vertices(self):
        """The refusal the builder raises, checked before the click."""
        from nanocarbon_lab.builders.supernetwork import named_graph

        for name, preset in self._supernetwork_presets().items():
            scale = float(preset["sn_scale"])
            graph = named_graph(str(preset["sn_graph"]), scale)
            eaten = 2.0 * (float(preset["sn_radius"])
                           + float(preset["sn_blend"]))
            shortest = float(graph.strut_lengths(scale).min())
            assert shortest > eaten, (
                f"preset {name!r} would be refused: {shortest:.1f} Å struts "
                f"and {eaten:.1f} Å eaten by the vertices")

    def test_the_hint_reports_the_size_and_not_only_the_shape(self):
        source = _source()
        assert "wall_area(" in source and "build_cost_note(" in source, (
            "the supernetwork hint no longer prices the build, so an "
            "enormous net reads the same as a small one")
