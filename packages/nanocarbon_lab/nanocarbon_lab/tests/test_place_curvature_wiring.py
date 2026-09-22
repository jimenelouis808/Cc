"""Every mode the window offers the placement switch for must accept it.

This is the class of bug that has bitten here twice: a keyword the GUI
passes and the builder does not take is a TypeError raised the moment
someone ticks the box, with nothing drawn. `parameter_names` alone does
not catch it either way -- a builder taking `**kwargs` accepts the
keyword without declaring it, and one declaring it may still not reach
the remesher -- so this calls the builders.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from nanocarbon_lab.jobs import builder_for

APP = Path(__file__).resolve().parents[1] / "gui" / "app.py"


def _modes_the_window_passes_it_for() -> set[str]:
    """Read the GUI's own parameter blocks rather than a second list.

    A hand-kept list would drift from the window, which is exactly the
    failure this is here to stop.
    """
    source = APP.read_text()
    found, mode = set(), None
    for line in source.splitlines():
        match = re.search(r'(?:el)?if mode == "([^"]+)"', line)
        if match:
            mode = match.group(1)
        elif "place_curvature=bool(" in line and mode:
            found.add(mode)
    return found


class TestPlacementSwitchWiring:
    def test_the_window_offers_it_for_something(self):
        assert _modes_the_window_passes_it_for()

    @pytest.mark.parametrize("mode", sorted(_modes_the_window_passes_it_for()))
    def test_the_builder_accepts_the_keyword(self, mode):
        """Accepting it via `**kwargs` counts -- `build_coil` forwards to
        the swept-tube builder, which declares it."""
        signature = inspect.signature(builder_for(mode))
        explicit = "place_curvature" in signature.parameters
        variadic = any(p.kind is inspect.Parameter.VAR_KEYWORD
                       for p in signature.parameters.values())
        assert explicit or variadic, (
            f"the window passes place_curvature to {mode!r}, whose builder "
            f"{builder_for(mode).__name__} does not take it")

    @pytest.mark.parametrize("mode", sorted(_modes_the_window_passes_it_for()))
    def test_the_keyword_reaches_the_remesher(self, mode):
        """Declared-but-ignored is the quieter half of this bug: the box
        ticks, nothing raises, and nothing moves either."""
        seen = {}
        import nanocarbon_lab.builders.remesh as rm

        real = rm.isotropic_remesh

        def spy(*args, **kwargs):
            seen["place_curvature"] = kwargs.get("place_curvature")
            raise _Stop

        rm.isotropic_remesh = spy
        try:
            with pytest.raises((_Stop, Exception)):
                builder_for(mode)(place_curvature=True)
        except _Stop:
            pass
        finally:
            rm.isotropic_remesh = real
        if seen:
            assert seen["place_curvature"] is True, (
                f"{mode} swallowed the flag before the remesher")


class _Stop(BaseException):
    """Stops a builder the moment the remesher is reached."""
