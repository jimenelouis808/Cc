"""Drawing a phase on the pattern without claiming it is there.

Identification answers "which of these does the program accept", and
that is a decision with a bar: the phase has to match positions, account
for enough of its own calculated intensity, and not be contradicted by a
reflection it should have shown. The bar is right for a claim the
program makes, and useless for the question left when the search comes
back with nothing -- "would THIS one line up?" -- which needs no verdict
at all, only the lines drawn on the same axis.

The pinning that already existed narrows the SEARCH, which is a
different thing: it still ends in a verdict, and a phase the search
would refuse stays invisible.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from ramancarbon.examples.demo_data import make_xrd_demo
from ramancarbon.gui.plots_xrd import plot_manual_sticks
from ramancarbon.gui.theme import PALETTES
from ramancarbon.gui.xrd_state import XRDSession


@pytest.fixture()
def session():
    state = XRDSession()
    state.add_pattern(make_xrd_demo(seed=3))
    return state


class TestChoosingPhasesByHand:
    def test_a_library_phase_can_be_added_and_taken_away(self, session):
        name = session.library()[0].crystal.name
        assert session.add_overlay(name)
        assert session.overlay_phases == [name]
        assert [c.name for c in session.overlay_crystals()] == [name]
        assert session.remove_overlay(name)
        assert session.overlay_phases == []

    def test_adding_the_same_one_twice_is_not_two_of_it(self, session):
        name = session.library()[0].crystal.name
        session.add_overlay(name)
        session.add_overlay(name)
        assert session.overlay_phases == [name]

    def test_a_phase_that_is_not_in_the_library_is_refused(self, session):
        assert not session.add_overlay("no_existe")
        assert not session.overlay_phases
        assert any("no está en la biblioteca" in text
                   for _, text in session.messages)

    def test_clearing_takes_them_all(self, session):
        for entry in session.library()[:3]:
            session.add_overlay(entry.crystal.name)
        assert len(session.overlay_phases) == 3
        session.clear_overlays()
        assert session.overlay_phases == []

    def test_it_is_separate_from_pinning_the_search(self, session):
        # Pinning narrows what the search will consider and still ends in
        # a verdict; this narrows nothing and asserts nothing.
        name = session.library()[0].crystal.name
        session.add_overlay(name)
        assert session.selected_phases == []
        session.selected_phases = ["otra"]
        assert session.overlay_phases == [name]


class TestTheOverlayDraws:
    def test_it_draws_without_any_identification_having_run(self, session):
        # The case it exists for: the search proposed nothing, so there
        # is no result to plot, and the question is still open.
        item = session.item
        assert item.result is None
        crystals = [session.library()[0].crystal]
        figure = Figure()
        ax = figure.add_subplot(111)
        plot_manual_sticks(ax, item.pattern, crystals, PALETTES["oscuro"])
        assert ax.collections or ax.lines
        assert "sin veredicto" in ax.get_title()

    def test_the_measured_curve_and_one_legend_entry_per_phase(self, session):
        item = session.item
        crystals = [e.crystal for e in session.library()[:2]]
        figure = Figure()
        ax = figure.add_subplot(111)
        plot_manual_sticks(ax, item.pattern, crystals, PALETTES["claro"])
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "medido (normalizado)" in labels
        for crystal in crystals:
            assert crystal.name in labels

    def test_the_axis_is_the_measured_range(self, session):
        item = session.item
        figure = Figure()
        ax = figure.add_subplot(111)
        plot_manual_sticks(ax, item.pattern,
                           [session.library()[0].crystal],
                           PALETTES["oscuro"])
        assert ax.get_xlim() == pytest.approx(item.pattern.range)

    def test_no_phases_still_draws_the_data(self, session):
        figure = Figure()
        ax = figure.add_subplot(111)
        plot_manual_sticks(ax, session.item.pattern, [], PALETTES["oscuro"])
        assert ax.lines
