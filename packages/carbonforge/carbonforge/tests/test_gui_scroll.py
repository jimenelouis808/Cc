"""Tests for the scrolling parameter column.

The bug these exist for was not a crash: the adsorption tab's three
numbered steps were taller than the window, its column did not scroll,
and steps 2 and 3 were unreachable with nothing on screen saying so.
So the assertions are about **reachability** -- that the column is
wired to scroll at all, and that the tab uses it -- rather than about
any value a user reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from carbonforge.gui.scroll import (
    WHEEL_STEP,
    _wheel_delta,
    scrollable_column,
)


class _Stub:
    """A Tk widget that records what was done to it.

    Children register with their parent, which is how a test finds the
    canvas the helper built without the helper having to return it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.children: list["_Stub"] = []
        self.packed: dict[str, Any] | None = None
        self.bindings: dict[str, Any] = {}
        self.all_bindings: dict[str, Any] = {}
        self.unbound: list[str] = []
        self.items: list[dict[str, Any]] = []
        self.scrolled: list[tuple[int, str]] = []
        parent = args[0] if args else kwargs.get("master")
        if isinstance(parent, _Stub):
            parent.children.append(self)

    def pack(self, **kwargs: Any) -> None:
        self.packed = kwargs

    def configure(self, **kwargs: Any) -> None:
        self.kwargs.update(kwargs)

    def bind(self, sequence: str, callback: Any) -> None:
        self.bindings[sequence] = callback

    def bind_all(self, sequence: str, callback: Any) -> None:
        self.all_bindings[sequence] = callback

    def unbind_all(self, sequence: str) -> None:
        self.unbound.append(sequence)

    def create_window(self, _at: Any, **kwargs: Any) -> int:
        self.items.append(dict(kwargs))
        return len(self.items) - 1

    def itemconfigure(self, index: int, **kwargs: Any) -> None:
        self.items[index].update(kwargs)

    def yview(self, *_a: Any) -> None:
        return None

    def set(self, *_a: Any) -> None:
        """A scrollbar's thumb setter, which the canvas is handed."""
        return None

    def yview_scroll(self, amount: int, what: str) -> None:
        self.scrolled.append((amount, what))

    def bbox(self, _what: str) -> tuple[int, int, int, int]:
        return (0, 0, 360, 1400)


class _Tk:
    Canvas = _Stub


class _Ttk:
    Scrollbar = _Stub
    Frame = _Stub


class _Event:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


@pytest.fixture
def built() -> tuple[_Stub, _Stub, _Stub, _Stub]:
    """``(parent, canvas, scrollbar, frame)`` for one built column."""
    parent = _Stub()
    frame = scrollable_column(_Tk, _Ttk, parent, width=360)
    canvas, scrollbar = parent.children
    assert canvas.items, "the canvas is the stub holding the inner frame"
    return parent, canvas, scrollbar, frame


def test_the_canvas_and_the_scrollbar_are_wired_both_ways(built) -> None:
    """One half of the wiring alone leaves the column stuck."""
    _parent, canvas, scrollbar, frame = built
    assert scrollbar.kwargs["command"] == canvas.yview
    assert canvas.kwargs["yscrollcommand"] == scrollbar.set
    assert canvas.items[0]["window"] is frame
    assert canvas.kwargs["width"] == 360
    assert scrollbar.packed == {"side": "right", "fill": "y"}


def test_the_contents_set_the_scrollregion(built) -> None:
    """A <Configure> on the contents is what makes the region known.

    This is the assertion that would have caught the original bug: with
    no scrollregion the canvas believes its contents end at its own
    edge, and everything below is unreachable.
    """
    _parent, canvas, _scrollbar, frame = built
    assert "scrollregion" not in canvas.kwargs
    frame.bindings["<Configure>"](None)
    assert canvas.kwargs["scrollregion"] == (0, 0, 360, 1400)


def test_the_wheel_binds_on_enter_and_unbinds_on_leave(built) -> None:
    """Bound globally and left bound, the wheel would hijack the page."""
    _parent, canvas, _scrollbar, _frame = built

    canvas.bindings["<Enter>"](None)
    # X11 sends buttons 4 and 5; everything else sends <MouseWheel>.
    for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        assert sequence in canvas.all_bindings

    canvas.bindings["<Leave>"](None)
    assert set(canvas.unbound) == {"<MouseWheel>", "<Button-4>", "<Button-5>"}


def test_the_wheel_actually_scrolls_both_ways(built) -> None:
    _parent, canvas, _scrollbar, _frame = built
    canvas.bindings["<Enter>"](None)

    canvas.all_bindings["<Button-5>"](None)
    canvas.all_bindings["<Button-4>"](None)
    assert canvas.scrolled == [(WHEEL_STEP, "units"), (-WHEEL_STEP, "units")]

    canvas.all_bindings["<MouseWheel>"](_Event(delta=-120))
    assert canvas.scrolled[-1] == (WHEEL_STEP, "units")


def test_only_the_wheels_sign_is_read() -> None:
    """Windows reports +-120 and macOS small integers.

    Scrolling by `delta` itself moves a mac trackpad a pixel and a
    Windows wheel a whole page, so the magnitude is deliberately
    discarded.
    """
    assert _wheel_delta(_Event(delta=120)) == -WHEEL_STEP
    assert _wheel_delta(_Event(delta=1)) == -WHEEL_STEP
    assert _wheel_delta(_Event(delta=-120)) == WHEEL_STEP
    assert _wheel_delta(_Event(delta=-1)) == WHEEL_STEP


def test_the_inner_frame_follows_the_canvas_width(built) -> None:
    """Otherwise labels wrap against a width nobody set."""
    _parent, canvas, _scrollbar, _frame = built
    canvas.bindings["<Configure>"](_Event(width=412))
    assert canvas.items[0]["width"] == 412


def test_the_adsorption_tab_uses_it() -> None:
    """The regression guard, read off the tab's own source.

    A fixed-width frame with its propagation off is exactly what
    clipped steps 2 and 3, and it reads as ordinary layout code, so the
    next person to touch this column needs the reason in a failing test
    rather than in a comment.
    """
    import carbonforge.gui.tabs.adsorption as tab

    source = Path(tab.__file__).read_text(encoding="utf-8")
    assert "scrollable_column(tk, ttk, column" in source
    assert "left.pack_propagate(False)" not in source
