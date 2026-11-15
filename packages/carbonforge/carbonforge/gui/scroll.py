"""A parameter column that scrolls, with the wheel as well as the bar.

Three tabs here grew the same shape independently -- a tall stack of
option frames down the left of the page -- and each rolled its own
canvas-plus-scrollbar around it. The adsorption tab did not, and its
column is a numbered three-step workflow: with the steps taller than
the window, steps 2 and 3 were simply **unreachable**. A fixed-width
frame with ``pack_propagate(False)`` clips without a hint that there is
anything below, so nothing on screen said the page was cut.

`scrollable_column` is that shape once, so a new tab does not have to
remember it. It also binds the wheel, which none of the hand-rolled
copies did: a column you can only move by dragging a 4-pixel bar is
reachable in principle and not in practice.
"""

from __future__ import annotations

from typing import Any

#: One wheel notch, in canvas scroll units.
WHEEL_STEP = 3


def _scroll_by(canvas: Any, step: int) -> None:
    """Scroll `canvas` vertically, stopping at the ends."""
    canvas.yview_scroll(step, "units")


def _wheel_delta(event: Any) -> int:
    """Notches to scroll for a ``<MouseWheel>`` event, sign included.

    Windows reports ``delta`` in multiples of 120 and macOS in small
    integers, so the sign is read and the magnitude ignored -- scrolling
    by ``delta`` itself moves a mac trackpad a pixel and a Windows wheel
    a whole page.
    """
    return -WHEEL_STEP if getattr(event, "delta", 0) > 0 else WHEEL_STEP


def scrollable_column(tk: Any, ttk: Any, parent: Any,
                      width: int = 360) -> Any:
    """Wrap `parent` in a vertically scrolling canvas; return the frame.

    Widgets go into the returned frame exactly as they would have gone
    into `parent`, so adopting this is a one-line change at the top of a
    tab builder rather than a rewrite of its layout.

    `width` is the column's width in pixels. The canvas carries it, so
    the caller must NOT also set ``pack_propagate(False)`` on `parent`:
    that pins the outer frame's height too, which is the clipping this
    exists to remove.

    The wheel is bound while the pointer is over the column and
    unbound when it leaves, because ``bind_all`` is global -- left
    bound, the wheel would scroll this column while the user was over
    the plot on the other side of the page.
    """
    canvas = tk.Canvas(parent, highlightthickness=0, width=width)
    scroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    frame = ttk.Frame(canvas)

    window = canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.configure(yscrollcommand=scroll.set)
    canvas.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    def _fit_contents(_event: Any = None) -> None:
        """Let the canvas know how tall its contents became."""
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _fit_width(event: Any) -> None:
        """Keep the inner frame as wide as the canvas.

        Without this the frame takes its requested width, so a long
        label wraps against a width nobody set and the column's
        contents and its scrollbar disagree about where the edge is.
        """
        canvas.itemconfigure(window, width=event.width)

    frame.bind("<Configure>", _fit_contents)
    canvas.bind("<Configure>", _fit_width)

    def _bind_wheel(_event: Any = None) -> None:
        canvas.bind_all("<MouseWheel>",
                        lambda e: _scroll_by(canvas, _wheel_delta(e)))
        # X11 reports the wheel as buttons 4 and 5, not as <MouseWheel>.
        canvas.bind_all("<Button-4>",
                        lambda _e: _scroll_by(canvas, -WHEEL_STEP))
        canvas.bind_all("<Button-5>",
                        lambda _e: _scroll_by(canvas, WHEEL_STEP))

    def _unbind_wheel(_event: Any = None) -> None:
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.unbind_all(sequence)

    canvas.bind("<Enter>", _bind_wheel)
    canvas.bind("<Leave>", _unbind_wheel)

    return frame
