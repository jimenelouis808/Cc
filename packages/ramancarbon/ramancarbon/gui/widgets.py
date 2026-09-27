"""Reusable widgets: cards, scrollable text, tables, labelled fields.

Tkinter is imported inside the functions rather than at module scope so that
importing :mod:`ramancarbon.gui` on a machine without Tk fails with the
package's own explanatory message instead of an ImportError traceback.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence

from .theme import PAD, Palette


def _fit(label, event, wrap: int) -> None:
    """Set a label's wrap width from the width it was just given."""
    target = min(max(int(event.width), MIN_WRAP), max(int(wrap), MIN_WRAP))
    if abs(target - _wraplength(label)) > 2:
        label.configure(wraplength=target)


def _wraplength(label) -> int:
    """A label's current wrap width, as a number.

    ``cget("wraplength")`` returns an empty string on a label that never
    had one set — which is every heading — and ``int("")`` raises. The
    resize callbacks read this value to decide whether anything changed,
    so an unset one has to mean "no wrapping yet", not a traceback in
    every ``<Configure>`` event.
    """
    try:
        return int(label.cget("wraplength") or 0)
    except (TypeError, ValueError):
        return 0


def card(parent, title: Optional[str] = None, subtitle: Optional[str] = None):
    """A bordered surface panel with an optional heading.

    Returns
    -------
    (ttk.Frame, ttk.Frame)
        The outer card and the inner body to put content into. Two frames
        rather than one so the padding is uniform and the caller never has
        to remember it.
    """
    from tkinter import ttk

    outer = ttk.Frame(parent, style="Card.TFrame", padding=PAD["md"])
    heads: list = []
    if title:
        # Headings wrap too. "Número de componentes" is a perfectly
        # ordinary card title and it was given 76 px of the 237 it needs,
        # because the card it heads is in a narrow column -- so the card
        # announced itself as "Número de com".
        head = ttk.Label(outer, text=title, style="Heading.TLabel",
                         justify="left")
        head.pack(anchor="w", fill="x",
                  pady=(0, PAD["xs"] if subtitle else PAD["sm"]))
        heads.append(head)
    if subtitle:
        note = ttk.Label(outer, text=subtitle, style="Muted.TLabel",
                         wraplength=520, justify="left")
        note.pack(anchor="w", fill="x", pady=(0, PAD["sm"]))
        heads.append(note)
    for head in heads:
        def _refit(event, label=head) -> None:
            width = max(int(event.width), MIN_WRAP)
            if abs(width - _wraplength(label)) > 2:
                label.configure(wraplength=width)

        head.bind("<Configure>", _refit, add="+")
    body = ttk.Frame(outer, style="Card.TFrame")
    body.pack(fill="both", expand=True)
    return outer, body


def split_column(parent, weights: Sequence[int] = (3, 2)):
    """A vertical stack of panes with draggable sashes between them.

    Two cards packed with ``fill="both", expand=True`` do NOT share the
    height fairly: the packer gives each its requested size first and
    only then divides what is left, so a 500 px main figure and a 280 px
    strip of companion figures come out as a full-size plot and a sliver
    — which is what every one of these tabs looked like, with the
    companion panels crushed against the bottom edge and no way to see
    them. Worse, the sliver has no minimum, so on a shorter window the
    companions disappear entirely.

    A paned window gives each pane a real share AND lets the user drag
    the boundary, which is the actual answer to "I cannot expand it".

    Returns
    -------
    (ttk.PanedWindow, list[ttk.Frame])
        The container and one frame per weight, already added.
    """
    from tkinter import ttk

    paned = ttk.PanedWindow(parent, orient="vertical")
    panes = []
    for weight in weights:
        frame = ttk.Frame(paned)
        paned.add(frame, weight=int(weight))
        panes.append(frame)
    return paned, panes


def scrollable_column(parent, width: int = 280):
    """A fixed-width column that scrolls when its contents outgrow it.

    Every sidebar in this package was a plain frame with
    ``pack_propagate(False)``, which fixes the width and lets the height
    overflow with nothing to reach it. Measured in a 1480x940 window: the
    electrochemistry section's column asked for 1344 px and was given 835,
    so ``Cargar GCD…``, ``Cargar EIS…``, ``Cargar polarización…``,
    ``Analizar`` and ``Guardar informe…`` were below the fold and simply
    not drawn -- which is why the section looked like it had one import
    button. The XPS column had cards rendered at **one pixel**.

    Returns the frame to put content in, not the outer container: callers
    pack their cards into it exactly as before.

    The wheel is bound on enter and unbound on leave rather than through
    ``bind_all`` permanently, so scrolling over a plot or a table still
    reaches that widget instead of this column.
    """
    import tkinter as tk
    from tkinter import ttk

    outer = ttk.Frame(parent)
    canvas = tk.Canvas(outer, width=width, highlightthickness=0, bd=0)
    scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)

    window = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scroll.set)
    canvas.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    def _resize_region(_event=None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))
        # Hide the scrollbar when everything fits: a bar that never moves
        # reads as a broken one.
        needed = inner.winfo_reqheight() > canvas.winfo_height()
        if needed and not scroll.winfo_ismapped():
            scroll.pack(side="right", fill="y")
        elif not needed and scroll.winfo_ismapped():
            scroll.pack_forget()

    def _resize_inner(event) -> None:
        canvas.itemconfigure(window, width=event.width)
        _resize_region()

    inner.bind("<Configure>", _resize_region)
    canvas.bind("<Configure>", _resize_inner)

    def _wheel(event) -> None:
        if inner.winfo_reqheight() <= canvas.winfo_height():
            return
        step = -1 if getattr(event, "delta", 0) > 0 or event.num == 4 else 1
        canvas.yview_scroll(step, "units")

    def _bind(_event=None) -> None:
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.bind_all(sequence, _wheel)

    def _unbind(_event=None) -> None:
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.unbind_all(sequence)

    outer.bind("<Enter>", _bind)
    outer.bind("<Leave>", _unbind)

    inner._scroll_container = outer  # type: ignore[attr-defined]
    return outer, inner


def scrolled_text(parent, palette: Palette, font, height: int = 20, width: int = 80):
    """A read-only text area with a scrollbar and the app's monospace font.

    Returns the ``Text`` widget; the caller writes through :func:`set_text`.
    """
    import tkinter as tk
    from tkinter import ttk

    frame = ttk.Frame(parent, style="Card.TFrame")
    frame.pack(fill="both", expand=True)
    text = tk.Text(
        frame,
        height=height,
        width=width,
        wrap="none",
        font=font,
        background=palette.surface,
        foreground=palette.text,
        insertbackground=palette.text,
        relief="flat",
        borderwidth=0,
        highlightthickness=0,
        padx=PAD["sm"],
        pady=PAD["sm"],
    )
    y_scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
    text.grid(row=0, column=0, sticky="nsew")
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

    # Colour tags for the report's own markers, so warnings stand out.
    text.tag_configure("warning", foreground=palette.warning)
    text.tag_configure("danger", foreground=palette.danger)
    text.tag_configure("success", foreground=palette.success)
    text.tag_configure("heading", foreground=palette.accent)
    text.configure(state="disabled")
    return text


def set_text(widget, content: str) -> None:
    """Replace a read-only text widget's contents, colouring known markers.

    The reports use a small visual vocabulary — ``⚠`` for a caveat, ``✓``
    and ``✗`` for a passed or failed cross-check, a line of box characters
    for a section rule. Tagging them here means the report renderer stays
    plain text (so it also works in a terminal and in a file) while the GUI
    still shows the warnings in warning colour.
    """
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", content)
    for index, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("⚠"):
            tag = "warning"
        elif stripped.startswith("✗"):
            tag = "danger"
        elif stripped.startswith("✓"):
            tag = "success"
        elif stripped.startswith(("═", "─")) or (
            stripped and stripped == stripped.upper() and len(stripped) > 3
            and any(c.isalpha() for c in stripped)
        ):
            tag = "heading"
        else:
            continue
        widget.tag_add(tag, f"{index}.0", f"{index}.end")
    widget.configure(state="disabled")


def table(parent, columns: Sequence[str], height: int = 12, stretch: bool = True):
    """A Treeview configured as a data table, with both scrollbars.

    Returns the ``Treeview``; fill it with :func:`fill_table`.
    """
    from tkinter import ttk

    frame = ttk.Frame(parent, style="Card.TFrame")
    frame.pack(fill="both", expand=True)
    tree = ttk.Treeview(frame, columns=list(columns), show="headings", height=height)
    for name in columns:
        tree.heading(name, text=name)
        tree.column(name, width=max(70, min(190, 9 * len(name) + 40)),
                    anchor="w", stretch=stretch)
    y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
    tree.grid(row=0, column=0, sticky="nsew")
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return tree


def fill_table(tree, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    """Replace a table's columns and contents."""
    tree.delete(*tree.get_children())
    tree.configure(columns=list(columns))
    for name in columns:
        tree.heading(name, text=name)
        tree.column(name, width=max(70, min(190, 9 * len(name) + 40)), anchor="w")
    for row in rows:
        tree.insert("", "end", values=list(row))


#: Narrowest a control may become before its label moves above it.
#:
#: A combobox 58 px wide shows about four characters and is useless for
#: choosing between "C turbostrático 3.44" and "C turbostrático 3.50",
#: which is what the Rietveld phase chooser came out as in a 1100 px
#: window. Below this the label and the control stop sharing a line.
MIN_CONTROL = 120


def labelled(parent, text: str, widget_factory: Callable[[Any], Any], width: int = 16):
    """A label and a control, side by side — or stacked when narrow.

    The side-by-side row is right until the column is too narrow for it,
    and then it is badly wrong: the label keeps its ``width`` characters
    and the control absorbs the whole shortfall. Rather than choose one
    arrangement for every window size, the row measures itself and moves
    the label above the control when the control would drop below
    :data:`MIN_CONTROL`. Stacking costs a line of height, which a
    scrolling sidebar has, and buys back the width, which it does not.

    Returns the created widget, so the caller can keep a reference without
    a temporary variable for the row.
    """
    from tkinter import ttk

    row = ttk.Frame(parent, style="Card.TFrame")
    row.pack(fill="x", pady=PAD["xs"])
    label = ttk.Label(row, text=text, style="Card.TLabel", width=width,
                      anchor="w")
    widget = widget_factory(row)
    state = {"stacked": None}

    def arrange(_event=None) -> None:
        available = row.winfo_width()
        if available <= 1:
            available = row.winfo_reqwidth()
        stacked = available - label.winfo_reqwidth() < MIN_CONTROL
        if stacked == state["stacked"]:
            return
        state["stacked"] = stacked
        label.grid_forget()
        widget.grid_forget()
        row.columnconfigure(0, weight=0)
        row.columnconfigure(1, weight=0)
        if stacked:
            label.grid(row=0, column=0, sticky="w")
            widget.grid(row=1, column=0, sticky="ew", pady=(PAD["xs"], 0))
            row.columnconfigure(0, weight=1)
        else:
            label.grid(row=0, column=0, sticky="w")
            widget.grid(row=0, column=1, sticky="ew")
            row.columnconfigure(1, weight=1)

    arrange()
    row.bind("<Configure>", arrange, add="+")
    return widget


class Flow:
    """A row of controls that wraps to the next line instead of squeezing.

    Tk has no flow manager, and its absence is what most of this
    application's layout damage came from. A sidebar is about 230 px wide
    once the scrollbar and the card padding are taken out; a row that
    packs a label, two spinboxes and a button side by side asks for 380,
    and ``pack`` resolves that by handing out what it has in order — so
    the last widget in the row gets whatever is left. Measured on the
    real widget tree: the "Quitar" button in the diffractogram panel was
    allocated **2 pixels** of the 129 it asked for, and "Guardar datos…"
    got 7 of 145. They were not misaligned, they were gone, and no amount
    of resizing the window brought them back because the row was packed,
    not wrapped.

    So the children are placed on a grid and the number of columns is
    recomputed whenever the row's width changes: everything keeps its
    natural size and the row grows downwards, which is the direction a
    scrolling sidebar has room in.

    Use it through :func:`flow`.
    """

    def __init__(self, frame, gap: int) -> None:
        self.frame = frame
        self.gap = gap
        self.items: list[tuple[Any, bool]] = []
        self._columns = 0
        frame.bind("<Configure>", self._relayout, add="+")

    def add(self, widget, grow: bool = False):
        """Put ``widget`` in the row. ``grow`` lets it take spare width."""
        self.items.append((widget, grow))
        self._relayout()
        return widget

    def _fits(self, width: int) -> int:
        """How many of the widgets fit across ``width``, at least one."""
        columns, used = 0, 0
        for widget, _grow in self.items:
            need = widget.winfo_reqwidth() + self.gap
            if used + need > width and columns:
                break
            used += need
            columns += 1
        return max(columns, 1)

    def _relayout(self, _event=None) -> None:
        if not self.items:
            return
        width = self.frame.winfo_width()
        if width <= 1:
            width = self.frame.winfo_reqwidth()
        columns = self._fits(width)
        if columns == self._columns:
            return
        # Re-gridding changes the row's own height, which fires another
        # <Configure>; without this the two chase each other forever.
        self._columns = columns
        for index in range(columns):
            self.frame.columnconfigure(index, weight=0)
        for index, (widget, grow) in enumerate(self.items):
            row, column = divmod(index, columns)
            widget.grid(row=row, column=column, sticky="ew",
                        padx=(0, self.gap), pady=(0, self.gap))
            if grow:
                self.frame.columnconfigure(column, weight=1)
        # A single column must stretch, or a wrapped row leaves its
        # buttons at their natural width against the left edge.
        if columns == 1:
            self.frame.columnconfigure(0, weight=1)


def flow(parent, gap: Optional[int] = None, style: str = "Card.TFrame") -> Flow:
    """A wrapping row of controls. See :class:`Flow`."""
    from tkinter import ttk

    frame = ttk.Frame(parent, style=style)
    frame.pack(fill="x", pady=PAD["xs"])
    return Flow(frame, PAD["xs"] if gap is None else gap)


def plot_toolbar(holder, toolbar, extras: Sequence[tuple[str, Callable]]):
    """Matplotlib's navigation bar plus this application's own buttons.

    Packed as a wrapping row, because a toolbar is exactly the case
    :class:`Flow` exists for. ``NavigationToolbar2Tk`` packs its own
    seven buttons and a coordinate label first, so anything added
    afterwards is last in the packing order and absorbs the whole
    shortfall: measured in a 1100 px window, "Restablecer zoom" was given
    49 px of the 156 it needs and "Guardar datos…" 67 of 145 — both
    unreadable, one barely clickable. Squeezing the navigation bar
    instead is no better; matplotlib's own "Forward" came out 10 px wide.

    Treating the navigation bar as one item in a wrapping row means the
    extra buttons drop to a second line when the pane is narrow, and
    nothing is ever cut.

    ``holder`` must already be the navigation bar's master and be packed:
    a widget can only be gridded beside its own siblings, and
    ``NavigationToolbar2Tk`` takes its master at construction.

    Returns the :class:`Flow` that manages the strip.
    """
    from tkinter import ttk

    row = Flow(holder, PAD["xs"])
    row.add(toolbar)
    for label, command in extras:
        row.add(ttk.Button(holder, text=label, command=command))
    return row


def lay_out(figure) -> None:
    """Fit a figure's decorations, and do something sensible when it cannot.

    ``tight_layout`` gives up when the axes decorations need more room
    than the figure has — "the bottom and top margins cannot be made
    large enough" — and gives up by doing NOTHING, leaving the default
    margins. The default margins are a fraction of the figure, so on a
    short pane the axis labels land on top of the ticks and the title on
    top of the axes: this is most of what "badly scaled" looks like from
    the outside, and the suite raised the warning on every redraw of the
    panes that are short by design.

    Falling back to explicit fractional margins is not as good as a real
    fit, but it is a layout rather than an absence of one, and the
    numbers are chosen so the labels have somewhere to go.
    """
    import warnings

    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        try:
            figure.tight_layout()
        except (ValueError, RuntimeError):
            raised.append(None)
        if not any(raised):
            return
    figure.subplots_adjust(left=0.16, right=0.97, bottom=0.20, top=0.90,
                           hspace=0.45, wspace=0.30)


def separator(parent) -> None:
    """A horizontal rule with the standard vertical margin."""
    from tkinter import ttk

    ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=PAD["sm"])


#: Narrowest a paragraph is allowed to get before it stops shrinking.
#:
#: Below about this the words break more often than they fit and the
#: column reads as a ladder. A panel narrower than this has a different
#: problem, and clipping the text is not the fix for it.
MIN_WRAP = 160


#: Longer than this, in characters, and a note folds itself away.
#:
#: This application explains its decisions at length and should: the
#: choice between areas and heights, or between three components and
#: five, changes the numbers, and a user who picks one at random is worse
#: off than one who reads three sentences first. But a note is only
#: guidance the first few times, and after that it is furniture standing
#: between the controls and the figure. Worse, once the notes wrap to
#: their real width they get TALL, and a panel with a fixed share of the
#: height clips them: the Rietveld note was showing 109 px of the 169 it
#: needed, so the last thing it said — the one about weight fractions
#: being of the modelled crystalline part only — was not on screen at all.
#:
#: Four hundred characters is about five lines in a sidebar.
FOLD_ABOVE = 400


def _first_sentence(text: str) -> tuple[str, str]:
    """Split a note into its opening claim and the rest.

    The opening sentence of every note in this package says what the
    control does; the rest says why and what goes wrong. So the split is
    not arbitrary truncation — it is the line that has to stay visible.
    """
    cleaned = " ".join(text.split())
    for stop in (". ", "? ", ": "):
        index = cleaned.find(stop)
        if 40 <= index <= 240:
            return cleaned[:index + 1], cleaned[index + 2:]
    if len(cleaned) > 240:
        cut = cleaned.rfind(" ", 0, 200)
        return cleaned[:cut] + "…", cleaned[cut + 1:]
    return cleaned, ""


def hint(parent, text: str, wrap: int = 380) -> None:
    """A small muted explanatory paragraph, wrapped to its container.

    Used liberally: this application makes a lot of decisions that change
    the numbers (area versus height, which RBM parameterisation, how many
    components), and a one-line explanation beside the control is what
    stops a user picking one at random.

    ``wrap`` is a MAXIMUM, not the width. It used to be the width, in
    pixels, chosen by hand at each call site — the codebase had fifteen
    different values from 230 to 900 — and every one of them was right
    for exactly one window size. Measured on the real widget tree at
    1480, 1280 and 1100 px, 377 widgets came back allocated less than
    they asked for, almost all of them these paragraphs: the note under
    the voltammetry controls wanted 520 px in a panel 121 px wide and
    simply vanished off the edge. A paragraph cannot be given a width in
    advance because the panel it sits in does not have one either, so the
    label asks the container at every resize instead.

    A long measure is as bad as a short one, which is why ``wrap`` stays:
    prose set across 900 px is hard to read even when it fits, so the
    caller's number still caps it.
    """
    from tkinter import ttk

    head, tail = ("", "")
    if len(" ".join(text.split())) > FOLD_ABOVE:
        head, tail = _first_sentence(text)

    label = ttk.Label(parent, text=head or text, style="Muted.TLabel",
                      wraplength=max(int(wrap), MIN_WRAP), justify="left")
    label.pack(anchor="w", fill="x", pady=(0, 0 if tail else PAD["sm"]))

    if tail:
        rest = ttk.Label(parent, text=tail, style="Muted.TLabel",
                         wraplength=max(int(wrap), MIN_WRAP), justify="left")
        toggle = ttk.Label(parent, text="▸ más", style="Muted.TLabel",
                           cursor="hand2")
        toggle.pack(anchor="w", pady=(0, PAD["sm"]))

        def flip(_event=None) -> None:
            if rest.winfo_ismapped():
                rest.pack_forget()
                toggle.configure(text="▸ más")
            else:
                rest.pack(anchor="w", fill="x", pady=(0, PAD["sm"]),
                          before=toggle)
                toggle.configure(text="▾ menos")

        toggle.bind("<Button-1>", flip)
        rest.bind("<Configure>", lambda e: _fit(rest, e, wrap), add="+")

    def refit(event) -> None:
        # Bound to the LABEL, not to its parent. Asking the parent means
        # guessing how much padding it will claim, and getting it wrong
        # wherever a card sits inside a paned window: four paragraphs in
        # the XPS and electrochemistry panels stayed cut because the
        # frame they were in reported a width the label never received.
        # The label is packed with fill="x", so its own width IS the
        # width available to it — and because the fill decides that width
        # rather than the text, setting wraplength from it cannot ratchet
        # the paragraph narrower on every resize.
        target = min(max(int(event.width), MIN_WRAP),
                     max(int(wrap), MIN_WRAP))
        if abs(target - _wraplength(label)) > 2:
            label.configure(wraplength=target)

    label.bind("<Configure>", refit, add="+")
    return label


__all__ = [
    "FOLD_ABOVE",
    "MIN_CONTROL",
    "MIN_WRAP",
    "Flow",
    "card",
    "fill_table",
    "flow",
    "hint",
    "labelled",
    "lay_out",
    "plot_toolbar",
    "scrollable_column",
    "scrolled_text",
    "separator",
    "set_text",
    "split_column",
    "table",
]
