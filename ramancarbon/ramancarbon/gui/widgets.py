"""Reusable widgets: cards, scrollable text, tables, labelled fields.

Tkinter is imported inside the functions rather than at module scope so that
importing :mod:`ramancarbon.gui` on a machine without Tk fails with the
package's own explanatory message instead of an ImportError traceback.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence

from .theme import PAD, Palette


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
    if title:
        ttk.Label(outer, text=title, style="Heading.TLabel").pack(
            anchor="w", pady=(0, PAD["xs"] if subtitle else PAD["sm"])
        )
    if subtitle:
        ttk.Label(outer, text=subtitle, style="Muted.TLabel", wraplength=520).pack(
            anchor="w", pady=(0, PAD["sm"])
        )
    body = ttk.Frame(outer, style="Card.TFrame")
    body.pack(fill="both", expand=True)
    return outer, body


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


def labelled(parent, text: str, widget_factory: Callable[[Any], Any], width: int = 16):
    """A left-aligned label followed by a widget, packed in a row.

    Returns the created widget, so the caller can keep a reference without
    a temporary variable for the row.
    """
    from tkinter import ttk

    row = ttk.Frame(parent, style="Card.TFrame")
    row.pack(fill="x", pady=PAD["xs"])
    ttk.Label(row, text=text, style="Card.TLabel", width=width, anchor="w").pack(side="left")
    widget = widget_factory(row)
    widget.pack(side="left", fill="x", expand=True)
    return widget


def separator(parent) -> None:
    """A horizontal rule with the standard vertical margin."""
    from tkinter import ttk

    ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=PAD["sm"])


def hint(parent, text: str, wrap: int = 380) -> None:
    """A small muted explanatory paragraph.

    Used liberally: this application makes a lot of decisions that change
    the numbers (area versus height, which RBM parameterisation, how many
    components), and a one-line explanation beside the control is what
    stops a user picking one at random.
    """
    from tkinter import ttk

    ttk.Label(parent, text=text, style="Muted.TLabel", wraplength=wrap,
              justify="left").pack(anchor="w", pady=(0, PAD["sm"]))


__all__ = [
    "card",
    "fill_table",
    "hint",
    "labelled",
    "scrolled_text",
    "separator",
    "set_text",
    "table",
]
