"""Measure the real widget tree and refuse a control nobody can use.

Written because the user's report was "some things come out on top of
each other or badly scaled", and because reading the layout code could
not have found it. Tk resolves a row that asks for more width than it has
by handing out the cavity in packing order and giving each widget its
request first, so the LAST widget in the row absorbs the whole shortfall.
Nothing in the source says which widget that is; it depends on the window
size, the font and what the widgets before it happen to contain.

So the tree is measured instead. After ``update_idletasks`` every widget
knows what it asked for and what it got, and two defects become
arithmetic rather than judgement:

    unusable   a control allocated less than it takes to operate it --
               measured: a 2 px "Quitar" button, a 58 px phase chooser
               showing four characters, a 6 px checkbox
    text cut   a label or button whose words do not fit in its allocation

The first audit of the suite found 104 of these across three window
sizes. This test is what stops them coming back: every one of them was
somebody adding a reasonable control to a reasonable row.
"""

from __future__ import annotations

import pytest

tkinter = pytest.importorskip("tkinter")


#: What each kind of control needs to be operable, in pixels.
#:
#: Deliberately not "what it requested". A combobox declared thirty
#: characters wide and given twenty-eight is fine; one given 58 px shows
#: four characters and cannot distinguish "C turbostrático 3.44" from
#: "C turbostrático 3.50", which is the whole decision it exists for.
#: Judging by the request instead flagged every widget that declared a
#: generous width and missed the buttons squeezed to two pixels.
USABLE = {
    "TButton": 70, "Button": 70, "TCombobox": 110, "TEntry": 90,
    "TSpinbox": 55, "TCheckbutton": 60, "TRadiobutton": 60,
}

CONTAINERS = {
    "Canvas", "Text", "Listbox", "Treeview", "TNotebook", "TFrame", "Frame",
    "TLabelframe", "Labelframe", "TPanedwindow", "Panedwindow", "TSeparator",
    "Separator", "Toplevel", "Tk",
}

TEXTUAL = {"TLabel", "Label", "TCheckbutton", "TRadiobutton", "TButton", "Button"}

#: Window sizes to check. The smallest is the suite's own ``minsize``:
#: a size the user can produce with the mouse has to work.
SIZES = ((1480, 940), (1100, 740))

#: Text allowed to be cut, in widgets, across the whole sweep.
#:
#: Not zero, and the four that survive say why: three are notes inside a
#: pane with a fixed share of the height, clipped by a line or two, and
#: one is a card title in the narrowest column in the suite. A budget
#: that is currently met with room to spare catches a regression --
#: 104 defects would blow through it instantly -- without demanding that
#: every paragraph in the application fit every window.
TEXT_CUT_BUDGET = 8


def _walk(widget, out):
    for kid in widget.winfo_children():
        try:
            if kid.winfo_ismapped():
                out.append(kid)
                _walk(kid, out)
        except tkinter.TclError:            # pragma: no cover - torn down
            continue


def _text_of(widget) -> str:
    try:
        return str(widget.cget("text"))[:40]
    except tkinter.TclError:
        return ""


def _inspect(root, where, unusable, cut):
    root.update_idletasks()
    root.update()
    widgets: list = []
    _walk(root, widgets)
    for widget in widgets:
        kind = widget.winfo_class()
        width, height = widget.winfo_width(), widget.winfo_height()
        want_w, want_h = widget.winfo_reqwidth(), widget.winfo_reqheight()
        if width <= 1 or height <= 1:
            if kind not in CONTAINERS and want_w > 4:
                unusable.append(f"{where}: {kind} «{_text_of(widget)}» "
                                f"is {width}x{height} px")
            continue
        floor = USABLE.get(kind)
        if floor is not None and width < floor <= want_w:
            unusable.append(f"{where}: {kind} «{_text_of(widget)}» got "
                            f"{width} px, needs {floor} to be operable "
                            f"(asked {want_w})")
            continue
        if kind in TEXTUAL and kind not in CONTAINERS:
            if want_w - width > 12 and width < 0.85 * want_w:
                cut.append(f"{where}: {kind} «{_text_of(widget)}» "
                           f"{width}/{want_w} px wide")
            elif want_h - height > 6 and height < 0.9 * want_h:
                cut.append(f"{where}: {kind} «{_text_of(widget)}» "
                           f"{height}/{want_h} px tall")


def _notebooks(widget, out=None):
    out = [] if out is None else out
    for kid in widget.winfo_children():
        if kid.winfo_class() == "TNotebook":
            out.append(kid)
        _notebooks(kid, out)
    return out


@pytest.fixture(scope="module")
def measured():
    """Every tab of every section, at every size, measured once."""
    import matplotlib

    matplotlib.use("Agg")
    from ramancarbon.gui.suite import SECTIONS, Suite

    try:
        root = tkinter.Tk()
    except tkinter.TclError as error:          # pragma: no cover - no display
        pytest.skip(f"no display: {error}")

    unusable: list[str] = []
    cut: list[str] = []
    crashes: list[str] = []
    try:
        suite = Suite(root)
        keys = [key for key, _label, _doc in SECTIONS]
        for width, height in SIZES:
            root.geometry(f"{width}x{height}")
            root.update_idletasks()
            root.update()
            for key, label, _doc in SECTIONS:
                try:
                    suite._ensure(key)
                except Exception as error:      # noqa: BLE001
                    crashes.append(f"{label}: {error}")
                    continue
                suite.notebook.select(keys.index(key))
                root.update_idletasks()
                root.update()
                inner = _notebooks(suite._frames[key])
                if not inner:
                    _inspect(root, f"{label}@{width}", unusable, cut)
                for notebook in inner:
                    for tab in notebook.tabs():
                        notebook.select(tab)
                        root.update_idletasks()
                        root.update()
                        name = notebook.tab(tab, "text").strip()
                        _inspect(root, f"{label}/{name}@{width}", unusable, cut)
    finally:
        root.destroy()
    return unusable, cut, crashes


def test_every_section_builds_at_every_size(measured):
    _unusable, _cut, crashes = measured
    assert not crashes, "a section failed to build:\n  " + "\n  ".join(crashes)


def test_no_control_is_too_small_to_use(measured):
    """The defect the user could see: buttons squeezed out of existence.

    "Quitar" in the diffractogram panel was allocated 2 px of the 129 it
    needs, "Guardar datos…" 7 of 145. They were not misaligned, they were
    gone, and no window size brought them back because the row was packed
    rather than wrapped.
    """
    unusable, _cut, _crashes = measured
    assert not unusable, (
        f"{len(unusable)} control(s) too small to operate:\n  "
        + "\n  ".join(unusable[:20])
    )


def test_text_is_not_cut_off(measured):
    """Within a budget, and the budget is generous on purpose.

    Wrapping every paragraph to its container fixed almost all of this;
    what is left is notes in panes with a fixed share of the height. A
    budget that today has room to spare still fails loudly if a change
    brings back the fifteen hand-tuned pixel wrap widths this replaced.
    """
    _unusable, cut, _crashes = measured
    assert len(cut) <= TEXT_CUT_BUDGET, (
        f"{len(cut)} widget(s) with their text cut off, budget "
        f"{TEXT_CUT_BUDGET}:\n  " + "\n  ".join(cut[:20])
    )
