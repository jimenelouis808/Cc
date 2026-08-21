"""Smoke tests for the Tkinter layer using a stubbed Tk.

We cannot open a real window in CI, but we can still catch the errors that
actually bite when writing GUI code by hand: typos in attribute names,
methods called before the widget exists, handlers referring to variables
that were never created.

Strategy: stub only the Tk widgets, and keep the **real** matplotlib Figure
and 3D axes. That means ``_render`` genuinely draws the structure — only the
window itself is fake.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


class _Widget:
    """Accept-anything stand-in for a Tk widget."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._children: list["_Widget"] = []
        self.kwargs = kwargs
        parent = args[0] if args else kwargs.get("master")
        if isinstance(parent, _Widget):
            parent._children.append(self)

    def __getattr__(self, name: str):
        # Any unknown method is a no-op returning a fresh widget, which covers
        # pack/grid/bind/configure/update/... without enumerating them.
        def _noop(*args: Any, **kwargs: Any):
            return _Widget()

        return _noop

    def winfo_children(self):
        return list(self._children)

    def destroy(self):
        self._children.clear()

    def bbox(self, *_a, **_k):
        return (0, 0, 100, 100)


class _Var:
    """Stand-in for StringVar / BooleanVar with real get/set semantics."""

    def __init__(self, value: Any = None, **kwargs: Any) -> None:
        self._value = kwargs.get("value", value)

    def get(self) -> Any:
        return self._value

    def set(self, value: Any) -> None:
        self._value = value


class _Root(_Widget):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.after_calls: list[int] = []

    def after(self, delay_ms: int, _callback=None):
        # Record but never invoke: _poll_queue reschedules itself and would
        # otherwise recurse forever.
        self.after_calls.append(delay_ms)


def _install_fake_tk(monkeypatch) -> None:
    tk = types.ModuleType("tkinter")
    tk.Tk = _Root
    tk.Canvas = _Widget
    tk.Text = _Widget
    tk.Frame = _Widget
    tk.StringVar = _Var
    tk.BooleanVar = _Var

    ttk = types.ModuleType("tkinter.ttk")
    for name in (
        "Frame", "LabelFrame", "Label", "Button", "Combobox",
        "Entry", "Checkbutton", "Scrollbar", "Notebook",
    ):
        setattr(ttk, name, _Widget)
    tk.ttk = ttk

    messagebox = types.ModuleType("tkinter.messagebox")
    messagebox.showerror = lambda *a, **k: None
    messagebox.showinfo = lambda *a, **k: None
    messagebox.showwarning = lambda *a, **k: None

    filedialog = types.ModuleType("tkinter.filedialog")
    filedialog.askdirectory = lambda *a, **k: ""
    filedialog.asksaveasfilename = lambda *a, **k: ""

    # Fake only the Tk-specific matplotlib backend; the Figure stays real.
    backend = types.ModuleType("matplotlib.backends.backend_tkagg")

    class _Canvas:
        def __init__(self, figure, master=None):
            self.figure = figure

        def get_tk_widget(self):
            return _Widget()

        def draw(self):
            return None

        def draw_idle(self):
            return None

    backend.FigureCanvasTkAgg = _Canvas
    backend.NavigationToolbar2Tk = _Widget

    for name, module in (
        ("tkinter", tk),
        ("tkinter.ttk", ttk),
        ("tkinter.messagebox", messagebox),
        ("tkinter.filedialog", filedialog),
        ("matplotlib.backends.backend_tkagg", backend),
    ):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture()
def app(monkeypatch):
    _install_fake_tk(monkeypatch)
    from nanocarbon_lab.gui.app import NanocarbonApp

    return NanocarbonApp(_Root())


class TestConstruction:
    def test_window_builds_without_error(self, app):
        """Exercises _build_layout, _build_preview and _rebuild_param_fields."""
        assert app.atoms is None
        assert app._param_vars  # default structure populated its fields

    def test_default_structure_is_cnt(self, app):
        assert app._current_structure_key() == "cnt"

    def test_switching_structure_rebuilds_fields(self, app):
        from nanocarbon_lab.gui.params import STRUCTURES

        app.structure_var.set(STRUCTURES["nanocoil"].label)
        app._rebuild_param_fields()
        assert app._current_structure_key() == "nanocoil"
        assert "coil_radius" in app._param_vars

    def test_foam_hides_modifier_fields(self, app):
        from nanocarbon_lab.gui.params import STRUCTURES

        app.structure_var.set(STRUCTURES["foam"].label)
        app._rebuild_param_fields()
        # foam sets supports_modifiers=False
        assert app._modifier_vars == {}

    def test_read_raw_returns_defaults(self, app):
        raw = app._read_raw(app._param_vars)
        assert raw["n"] == "6"


class TestBuildAndRender:
    def test_build_and_render_real_figure(self, app):
        """The worker path plus a genuine matplotlib 3D render."""
        from nanocarbon_lab.gui.params import build_structure, STRUCTURES

        values = {s.key: s.default for s in STRUCTURES["cnt"].params}
        atoms = build_structure("cnt", values)
        app._on_built(atoms)

        assert app.atoms is atoms
        # Something was actually drawn onto the real axes.
        assert app.axes.collections

    def test_switching_structure_discards_previous(self, app):
        from nanocarbon_lab.gui.params import build_structure, STRUCTURES

        values = {s.key: s.default for s in STRUCTURES["cnt"].params}
        app._on_built(build_structure("cnt", values))
        assert app.atoms is not None

        app.structure_var.set(STRUCTURES["graphene"].label)
        app._rebuild_param_fields()
        # Otherwise "Exportar" would write the stale CNT.
        assert app.atoms is None

    def test_error_path_reports_without_crashing(self, app):
        app._show_error(ValueError("radio demasiado pequeño"), "traceback")
        # Unexpected types take the traceback branch; must not raise either.
        app._show_error(KeyError("boom"), "traceback")

    def test_poll_queue_handles_error_payload(self, app):
        app._queue.put(("error", (ValueError("mal"), "tb")))
        app._poll_queue()
        assert app._busy is False
