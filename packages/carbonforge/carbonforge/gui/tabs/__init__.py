"""The tabs of the carbonforge window, one module each.

Each tab is a mixin: :class:`~carbonforge.gui.app.CarbonForgeApp` inherits
them all and supplies what they share (Tk modules, the root window, the
worker queue, the form helpers). A tab never imports another tab; what one
needs from another goes through the host, and each class docstring lists it.
Keeping them apart is the first step towards one window for carbonforge and
vibspec (docs/PLAN_SUITE.md, phase B).
"""

from .analysis import AnalysisTab
from .builder import BuilderTab
from .edlc import EdlcTab
from .importing import ImportTab
from .preview import PreviewPanel

__all__ = ["AnalysisTab", "BuilderTab", "EdlcTab", "ImportTab", "PreviewPanel"]
