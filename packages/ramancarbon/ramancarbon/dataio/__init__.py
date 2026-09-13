"""Getting data in and out: detection, universal reading, export, projects.

The four instruments have four readers, and a user with a folder of
measurements should not need to know which is which. This package is the
layer above them:

:func:`~ramancarbon.dataio.detect.detect`
    What is this file? Decided from the numbers, not the extension.
:func:`~ramancarbon.dataio.load.load`
    Open anything the suite understands, returning the right object.
:mod:`~ramancarbon.dataio.jcamp`
    JCAMP-DX, the one interchange format every spectrometer can write.
:class:`~ramancarbon.dataio.export.Table`
    Any result as CSV, TSV, JSON, Markdown, LaTeX or HTML.
:class:`~ramancarbon.dataio.project.Project`
    A whole session in one ``.rcproj`` file: measurements, settings,
    figures and results.
"""

from __future__ import annotations

from .detect import Detection, detect, scan
from .export import Table, export, series_table, significant, summary_table
from .jcamp import JCAMPError, read_jcamp, write_jcamp
from .load import LoadError, Loaded, load, load_folder
from .project import Dataset, Project, ProjectError

__all__ = [
    "Dataset",
    "Detection",
    "JCAMPError",
    "LoadError",
    "Loaded",
    "Project",
    "ProjectError",
    "Table",
    "detect",
    "export",
    "load",
    "load_folder",
    "read_jcamp",
    "scan",
    "series_table",
    "significant",
    "summary_table",
    "write_jcamp",
]
