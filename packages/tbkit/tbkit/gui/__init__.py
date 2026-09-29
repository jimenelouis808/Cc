"""tbkit's window (``tbkit-gui``): PySide6 + pyvista, the optional ``gui`` extra.

Install with ``pip install "tbkit[gui]"`` (or ``uv sync --extra gui``). The
calculations live in :mod:`tbkit.gui.actions`, which has no Qt.
"""


def main(argv=None):
    try:
        from .app import main as run
    except ImportError as error:              # PySide6/pyvista missing, or no libEGL
        raise SystemExit(f"La GUI de tbkit necesita PySide6 y pyvista "
                         f"(pip install \"tbkit[gui]\"): {error}") from error
    return run(argv)
