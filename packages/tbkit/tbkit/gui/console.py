"""The window's terminal: what each calculation runs and prints, as it happens."""

from __future__ import annotations

import time
from pathlib import Path

from ase import Atoms
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class Stream(QObject):
    """A file-like object whose writes become a signal (safe from any thread:
    the connection to the panel is queued into the GUI thread)."""

    text = Signal(str)

    def write(self, text: str) -> int:
        if text:
            self.text.emit(text)
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def describe(value) -> str:
    """A short, readable form of an argument (a structure by its formula, a model
    by its name), for the command line shown in the terminal."""
    if isinstance(value, Atoms):
        return value.get_chemical_formula()
    if hasattr(value, "orbitals") and hasattr(value, "name"):
        return f"<{value.name}>"
    if hasattr(value, "solution"):
        return "<estado fundamental>"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)) and len(value) > 6:
        return f"[{len(value)} valores]"
    if isinstance(value, (list, tuple)) and any(hasattr(v, "shape") for v in value):
        return "<modos>"
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "…"


def command_line(func, args, kwargs) -> str:
    parts = [describe(a) for a in args]
    parts += [f"{k}={describe(v)}" for k, v in kwargs.items() if v is not None]
    return f"{func.__name__}({', '.join(parts)})"


class ConsolePanel(QWidget):
    """Read-only, monospaced, dark: the log of the session's calculations."""

    MAX_LINES = 5000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(self.MAX_LINES)
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        font = QFont("Monospace")
        font.setStyleHint(QFont.TypeWriter)
        font.setPointSize(9)
        self.view.setFont(font)
        self.view.setStyleSheet("QPlainTextEdit { background: #1e1f22; color: #d7dae0; }")
        clear = QPushButton("Limpiar")
        clear.clicked.connect(self.view.clear)
        save = QPushButton("Guardar…")
        save.clicked.connect(self.save_dialog)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(clear)
        buttons.addWidget(save)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        layout.addLayout(buttons)

    def write(self, text: str) -> None:
        """Append raw text (the job's own prints), keeping the view at the end."""
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.view.setTextCursor(cursor)
        self.view.ensureCursorVisible()

    def line(self, text: str) -> None:
        """One stamped line of the window's own (start, end, error)."""
        stamp = time.strftime("%H:%M:%S")
        existing = self.view.toPlainText()
        prefix = "" if not existing or existing.endswith("\n") else "\n"
        self.write(f"{prefix}[{stamp}] {text}\n")

    def text(self) -> str:
        return self.view.toPlainText()

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(self.text(), encoding="utf-8")
        return path

    def save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Guardar registro de la terminal",
                                              "tbkit-terminal.log", "Texto (*.log *.txt)")
        if path:
            self.save(path)
