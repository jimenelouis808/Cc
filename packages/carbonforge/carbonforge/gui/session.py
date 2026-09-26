"""The structure the whole window is working on, shared between tabs.

Building a ribbon, importing a file and building a vibspec model all produce
a structure; exporting it to QE, turning it into an EDLC cell or computing
its IR all consume one. :class:`Session` is where one tab leaves it and
another picks it up, so a tab never has to know about the others.

Handing a structure over is always the user's choice (a button): a tab that
silently replaced what another tab shows would be worse than no sharing.
No Tk here, so it is tested without a display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ase import Atoms


@dataclass(frozen=True)
class Current:
    """The shared structure and where it came from (``"Construir"``...)."""

    atoms: Atoms
    origin: str

    def describe(self) -> str:
        return (f"{self.atoms.get_chemical_formula()} · {len(self.atoms)} átomos "
                f"· de «{self.origin}»")


class Session:
    """Holds the current structure and tells subscribers when it changes."""

    def __init__(self) -> None:
        self._current: Optional[Current] = None
        self._listeners: list[Callable[[Optional[Current]], None]] = []

    @property
    def current(self) -> Optional[Current]:
        return self._current

    def publish(self, atoms: Atoms, origin: str) -> Current:
        """Make ``atoms`` the current structure (a copy: later edits by the
        publishing tab do not leak into the others)."""
        self._current = Current(atoms.copy(), origin)
        self._notify()
        return self._current

    def take(self) -> Optional[Atoms]:
        """A copy of the current structure for a tab to work on, or None."""
        return None if self._current is None else self._current.atoms.copy()

    def clear(self) -> None:
        self._current = None
        self._notify()

    def subscribe(self, listener: Callable[[Optional[Current]], None]) -> None:
        """Call ``listener(current)`` now and on every change."""
        self._listeners.append(listener)
        listener(self._current)

    def _notify(self) -> None:
        for listener in list(self._listeners):
            listener(self._current)
