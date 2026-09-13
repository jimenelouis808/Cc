"""Undo, redo, and preferences that survive closing the program.

Two small things that separate a tool people use from a script with
buttons.

**Undo.** Every analysis step here is a pure function from one state to
another, so undo is not a matter of reversing operations — it is a matter
of keeping the previous state. That is cheap for settings and dear for
data, which is why what goes on the stack is the *settings and the
recipe*, and the spectra are recomputed from the file when needed.

**Preferences.** A window that comes back the size it was, with the same
palette, the same laser and the same baseline method, is the difference
between software and a demonstration. They live in a JSON file in the
platform's own configuration directory, and a corrupt or unreadable one
is ignored rather than fatal: preferences are a convenience, and losing
them must never stop the program from starting.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar

State = TypeVar("State")

#: How many states to keep. Fifty is more than anybody undoes, and the
#: states are settings dictionaries rather than data.
DEFAULT_DEPTH = 50

APPLICATION = "ramancarbon"


class History(Generic[State]):
    """An undo/redo stack of application states.

    Records whole states rather than reversible operations. It is the
    boring implementation and it is right for this: an operation-based
    undo has to be correct for every pair of operations, and a state-based
    one has to be correct once.

    Consecutive pushes of an *equal* state are collapsed, so dragging a
    slider does not fill the stack with fifty identical entries.
    """

    def __init__(self, initial: State, depth: int = DEFAULT_DEPTH,
                 copier: Optional[Callable[[State], State]] = None) -> None:
        self._copy = copier or copy.deepcopy
        self._states: list[State] = [self._copy(initial)]
        self._position = 0
        self._depth = max(2, int(depth))
        self._labels: list[str] = ["inicio"]

    # -- state ---------------------------------------------------------
    @property
    def current(self) -> State:
        return self._copy(self._states[self._position])

    @property
    def can_undo(self) -> bool:
        return self._position > 0

    @property
    def can_redo(self) -> bool:
        return self._position < len(self._states) - 1

    @property
    def depth(self) -> int:
        return len(self._states)

    def label(self, offset: int = 0) -> str:
        index = self._position + offset
        return self._labels[index] if 0 <= index < len(self._labels) else ""

    def undo_label(self) -> str:
        return self.label() if self.can_undo else ""

    def redo_label(self) -> str:
        return self.label(1) if self.can_redo else ""

    # -- editing -------------------------------------------------------
    def push(self, state: State, label: str = "") -> bool:
        """Record a new state. Returns whether anything was recorded.

        Pushing after an undo **discards the redo branch**, which is what
        every editor does and what people expect; keeping it would need a
        tree and a way to show it.
        """
        if state == self._states[self._position]:
            return False
        del self._states[self._position + 1:]
        del self._labels[self._position + 1:]
        self._states.append(self._copy(state))
        self._labels.append(label or f"paso {len(self._states) - 1}")
        if len(self._states) > self._depth:
            excess = len(self._states) - self._depth
            del self._states[:excess]
            del self._labels[:excess]
        self._position = len(self._states) - 1
        return True

    def undo(self) -> State:
        if not self.can_undo:
            raise IndexError("no hay nada que deshacer")
        self._position -= 1
        return self.current

    def redo(self) -> State:
        if not self.can_redo:
            raise IndexError("no hay nada que rehacer")
        self._position += 1
        return self.current

    def reset(self, state: State, label: str = "inicio") -> None:
        """Start again from one state, forgetting everything."""
        self._states = [self._copy(state)]
        self._labels = [label]
        self._position = 0

    def labels(self) -> list[str]:
        """Every state's label, for a history panel."""
        return list(self._labels)

    def __len__(self) -> int:
        return len(self._states)


def config_directory(application: str = APPLICATION) -> Path:
    """Where this platform keeps a program's configuration.

    ``$XDG_CONFIG_HOME`` or ``~/.config`` on Linux, ``%APPDATA%`` on
    Windows, ``~/Library/Application Support`` on macOS. Written out
    rather than pulled from a dependency, because one directory is not
    worth a package.
    """
    override = os.environ.get("RAMANCARBON_CONFIG")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / application
    elif os.uname().sysname == "Darwin":            # pragma: no cover - macOS
        return Path.home() / "Library" / "Application Support" / application
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / application


@dataclass
class Preferences:
    """Settings that persist between sessions.

    Deliberately a flat dictionary with typed accessors rather than a
    dataclass per panel: a preferences file has to survive the program
    growing new settings and losing old ones, and a strict schema turns
    every such change into a migration.
    """

    values: dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None
    loaded: bool = False
    problem: str = ""
    """Why the file could not be read, when it could not. Shown once and
    then ignored: preferences are a convenience and must never stop the
    program from starting."""

    #: The defaults, and the list of what is remembered at all.
    DEFAULTS: dict[str, Any] = field(default_factory=lambda: {
        "laser_nm": 532.0,
        "palette": "claro",
        "baseline_method": "asls",
        "normalise": "none",
        "check_phases": True,
        "check_interferences": False,
        "plot_preset": "predeterminado",
        "export_format": "csv",
        "window_size": [1200, 800],
        "last_directory": "",
        "recent_files": [],
        "xrd_anode": "Cu",
        "echem_reference": "Ag/AgCl_3M",
    })

    def __post_init__(self) -> None:
        merged = dict(self.DEFAULTS)
        merged.update(self.values)
        self.values = merged

    # -- access --------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, self.DEFAULTS.get(key, default))

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def remember_file(self, path: str | Path, limit: int = 12) -> None:
        """Put a file at the top of the recent list, without duplicates."""
        text = str(Path(path).resolve())
        recent = [item for item in self.values.get("recent_files", [])
                  if item != text]
        recent.insert(0, text)
        self.values["recent_files"] = recent[:limit]
        self.values["last_directory"] = str(Path(text).parent)

    def existing_recent(self) -> list[Path]:
        """Recent files that are still there.

        Checked on the way out rather than pruned on the way in: a file on
        a network share that is temporarily unmounted should come back
        when it is mounted again, not be forgotten.
        """
        return [Path(item) for item in self.values.get("recent_files", [])
                if Path(item).is_file()]

    # -- files ---------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "Preferences":
        """Read the preferences, or return the defaults with a reason."""
        location = Path(path) if path else config_directory() / "preferencias.json"
        preferences = cls(path=location)
        if not location.is_file():
            return preferences
        try:
            payload = json.loads(location.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            preferences.problem = (
                f"no se han podido leer las preferencias de {location}: "
                f"{error}. Se usan las de fábrica"
            )
            return preferences
        if not isinstance(payload, dict):
            preferences.problem = (
                f"{location} no contiene un objeto JSON; se usan las de fábrica")
            return preferences
        # Unknown keys are KEPT, not dropped: a preferences file shared
        # between two versions of the program must not lose the newer
        # one's settings every time the older one saves it.
        preferences.values.update(payload)
        preferences.loaded = True
        return preferences

    def save(self, path: Optional[str | Path] = None) -> Optional[Path]:
        """Write the preferences. Returns ``None`` if it could not.

        Writing is best-effort on purpose: a read-only home directory, a
        full disc or a locked profile are all real, and none of them is a
        reason to lose the user's work.
        """
        location = Path(path) if path else (
            self.path or config_directory() / "preferencias.json")
        try:
            location.parent.mkdir(parents=True, exist_ok=True)
            temporary = location.with_suffix(location.suffix + ".tmp")
            temporary.write_text(
                json.dumps(self.values, ensure_ascii=False, indent=2),
                encoding="utf-8")
            temporary.replace(location)
        except OSError as error:
            self.problem = f"no se han podido guardar las preferencias: {error}"
            return None
        self.path = location
        return location

    def reset(self, keys: Optional[Iterable[str]] = None) -> None:
        """Back to the factory settings, all of them or some."""
        if keys is None:
            self.values = dict(self.DEFAULTS)
            return
        for key in keys:
            if key in self.DEFAULTS:
                self.values[key] = self.DEFAULTS[key]
            else:
                self.values.pop(key, None)


__all__ = [
    "APPLICATION",
    "DEFAULT_DEPTH",
    "History",
    "Preferences",
    "config_directory",
]
