"""Every number a recipe uses, adjustable from the command line or a file, and explained.

A recipe keeps its settings as module-level names (``KZ = 4``, ``WINDOW = 3.2``) and
declares them in ``PARAMS``: what each one is, its unit and why it has the value it
has. Then, without changing the recipe's code, any run can override them::

    python -m tbkit.recipes.coil_double_resonance pairs 2.33 --ajuste NK=32 --ajuste GAMMA=0.05
    python -m tbkit.recipes.doped_raman raman N --ajustes mis_ajustes.json

and ``tbkit recetas NOMBRE`` lists them. The values actually used are written next to
the results (``ajustes_usados.json`` in the recipe's work folder, one entry per run), so
a result can always be traced to the settings that produced it.

Values are parsed as JSON first (``4``, ``0.05``, ``[1.96, 2.33]``, ``"xu_chn"``,
``true``) and then as plain text; a value is converted to the type of the current one
(a list becomes a tuple if the default is a tuple, a string a ``Path`` if it is a path),
and a name the recipe did not declare is refused rather than silently ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass(frozen=True)
class Param:
    """One adjustable setting of a recipe (a module-level name)."""

    name: str
    description: str
    unit: str = ""
    why: str = ""
    group: str = "cálculo"          # cálculo, convergencia, recursos, rutas, salida


def _convert(value: str, current):
    try:
        parsed = json.loads(value)
    except ValueError:
        parsed = value
    if current is None:
        return parsed
    if isinstance(current, bool):
        if isinstance(parsed, str):
            return parsed.lower() in ("1", "true", "si", "sí", "yes")
        return bool(parsed)
    if isinstance(current, Path):
        return Path(str(parsed))
    if isinstance(current, tuple) and isinstance(parsed, list):
        return tuple(tuple(x) if isinstance(x, list) else x for x in parsed)
    if isinstance(current, float) and isinstance(parsed, (int, float)):
        return float(parsed)
    if isinstance(current, int) and isinstance(parsed, (int, float)) \
            and float(parsed).is_integer():
        return int(parsed)
    if isinstance(current, dict) and isinstance(parsed, dict):
        return {k: (tuple(v) if isinstance(v, list) and isinstance(current.get(k), tuple)
                    else v) for k, v in {**current, **parsed}.items()}
    if type(parsed) is not type(current) and not isinstance(current, (list, dict)):
        raise ValueError(f"esperaba {type(current).__name__}, recibí {parsed!r}")
    return parsed


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """``--ajuste NOMBRE=VALOR`` (repetible), ``--ajustes ARCHIVO.json``, ``--ver-ajustes``."""
    group = parser.add_argument_group("ajustes de la receta (tbkit recetas NOMBRE los lista)")
    group.add_argument("--ajuste", action="append", default=[], metavar="NOMBRE=VALOR",
                       help="cambia un ajuste de la receta para esta ejecución")
    group.add_argument("--ajustes", type=Path, default=None, metavar="ARCHIVO.json",
                       help="varios ajustes en un JSON {nombre: valor}")
    group.add_argument("--ver-ajustes", action="store_true",
                       help="muestra los ajustes (con los cambios) y sale")


def apply(module: ModuleType, args: argparse.Namespace, record: Path | None = None) -> dict:
    """Override the module's declared settings from ``args``; return {name: value} changed.

    With ``record`` (a folder), append the full set in use to ``ajustes_usados.json``.
    With ``--ver-ajustes``, print them and exit."""
    declared = {p.name: p for p in getattr(module, "PARAMS", [])}
    wanted: dict = {}
    if getattr(args, "ajustes", None):
        wanted.update(json.loads(Path(args.ajustes).read_text(encoding="utf-8")))
    for item in getattr(args, "ajuste", []) or []:
        if "=" not in item:
            raise SystemExit(f"--ajuste espera NOMBRE=VALOR, recibió {item!r}")
        name, value = item.split("=", 1)
        wanted[name.strip()] = value.strip()
    changed = {}
    for name, value in wanted.items():
        if name not in declared:
            raise SystemExit(f"'{name}' no es un ajuste de esta receta. Los que hay: "
                             f"{', '.join(declared) or 'ninguno'}.")
        current = getattr(module, name)
        try:
            new = _convert(value, current) if isinstance(value, str) else \
                _convert(json.dumps(value), current)
        except ValueError as error:
            raise SystemExit(f"{name}: {error}") from error
        setattr(module, name, new)
        changed[name] = new
    if getattr(args, "ver_ajustes", False):
        print(describe(module))
        raise SystemExit(0)
    if record is not None:
        record = Path(record)
        record.mkdir(parents=True, exist_ok=True)
        path = record / "ajustes_usados.json"
        history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        history.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "command": " ".join(sys.argv),
                        "changed": {k: _jsonable(v) for k, v in changed.items()},
                        "values": current_values(module)})
        path.write_text(json.dumps(history, indent=1, ensure_ascii=False), encoding="utf-8")
    return changed


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def current_values(module: ModuleType) -> dict:
    return {p.name: _jsonable(getattr(module, p.name)) for p in getattr(module, "PARAMS", [])}


def describe(module: ModuleType) -> str:
    """The declared settings with their current values, by group."""
    params = getattr(module, "PARAMS", [])
    if not params:
        return f"{module.__name__}: sin ajustes declarados."
    spec = getattr(module, "__spec__", None)
    lines = [getattr(spec, "name", None) or module.__name__]
    for group in dict.fromkeys(p.group for p in params):
        lines.append(f"\n  [{group}]")
        for p in (q for q in params if q.group == group):
            value = json.dumps(_jsonable(getattr(module, p.name)), ensure_ascii=False)
            unit = f" {p.unit}" if p.unit else ""
            lines.append(f"  {p.name} = {value}{unit}")
            lines.append(f"      {p.description}")
            if p.why:
                lines.append(f"      por qué: {p.why}")
    return "\n".join(lines)


#: Recipes that declare PARAMS (for ``tbkit recetas``).
RECIPES = ("coil_double_resonance", "doped_raman", "doped_gpaw", "born_references",
           "electronic_compare", "ir_charge_fit", "n_references", "structure_screening",
           "nanocoil")
