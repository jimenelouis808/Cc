"""Read a code's own documentation into a :class:`~carbonforge.codes.catalog.Catalog`.

Each code documents its input in a different form; each has one reader here:

* **Quantum ESPRESSO** -- the ``INPUT_PW.def`` helpdoc file shipped in every
  QE source tree (``PW/Doc/INPUT_PW.def``), the machine-readable source of
  the official input documentation.
* **SIESTA** -- the LaTeX manual (``Docs/siesta.tex``), whose keywords are
  ``fdfentry`` environments with a type and a default.
* **GPAW** -- the installed package itself: the parameters its calculator
  accepts, with their defaults.

The result is stored per user (:func:`~carbonforge.codes.catalog.save_imported`)
and merged over the curated core by :func:`~carbonforge.codes.catalog.load_catalog`.
Nothing of the manuals is shipped with carbonforge.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

from .catalog import Catalog, Parameter

# --------------------------------------------------------------------------
# Quantum ESPRESSO: helpdoc (.def)
# --------------------------------------------------------------------------

_QE_KINDS = {"INTEGER": "integer", "REAL": "real", "LOGICAL": "logical",
             "CHARACTER": "string", "STRING": "string"}


def _tokens(text: str) -> Iterator[str]:
    """Words and single braces of a helpdoc file; ``#`` comments removed."""
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        for match in re.finditer(r"\{|\}|[^\s{}]+", line):
            yield match.group(0)


def _parse_block(tokens: list[str], start: int) -> tuple[list, int]:
    """Parse from after an opening brace to its closing brace.

    Returns a nested list: plain words are strings, sub-blocks are lists.
    """
    out: list = []
    i = start
    while i < len(tokens):
        token = tokens[i]
        if token == "{":
            block, i = _parse_block(tokens, i + 1)
            out.append(block)
        elif token == "}":
            return out, i + 1
        else:
            out.append(token)
            i += 1
    return out, i


def _text(block) -> str:
    """The words of a block (recursively), as one line."""
    if isinstance(block, str):
        return block
    return " ".join(_text(item) for item in block)


def _clean_info(text: str) -> str:
    text = re.sub(r"@ref\s+", "", text)
    text = re.sub(r"@b\s*\{?([^}\s]*)\}?", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _var_fields(body: list) -> dict[str, object]:
    """``default``, ``info``, ``units`` and the ``opt -val`` options of a var body."""
    fields: dict[str, object] = {"choices": [], "option_info": []}
    i = 0
    while i < len(body):
        item = body[i]
        if item in ("default", "info", "units", "status") and i + 1 < len(body):
            j = i + 1
            while j < len(body) and isinstance(body[j], str):   # "-kind expr"
                j += 1
            if j < len(body):
                fields[item] = _clean_info(_text(body[j]))
            i = j + 1
            continue
        if item == "options" and i + 1 < len(body) and isinstance(body[i + 1], list):
            fields.update(_var_fields(body[i + 1]))
            i += 2
            continue
        if item == "opt" and i + 2 < len(body) and body[i + 1] == "-val":
            value = body[i + 2]
            if isinstance(value, str):
                fields["choices"].append(value.strip("'\""))
                if i + 3 < len(body) and isinstance(body[i + 3], list):
                    text = _clean_info(_text(body[i + 3]))
                    if text:
                        fields["option_info"].append(f"{value.strip(chr(39) + chr(34))}: {text}")
                    i += 1
            i += 3
            continue
        i += 1
    return fields


def _walk_vars(block: list, section: str) -> Iterator[Parameter]:
    """Every ``var``/``dimension`` in a namelist, inside groups and choose blocks."""
    i = 0
    while i < len(block):
        item = block[i]
        if item in ("var", "dimension") and i + 1 < len(block) and isinstance(block[i + 1], str):
            name = block[i + 1]
            kind = "any"
            j = i + 2
            while j < len(block) and isinstance(block[j], str):
                if block[j] == "-type" and j + 1 < len(block):
                    kind = _QE_KINDS.get(str(block[j + 1]).upper(), "any")
                j += 1
            body = block[j] if j < len(block) and isinstance(block[j], list) else []
            fields = _var_fields(body)
            choices = tuple(dict.fromkeys(fields["choices"]))
            if kind == "logical":
                choices = ()
            description = "\n".join([str(fields.get("info", "")), *fields["option_info"]])
            yield Parameter(
                code="qe", name=name + ("(i)" if item == "dimension" else ""),
                section=section, kind=kind,
                default=fields.get("default") or None, choices=choices,
                units=str(fields.get("units", "")), description=description.strip()[:1500],
                source="manual INPUT_PW.def",
            )
            i = j + 1
            continue
        if item == "vargroup":
            # A group of variables sharing a type and one description.
            j = i + 1
            kind = "any"
            while j < len(block) and isinstance(block[j], str):
                if block[j] == "-type" and j + 1 < len(block):
                    kind = _QE_KINDS.get(str(block[j + 1]).upper(), "any")
                j += 1
            body = block[j] if j < len(block) and isinstance(block[j], list) else []
            fields = _var_fields(body)
            k = 0
            while k < len(body):
                if body[k] == "var" and k + 1 < len(body) and isinstance(body[k + 1], str):
                    yield Parameter(code="qe", name=body[k + 1], section=section, kind=kind,
                                    default=fields.get("default") or None,
                                    description=str(fields.get("info", "")),
                                    source="manual INPUT_PW.def")
                    k += 2
                else:
                    k += 1
            i = j + 1
            continue
        if isinstance(item, list):
            yield from _walk_vars(item, section)
        i += 1


def import_qe_helpdoc(path: str | Path) -> Catalog:
    """Read a QE ``INPUT_PW.def`` (helpdoc) file into a catalogue.

    Only namelists are read (``&CONTROL``, ``&SYSTEM``, ...); cards
    (``ATOMIC_POSITIONS``...) are structural and written by carbonforge.
    """
    path = Path(path)
    tokens = list(_tokens(path.read_text(encoding="utf-8", errors="replace")))
    tree, _ = _parse_block(tokens, 0)
    catalog = Catalog("qe", sources=[f"manual {path.name}"])

    def find_namelists(block: list) -> Iterator[tuple[str, list]]:
        for k, item in enumerate(block):
            if item == "namelist" and k + 2 < len(block) and isinstance(block[k + 2], list):
                yield str(block[k + 1]).lstrip("&").lower(), block[k + 2]
            elif isinstance(item, list):
                yield from find_namelists(item)

    for section, body in find_namelists(tree):
        for parameter in _walk_vars(body, section):
            catalog.add(parameter)
    if not catalog.parameters:
        raise ValueError(f"{path.name}: no se encontró ningún namelist; ¿es un INPUT_*.def de QE?")
    return catalog


# --------------------------------------------------------------------------
# SIESTA: LaTeX manual (fdfentry environments)
# --------------------------------------------------------------------------

_FDF_ENTRY = re.compile(
    r"^\\begin\{fdf(?:entry|logicalT|logicalF)\}\{([^}]+)\}(?:\[([^\]]*)\])?(?:<([^\n]*))?",
    re.MULTILINE,
)
_FDF_END = re.compile(r"\\end\{fdf(?:entry|logicalT|logicalF)\}")
#: Physical dimensions: SIESTA reads these as "<number> <unit>".
_SIESTA_QUANTITIES = {"energy", "length", "time", "temperature", "force", "pressure",
                      "mass", "charge", "angle", "velocity", "bfield", "efield",
                      "dipole", "torque", "mominert"}
_SIESTA_KINDS = {"integer": "integer", "real": "real", "logical": "logical",
                 "string": "string", "block": "any", "list": "any",
                 "file": "string", "path": "string"}


def _siesta_name(raw: str) -> str:
    """The keyword as written in an fdf file: the manual's ``!`` index
    separators are dots (``Mesh!Cutoff`` -> ``Mesh.Cutoff``)."""
    return raw.strip().replace("!", ".")


def _latex_to_text(text: str) -> str:
    """Enough LaTeX removal to read a description or a default."""
    text = re.sub(r"(?m)(?<!\\)%.*$", "", text)
    text = re.sub(r"\\(?:begin|end)\{[^}]*\}", " ", text)
    text = re.sub(r"\\fdfvalue\{([^}]*)\}", lambda m: "valor de " + _siesta_name(m.group(1)),
                  text)
    text = re.sub(r"\\fdf\*?\{([^}]*)\}", lambda m: _siesta_name(m.group(1)), text)
    text = re.sub(r"\\(?:fdfindex|index|fdfdeprecates|fdfdepend|label|ref|cite)\*?\{[^}]*\}",
                  "", text)
    text = re.sub(r"\\(?:textbf|emph|textit|texttt|shell|program|file|code|url|mathrm|"
                  r"nonvalue|text)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*\\times\s*10\^\{?(-?\d+)\}?", r"\1e\2", text)
    text = re.sub(r"(?<![\d.])10\^\{?(-?\d+)\}?", r"1e\1", text)
    text = text.replace("\\note", "Nota:").replace("~", " ")
    text = re.sub(r"\\[,;:! ]", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", "", text)
    text = re.sub(r"[{}$]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _siesta_description(body: str) -> str:
    """The first paragraph of prose; option lists are left to ``choices``."""
    for paragraph in re.split(r"\n\s*\n", body):
        if "\\option" in paragraph or "fdfexample" in paragraph:
            continue
        text = _latex_to_text(paragraph)
        if len(text) > 3:
            return text[:600]
    return ""


def import_siesta_tex(path: str | Path) -> Catalog:
    """Read SIESTA's LaTeX manual (``siesta.tex``) into a catalogue.

    Each ``fdfentry`` (and ``fdflogicalT/F``) becomes a parameter: name (as
    written in an fdf file), type, default, the ``\\option[...]`` values when
    present, and the first paragraph of its text. Physical quantities
    (energies, lengths, temperatures...) get kind ``"quantity"``: a number
    followed by a unit, as SIESTA reads them.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    catalog = Catalog("siesta", sources=[f"manual {path.name}"])
    for match in _FDF_ENTRY.finditer(text):
        name, raw_kind, default = match.group(1), match.group(2) or "", match.group(3)
        default = re.sub(r">?\s*(%.*)?$", "", default.strip()) if default else None
        env = "logicalT" if "logicalT" in match.group(0) else (
            "logicalF" if "logicalF" in match.group(0) else "entry")
        end = _FDF_END.search(text, match.end())
        body = text[match.end(): end.start() if end else match.end() + 4000]
        dimensions = [k.strip().lower() for k in re.split(r"[,/]", raw_kind) if k.strip()]
        units = ""
        if env != "entry":
            kind, default = "logical", "true" if env == "logicalT" else "false"
        elif dimensions and dimensions[0] in _SIESTA_QUANTITIES:
            kind, units = "quantity", "/".join(dimensions)
        else:
            kind = _SIESTA_KINDS.get(dimensions[0], "any") if dimensions else "any"
        # "\option[SZ|minimal]": each alias is a valid spelling.
        options = tuple(dict.fromkeys(
            alias.strip() for o in re.findall(r"\\option\[([^\]]+)\]", body)
            for alias in _latex_to_text(o).split("|")))
        parameter = Parameter(
            code="siesta", name=_siesta_name(name), kind=kind,
            default=_latex_to_text(default) if default else None,
            choices=tuple(o for o in options if o) if kind in ("string", "any") else (),
            units=units, description=_siesta_description(body),
            source=f"manual {path.name}",
        )
        catalog.add(parameter)
        # Older spellings still accepted by SIESTA (DM.Tolerance for
        # SCF.DM.Tolerance) are indexed right after the entry.
        head = body[:400]
        for alias in re.findall(r"\\fdfindex\*\{([^}:]+)\}", head):
            alias = _siesta_name(alias)
            if alias.lower() != parameter.name.lower():
                catalog.add(Parameter(**{**asdict(parameter), "name": alias,
                                         "description": f"Alias de {parameter.name}. "
                                                        + parameter.description}),
                            replace=False)
    if not catalog.parameters:
        raise ValueError(f"{path.name}: no se encontró ninguna entrada fdfentry; ¿es siesta.tex?")
    return catalog


# --------------------------------------------------------------------------
# GPAW: introspection of the installed package
# --------------------------------------------------------------------------

def introspect_gpaw() -> Catalog:
    """The parameters the installed GPAW accepts, with their defaults.

    GPAW documents its input in Python, not in a manual; the calculator's
    parameter table is the authority. Raises ``ImportError`` without GPAW.
    """
    import gpaw

    table: Optional[dict] = None
    for module_name in ("gpaw.old.calculator", "gpaw.calculator"):
        try:
            module = __import__(module_name, fromlist=["GPAW"])
            table = dict(getattr(module.GPAW, "default_parameters", {}) or {})
            if table:
                break
        except ImportError:
            continue
    if not table:
        raise ImportError("Esta versión de GPAW no expone su tabla de parámetros.")
    catalog = Catalog("gpaw", sources=[f"GPAW {gpaw.__version__} instalado"])
    for name, default in table.items():
        kind = ("logical" if isinstance(default, bool) else "integer" if isinstance(default, int)
                else "real" if isinstance(default, float) else "any")
        catalog.add(Parameter(code="gpaw", name=name, kind=kind, default=repr(default),
                              description="", source=f"GPAW {gpaw.__version__}"))
    return catalog


def import_manual(path: str | Path) -> Catalog:
    """Pick the reader from the file: ``*.def`` -> QE, ``*.tex`` -> SIESTA."""
    path = Path(path)
    if path.suffix.lower() == ".def":
        return import_qe_helpdoc(path)
    if path.suffix.lower() == ".tex":
        return import_siesta_tex(path)
    raise ValueError(
        f"{path.name}: formato no reconocido. QE: INPUT_PW.def (PW/Doc/ de las fuentes); "
        "SIESTA: siesta.tex (Docs/ de las fuentes). GPAW se lee del paquete instalado."
    )
