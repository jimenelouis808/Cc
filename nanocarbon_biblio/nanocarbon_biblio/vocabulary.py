"""Human-readable vocabulary of everything the classifier recognises.

The search terms live in two places that serve different jobs: the query in
``queries/scopus.md`` **retrieves** the corpus, and the lexicon in
:mod:`nanocarbon_biblio.lexicons` **describes** what was retrieved. They are not
the same list and should not be — application and method terms belong in the
lexicon but would wreck the query's precision if added to its core blocks.

What they do need is to stay compatible on the facets that gate retrieval:
every dopant and every defect the classifier can label must be reachable by the
query, or the label can never fire. :func:`coverage_report` checks that, and
:func:`write_dictionary` renders the whole lexicon as a reference table — which
doubles as the description of the classification scheme that a Methods section
or a reviewer will ask for.

Generated from the lexicon itself, so the document cannot drift away from the
code the way a hand-maintained list would.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from .lexicons import FACETS

__all__ = [
    "readable_terms",
    "facet_dictionary",
    "write_dictionary",
    "coverage_report",
    "BY_DESIGN_ABSENT",
    "RETRIEVAL_FACETS",
]

#: Labels deliberately absent from the core query, with the reason. Without
#: this, a coverage check reports a scope decision as a defect every time.
BY_DESIGN_ABSENT: dict[str, str] = {
    "host/graphene_2d": (
        "El grafeno queda fuera del corpus núcleo por decisión de alcance "
        "(docs/PROTOCOL.md §2.1). La etiqueta existe para marcar híbridos "
        "DENTRO de registros de CNT ya recuperados, no para recuperarlos."
    ),
}

#: Facets whose terms must be reachable by the query, and the block that needs
#: to contain them.
RETRIEVAL_FACETS = {
    "host": "A (nanoestructura)",
    "dopant": "B (dopaje)",
    "defect": "B (defectos)",
}

_FACET_TITLES = {
    "dopant": "Dopantes",
    "defect": "Tipos de defecto",
    "doping_mode": "Modo de dopaje",
    "method_theory": "Métodos teóricos y computacionales",
    "method_experiment": "Métodos experimentales y de caracterización",
    "morphology": "Morfología y escala",
    "application": "Aplicaciones",
    "host": "Material huésped",
}

#: Regex glue and English connectives; never search terms in their own right.
_STOPWORDS = frozenset({
    "with", "and", "the", "for", "of", "in", "on", "to", "by", "or",
    "were", "was", "is", "are", "we", "ies", "ous", "ed", "ing", "ant",
})

#: Words so common across this topic that their presence proves nothing: every
#: label would look covered if they counted.
_UBIQUITOUS = frozenset({
    "doped", "doping", "dopant", "dopants", "defect", "defects", "carbon",
    "nanotube", "nanotubes", "nano", "material", "materials", "site", "sites",
})

_ALTERNATION = re.compile(r"\(\?:([^()]*?)\)")
_CHARCLASS = re.compile(r"\[[^\]]*\]")
_QUANTIFIER = re.compile(r"\{\d+(?:,\d*)?\}")


def _expand_groups(text: str, limit: int) -> list[str]:
    """Recursively replace each innermost ``(?:a|b)`` group with its options.

    Recursion rather than a loop because the lexicon nests groups:
    ``nano(?:tube|fib(?:er|re)|coil)``. Expanding the innermost group in place
    and re-scanning the whole rebuilt string handles any depth; consuming the
    text left to right does not, and leaves unbalanced fragments behind.
    """
    match = _ALTERNATION.search(text)
    if not match:
        return [text]
    options = match.group(1).split("|")
    head, tail = text[: match.start()], text[match.end():]
    if tail.startswith("?"):          # the whole group is optional
        tail = tail[1:]
        options.append("")
    out: list[str] = []
    for option in options:
        for expanded in _expand_groups(head + option + tail, limit):
            out.append(expanded)
            if len(out) >= limit:
                return out
    return out


def _expand(pattern: str, limit: int = 40) -> list[str]:
    """Expand a simple regex into the literal strings it can match.

    Handles what the lexicon actually uses: nested non-capturing alternations,
    optional separator classes, and top-level ``|``. The point is to turn
    ``dop(?:ed|ing|ant)`` into *doped, doping, dopant* rather than into the
    fragments ``dop``, ``ed``, ``ing``, ``ant``.

    ``limit`` caps the expansion so a pathological pattern cannot explode; the
    lexicon's own patterns stay far below it.
    """
    text = pattern
    # Separator classes become a real space before the generic class strip.
    text = re.sub(r"\[\\s\\?-\]\??", " ", text)
    text = text.replace(r"\s", " ")
    # A short letter class is a spelling choice, not punctuation: `sul[fp]h?ur`
    # is sulfur/sulphur. Stripping it to a space produced "sul hur".
    text = re.sub(r"\[([a-zA-Z]{2,4})\]",
                  lambda m: "(?:" + "|".join(m.group(1)) + ")", text)
    # An optional single letter is also a spelling choice: `sul(?:f|p)h?ur`
    # is sulfur or sulphur. No guard is needed against group syntax, since
    # `(?:` and `(?i:` never put `?` directly after a letter.
    text = re.sub(r"([a-zA-Z])\?", r"(?:\1|)", text)
    text = _CHARCLASS.sub(" ", text)
    text = _QUANTIFIER.sub("", text)
    text = text.replace(r"\b", "").replace("(?i:", "(?:")

    out: list[str] = []
    for variant in _expand_groups(text, limit):
        for piece in variant.split("|"):
            piece = re.sub(r"[()\[\]{}?*+^$\\]", "", piece)
            piece = re.sub(r"\s+", " ", piece).strip(" -.:")
            if len(piece) < 3 or piece.lower() in _STOPWORDS:
                continue
            if not re.search(r"[a-zA-Z]{2}", piece):
                continue
            if piece not in out:
                out.append(piece)
    return out


def readable_terms(patterns: list[tuple]) -> list[str]:
    """Turn a label's regex patterns into the terms a human would recognise.

    For reading, and for checking a query against — never for matching, which
    always uses the compiled patterns in :mod:`nanocarbon_biblio.lexicons`.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for pattern, _guard, _case_sensitive in patterns:
        for term in _expand(pattern):
            lowered = term.lower()
            if lowered not in seen:
                seen.add(lowered)
                terms.append(lowered)
    return terms


def facet_dictionary() -> dict[str, dict[str, list[str]]]:
    """Return ``{facet: {label: [terms]}}`` for the whole lexicon."""
    return {
        facet: {label: readable_terms(specs) for label, specs in labels.items()}
        for facet, labels in FACETS.items()
    }


def _query_vocabulary(query_text: str) -> tuple[set[str], list[str]]:
    """Split a query into its literal tokens and its truncation prefixes.

    Scopus and WoS queries are written with wildcards (``vacanc*``), hyphenated
    compounds (``defect-free``) and proximity operators (``nitrogen W/3
    doped``), so a literal substring test against the query text reports almost
    every label as missing. Hyphenated tokens are indexed both whole and split,
    because the lexicon and the query disagree about hyphens constantly.
    """
    tokens: set[str] = set()
    prefixes: list[str] = []
    for raw in re.findall(r"[a-z][a-z0-9\-]*\*?", query_text.lower()):
        if raw.endswith("*"):
            stem = raw[:-1].strip("-")
            if len(stem) >= 3:
                prefixes.append(stem)
            continue
        for piece in [raw, *raw.split("-")]:
            piece = piece.strip("-")
            if len(piece) >= 3:
                tokens.add(piece)
    return tokens, prefixes


def _word_in_query(word: str, tokens: set[str], prefixes: list[str]) -> bool:
    """True when ``word`` is retrievable by the query's tokens or wildcards.

    Three ways to match, because truncation runs in both directions: the query
    may hold ``vacanc*`` for the lexicon's ``vacancy``, or the full word
    ``engineering`` where the lexicon stores the stem ``engineer``.
    """
    if word in tokens:
        return True
    if any(word.startswith(stem) for stem in prefixes):
        return True
    return len(word) >= 4 and any(token.startswith(word) for token in tokens)


def coverage_report(query_text: str) -> list[dict[str, object]]:
    """Check that the query can retrieve what the classifier can label.

    Only the retrieval facets are checked (:data:`RETRIEVAL_FACETS`): an
    ``application`` label missing from the core query is correct, because
    applications describe the retrieved corpus rather than select it.

    A label counts as covered when any **distinctive word** of any of its terms
    is reachable from the query — present as a token, or matched by one of its
    truncation wildcards. Words shared across the whole topic (``doped``,
    ``defect``, ``carbon``) are ignored, since they match everything and would
    make every label look covered.

    Deliberately permissive about *how* a term is reachable, because the query
    uses proximity operators this cannot model. Its job is to catch a dopant
    added to the lexicon and forgotten in the query, not to audit query syntax.
    """
    tokens, prefixes = _query_vocabulary(query_text)
    rows: list[dict[str, object]] = []
    dictionary = facet_dictionary()
    for facet in RETRIEVAL_FACETS:
        for label, terms in dictionary[facet].items():
            matched: list[str] = []
            for term in terms:
                # The query may write a compound solid ("codoped") where the
                # lexicon separates it ("co doped"); try the joined form too.
                joined = re.sub(r"[\s\-]+", "", term)
                if len(joined) >= 5 and _word_in_query(joined, tokens, prefixes):
                    matched.append(term)
                    continue
                words = [w for w in re.split(r"[\s\-]+", term) if len(w) >= 3]
                distinctive = [w for w in words if w not in _UBIQUITOUS]
                # When every word is topic-ubiquitous — "carbon nanotube" in a
                # review about carbon nanotubes — demand all of them instead of
                # none: dropping them all would make the label unmatchable.
                required = distinctive or words
                if required and all(
                    _word_in_query(w, tokens, prefixes) for w in required
                ):
                    matched.append(term)
            key = f"{facet}/{label}"
            rows.append({
                "facet": facet,
                "label": label,
                "covered": bool(matched) or key in BY_DESIGN_ABSENT,
                "matched": matched[:5],
                "n_terms": len(terms),
                "by_design": BY_DESIGN_ABSENT.get(key, ""),
            })
    return rows


def write_dictionary(path: str | Path, *, query_text: str = "") -> Path:
    """Write the vocabulary reference to ``path`` as Markdown.

    With ``query_text``, a coverage section is appended so the document also
    records whether the query can reach every dopant and defect label.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionary = facet_dictionary()
    n_terms = sum(len(terms) for labels in dictionary.values() for terms in labels.values())
    n_labels = sum(len(labels) for labels in dictionary.values())

    lines: list[str] = [
        "# Diccionario del clasificador",
        "",
        "> **Generado automáticamente** desde `nanocarbon_biblio/lexicons.py` con",
        "> `python -m nanocarbon_biblio.cli vocab`. No lo edites a mano: cambia el",
        f"> léxico y vuelve a generarlo. Última generación: {date.today().isoformat()}.",
        "",
        f"{len(dictionary)} facetas · {n_labels} etiquetas · {n_terms} términos.",
        "",
        "## Para qué sirve",
        "",
        "1. **Al escribir o revisar la consulta** (`queries/scopus.md`): si un",
        "   dopante o un defecto está aquí pero no en la consulta, el clasificador",
        "   nunca podrá etiquetarlo, porque esos registros no se recuperarán.",
        "2. **En Métodos**: es la descripción del esquema de clasificación que un",
        "   revisor va a pedir. Publícalo como material suplementario.",
        "",
        "Esta lista **no es la consulta**. La consulta recupera; el léxico describe",
        "lo recuperado. Los términos de aplicación y de método están aquí a",
        "propósito y **no** deben añadirse a los bloques núcleo de la consulta:",
        "arruinarían su precisión.",
        "",
        "Los términos son una lectura legible de los patrones, no los patrones",
        "mismos. La clasificación usa siempre las expresiones compiladas del",
        "léxico, con sus guardas de desambiguación.",
        "",
        "Las variantes ortográficas se expanden mecánicamente, así que alguna",
        "combinación imposible se cuela (`sul(?:f|p)h?ur` da *sulfur* y *sulphur*,",
        "pero también *sulpur*). No es un error: el patrón acepta las cuatro y la",
        "tabla las lista todas.",
        "",
    ]

    for facet, labels in dictionary.items():
        lines.append(f"## {_FACET_TITLES.get(facet, facet)} — `{facet}`")
        lines.append("")
        if facet in RETRIEVAL_FACETS:
            lines.append(
                f"*Faceta de recuperación: sus términos tienen que estar en el "
                f"bloque {RETRIEVAL_FACETS[facet]} de la consulta.*"
            )
            lines.append("")
        lines.append("| Etiqueta | Términos que reconoce |")
        lines.append("|---|---|")
        for label, terms in labels.items():
            shown = ", ".join(f"`{t}`" for t in terms) or "—"
            lines.append(f"| **{label}** | {shown} |")
        lines.append("")

    if query_text:
        rows = coverage_report(query_text)
        uncovered = [r for r in rows if not r["covered"]]
        lines += [
            "## Cobertura de la consulta",
            "",
            "¿Puede la consulta recuperar lo que el clasificador sabe etiquetar?",
            "",
            "| Faceta | Etiqueta | ¿Cubierta? | Términos encontrados en la consulta |",
            "|---|---|---|---|",
        ]
        for row in rows:
            mark = "por diseño" if row["by_design"] else ("sí" if row["covered"] else "**NO**")
            examples = ", ".join(f"`{t}`" for t in row["matched"]) or "—"
            lines.append(f"| {row['facet']} | {row['label']} | {mark} | {examples} |")
        lines.append("")
        if uncovered:
            lines += ["### Hay que arreglar", ""]
            for row in uncovered:
                lines.append(
                    f"- **{row['facet']}/{row['label']}**: ninguno de sus "
                    f"{row['n_terms']} términos aparece en la consulta. O lo añades "
                    "a la consulta, o quitas la etiqueta del léxico."
                )
            lines.append("")
        else:
            lines += [
                "Todas las etiquetas de recuperación están cubiertas, o declaradas "
                "ausentes por diseño.", "",
            ]
        for key, reason in BY_DESIGN_ABSENT.items():
            lines += [f"> **{key}** — {reason}", ""]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
