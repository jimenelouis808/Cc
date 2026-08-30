"""Streamlit GUI for the Scopus + WoS → biblioshiny pipeline.

Run with::

    streamlit run nanocarbon_biblio/app.py

Seven steps, left to right, each one a tab. State lives in ``st.session_state``
so you can move back and forth without re-loading a 30 000-record corpus.

The GUI is a front end over the library, not a reimplementation of it: every
button calls the same function the CLI does, so anything you do here is
reproducible from a script. That matters — a review whose corpus was assembled
by clicking is not reproducible, so the last tab writes a ``manifest.json``
recording exactly which parameters were used.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Allow `streamlit run nanocarbon_biblio/app.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanocarbon_biblio.classify import (  # noqa: E402
    classify_all, crosstab, sample_for_validation, to_dataframe,
)
from nanocarbon_biblio.dedupe import DedupeResult, deduplicate, overlap_table  # noqa: E402
from nanocarbon_biblio.demo import DemoConfig, generate_demo_corpus  # noqa: E402
from nanocarbon_biblio.exporters import export_bundle  # noqa: E402
from nanocarbon_biblio.loaders import load_any, load_directory  # noqa: E402
from nanocarbon_biblio.thesaurus import (  # noqa: E402
    SEED_GROUPS, suggest_synonyms, write_thesaurus,
)

st.set_page_config(page_title="nanocarbon_biblio", page_icon="⬡", layout="wide")

_STATE_DEFAULTS = {
    "records": None,
    "dedupe_result": None,
    "labels": None,
    "thesaurus_groups": None,
    "excluded_keys": set(),
}
for _key, _value in _STATE_DEFAULTS.items():
    st.session_state.setdefault(_key, _value)


def _project_root() -> Path:
    """Repository root, used to default every path box."""
    return Path(__file__).resolve().parent.parent


# Chart palette: the three leading categorical slots (blue, orange, aqua) and a
# single-hue blue sequential ramp. Validated for colour-vision deficiency —
# worst adjacent pair ΔE 9.2 (deutan), 27.6 (normal vision). Aqua sits below 3:1
# against a light surface, so every chart that uses it also offers a table view.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQUENTIAL_BLUE = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]


def _series_colors(columns: list[str]) -> list[str]:
    """Assign categorical hues in fixed slot order, never cycled.

    Colour follows the entity, so the mapping is by sorted column name: a filter
    that removes one series must not repaint the survivors.
    """
    return [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(columns))]


def _run_rscript(script: str, cwd: Path, timeout: int = 1800) -> tuple[int, str]:
    """Run an R script from the GUI and return ``(returncode, combined output)``.

    R is invoked as a subprocess rather than through a Python-R bridge on
    purpose: the same command the user would type in a terminal, so a failure
    here reproduces there. stderr is folded into stdout because bibliometrix
    writes its progress messages to stderr.
    """
    executable = shutil.which("Rscript")
    if not executable:
        return 127, (
            "Rscript no está en el PATH. Instala R y luego:\n"
            "  Rscript R/install_deps.R"
        )
    try:
        completed = subprocess.run(
            [executable, script], cwd=str(cwd), capture_output=True,
            text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, f"{script} superó el límite de {timeout} s y fue interrumpido."
    return completed.returncode, (completed.stdout or "") + (completed.stderr or "")


st.title("⬡ nanocarbon_biblio")
st.caption(
    "Defectos y dopaje en nanoestructuras de carbono 1D — "
    "pipeline Scopus + WoS → bibliometrix / biblioshiny"
)

tabs = st.tabs([
    "1 · Cargar", "2 · Deduplicar", "3 · Clasificar", "4 · Cribado",
    "5 · Tesauro", "6 · Exportar a R", "7 · Validación",
])

# ---------------------------------------------------------------- 1 · Cargar
with tabs[0]:
    st.subheader("Cargar exportaciones crudas")
    st.markdown(
        "Formatos aceptados: **Scopus CSV**, **WoS plain text (tagged)**, "
        "**WoS tab-delimited**. El formato se detecta por contenido, no por extensión.\n\n"
        "> ⚠️ La exportación de Scopus debe incluir el campo **References**, y la de WoS "
        "debe ser **Full Record and Cited References**. Sin eso no hay co-citación ni RPYS, "
        "y no se puede añadir después."
    )
    mode = st.radio(
        "Origen",
        ["Carpeta del proyecto", "Subir ficheros", "Corpus de demostración"],
        horizontal=True,
    )

    if mode == "Carpeta del proyecto":
        directory = st.text_input("Carpeta", value=str(_project_root() / "data" / "raw"))
        if st.button("Cargar carpeta", type="primary"):
            with st.spinner("Leyendo…"):
                buffer = io.StringIO()
                original_stdout, sys.stdout = sys.stdout, buffer
                try:
                    records = load_directory(directory)
                finally:
                    sys.stdout = original_stdout
            st.session_state.records = records
            st.session_state.dedupe_result = None
            st.code(buffer.getvalue() or "(sin salida)", language="text")
    elif mode == "Subir ficheros":
        uploads = st.file_uploader(
            "Ficheros", accept_multiple_files=True, type=["csv", "txt", "tsv"]
        )
        if uploads and st.button("Cargar ficheros", type="primary"):
            tmpdir = Path(st.session_state.get("_tmpdir") or Path.cwd() / ".upload_cache")
            tmpdir.mkdir(parents=True, exist_ok=True)
            st.session_state["_tmpdir"] = str(tmpdir)
            records = []
            for upload in uploads:
                target = tmpdir / upload.name
                target.write_bytes(upload.getvalue())
                try:
                    loaded = load_any(target)
                except Exception as exc:  # noqa: BLE001 - surface, don't crash the app
                    st.error(f"{upload.name}: {exc}")
                    continue
                st.write(f"`{upload.name}` → {len(loaded)} registros")
                records.extend(loaded)
            st.session_state.records = records
            st.session_state.dedupe_result = None
    else:
        st.warning(
            "**Datos sintéticos.** Sirven para aprender la interfaz y probar el pipeline "
            "sin gastar cuota de exportación. Ningún número que salga de aquí vale para "
            "el manuscrito."
        )
        col_a, col_b = st.columns(2)
        n_works = col_a.number_input("Obras a generar", 100, 5000, 1200, 100)
        seed = col_b.number_input("Semilla", 1, 10**9, 20260830)
        if st.button("Generar y cargar corpus de demostración", type="primary"):
            demo_dir = _project_root() / "data" / "raw" / "demo"
            with st.spinner("Generando…"):
                summary = generate_demo_corpus(
                    demo_dir, DemoConfig(n_works=int(n_works), seed=int(seed))
                )
                st.session_state.records = load_directory(demo_dir)
                st.session_state.dedupe_result = None
                st.session_state.labels = None
                st.session_state["_demo_truth"] = summary["true_overlap"]
            st.success(
                f"{summary['n_works']} obras → {summary['n_scopus_records']} registros Scopus "
                f"+ {summary['n_wos_records']} de WoS, en {demo_dir}"
            )
            st.caption(
                "Solapamiento real generado: "
                f"{summary['true_overlap']}. La pestaña 2 debería recuperarlo casi exacto — "
                "es una comprobación de que la deduplicación funciona."
            )

    records = st.session_state.records
    if records:
        summary = pd.DataFrame([
            {"source": r.source, "year": r.year, "has_abstract": r.has_abstract(),
             "has_doi": bool(r.doi_key), "has_refs": bool(r.n_references)}
            for r in records
        ])
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Registros cargados", len(records))
        col_b.metric("Con DOI", f"{summary.has_doi.mean():.0%}")
        col_c.metric("Con resumen", f"{summary.has_abstract.mean():.0%}")
        col_d.metric("Con referencias", f"{summary.has_refs.mean():.0%}")
        if summary.has_refs.mean() < 0.5:
            st.error(
                "Menos de la mitad de los registros traen referencias citadas. "
                "Re-exporta incluyendo las referencias antes de seguir."
            )
        by_year_source = summary.groupby(["year", "source"]).size().unstack(fill_value=0)
        st.bar_chart(by_year_source, color=_series_colors(list(by_year_source.columns)))

# ----------------------------------------------------------- 2 · Deduplicar
with tabs[1]:
    st.subheader("Deduplicar Scopus ∪ WoS")
    if not st.session_state.records:
        st.info("Carga registros en la pestaña 1.")
    else:
        col_a, col_b = st.columns(2)
        threshold = col_a.slider(
            "Umbral de similitud de título", 80.0, 100.0, 92.0, 0.5,
            help="token_set_ratio. Por debajo de ~88 empieza a fusionar artículos distintos "
                 "del mismo grupo (parte I / parte II).",
        )
        window = col_b.slider(
            "Ventana de años", 0, 3, 1,
            help="Absorbe el desfase entre año online-first y año de número.",
        )
        if st.button("Deduplicar", type="primary"):
            with st.spinner("Comparando títulos…"):
                st.session_state.dedupe_result = deduplicate(
                    st.session_state.records, title_threshold=threshold, year_window=window
                )
                st.session_state.labels = None

        result = st.session_state.dedupe_result
        if result:
            overlap = overlap_table(result)
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Únicos", overlap["total_unique"])
            col_b.metric("Duplicados eliminados", overlap["duplicates_removed"])
            col_c.metric("Solo Scopus", overlap.get("scopus_only", 0))
            col_d.metric("Solo WoS", overlap.get("wos_only", 0))
            st.metric("En ambas bases", overlap.get("both", 0))
            total = max(1, overlap["total_unique"])
            exclusive = (overlap.get("scopus_only", 0) + overlap.get("wos_only", 0)) / total
            st.markdown(
                f"**{exclusive:.1%} de los registros únicos son exclusivos de una sola base.** "
                "Este número justifica por sí solo buscar en las dos, y va en Métodos."
            )
            truth = st.session_state.get("_demo_truth")
            if truth:
                comparison = pd.DataFrame({
                    "real (generado)": truth,
                    "recuperado": {k: overlap.get(k, 0) for k in truth},
                })
                comparison["error"] = comparison["recuperado"] - comparison["real (generado)"]
                st.markdown("#### Validación contra la verdad conocida del corpus de demostración")
                st.dataframe(comparison, width="stretch")
            st.json(overlap)

# ------------------------------------------------------------ 3 · Clasificar
with tabs[2]:
    st.subheader("Clasificar por dopante, defecto, método, morfología y aplicación")
    result = st.session_state.dedupe_result
    if not result:
        st.info("Deduplica primero (pestaña 2).")
    else:
        if st.button("Clasificar", type="primary"):
            with st.spinner("Aplicando reglas…"):
                classify_all(result.unique)
                st.session_state.labels = to_dataframe(result.unique)

        labels = st.session_state.labels
        if labels is not None:
            st.markdown("#### Tipo de estudio (RQ2)")
            counts = labels.study_type.value_counts().rename_axis("study_type").reset_index(name="n")
            base = alt.Chart(counts).encode(
                x=alt.X("n:Q", title="documentos"),
                y=alt.Y("study_type:N", sort="-x", title=None),
                tooltip=["study_type", "n"],
            )
            st.altair_chart(
                (
                    base.mark_bar(color=SERIES_COLORS[0], cornerRadiusEnd=4, height=18)
                    + base.mark_text(align="left", dx=6, color="#52514e").encode(text="n:Q")
                ).properties(height=150),
                width="stretch",
            )

            st.markdown("#### Cuota anual por tipo de estudio")
            min_per_year = st.slider(
                "Mínimo de documentos por año", 1, 40, 5,
                help="Una cuota calculada sobre 1 o 2 documentos es ruido: ese año valdría "
                     "0 % o 100 % por azar. Los años por debajo del umbral se ocultan.",
            )
            evolution = (
                labels.dropna(subset=["year"])
                .assign(year=lambda d: d.year.astype(int))
                .groupby(["year", "study_type"]).size().unstack(fill_value=0)
            )
            totals = evolution.sum(axis=1)
            dense = evolution[totals >= min_per_year]
            if dense.empty:
                st.warning("Ningún año alcanza el umbral. Bájalo o carga más registros.")
            else:
                share = dense.div(dense.sum(axis=1), axis=0)
                # Lines, not stacked areas: the question is "is the combined
                # share rising?", which is a trend read. Three translucent
                # overlapping areas turn to mud, and a stacked band hides the
                # middle series' slope, which is the one that matters.
                st.line_chart(share, color=_series_colors(list(share.columns)))
                hidden = int((totals < min_per_year).sum())
                if hidden:
                    st.caption(
                        f"{hidden} año(s) ocultos por tener menos de {min_per_year} documentos."
                    )
            st.caption(
                "La cuota de estudios *combined* a lo largo del tiempo es la medida directa "
                "del acoplamiento teoría↔experimento (RQ2 del protocolo)."
            )
            with st.expander("Recuento absoluto por año (gráfico y tabla)"):
                st.bar_chart(evolution, color=_series_colors(list(evolution.columns)))
                st.dataframe(evolution, width="stretch")

            st.markdown("#### Matriz dopante × aplicación")
            col_a, col_b = st.columns(2)
            row_facet = col_a.selectbox("Filas", ["dopant", "defect", "morphology", "doping_mode"], 0)
            col_facet = col_b.selectbox("Columnas", ["application", "method_experiment", "method_theory", "morphology"], 0)
            matrix = crosstab(labels, rows=row_facet, cols=col_facet)
            if matrix.empty:
                st.warning("Sin celdas con datos para esa combinación.")
            else:
                long = matrix.reset_index().melt(id_vars=row_facet, var_name=col_facet, value_name="n")
                st.altair_chart(
                    alt.Chart(long).mark_rect(stroke="#fcfcfb", strokeWidth=2).encode(
                        x=alt.X(f"{col_facet}:N", title=None,
                                axis=alt.Axis(labelAngle=-90, labelLimit=200,
                                              labelOverlap=False, labelPadding=6)),
                        y=alt.Y(f"{row_facet}:N", title=None,
                                axis=alt.Axis(labelLimit=200)),
                        color=alt.Color(
                            "n:Q", title="docs",
                            scale=alt.Scale(range=SEQUENTIAL_BLUE),
                        ),
                        tooltip=[row_facet, col_facet, "n"],
                    ).properties(height=36 * max(2, len(matrix))),
                    width="stretch",
                )
                st.caption(
                    "Las celdas vacías **con señal teórica y sin señal experimental** son tu "
                    "sección de perspectivas. Ojo: un documento con dos dopantes y dos "
                    "aplicaciones cuenta en cuatro celdas — dilo en el pie de figura."
                )
                st.dataframe(matrix, width="stretch")

# --------------------------------------------------------------- 4 · Cribado
with tabs[3]:
    st.subheader("Cribado y exclusiones")
    labels = st.session_state.labels
    if labels is None:
        st.info("Clasifica primero (pestaña 3).")
    else:
        col_a, col_b, col_c = st.columns(3)
        years = labels.year.dropna()
        year_range = col_a.slider(
            "Años", int(years.min()), int(years.max()),
            (int(years.min()), int(years.max())),
        ) if len(years) else (0, 9999)
        types = sorted(t for t in labels.doc_type.dropna().unique() if t)
        keep_types = col_b.multiselect("Tipos de documento", types, default=types)
        require_topic = col_c.checkbox(
            "Exigir dopante o defecto detectado", value=True,
            help="Descarta registros donde ninguna regla de dopaje ni de defectos disparó.",
        )
        drop_ambiguous = st.checkbox(
            "Apartar los marcados como 'dopante en huésped no-carbono'", value=False,
            help="Falso positivo típico: 'TiO2 dopado con N soportado en MWCNT'. "
                 "Recomendado: NO descartarlos automáticamente, revisarlos a mano.",
        )

        mask = pd.Series(True, index=labels.index)
        if len(years):
            mask &= labels.year.between(*year_range) | labels.year.isna()
        if keep_types:
            mask &= labels.doc_type.isin(keep_types) | labels.doc_type.isna() | labels.doc_type.eq("")
        if require_topic:
            mask &= labels.has_dopant.astype(bool) | labels.has_defect.astype(bool)
        if drop_ambiguous:
            mask &= ~labels.dopant_host_ambiguous.astype(bool)

        kept = labels[mask]
        st.metric("Registros que pasan el cribado", f"{len(kept)} / {len(labels)}")
        st.session_state.excluded_keys = set(labels.loc[~mask, "key"])

        ambiguous = labels[labels.dopant_host_ambiguous.astype(bool)]
        st.markdown(f"#### Revisión manual obligatoria — {len(ambiguous)} registros marcados")
        st.caption(
            "El heurístico es ruidoso a propósito, en la dirección segura. Descarga, "
            "codifica a mano y reincorpora la decisión."
        )
        st.dataframe(
            ambiguous[["key", "year", "title", "dopant", "application"]].head(300),
            width="stretch", hide_index=True,
        )
        st.download_button(
            "Descargar marcados para revisión (CSV)",
            ambiguous.to_csv(index=False).encode("utf-8"),
            "revision_manual.csv", "text/csv",
        )

# --------------------------------------------------------------- 5 · Tesauro
with tabs[4]:
    st.subheader("Tesauro de keywords para biblioshiny")
    st.markdown(
        "Formato de biblioshiny: un grupo por línea, términos separados por `;`, "
        "**el primer término es el que sustituye a los demás**. "
        "Se carga en biblioshiny en *Data → Filters → synonyms*."
    )
    if not st.session_state.records:
        st.info("Carga registros en la pestaña 1.")
    else:
        col_a, col_b = st.columns(2)
        min_count = col_a.number_input("Frecuencia mínima del término", 2, 200, 5)
        fuzz_threshold = col_b.slider("Umbral de agrupación", 70.0, 100.0, 88.0, 1.0)
        source_records = (
            st.session_state.dedupe_result.unique if st.session_state.dedupe_result
            else st.session_state.records
        )
        if st.button("Sugerir grupos de sinónimos"):
            with st.spinner("Agrupando keywords…"):
                st.session_state.thesaurus_groups = suggest_synonyms(
                    source_records, min_count=int(min_count), threshold=fuzz_threshold
                )
        groups = st.session_state.thesaurus_groups
        if groups is not None:
            st.warning(
                f"{len(groups)} grupos sugeridos. **Revísalos uno a uno**: `n-doped` y "
                "`p-doped` puntúan altísimo entre sí y son opuestos."
            )
            editable = "\n".join(";".join(g) for g in groups)
            edited = st.text_area("Grupos (editable)", editable, height=320)
            out_path = st.text_input(
                "Guardar en", str(_project_root() / "queries" / "thesaurus.txt")
            )
            if st.button("Guardar tesauro", type="primary"):
                reviewed = [
                    [t.strip() for t in line.split(";") if t.strip()]
                    for line in edited.splitlines() if line.strip()
                ]
                path = write_thesaurus(reviewed, out_path, include_seed=True)
                st.success(f"Escrito {path} ({len(SEED_GROUPS)} grupos semilla + {len(reviewed)} revisados)")

# --------------------------------------------------------- 6 · Exportar a R
with tabs[5]:
    st.subheader("Exportar el corpus para bibliometrix / biblioshiny")
    result = st.session_state.dedupe_result
    if not result:
        st.info("Deduplica y clasifica primero.")
    else:
        outdir = st.text_input("Carpeta de salida", str(_project_root() / "data" / "processed"))
        note = st.text_area("Nota de consulta (queda en manifest.json)", "", height=80)
        apply_screening = st.checkbox(
            "Aplicar el cribado de la pestaña 4", value=True,
            help="Excluye los registros que no pasaron los filtros.",
        )
        if st.button("Exportar", type="primary"):
            export_result = result
            if apply_screening and st.session_state.excluded_keys:
                excluded = st.session_state.excluded_keys
                kept_records = [r for r in result.unique if r.key not in excluded]
                export_result = DedupeResult(
                    unique=kept_records, clusters=result.clusters, n_input=result.n_input
                )
            manifest = export_bundle(export_result, outdir, query_note=note)
            st.success(f"Escrito en {outdir}")
            st.json(manifest)
            st.download_button(
                "Descargar manifest.json",
                json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
                "manifest.json", "application/json",
            )

            st.session_state["_last_manifest"] = manifest

        manifest = st.session_state.get("_last_manifest")
        if manifest:
            st.markdown("#### Flujo PRISMA")
            prisma = manifest["prisma"]
            overlap = manifest["overlap"]
            flow = pd.DataFrame([
                ("Registros identificados en Scopus + WoS", prisma["records_identified"]),
                ("Duplicados eliminados", -prisma["duplicates_removed"]),
                ("Registros cribados", prisma["records_screened"]),
                ("  · solo Scopus", overlap.get("scopus_only", 0)),
                ("  · solo WoS", overlap.get("wos_only", 0)),
                ("  · en ambas bases", overlap.get("both", 0)),
                ("Sin resumen (no clasificables)", prisma["records_without_abstract"]),
                ("Marcados para revisión manual", prisma["flagged_host_ambiguous"]),
            ], columns=["Etapa", "n"])
            st.dataframe(flow, width="stretch", hide_index=True)
            st.download_button(
                "Descargar el flujo PRISMA (CSV)",
                flow.to_csv(index=False).encode("utf-8"),
                "prisma_flow.csv", "text/csv",
            )
            st.caption(
                "Estas cifras van al diagrama PRISMA 2020 tal cual. El número de "
                "excluidos en el cribado manual lo añades tú tras revisar los marcados."
            )

        st.divider()
        st.markdown("#### Siguiente paso, en R")

        root = _project_root()
        has_r = shutil.which("Rscript") is not None
        if not has_r:
            st.warning(
                "`Rscript` no está en el PATH, así que no puedo lanzarlo desde aquí. "
                "Instala R y ejecuta `Rscript R/install_deps.R`; los botones se activarán solos."
            )

        col_a, col_b = st.columns(2)
        if col_a.button("Construir M.rds (00_build_M.R)", disabled=not has_r, type="primary"):
            with st.spinner("convert2df + mergeDbSources + join de etiquetas…"):
                code, output = _run_rscript("R/00_build_M.R", root)
            (st.success if code == 0 else st.error)(
                "M.rds construido." if code == 0 else f"R terminó con código {code}."
            )
            st.code(output or "(sin salida)", language="text")
            report = root / "data" / "processed" / "join_report.txt"
            if code == 0 and report.exists():
                st.text(report.read_text(encoding="utf-8"))

        if col_b.button("Análisis guionizados (01_core_analyses.R)", disabled=not has_r):
            with st.spinner("Ejecutando análisis…"):
                code, output = _run_rscript("R/01_core_analyses.R", root)
            (st.success if code == 0 else st.error)(
                "Análisis terminados, salidas en results/." if code == 0
                else f"R terminó con código {code}."
            )
            st.code(output or "(sin salida)", language="text")
            failed = root / "results" / "failed_steps.txt"
            if failed.exists():
                st.warning("Pasos que fallaron:\n\n" + failed.read_text(encoding="utf-8"))

        st.markdown("##### Y para abrir biblioshiny")
        st.code("Rscript R/launch_biblioshiny.R", language="bash")
        st.caption(
            "biblioshiny abre su propio servidor, así que se lanza desde la terminal, "
            "no desde aquí. Dentro: *Data → Load bibliometrix file* → "
            "`data/processed/M.rds`. **No** uses *Import raw files*: perderías las "
            "etiquetas de Python."
        )

# ------------------------------------------------------------ 7 · Validación
with tabs[6]:
    st.subheader("Muestra estratificada para validar las reglas")
    labels = st.session_state.labels
    if labels is None:
        st.info("Clasifica primero (pestaña 3).")
    else:
        col_a, col_b = st.columns(2)
        size = col_a.number_input("Tamaño de la muestra", 20, 500, 100, 10)
        stratify = col_b.selectbox("Estratificar por", ["study_type", "dopant", "morphology"], 0)
        sample = sample_for_validation(labels, n=int(size), stratify=stratify)
        st.dataframe(
            sample[["key", "year", "title", "study_type", "dopant", "defect", "application"]],
            width="stretch", hide_index=True,
        )
        blank = sample.copy()
        for column in ("manual_study_type", "manual_dopant", "manual_relevant", "coder", "notes"):
            blank[column] = ""
        st.download_button(
            "Descargar hoja de codificación (CSV)",
            blank.to_csv(index=False).encode("utf-8"),
            f"validacion_{stratify}_{size}.csv", "text/csv", type="primary",
        )
        st.markdown(
            "Codifica a mano, calcula **kappa de Cohen** contra la salida de las reglas y "
            "repórtalo en Métodos. Un review que dice *«clasificación validada sobre 100 "
            "registros estratificados, κ = 0.87»* está en otra categoría de credibilidad."
        )
