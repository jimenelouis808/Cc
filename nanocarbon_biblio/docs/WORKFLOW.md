# Workflow completo

Python hace lo que Python hace bien (parsear, deduplicar, clasificar con reglas,
minar texto). R hace lo que R hace bien (bibliometrix y todo el aparato de
science mapping). El puente entre los dos está diseñado para **no perder las
referencias citadas**, que es donde se rompen la mayoría de los pipelines
mixtos que se ven por ahí.

```
Scopus  ─┐
         ├─► data/raw/ ─► [Python] ─► data/processed/ ─► [R] ─► M.rds ─► biblioshiny
WoS     ─┘                 dedup            formato          convert2df
                        clasificar          NATIVO         mergeDbSources
                          tesauro         + labels.csv       join labels
```

---

## Paso 0 — Instalar

```bash
cd nanocarbon_biblio
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
Rscript R/install_deps.R
```

## Paso 1 — Calibrar la consulta (no te lo saltes)

1. Arma el *gold standard* de 40–60 artículos:
   ```bash
   cp queries/gold_standard_template.csv queries/gold_standard.csv
   ```
   Complétala — instrucciones en `queries/GOLD_STANDARD.md`.
2. Corre las variantes de `queries/scopus.md` §1 y §2 solo para contar N, y
   exporta una muestra de cada una.
3. Mide el recall relativo de cada variante:
   ```bash
   python -m nanocarbon_biblio.cli recall --raw data/raw --gold queries/gold_standard.csv
   ```
   Objetivo ≥ 0.95. Revisa además 100 registros aleatorios para la precisión.
4. Fija la consulta. Anota fecha, N, recall y versión en `docs/PROTOCOL.md` §5.

## Paso 2 — Exportar

Sigue los checklists de `queries/scopus.md` §5 y `queries/wos.md` §6.
Los dos puntos que arruinan el análisis si se olvidan:

- Scopus: marcar **References** en el panel de campos.
- WoS: elegir **Full Record and Cited References**.

Guarda todo en `data/raw/` (subcarpetas si quieres, se busca recursivamente).

## Paso 3 — Pipeline de Python

Exploratorio, con GUI:

```bash
streamlit run nanocarbon_biblio/app.py
```

Si aún no tienes exportaciones y quieres aprender la interfaz primero, la
pestaña 1 tiene un botón de **corpus de demostración** que genera datos
sintéticos realistas (1991-2025, con duplicados entre bases, registros sin DOI y
los falsos positivos típicos). Son inventados: para practicar, no para analizar.

Desde la pestaña 6 puedes además lanzar `00_build_M.R` y `01_core_analyses.R`
sin salir de la interfaz, y ver su salida en la página.

Definitivo, reproducible (**el que va al paper**):

```bash
python -m nanocarbon_biblio.cli run \
  --raw data/raw --out data/processed \
  --title-threshold 92 --year-window 1 \
  --require-topic --crosstab --indicators \
  --note "Scopus 2026-XX-XX + WoS 2026-XX-XX, consulta CORE v3"
```

`--indicators` añade las tablas de RQ2 (`rq2_dopant_lag.csv`,
`rq2_study_type_share.csv`) y RQ3 (`rq3_gap_matrix.csv`). Son las que sostienen
las dos aportaciones originales del review; sin ellas tienes un mapeo temático
más, que es justo lo que un editor rechaza.

> **`--out` nunca dentro de `--raw`.** La corrida siguiente re-ingeriría su
> propia salida y duplicaría el corpus. La CLI lo rechaza.

Guarda ese comando en el repo. Un corpus montado a clics no es reproducible.

## Paso 4 — Tesauro

```bash
python -m nanocarbon_biblio.cli thesaurus --raw data/raw --out queries/thesaurus.txt
```

**Revísalo a mano línea por línea.** El agrupador es difuso y `n-doped` /
`p-doped` puntúan altísimo entre sí siendo opuestos. Versiona el fichero
revisado: forma parte del material suplementario.

## Paso 5 — Cribado manual

`data/processed/labels.csv` trae la columna `dopant_host_ambiguous`. Son los
casos tipo *"TiO2 dopado con N soportado en MWCNT"*, el falso positivo dominante
de este tema. Codifícalos a mano (la pestaña 4 de la GUI te los exporta) y
reincorpora la decisión.

Con dos codificadores, calcula **kappa de Cohen** y repórtalo.

## Paso 6 — Construir M para bibliometrix

```bash
Rscript R/00_build_M.R
```

Comprueba `data/processed/join_report.txt`:
- cobertura de `CR` alta (si es baja, re-exporta: no hay arreglo posterior);
- ≥90 % de los registros con etiquetas unidas.

## Paso 7 — biblioshiny

```bash
Rscript R/launch_biblioshiny.R
```

En la app: **Data → Load bibliometrix file → `data/processed/M.rds`**.
**No** uses *Import raw files*: volverías a parsear los crudos y perderías las
columnas `dopant`, `defect`, `study_type`, `morphology`, `application`.

Lo que merece la pena en biblioshiny para este review:

| Menú | Análisis | Para qué |
|---|---|---|
| Overview | Main information, annual production | Curva de crecimiento, cortes temporales |
| Overview | **Reference Spectroscopy (RPYS)** | Raíces intelectuales: picos en 1991, 1986 |
| Sources | Bradford's law | Revistas núcleo |
| Authors | Lotka, most cited, países | Estructura social |
| Documents | **Co-word / co-occurrence network** | RQ1: temas (con el tesauro cargado) |
| Documents | **Thematic map** | Motor / nicho / emergente / declive |
| Documents | **Thematic evolution** | Migración temática con cortes físicos |
| Clustering | Coupling / co-citation | Frentes de investigación vs base intelectual |
| Conceptual | MCA / correspondence analysis | Estructura conceptual |

**Filtro por faceta:** biblioshiny no sabe filtrar por columnas propias, así que
para analizar solo el subcorpus teórico, genera un `M` filtrado en R y guárdalo
aparte:

```r
M <- readRDS("data/processed/M.rds")
Mt <- M[M$study_type %in% c("theoretical", "combined"), ]
class(Mt) <- c("bibliometrixDB", "data.frame")
saveRDS(Mt, "data/processed/M_theory.rds")
```
Luego cárgalo como otro fichero. Haz lo mismo para `M_experimental.rds`,
`M_3d.rds`, `M_nitrogen.rds`. **Comparar los mapas temáticos de teoría vs
experimento lado a lado es la figura que responde RQ2.**

## Paso 7b — Validar las reglas y cerrar el PRISMA

Descarga la hoja de codificación de la pestaña 7, codifica a mano (dos
codificadores si puedes) y puntúala:

```bash
python -m nanocarbon_biblio.cli agreement --sheet validacion_codificada.csv
```

Con el número de excluidos del cribado manual ya decidido, cierra la figura:

```bash
python -m nanocarbon_biblio.cli prisma --excluded <N> --out results/prisma_flow.svg
```

## Paso 8 — Análisis guionizados

```bash
Rscript R/01_core_analyses.R
```

Salidas en `results/`. Lo que va al paper sale de aquí, no de un clic.

---

## Herramientas complementarias que vale la pena añadir

Ninguna es imprescindible; cada una añade algo que ni bibliometrix ni este
pipeline dan.

| Herramienta | Qué aporta | Cuándo |
|---|---|---|
| **CRExplorer** (Java, gratis) | RPYS fino, con desambiguación de referencias citadas y detección de picos. Mejor que `rpys()` para la sección de raíces intelectuales. | Si RQ sobre orígenes conceptuales pesa |
| **VOSviewer** | Mapas de co-ocurrencia y co-citación de calidad de publicación; overlay maps. Acepta ficheros de Scopus/WoS directamente. | Para las figuras finales |
| **CiteSpace** | **Detección de bursts** (Kleinberg): términos y referencias con explosión de uso. Es el análisis que identifica temas emergentes con estadística, no a ojo. | Muy recomendable para "perspectivas" |
| **ScientoPy** (Python) | Tendencias de tópicos normalizadas (*% documents per year*, *average growth rate*). Complementa bien el mapa temático. | Curvas de tendencia por dopante |
| **pyalex** / OpenAlex | Tercera fuente gratuita: sesgo de cobertura, ROR de instituciones, estado de acceso abierto, rescate de DOIs. | Sección de robustez |
| **Crossref + Retraction Watch** | Cruce de DOIs contra retractaciones. En electrocatálisis de nanocarbono la huella no es despreciable. | Integridad; casi nadie lo hace |
| **BERTopic + MatSciBERT** | Tópicos guiados por embeddings del dominio de materiales sobre los resúmenes; se contrastan con los clusters de co-palabras. Si coinciden, resultado robusto; si no, hallazgo. | Validación cruzada de RQ1 |
| **Quarto** | Manuscrito con R y Python en el mismo documento, cifras generadas en compilación. | Reproducibilidad |
| **renv + uv/conda** | Congelar versiones de R y de Python. | Obligatorio si el review tarda un año |
| **OSF + Zenodo** | Protocolo registrado y DOI del código/tesauro/lista de DOIs. | Reporte |

### Sobre BERTopic y la validación cruzada de clusters

Es el añadido con mejor relación valor/esfuerzo después de RQ2. La idea:

1. Embeddings de los resúmenes con un modelo del dominio (MatSciBERT o
   SciBERT; `sentence-transformers` sirve como línea base).
2. BERTopic → tópicos guiados por datos.
3. Tabla de contingencia tópicos-BERTopic × clusters-de-co-palabras, con
   información mutua ajustada.

Si las dos particiones coinciden, tienes una afirmación fuerte: *la estructura
temática del campo es robusta al método*. Si divergen, la divergencia es el
hallazgo — normalmente significa que las keywords de autor van por detrás de lo
que realmente se está investigando, que en un campo con deriva terminológica
como este es muy plausible.

---

## Errores que cuestan un mes

1. **Exportar sin referencias citadas.** No se puede arreglar después: hay que
   volver a bajar todo. Compruébalo en el primer fichero, no en el vigésimo.
2. **No fijar la fecha de corte.** Las bases se actualizan a diario; si repites
   la búsqueda dos semanas después, los números cambian y ya no cuadran con lo
   que escribiste.
3. **Analizar sin tesauro.** El mapa de co-palabras sale fragmentado y hay que
   rehacer todas las figuras.
4. **Citas brutas en lugar de normalizadas por campo.** Un paper de 2005 tiene
   veinte años para acumular citas; uno de 2024 no. Cualquier ranking sin
   normalizar mide antigüedad, no impacto.
5. **Perder la trazabilidad del cribado.** Si no puedes decir por qué se
   excluyeron exactamente N registros, no tienes una revisión sistemática.
   `manifest.json` existe para eso: consérvalo con cada corrida.
