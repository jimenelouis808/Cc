# nanocarbon_biblio

Pipeline bibliométrico para un artículo de revisión sobre **defectos y dopaje
con heteroátomos en nanoestructuras de carbono 1D** (nanotubos y nanofibras),
desde el punto de vista experimental y teórico.

Scopus + Web of Science entran; sale un corpus deduplicado, clasificado y
listo para `bibliometrix` / **biblioshiny**, sin perder las referencias citadas.

Proyecto hermano de [`nanocarbon_lab`](../nanocarbon_lab): el mismo objeto de
estudio — dopantes sustitucionales, vacantes, Stone-Wales, redes 3D — visto
desde la literatura en lugar de desde la estructura atómica.

---

## Qué hay aquí

| Ruta | Contenido |
|---|---|
| `queries/scopus.md` | Consultas de Scopus: núcleo, alta sensibilidad, facetas, corpus de contexto, checklist de exportación |
| `queries/wos.md` | Lo mismo para WoS, con las diferencias de sintaxis que sí cambian los resultados |
| `queries/thesaurus_seed.txt` | Tesauro de arranque para biblioshiny (29 grupos curados) |
| `docs/PROTOCOL.md` | Preguntas de investigación, decisiones de alcance, criterios, sesgos, reporte |
| `docs/WORKFLOW.md` | El recorrido completo paso a paso, y las herramientas complementarias |
| `nanocarbon_biblio/` | El paquete de Python |
| `R/` | `00_build_M.R`, `01_core_analyses.R`, `launch_biblioshiny.R` |
| `data/` | `raw/` (exportaciones crudas), `processed/` (bundle para R) |

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
Rscript R/install_deps.R
```

Python ≥ 3.10, R ≥ 4.2.

## La GUI

```bash
streamlit run nanocarbon_biblio/app.py
```

Siete pestañas, de izquierda a derecha. El estado vive en `st.session_state`, así
que puedes ir y volver sin recargar un corpus de 30 000 registros.

| Pestaña | Qué hace |
|---|---|
| **1 · Cargar** | Carpeta del proyecto, subida de ficheros, o **corpus de demostración** generado al vuelo. Métricas de cobertura de DOI, resumen y referencias, con aviso en rojo si faltan referencias citadas |
| **2 · Deduplicar** | Umbral de similitud y ventana de años ajustables; solapamiento Scopus/WoS. Con el corpus de demostración, compara lo recuperado contra la verdad conocida |
| **3 · Clasificar** | Tipo de estudio, cuota anual (con umbral de documentos/año, porque una cuota sobre 2 documentos es ruido) y la matriz dopante × aplicación interactiva |
| **4 · Cribado** | Filtros por año, tipo y señal temática; exporta los marcados como `dopant_host_ambiguous` para revisión manual |
| **5 · Tesauro** | Sugiere grupos de sinónimos del propio corpus, editables en la página, y los guarda en formato biblioshiny |
| **6 · Exportar a R** | Escribe el bundle, muestra el **flujo PRISMA** descargable y **lanza los scripts de R** (`00_build_M.R`, `01_core_analyses.R`) mostrando su salida |
| **7 · Validación** | Muestra estratificada y reproducible, con hoja de codificación en blanco para calcular kappa |
| **8 · RQ2 · RQ3** | Las dos preguntas originales: desfase teoría→experimento por dopante, matriz de huecos, y citas normalizadas |

La GUI es una fachada sobre la librería: **todo lo que se puede hacer clicando se
puede hacer desde la CLI**. Un corpus montado a clics no es reproducible.

### Probarla sin tener nada exportado

```bash
python -m nanocarbon_biblio.cli demo --out data/raw/demo --n-works 1200
streamlit run nanocarbon_biblio/app.py
```

O directamente el botón *Corpus de demostración* de la pestaña 1. Genera
exportaciones sintéticas de Scopus y WoS de 1991 a 2025, con curva de
crecimiento realista (incluido el bache del grafeno en 2010-2014), duplicados
entre bases, registros sin DOI, señuelos `p-doped` que no son fósforo y casos de
dopante en huésped no-carbono. **Son datos inventados**: sirven para aprender la
interfaz y probar el pipeline, nunca para el manuscrito.

Como el generador conoce su propio solapamiento real, la pestaña 2 lo usa para
comprobarse a sí misma — y así es como se detectó y corrigió un fallo real de la
deduplicación cuando falta el DOI.

## Uso desde la línea de comandos

```bash
# 1. Corpus sintético para probar el pipeline sin exportar nada
python -m nanocarbon_biblio.cli demo --out data/raw/demo

# 2. La corrida definitiva, reproducible
python -m nanocarbon_biblio.cli run --raw data/raw --out data/processed \
    --require-topic --crosstab --indicators --note "Scopus + WoS, consulta CORE v3"

# 3. Tesauro (revísalo a mano antes de usarlo)
python -m nanocarbon_biblio.cli thesaurus --raw data/raw --out queries/thesaurus.txt

# 4. Construir el data frame de bibliometrix y lanzar biblioshiny
#    (los dos primeros también se lanzan desde la pestaña 6 de la GUI)
Rscript R/00_build_M.R
Rscript R/01_core_analyses.R
Rscript R/launch_biblioshiny.R      # Data -> Load bibliometrix file -> M.rds
```

## La decisión de diseño que importa

**El corpus filtrado se reescribe en formato nativo**, no como un data frame
reconstruido. `convert2df()` parsea el CSV de Scopus y el texto etiquetado de
WoS exactamente como fue diseñado para hacerlo, así que el campo `CR`
(referencias citadas), el `C1` (afiliaciones) y todas las manías del parser de
bibliometrix se conservan intactas. Python aporta solo las **etiquetas**, en una
tabla aparte que R une por DOI y, en su defecto, por título normalizado.

Reconstruir `CR` desde Python rompe co-citación, acoplamiento bibliográfico y
RPYS — y lo rompe en silencio, que es peor.

## Las dos preguntas originales (pestaña 8)

Lo que separa este review de "el enésimo N-doped carbon for ORR". Ambas se
calculan sin red, desde el propio corpus (`indicators.py`).

**RQ2 — ¿predice la teoría, o documenta?** Por cada dopante, el año del primer
estudio computacional frente al del primero experimental, y el desfase entre
ambos. Positivo = la teoría llegó antes (predicción → realización); negativo =
el experimento llegó antes y la teoría vino a explicarlo. El desfase se ancla en
el **k-ésimo** documento, no en el primero: colgar una afirmación de una década
de un único registro que podría estar mal clasificado es indefendible. Se
acompaña de la correlación de Spearman de la cuota de estudios *combined* contra
el año, que es la forma cuantitativa de "teoría y experimento se han acoplado".

**RQ3 — ¿qué está predicho pero sin hacer?** Cada celda (dopante × aplicación)
se clasifica como `covered`, `experiment_only` o `theory_only`. Las
`theory_only` —con trabajo computacional y **cero** experimental— son la sección
de perspectivas con evidencia detrás. Se ordenan por `n_theory × cnorm medio`,
así que una celda predicha por teoría bien citada pesa más que una mencionada de
pasada. Una celda vacía no es un hueco por sí sola: puede ser físicamente poco
interesante o estar fuera del vocabulario de las reglas, y la interfaz lo avisa.

**Citas normalizadas.** La cita bruta mide antigüedad, no impacto. Se calcula
`cnorm_year` (citas / media del año) y el percentil dentro del año. Es
**normalización por corpus, no por campo**: sirve para ordenar documentos dentro
de este corpus, que es lo que el review necesita, pero no se puede reportar como
CNCI ni MNCS —eso exige una línea base de toda la ciencia, de SciVal o InCites—.
El código lo dice en el docstring y la interfaz lo repite en pantalla.

## Lo que el pipeline etiqueta

Reglas transparentes y auditables (`nanocarbon_biblio/lexicons.py`), no un
clasificador opaco:

- **dopante** — N, B, P, S, F, halógenos, Si, O, Se, metales de transición, co-dopaje
- **defecto** — vacantes, Stone-Wales, topológicos, fronteras de grano, bordes, sp3, adátomos, irradiación
- **modo de dopaje** — sustitucional / transferencia de carga / funcionalización
- **tipo de estudio** — teórico / experimental / **combinado** / poco claro
- **morfología** — 1D individual, array, bosque VACNT, fibra, buckypaper, esponja/aerogel, red/juntura, composite
- **aplicación** — ORR, HER/OER, supercondensadores, baterías, sensores, emisión de campo, térmicas, mecánicas, adsorción, EMI…
- **banderas** — híbrido con grafeno, ensamblaje 3D, y `dopant_host_ambiguous`

### Las tres trampas que las reglas evitan a propósito

1. `p-doped` ≠ fósforo. Casi siempre es *p-type*. La abreviatura solo dispara si
   `phosphor*` aparece en el registro.
2. `CNF` ≠ nanofibra de carbono en la mitad de la literatura de materiales: es
   *cellulose nanofibril*. Las consultas usan `"carbon nanofib*"` desarrollado.
3. `"TiO2 dopado con N soportado en MWCNT"` no es un nanotubo dopado. La bandera
   `dopant_host_ambiguous` los aparta para revisión manual en lugar de contarlos.

## Tests

```bash
pytest nanocarbon_biblio/tests -q
```

50 tests. Cubren el parseo de ambos formatos, la conservación literal de `CR` en
el viaje de ida y vuelta, la deduplicación entre bases puntuada **contra la
verdad conocida** del corpus sintético, los guardas de desambiguación, la
protección contra re-ingerir la propia salida, y el arranque de la GUI.

## Estado de verificación

- **Python**: probado y ejecutado en este entorno; los 50 tests pasan.
- **GUI**: recorrida de principio a fin (cargar → deduplicar → clasificar →
  cribar → exportar) contra el corpus de demostración, y capturada en imágenes.
  Los botones de R se activan solos cuando `Rscript` está en el PATH.
- **R**: los cuatro scripts están **comprobados sintácticamente** con `parse()`,
  pero **no ejecutados**, porque CRAN no es alcanzable desde el entorno donde se
  escribieron y `bibliometrix` no se pudo instalar. Los pasos frágiles van
  envueltos en `step()` y registran el fallo en `results/failed_steps.txt` en
  lugar de tumbar el script, pero **la primera corrida de `R/00_build_M.R` con
  datos reales hay que mirarla con atención**, en particular la cobertura de
  `CR` y el porcentaje de etiquetas unidas que reporta `join_report.txt`.

## Lo que deliberadamente no está

**Enriquecimiento por API** (OpenAlex para cobertura y ROR, Crossref y Retraction
Watch para retractaciones). Está recomendado en `docs/WORKFLOW.md` y sigue siendo
buena idea, pero no lo implementé porque el entorno donde se escribió este código
tiene bloqueado el acceso a esas APIs por política de red, y no quiero entregar
código de red que no he podido ejecutar ni una vez. Cuando lo añadas, ponlo en
`enrich.py` con caché en disco y respetando el `mailto` de la *polite pool* de
ambas APIs.

## Licencia y datos

El código es tuyo. Los metadatos de Scopus y WoS **no se pueden redistribuir**;
`data/raw/` y `data/processed/` están fuera del control de versiones. Lo que sí
puedes publicar en Zenodo, y deberías: las cadenas de búsqueda, el tesauro, el
código, el `manifest.json` y la lista de DOIs del corpus.
