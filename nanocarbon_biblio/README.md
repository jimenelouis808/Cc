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

## Uso

```bash
# 1. GUI para explorar, deduplicar, clasificar y decidir el cribado
streamlit run nanocarbon_biblio/app.py

# 2. La corrida definitiva, reproducible
python -m nanocarbon_biblio.cli run --raw data/raw --out data/processed \
    --require-topic --crosstab --note "Scopus + WoS, consulta CORE v3"

# 3. Tesauro (revísalo a mano antes de usarlo)
python -m nanocarbon_biblio.cli thesaurus --raw data/raw --out queries/thesaurus.txt

# 4. Construir el data frame de bibliometrix y lanzar biblioshiny
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

34 tests. Cubren el parseo de ambos formatos, la conservación literal de `CR` en
el viaje de ida y vuelta, la deduplicación entre bases, los guardas de
desambiguación y el arranque de la GUI.

## Estado de verificación

- **Python**: probado y ejecutado en este entorno; los 34 tests pasan.
- **R**: los cuatro scripts están **comprobados sintácticamente** con `parse()`,
  pero **no ejecutados**, porque CRAN no es alcanzable desde el entorno donde se
  escribieron y `bibliometrix` no se pudo instalar. Los pasos frágiles van
  envueltos en `step()` y registran el fallo en `results/failed_steps.txt` en
  lugar de tumbar el script, pero **la primera corrida de `R/00_build_M.R` con
  datos reales hay que mirarla con atención**, en particular la cobertura de
  `CR` y el porcentaje de etiquetas unidas que reporta `join_report.txt`.

## Licencia y datos

El código es tuyo. Los metadatos de Scopus y WoS **no se pueden redistribuir**;
`data/raw/` y `data/processed/` están fuera del control de versiones. Lo que sí
puedes publicar en Zenodo, y deberías: las cadenas de búsqueda, el tesauro, el
código, el `manifest.json` y la lista de DOIs del corpus.
