# CLAUDE.md

Instrucciones para Claude Code (o cualquier asistente) trabajando en este subproyecto.

## Alcance
`nanocarbon_biblio` es el pipeline bibliométrico del artículo de revisión sobre
**defectos y dopaje en nanoestructuras de carbono 1D**. Convierte exportaciones
crudas de Scopus y Web of Science en un corpus deduplicado, clasificado y
cargable en `bibliometrix`/biblioshiny.

Es un instrumento de investigación: los números que produce acaban en un
manuscrito. **Un cambio silencioso en una regla cambia una figura publicada.**

## Reglas de oro
1. **Python ≥ 3.10**, type hints en toda función pública, docstrings obligatorias.
2. **Nunca reconstruir el data frame de bibliometrix en Python.** El corpus
   filtrado se reescribe en formato NATIVO (CSV de Scopus, texto etiquetado de
   WoS) y lo parsea `convert2df`. Las etiquetas van aparte, en `labels.csv`.
   Romper esto destruye `CR` y con él la co-citación, el acoplamiento
   bibliográfico y el RPYS — en silencio.
3. **Toda regla nueva de léxico ship con test.** Ver `tests/test_classify.py`.
4. **Los guardas de desambiguación no se tocan sin justificar.** En particular
   el de `P-doped` (fósforo vs p-type) y el de `dopant_host_ambiguous`.
5. Nada de rutas absolutas: `pathlib.Path` y directorios que da el usuario.
6. Todo lo aleatorio (muestras de validación, agrupación de sinónimos) acepta
   `seed` y es reproducible.
7. La GUI es una fachada sobre la librería: **no** puede tener lógica que la CLI
   no tenga. Si algo solo se puede hacer clicando, el análisis no es reproducible.
8. Los gráficos usan la paleta validada de `app.py` (`SERIES_COLORS`,
   `SEQUENTIAL_BLUE`): categórica en orden fijo de ranura, nunca cíclica;
   secuencial de **un solo tono**, nunca arcoíris. Los colores están validados
   para daltonismo (peor par adyacente ΔE 9.2 deutan). Si cambias un color,
   revalida el conjunto.

## Guardas científicas
- **La deduplicación usa `title_sort_key` + `fuzz.ratio`, no `token_set_ratio`.**
  Esto NO es una micro-optimización: `token_set_ratio` puntúa 100 cuando un
  título es subconjunto de otro, y fusiona artículos distintos. Si alguien lo
  «simplifica» de vuelta, `test_subset_titles_do_not_merge` falla. Que siga así.
- **La plantilla del conjunto de oro no lleva DOIs inventados.** Rellenarla con
  datos bibliográficos sin verificar convierte la calibración en una medida de
  la memoria del asistente, no de la consulta del usuario. Si añades entradas,
  van con `verified=FALSE` hasta que un humano las compruebe.
- **El diagrama PRISMA comprueba su propia aritmética.** Identificados −
  duplicados = cribados. Si no cuadra, el aviso va impreso sobre la figura y la
  CLI devuelve código 1. No silencies ninguno de los dos.
- **Los identificados de PRISMA son los de ANTES de deduplicar**, por base
  (`records_identified_by_source`). El reparto posterior es otro número y
  subestima la identificación exactamente en el solapamiento entre bases.
- **Nunca reportar `cnorm_year` como CNCI o MNCS.** Es normalización por corpus,
  no por campo. Sirve para ordenar dentro de este corpus; una cifra
  field-normalised de verdad sale de SciVal o InCites. El docstring de
  `annotate_records` lo dice y la GUI lo repite en pantalla: no quites ninguno.
- **El desfase de RQ2 se reporta con `lag_at_k`, no con `lag_first`.** Un solo
  documento temprano puede ser una mala clasificación.
- **Una celda vacía de la matriz de RQ3 no es un hueco de investigación.** Puede
  ser físicamente poco interesante o estar fuera del léxico. El umbral
  `min_theory` existe para que un artículo suelto no fabrique un hueco.
- `dop*` como comodín recupera `dopamine`: en las consultas se enumeran las
  formas (`doped OR doping OR dopant*`). No "simplificar" esto.
- `CNF` significa *cellulose nanofibril* con más frecuencia que *carbon
  nanofiber*. Las consultas usan `"carbon nanofib*"` desarrollado.
- `p-doped` / `n-doped` colisionan con dopaje tipo p/n. Las abreviaturas de una
  letra se compilan **case-sensitive** y la de fósforo lleva guarda obligatoria.
- La detección de co-dopaje (`codoping_dopants`) exige que el hueco entre
  elementos contenga **solo** nombres de elementos y separadores. Aflojarlo hace
  que "oxygen reduction over nitrogen co-doped carbon" cuente como dopaje con
  oxígeno.
- El corpus de grafeno/GO es **separado** por decisión de protocolo
  (`docs/PROTOCOL.md` §2.1). No fusionarlo con el núcleo.

## Comandos
```bash
pip install -r requirements.txt
pytest nanocarbon_biblio/tests -q
streamlit run nanocarbon_biblio/app.py
python -m nanocarbon_biblio.cli run --raw data/raw --out data/processed
Rscript R/00_build_M.R && Rscript R/01_core_analyses.R
```

## Dónde añadir cosas
- Nuevo dopante o tipo de defecto → `lexicons.py` (`FACETS`) + test en `tests/test_classify.py`.
- Nueva base de datos (Dimensions, Lens, OpenAlex) → `loaders.py` con un
  `load_<db>()` que preserve el crudo, más una rama en `exporters.py` y el
  `dbsource` correspondiente en `R/00_build_M.R`.
- Nuevo análisis de R → `R/01_core_analyses.R`, envuelto en `step()`.

## Qué no hacer
- No usar el corpus de demostración para nada que acabe en el manuscrito: son
  datos inventados. Existe para ejercitar el pipeline y aprender la GUI.
- No apuntar `--out` dentro de `--raw`: la corrida siguiente re-ingeriría su
  propia salida. La CLI lo rechaza y `load_directory` salta esos ficheros; no
  desactives ninguna de las dos protecciones.
- No fusionar el corpus de grafeno con el núcleo.
- No descartar automáticamente los `dopant_host_ambiguous`: son para revisión manual.
- No añadir dependencias fuera de `requirements.txt` sin justificarlo.
- No versionar nada bajo `data/`: los metadatos de Scopus y WoS no son
  redistribuibles.
