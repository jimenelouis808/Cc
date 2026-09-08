# Guía de desarrollo

Para quien vaya a tocar el código — persona o asistente.


## Arquitectura

Cuatro instrumentos sobre una base común. Cada capa es usable por separado
y las dependencias van en un solo sentido:

```
                        ┌──▶  analysis  ──┐          (Raman)
core   ──▶  models  ────┤                 ├──▶  gui / cli
  │           │         ├──▶  xrd  ───────┤          (difracción)
  │           │         ├──▶  echem  ─────┤          (electroquímica)
  │           │         └──▶  mapping  ───┤          (mapas Raman)
  └───────────┴─────────────────┬─────────┘
                            database

              plotting  ·  dataio  ·  benchmarks
              (transversales: no dependen de la instrumentación)
```

`xrd` y `echem` son hermanos de `analysis`, no capas por encima: comparten
`core` (la línea base asimétrica, el shim de `trapezoid`) y `database` (los
JSON), y no se importan entre sí.

* **`core`** — `Spectrum`, lectura de archivos, línea base, despiking,
  detección de picos. No sabe nada de carbono.
* **`models`** — perfiles y motor de ajuste. No sabe nada de bandas
  concretas salvo por lo que le pasa `deconvolution`.
* **`database`** — los JSON y su API tipada. **No importa nada del resto.**
* **`analysis`** — asignación, cocientes, diámetros, desplazamientos,
  clasificador, fases no carbonosas, dicalcogenuros y heteroestructuras, y
  `report.analyse` que los encadena.
* **`xrd`** — simetría, estructuras, CIF, factores de forma, patrón
  calculado, búsqueda de fases y Rietveld. No hay tablas de posiciones de
  pico: todo se calcula de la estructura.
* **`echem`** — curvas, lectores de potenciostato, CV, carga-descarga,
  impedancia con circuitos equivalentes, mecanismo de almacenamiento y
  electrocatálisis.
* **`mapping`** — el cubo de un mapa, las imágenes por píxel y la
  quimiometría. Depende de `core` y de nada más de la instrumentación.
* **`plotting`** — el motor de figuras. No importa nada del resto del
  paquete salvo para los constructores de conveniencia
  (`from_spectrum`, `from_pattern`), que son importaciones perezosas.
* **`dataio`** — detección de formato, lector universal, exportación y
  proyectos. Está *encima* de los lectores de cada instrumento, no en
  lugar de ellos.
* **`benchmarks`** — cronometra la suite. Sin él, «optimizado» es una
  afirmación.
* **`gui` / `cli`** — presentación. La ventana es una suite de cuatro
  secciones (`gui/suite.py`) sobre `gui/base.py`, y toda la lógica vive en
  `gui/state.py`, `gui/xrd_state.py`, `gui/echem_state.py` y los módulos de
  `plots*`, **sin Tkinter**, para que se pueda probar
  sin pantalla.

## Reglas

1. **Python ≥ 3.10**, anotaciones de tipo en todo lo público.
2. **Docstrings obligatorios.** Entradas, salidas y, cuando toque, la
   suposición física. Si un número sale de un artículo, la fuente va en el
   JSON, no en el código.
3. **Ningún valor de literatura se codifica en Python.** Va en
   `database/data/*.json` con `source` y `confidence`.
4. Todo lo que devuelva un número que dependa del láser **debe** aceptar
   la energía de excitación y corregir por dispersión.
5. Toda función que pueda fallar por falta de datos debe **degradar, no
   reventar**: devolver `None` con una razón, o un objeto con
   `available=False` y `reason`.
6. Los avisos son parte del resultado, no ruido. Si un número puede estar
   sesgado, dilo en el propio objeto que lo lleva.
7. Cada función nueva viene con su prueba en `tests/`.
8. Nada de dependencias fuera de `numpy`, `scipy`, `matplotlib` sin
   justificarlo.

## Barreras que no hay que bajar

Cada una de estas está ahí porque su ausencia produjo un resultado
incorrecto durante el desarrollo. Están cubiertas por pruebas.

* **El despiking usa mediana móvil, no primera diferencia.** El criterio de
  primera diferencia escala con la pendiente, y en una banda G intensa y
  estrecha marcaba los flancos como rayos cósmicos, dejando muescas en los
  costados de la banda.
* **La detección de picos usa significancia de filtro adaptado con
  corrección por número de pruebas.** Un umbral de «3σ de altura» encuentra
  picos en ruido puro casi siempre: el máximo de 320 muestras gaussianas
  *es* ~3σ por construcción. Con el umbral antiguo aparecían cuatro RBM
  fantasma en todos los espectros de grafeno. El valor por defecto (18) está
  calibrado sobre 400 ventanas de ruido puro.
* **La ausencia de RBM solo es prueba si el espectro cubrió esa región.**
  `Spectrum.covers` existe para eso. Sin esa comprobación, todo espectro de
  400–3000 cm⁻¹ se clasifica como multipared.
* **Los cocientes se comparan en la base en que se publicó el rango.** Un
  I_D/I_G de áreas es 2–3 veces el de alturas. `Ratio.on_basis()` existe
  para que el clasificador pida la que necesita.
* **La metalicidad se decide ajustando, no adivinando.** Se ajusta G⁻ con
  BWF y con lorentziana y se comparan por BIC. Las dos constantes del
  desdoblamiento G difieren un 40 %, así que fallar aquí mueve el diámetro
  un 30 %.
* **`bwf_peak_position` es `ω₀ + Γ/q`.** La derivada del perfil BWF factoriza
  como `(1 + b u)(b − u)`; el máximo está en `u = b`. Una versión anterior
  resolvía una cuadrática mal planteada y devolvía la rama del mínimo,
  poniendo el pico del lado equivocado.
* **Un modelo D–G para nanotubos tiene que incluir G⁻.** Sin ella la banda D
  se estira hasta su límite de anchura absorbiendo esa intensidad, y
  I_D/I_G sale varias veces demasiado grande. Por eso existe el preajuste
  `swcnt_full` y por eso se añade a la comparación cuando hay RBM.
* **La banda 2D tiene que estar *detectada* antes de ajustarla.** Si no, en
  un óxido —donde no hay 2D— el ajustador clava una componente ancha en el
  borde de la ventana y llama banda a la cola de D+G.
* **El desdoblamiento G no se aplica a material multipared.** Allí lo que
  acompaña a la G es D', una banda de defectos; la fórmula convertiría un
  defecto en un diámetro. `diameter_from_g_splitting` lanza excepción si le
  pasas `walls >= 3`.
* **Los rangos de FWHM de la base de datos son también los límites del
  ajuste.** Si los estrechas, los ajustes se quedan pegados a ellos y el
  ancho informado queda sesgado.
* **La asignación se compone de todas las regiones ajustadas más el
  detector de picos.** Si solo se pasa el ajuste D–G, la banda 2D queda sin
  asignar y el clasificador concluye que no hay red conjugada.
* **`background_mask` detrenda antes de ordenar.** Ordenar por intensidad
  bruta en un espectro fluorescente marca solo el extremo lejano y excluye
  justo la zona de pendiente fuerte, que es donde una línea base falla.
* **La rigidez sale del corte del suavizador de Whittaker**, no de una
  búsqueda. Ver `CLAUDE.md` para por qué la búsqueda no funciona.
* **`_tab_canvases` indexa por el orden en que se AÑADEN las pestañas**, no
  por el orden en que están definidos los métodos en el archivo. Hay una
  prueba que lo comprueba.

## Dónde tocar cada cosa

| Quiero… | Va en… |
|---------|--------|
| Añadir una banda | `database/data/bands.json` |
| Añadir un material de referencia | `database/data/materials.json` |
| Añadir una relación RBM ↔ diámetro | `database/data/rbm.json` |
| Añadir una firma de dopado | `database/data/perturbations.json` |
| Un perfil nuevo | `models/lineshapes.py` + entrada en `PROFILES` |
| Un preajuste de deconvolución | `models/deconvolution.py`: `PRESETS`, `PRESET_BANDS`, `PRESET_WINDOWS`, `PRESET_LABELS` |
| Una regla del clasificador | `analysis/classify.py`, con su `Evidence` y su peso en `WEIGHTS` |
| Un formato de archivo | `core/io.py` |
| Una pestaña de la interfaz | el `*_app.py` de su sección (widgets) + su `*_state.py` (lógica) |
| Una sección nueva de la suite | `gui/suite.py`: `SECTIONS` y `_ensure` |
| Un subcomando | `cli/main.py` |
| Un índice estructural | `analysis/indices.py` |
| Una firma de dopante | `database/data/perturbations.json` → `dopants` |
| Un formato de exportación | `analysis/export.py`, o `dataio/export.py` para las tablas |
| Un formato de archivo que hay que reconocer solo | `dataio/detect.py` |
| Un preajuste de figura | `plotting/style.py`: `PRESETS` y `COLUMN_WIDTHS` |
| Un eje secundario nuevo | `plotting/transforms.py`: `TRANSFORMS` |
| Una medida por píxel de un mapa | `mapping/images.py` |
| Una ligadura entre parámetros | ya existe: `models/constraints.py` |
| Un ajuste que hay que cronometrar | `benchmarks.py`: `run()` |
| Una fase no carbonosa (Raman) | `database/data/phases.json` |
| Un óxido que acompaña a un TMD | `database/data/tmd.json` → `oxides` |
| **Una fase de referencia de DRX** | suelta su CIF en una carpeta y pásala con `--cif`; para incluirla, `database/data/cif/` y su generador |
| Un elemento de circuito equivalente | `echem/eis.py`: `ELEMENTS` |
| Un electrodo de referencia | `database/data/echem.json` |

## Pruebas

```bash
pytest ramancarbon/tests -q            # ~1030 pruebas, unos 3 min
pytest ramancarbon/tests -q -k rbm     # solo lo del RBM
pytest ramancarbon/tests -q -k "xrd or rietveld"
pytest ramancarbon/tests -q -k echem
ruff check ramancarbon                 # estilo
```

La interfaz gráfica no se puede probar sin pantalla, así que toda su lógica
está en los `*_state.py` y su dibujo en los `plots*.py`, y ambos sí se
prueban (con el backend `Agg`). Del cableado de Tkinter hay dos cosas:
comprobaciones **estáticas** en `test_gui_wiring.py` — cada lienzo tiene su
dibujante y su pestaña, cada `command=` nombra un método real, cada
`self.x` que se lee se asigna en algún sitio — y una prueba que **abre la
ventana de verdad**, recorre las cuatro secciones y ejecuta sus análisis,
que se salta sola donde no hay Tkinter ni pantalla. Sin pantalla pero con Tkinter, un servidor X virtual basta:

```bash
xvfb-run -a .venv/bin/python -m pytest ramancarbon/tests/test_gui_wiring.py -q
```

Y si el Python del sistema no trae Tkinter, otro que sí:

```bash
python3.12 -m venv --system-site-packages /tmp/guienv
/tmp/guienv/bin/pip install numpy scipy matplotlib pytest
xvfb-run -a /tmp/guienv/bin/python -m pytest ramancarbon/tests/test_gui_wiring.py -q
```

Muchas pruebas son **viajes de ida y vuelta**: el generador de datos
sintéticos construye la señal a partir de la física que se quiere medir, y
el análisis tiene que devolver los mismos números. Una capacitancia de
50 mF vuelve como 50 mF; una pendiente de Tafel de 60 mV/dec vuelve como
60.4; los parámetros de red de un difractograma calculado vuelven con seis
cifras. Cuando eso falla, falla por una razón concreta.

## Lo que falta por validar

Esto es lo importante de esta sección: **todo se ha comprobado contra datos
sintéticos**, generados por `examples/demo_data.py`. Eso demuestra que las
fórmulas están bien implementadas y que las piezas encajan. No demuestra
que los resultados sean correctos sobre espectros reales, que tienen ruido
correlacionado, respuesta instrumental, líneas de sustrato, inhomogeneidad y
alas no lorentzianas.

Lo que haría falta para cerrarlo:

* Un conjunto de espectros medidos con composición conocida por otra
  técnica (TEM para el número de paredes y para separar MWCNT de nanofibra,
  XPS para el dopado y su configuración).
* Los rangos de S, P y Se en la base de datos son extrapolaciones razonadas
  desde muy poca literatura, no compilaciones de medidas. El de selenio en
  particular es una hipótesis.
* Contrastar los diámetros por RBM con TEM sobre la misma muestra.
* Comprobar la separación deformación/dopado sobre grafeno con dopado
  electroquímico controlado.
* Medir la misma muestra con dos o tres láseres y verificar que las
  posiciones corregidas por dispersión coinciden.

## Rendimiento

```bash
python -m ramancarbon.benchmarks --rapido       # sin los refinamientos
python -m ramancarbon.benchmarks --json t.json  # para comparar entre ramas
```

Se informa el **mejor** de varias repeticiones, no la media: en una máquina
compartida la media mide lo que estuviera haciendo la máquina. Si mides con
la batería de pruebas corriendo al lado, todo sale unas siete veces más
lento — uniformemente, lo cual es a la vez la trampa y la pista.

Referencia en esta máquina (Python 3.11, NumPy 2.4, SciPy 1.17):

| Operación | Tiempo |
|---|---|
| `analyse()` de un espectro de 3200 puntos | 190 ms |
| línea base asLS / arPLS / SNIP | 5 / 28 / 6 ms |
| despicado de un mapa de 480 píxeles | 270 ms |
| PCA de ese mapa | 125 ms |
| identificación de fases en un difractograma | 110 ms |
| Rietveld automático de una fase | 3.0 s |
| ajuste de circuito equivalente | 43 ms |
| DRT (Tikhonov + curva L) | 34 ms |
