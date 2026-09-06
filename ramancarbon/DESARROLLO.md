# Guía de desarrollo

Para quien vaya a tocar el código — persona o asistente.

## Arquitectura

Cuatro capas, cada una usable por separado, con dependencias en un solo
sentido:

```
core      ──▶  models  ──▶  analysis  ──▶  gui / cli
  │              │             ▲
  └──────────────┴─────────────┘
                database
```

* **`core`** — `Spectrum`, lectura de archivos, línea base, despiking,
  detección de picos. No sabe nada de carbono.
* **`models`** — perfiles y motor de ajuste. No sabe nada de bandas
  concretas salvo por lo que le pasa `deconvolution`.
* **`database`** — los JSON y su API tipada. **No importa nada del resto.**
* **`analysis`** — asignación, cocientes, diámetros, desplazamientos,
  clasificador, y `report.analyse` que los encadena.
* **`gui` / `cli`** — presentación. Toda la lógica de la interfaz vive en
  `gui/state.py` y `gui/plots.py`, **sin Tkinter**, para que se pueda probar
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
| Una pestaña de la interfaz | `gui/app.py` (widgets) + `gui/state.py` (lógica) |
| Un subcomando | `cli/main.py` |

## Pruebas

```bash
pytest ramancarbon/tests -q            # ~200 pruebas, unos 15 s
pytest ramancarbon/tests -q -k rbm     # solo lo del RBM
ruff check ramancarbon                 # estilo
```

La interfaz gráfica no se puede probar sin pantalla, así que toda su lógica
está en `gui/state.py` y su dibujo en `gui/plots.py`, y ambos sí se prueban
(con el backend `Agg`). Lo único sin cubrir es el cableado de Tkinter.

## Lo que falta por validar

Esto es lo importante de esta sección: **todo se ha comprobado contra datos
sintéticos**, generados por `examples/demo_data.py`. Eso demuestra que las
fórmulas están bien implementadas y que las piezas encajan. No demuestra
que los resultados sean correctos sobre espectros reales, que tienen ruido
correlacionado, respuesta instrumental, líneas de sustrato, inhomogeneidad y
alas no lorentzianas.

Lo que haría falta para cerrarlo:

* Un conjunto de espectros medidos con composición conocida por otra
  técnica (TEM para el número de paredes, XPS para el dopado).
* Contrastar los diámetros por RBM con TEM sobre la misma muestra.
* Comprobar la separación deformación/dopado sobre grafeno con dopado
  electroquímico controlado.
* Medir la misma muestra con dos o tres láseres y verificar que las
  posiciones corregidas por dispersión coinciden.
