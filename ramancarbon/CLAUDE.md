# CLAUDE.md

Instrucciones para Claude Code (o cualquier asistente) que trabaje en este
proyecto.

## Alcance

`ramancarbon` **analiza espectros Raman experimentales** de nanomateriales de
carbono: identifica SWCNT/DWCNT/MWCNT y material tipo grafeno, deconvoluciona
las bandas D y G, calcula cocientes de intensidad, deduce diámetros del RBM y
compara posiciones con la literatura.

No confundir con `carbonforge`, el proyecto hermano en este mismo repositorio:
aquél **prepara cálculos** de primeros principios (Quantum ESPRESSO, SIESTA,
LAMMPS). Éste analiza medidas. Mantén la distinción en nombres y documentación
y no muevas funcionalidad de uno a otro.

La interfaz de usuario (GUI, CLI, informes, mensajes de error) está **en
español**. El código, los nombres de identificadores y los docstrings están en
inglés. No mezcles.

## Estructura

```
ramancarbon/
├── core/        # Spectrum, lectura de archivos, línea base, despiking, picos
├── models/      # perfiles (incl. Breit-Wigner-Fano) y motor de ajuste
├── database/    # JSON de literatura + API tipada; NO importa del resto
├── analysis/    # asignación, cocientes, índices, diámetros, desplazamientos,
│                #   clasificador, multiláser, exportación
├── gui/         # app Tkinter; la lógica vive en state.py y plots.py, sin Tk
├── cli/         # ramancarbon analizar / lote / deconvolucionar / bd / demo
├── examples/    # scripts ejecutables + demo_data.py (espectros sintéticos)
└── tests/       # pytest
```

Dependencias en un solo sentido: `core → models → analysis → gui/cli`, y
`database` no importa de ninguno.

## Reglas

1. **Python ≥ 3.10**, anotaciones de tipo en todo lo público.
2. **Docstrings obligatorios**, con la suposición física cuando la haya.
3. **Ningún valor de literatura codificado en Python.** Va en
   `database/data/*.json` con su `source` y su `confidence`. Esto es
   deliberado: el usuario es especialista y tiene que poder corregir la base
   de datos sin tocar código.
4. Todo número que dependa de la excitación **debe** corregirse por
   dispersión antes de compararse con nada.
5. Degradar, no reventar: si faltan datos, devuelve `None` con una razón o un
   objeto con `available=False` y `reason`, y dilo en el informe.
6. Los avisos son parte del resultado. Si un número puede estar sesgado, el
   objeto que lo lleva tiene que decirlo.
7. Nada estocástico sin `seed`.
8. Cada función nueva con su prueba en `tests/`.
9. Sin dependencias fuera de `numpy`, `scipy`, `matplotlib` sin justificarlo.

## Barreras físicas — no las bajes

Cada una está ahí porque su ausencia produjo un resultado incorrecto, y cada
una tiene una prueba que la protege.

- **Despiking por mediana móvil, no por primera diferencia.** El criterio de
  primera diferencia escala con la pendiente y marcaba los flancos de una
  banda G intensa como rayos cósmicos, dejando muescas en la banda.
- **Detección de picos por significancia de filtro adaptado con corrección
  por número de pruebas.** Un umbral de «3σ de altura» encuentra picos en
  ruido puro casi siempre; con él aparecían cuatro RBM fantasma en todo
  espectro de grafeno. El valor por defecto está calibrado sobre 400
  ventanas de ruido puro.
- **La ausencia de RBM solo es prueba si el espectro cubrió esa región.** Sin
  esa comprobación (`Spectrum.covers`), todo espectro de 400–3000 cm⁻¹ se
  clasifica como multipared.
- **Los cocientes se comparan en la base en que se publicó el rango.** Áreas
  y alturas difieren en un factor 2–3. Usa `Ratio.on_basis()`.
- **La metalicidad se decide ajustando BWF contra lorentziana y comparando
  por BIC**, no adivinando por la altura del hombro. Las dos constantes del
  desdoblamiento G difieren un 40 %.
- **`bwf_peak_position` es `ω₀ + Γ/q`.** La derivada factoriza como
  `(1 + b u)(b − u)`; el máximo está en `u = b`. Una versión anterior
  devolvía la rama del mínimo.
- **Un modelo D–G de nanotubo tiene que incluir G⁻** (preajuste
  `swcnt_full`). Sin ella la D se estira absorbiendo esa intensidad y
  I_D/I_G sale varias veces demasiado grande.
- **La 2D tiene que estar detectada antes de ajustarla.** Si no, en un óxido
  el ajustador clava una componente en el borde de la ventana y llama banda
  a la cola de D+G.
- **El desdoblamiento G no se aplica a multipared**: allí lo que acompaña a
  la G es D', una banda de defectos, y la fórmula convertiría un defecto en
  un diámetro.
- **Los rangos de FWHM de `bands.json` son también los límites del ajuste.**
  Estrecharlos hace que los ajustes se peguen a ellos y sesga las anchuras.
- **La asignación se compone de todas las regiones ajustadas más el detector
  de picos.** Con solo el ajuste D–G, la 2D queda sin asignar.
- **I_D/I_G no es monótona con el desorden.** Informa siempre las dos ramas
  de Tuinstra–Koenig; citar solo la de bajo desorden puede errar un orden de
  magnitud.
- **La G sube con electrones y con huecos.** El signo del dopado solo lo da
  la 2D. No escribas nada que sugiera lo contrario.
- **La separación deformación/dopado está calibrada para grafeno monocapa.**
  Aplicarla a nanotubos exige el aviso correspondiente.
- **Raman no separa las configuraciones del nitrógeno.** Para eso hace falta
  XPS, y la base de datos lo dice explícitamente. No lo suavices.
- **asLS es la línea base por defecto, medido, no por reputación.** arPLS se
  recomienda mucho para fluorescencia; contrastado contra fondos sintéticos
  conocidos a la rigidez que cada uno usaría de verdad, asLS ganó en todos
  los casos y arPLS arrastraba un desplazamiento sistemático. Si alguien lo
  revierte, que sea con medidas nuevas, no con la cita.
- **La rigidez de la línea base sale de la función de transferencia del
  suavizador de Whittaker**, fijando el corte en cinco anchuras de banda. El
  primer intento puntuaba rigideces por cómo seguían al fondo: eso NO
  funciona, y el módulo explica por qué (el residuo lo domina el ruido, y su
  parte sistemática se minimiza con la línea base más blanda, la que se come
  los picos). No vuelvas a la búsqueda en rejilla.
- **λ depende del paso de muestreo** porque el corte es un periodo en
  puntos. Copiar un λ de un artículo sin reescalarlo es un error de 256×
  entre 1 y 4 cm⁻¹/punto.
- **La normalización 0-100 resta un desplazamiento y eso cambia los
  cocientes.** Es la única de las normalizaciones que lo hace. Avisa.
- **Un modelo D–G de nanotubo con G⁻ hace falta también con perfil
  forzado**: la BWF de un G⁻ metálico sobrevive al override, porque es
  mecanismo, no preferencia de forma.
- **Γ_G es monótona con el desorden e I_D/I_G no.** En material muy
  desordenado, Γ_G manda. No lo entierres entre los demás índices.
- **Raman no distingue MWCNT de CNF con fiabilidad.** Peso bajo y aviso
  explícito cuando cae en el solape. No subas ese peso.
- **Con S, P y Se la G puede BAJAR aunque el dopado sea de tipo n**, porque
  el enlace C–X más largo ablanda la red más de lo que la carga la endurece.
  Los rangos de esos dopantes cruzan el cero a propósito.
- **Los materiales con `role: "reference"` no compiten en la clasificación.**
  Un MWCNT dopado con N sigue siendo un MWCNT.
- **Un ajuste por mínimos cuadrados SIEMPRE devuelve componentes.** Sobre un
  espectro de ceros el modelo de cinco bandas devolvía D4, D, D3, G y D′ y el
  informe anunciaba «MWCNT, confianza media, I_D/I_G = 1.43». Las componentes
  ajustadas pasan por un umbral de significancia antes de asignarse
  (`MIN_COMPONENT_SIGNIFICANCE`). No lo quites.
- **Sin banda G no se clasifica nada.** Es el único modo permitido en primer
  orden del carbono sp² y la referencia de todos los cocientes. Un espectro
  que solo cubría la región 2D se clasificaba como grafito.
- **`Spectrum` rechaza intensidades no finitas.** Un NaN atravesaba la línea
  base y el ajuste y salía como una clasificación con aspecto de válida.
- **Usa `core.compat.trapezoid`, nunca `np.trapezoid`.** El paquete declara
  `numpy>=1.24` y `np.trapezoid` es de NumPy 2.0. Hay una prueba que audita
  esto.
- **La detección de fases no carbonosas exige corroboración**: varias líneas
  fuertes de la especie, no una. Con una sola, en la ventana RBM la
  coincidencia es más probable que la especie, y la primera versión tiró
  cuatro de cinco RBM auténticos.
- **El escaneo de interferencias va APAGADO por defecto**, en la API, en la
  CLI y en la GUI. Es una decisión, no un descuido.
- **`analyse_many` fija el BLAS a un hilo por proceso y usa `spawn`.** Sin lo
  primero el paralelismo es 8× más lento que el serie; sin lo segundo la
  variable de entorno no llega al hijo. Y si el script del usuario no tiene
  guarda `__main__`, degrada a serie con un aviso: no lo conviertas en error.
- **Las incertidumbres analíticas del ajuste son cortas.** Para números que
  se publican, usa `models.bootstrap`. Documenta siempre que el remuestreo
  mide reproducibilidad frente al ruido, no acierto.

## Honestidad sobre la validación

Todo está validado contra espectros **sintéticos** generados por
`examples/demo_data.py`. Eso comprueba las fórmulas y la lógica, no la
exactitud sobre datos reales. Cualquier documentación que escribas debe
decirlo; no escribas nada que sugiera que el paquete está contrastado contra
medidas.

## Comandos

```bash
pip install -e ".[dev]"
pytest ramancarbon/tests -q
ruff check ramancarbon
ramancarbon demo datos/
ramancarbon analizar datos/demo_DWCNT_532nm.txt --laser 532 --auto --perfil pseudo_voigt
ramancarbon laseres datos/m_532nm.txt datos/m_633nm.txt
```

## La capa Tk no tiene pruebas de ejecución

No hay pantalla en el entorno donde se construyó esto, así que
`gui/app.py` nunca se ha ejecutado. Toda su lógica vive en `gui/state.py` y
`gui/plots.py`, que sí se prueban. `tests/test_gui_wiring.py` comprueba por
análisis del código que las piezas encajan — cada canvas tiene su pestaña y
su función de dibujo, cada `command=self._x` existe, cada `self.attr` que se
lee se asigna en algún sitio. Eso ya ha cazado dos errores reales que ni el
linter ni las pruebas veían. No cubre si la ventana se ve bien.
