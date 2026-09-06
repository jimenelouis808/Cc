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
├── analysis/    # asignación, cocientes, diámetros, desplazamientos, clasificador
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
ramancarbon demo datos/ && ramancarbon analizar datos/demo_DWCNT_532nm.txt --laser 532
```
