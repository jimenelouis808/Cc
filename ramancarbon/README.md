# ramancarbon

Analiza **espectros Raman experimentales** de nanomateriales de carbono:
distingue SWCNT / DWCNT / MWCNT, deconvoluciona las bandas D y G, calcula
I_D/I_G, I_2D/I_G e I_D/I_D', deduce diámetros a partir del RBM y compara
las posiciones medidas con la literatura para detectar dopado o deformación.

> Este proyecto es hermano de [`carbonforge`](../carbonforge), pero hace lo
> contrario: `carbonforge` **prepara cálculos** de primeros principios;
> `ramancarbon` **analiza lo que sale del espectrómetro**.

## Qué lo diferencia

**Nada está codificado a fuego.** Cada número que viene de un artículo vive
en un JSON editable con su `source` y su `confidence` al lado
(`ramancarbon/database/data/`). Puedes corregir una ventana de banda para
que encaje con tu equipo, o añadir tu propia parametrización del RBM, sin
tocar una línea de Python — y el cambio llega a la interfaz, a la línea de
comandos y a los informes de golpe. Los informes citan la referencia de
cada cosa que afirman.

**Corrige por dispersión antes de comparar nada.** La banda D está a
1350 cm⁻¹ a 532 nm y a 1331 cm⁻¹ a 633 nm; la 2D se mueve el doble. Todas
las ventanas de la base de datos están definidas a 2.33 eV y se trasladan
al láser que hayas usado. Sin eso, un espectro a 785 nm «no tiene banda D»
y muestra un «desplazamiento» de 38 cm⁻¹ que es pura aritmética.

**Distingue «ausente» de «no medido».** Un espectro que empieza en
400 cm⁻¹ no dice absolutamente nada sobre si hay RBM. El clasificador se
niega a usar esa ausencia como prueba y lo dice en el informe, en vez de
llamar multipared a todo lo que no llega abajo.

**Dice por qué.** La identificación no es una probabilidad opaca: es una
lista de evidencias con su peso y una frase que explica cada una.

```
Identificación: Nanotubo de pared doble (DWCNT)  (confianza alta)

Evidencia:
  [+4.0] hay 4 modo(s) de respiración radial (158, 178, 265, 290 cm⁻¹). El RBM
         solo existe en tubos de diámetro pequeño: descarta material multipared
         grueso, grafeno, grafito y carbono amorfo
  [+4.0] dos RBM emparejan como paredes concéntricas: 158 cm⁻¹ (d=1.58 nm) y
         265 cm⁻¹ (d=0.86 nm), separación 0.362 nm, dentro del rango
         pared-pared 0.335–0.36 nm
  [+2.5] la banda G aparece desdoblada en G⁻/G⁺ por la curvatura de la pared
```

**Avisa de lo que puede salir mal.** Parámetros pegados a sus límites,
componentes casi degeneradas, cocientes tomados de alturas cuando la
fórmula se calibró con áreas, incertidumbres optimistas porque suavizaste
antes de ajustar. Todo eso sale escrito.

## Instalación

```bash
./install.sh          # Linux / macOS
install.bat           # Windows
```

Manualmente:

```bash
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate en Windows
pip install -e ".[dev]"
```

Python 3.10+. Dependencias: `numpy`, `scipy`, `matplotlib`. La interfaz
gráfica usa Tkinter, que viene con Python (en Debian/Ubuntu:
`sudo apt install python3-tk`).

## Empezar

```bash
ramancarbon demo datos/                       # genera espectros de prueba
ramancarbon analizar datos/demo_DWCNT_532nm.txt --laser 532
ramancarbon-gui                               # la interfaz gráfica
```

Desde Python:

```python
from ramancarbon import read_spectrum, analyse

spectrum = read_spectrum("muestra.txt", laser_nm=532)
result = analyse(spectrum)

print(result.report())
print(result.id_ig, result.i2d_ig, result.id_idprime)
print(result.classification.label, result.classification.confidence)
```

## Qué hace

| Módulo | Qué contiene |
|--------|--------------|
| `core` | Contenedor `Spectrum`, lectores de archivo, línea base, eliminación de rayos cósmicos, detección de picos |
| `models` | Lorentziana, gaussiana, pseudo-Voigt y **Breit-Wigner-Fano**; motor de ajuste con límites; preajustes de deconvolución (2 a 5 bandas, región G de nanotubo, RBM, 2D) |
| `database` | Bandas y dispersiones, 13 materiales de referencia, 5 parametrizaciones RBM, firmas de dopado y deformación — todo en JSON con su fuente |
| `analysis` | Asignación de bandas, cocientes, diámetros, desplazamientos, clasificador |
| `gui` | Aplicación de escritorio Tkinter |
| `cli` | `ramancarbon analizar / lote / deconvolucionar / bd / demo` |

### Bandas y cocientes

Identifica RBM, D4, D, D3, G⁻, G, G⁺, D', M, iTOLA, D+D'', 2D, D+D', 2D' y
la línea del diamante a 1332 cm⁻¹ — cada una con su ventana, su dispersión,
su anchura típica y su origen físico.

* **I_D/I_G** → tamaño de cristalito `L_a` y densidad de defectos `n_D`
  (Cançado). Se informan **las dos ramas** de la relación de
  Tuinstra–Koenig, porque I_D/I_G no crece de forma monótona con el
  desorden: sube, alcanza un máximo cuando los defectos están a ~3 nm, y
  luego baja. Citar solo la rama de bajo desorden puede errar en un orden
  de magnitud.
* **I_2D/I_G** → número de capas en material tipo grafeno, con el aviso de
  que el dopado baja este cociente sin añadir ni una capa: manda la
  **anchura** de la 2D, no el cociente.
* **I_D/I_D'** → **tipo** de defecto: sp³ (~13), vacantes (~7), bordes
  (~3.5), sustitucional (~1.3). Requiere deconvolución, porque D' es un
  hombro sobre G.

Cada cociente lleva su valor en **las dos bases** (áreas y alturas), porque
la literatura cita unas veces una y otras veces la otra, y difieren en un
factor 2–3.

### Diámetros

Dos rutas independientes, y se cruzan:

1. **RBM**: `ω = A/d + B`, con las cinco parametrizaciones estándar y la
   dispersión entre ellas como incertidumbre sistemática honesta.
2. **Desdoblamiento G**: `ω_G⁻ = ω_G⁺ − C/d²`. No necesita bajar a
   100 cm⁻¹. Si es metálico o semiconductor cambia el resultado un 30 %, así
   que la metalicidad **se decide ajustando** G⁻ con perfil BWF y con
   lorentziana y comparando, no adivinando.

Si las dos rutas no coinciden, el informe lo dice y sugiere qué revisar, en
lugar de promediarlas.

También lista los `(n,m)` compatibles — dejando claro que el diámetro por sí
solo no fija la quiralidad — y comprueba si dos RBM emparejan como **paredes
concéntricas**, que es lo único que distingue un DWCNT de una mezcla de
SWCNT de dos diámetros.

### Deconvolución

Preajustes de 2, 3, 4 y 5 bandas (el modelo de Sadezky para hollín), un
modelo específico de nanotubo con G⁻ y G⁺, y modelos para el RBM y la 2D.
El número de componentes se elige por **criterio de información**, no por
costumbre: añadir componentes siempre mejora el R², así que el R² no puede
decidirlo.

El perfil también es una afirmación física. La G⁻ de un nanotubo **metálico**
es Breit-Wigner-Fano, no lorentziana; ajustarla mal desplaza la posición
varios cm⁻¹ e infla la D' de al lado.

### Desplazamientos, dopado y deformación

La banda G **sube con electrones y con huecos** (bloqueo de Pauli de la
anomalía de Kohn), así que ΔG sola no da el signo del dopado. La 2D sí: sube
con huecos y baja con electrones. Y como la deformación y el dopado mueven
la pareja (ω_G, ω_2D) en direcciones distintas del plano (pendientes 2.2 y
0.7), un desplazamiento medido **se descompone** en sus dos contribuciones.

## Limitaciones — léelas

* **Todo se ha validado contra datos sintéticos**, generados por el propio
  programa. Los espectros reales tienen ruido correlacionado, respuesta del
  instrumento, líneas del sustrato, inhomogeneidad de muestra y alas no
  lorentzianas. Nada de eso está en las pruebas. Contrasta con patrones
  propios antes de fiarte de un número para publicar.
* **No lee formatos binarios de fabricante** (Renishaw `.wxd`, Thermo
  `.spa`, Bruker `.opus`). Expórtalos como texto.
* **La asignación de quiralidad devuelve un conjunto, no una respuesta.**
  Fijar `(n,m)` requiere la condición de resonancia (Kataura), y eso
  necesita medir con varios láseres.
* **La separación deformación/dopado está calibrada para grafeno
  monocapa.** En nanotubos y multipared da números que parecen precisos y no
  lo son; el programa lo avisa cada vez.
* **Raman no separa las configuraciones del nitrógeno** (grafítico,
  piridínico, pirrólico). Para eso hace falta XPS. La base de datos lo dice
  en vez de fingir lo contrario.
* **Calibra tu equipo** en la misma sesión (la línea de 520.7 cm⁻¹ del
  silicio) antes de interpretar desplazamientos de pocos cm⁻¹: la deriva
  típica de un espectrómetro es de ese mismo orden. Y comprueba la potencia
  del láser: unos pocos mW sobre un polvo negro lo calientan cientos de
  kelvin y bajan la G, imitando una deformación de tracción.

## Documentación

* [**Guía rápida**](GUIA_RAPIDA.md) — instalación y uso paso a paso.
* [**Guía de desarrollo**](DESARROLLO.md) — arquitectura, pruebas, dónde
  tocar cada cosa.

## Pruebas

```bash
pytest ramancarbon/tests -q
```

## Licencia

MIT.
