# ramancarbon

Suite de caracterización de nanomateriales, con **cuatro instrumentos** en
una sola aplicación:

| Sección | Qué hace |
|---|---|
| **Raman · carbono** | SWCNT / DWCNT / MWCNT, deconvolución D–G configurable, I_D/I_G, I_2D/I_G, I_D/I_D′, diámetros por RBM, dopado y deformación, y las fases no carbonosas de la muestra (FeSe, Se, carburos de hierro) |
| **Raman · TMD** | MoS₂, WS₂, MoSe₂, WSe₂, MoTe₂: número de capas, fase 2H/1T′, y las heteroestructuras óxido/calcogenuro (MoO₃@MoSe₂, MoO₂@MoSe₂…) |
| **DRX** | Identificación de fases contra estructuras cristalinas reales (CIF de la COD), patrón teórico, residual y refinamiento **Rietveld** automático o a mano |
| **Electroquímica** | CV, carga-descarga, impedancia con circuitos equivalentes, mecanismo de almacenamiento (condensador / pseudocondensador / batería), capacitancia y capacidad, energía y potencia, HER y OER |

> Este proyecto es hermano de [`carbonforge`](../carbonforge), pero hace lo
> contrario: `carbonforge` **prepara cálculos** de primeros principios;
> `ramancarbon` **analiza lo que sale del equipo**.

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

**Se niega a dar el número equivocado.** Un material con picos redox tiene
*capacidad*, no capacitancia: dividir la carga por la ventana entera y dar
F/g infla la cifra y describe algo que el material no hace. El programa
clasifica el mecanismo y, si es de tipo batería, dice explícitamente que no
se informe en faradios. Lo mismo con una pendiente de Tafel ajustada sobre
menos de una década, con un tamaño de cristalito por encima de la
resolución del equipo, o con un ECSA presentado sin el factor de tres que
arrastra.

**En difracción no hay tablas de posiciones de pico.** Una fase es una
*estructura cristalina* y cada posición e intensidad se calcula de ella, así
que un parámetro de red refinado mueve los picos como los movería la
física. Añadir una fase de referencia es soltar su CIF en una carpeta.

**Los picos sin explicar son el resultado, no el resto.** Son la fase que no
esperabas. Aparecen en su propia lista, marcados, y con la advertencia de
que un Rietveld sin ellos reparte su intensidad entre las fases que sí están.

**Avisa de lo que puede salir mal.** Parámetros pegados a sus límites,
componentes casi degeneradas, cocientes tomados de alturas cuando la
fórmula se calibró con áreas, incertidumbres optimistas porque suavizaste
antes de ajustar. Todo eso sale escrito.

## Instalación

```bash
./install.sh          # Linux / macOS
install.bat           # Windows
```

Guía completa paso a paso, incluida la resolución de problemas:
[**INSTALACION.md**](INSTALACION.md).

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
ramancarbon-gui                               # la suite entera, con sus cuatro secciones
```

O por línea de comandos, una sección cada vez:

```bash
# Raman
ramancarbon demo datos/                       # genera espectros de prueba
ramancarbon analizar datos/demo_DWCNT_532nm.txt --laser 532
ramancarbon analizar datos/demo_MWCNT_FeSe_532nm.txt --laser 532 --picos-d 3 --picos-g 2
ramancarbon tmd datos/demo_MoS2_2capa_532nm.txt

# Difracción
ramancarbon demo-datos drx drx/
ramancarbon drx drx/demo_drx_CNT_FeSe.xye --textura 001
ramancarbon drx --biblioteca                  # qué fases de referencia hay
ramancarbon drx-lote drx/ --csv fases.csv

# Electroquímica
ramancarbon demo-datos echem ec/
ramancarbon echem --cv ec/demo_cv_condensador_20mVs.txt --masa 2 --area 1
ramancarbon echem --gcd ec/demo_gcd_bateria.txt --eis ec/demo_eis.txt --masa 2
ramancarbon echem --polarizacion ec/demo_lsv_OER.txt --referencia RHE \
                  --area 1 --resistencia 3 --reaccion OER
```

Desde Python:

```python
from ramancarbon import read_spectrum, analyse

spectrum = read_spectrum("muestra.txt", laser_nm=532)
result = analyse(spectrum)
print(result.report())
print(result.id_ig, result.i2d_ig, result.id_idprime)
print(result.classification.label, result.classification.confidence)
print(result.phases.summary())          # FeSe, Se, carburos…
```

```python
from ramancarbon.xrd import read_pattern
from ramancarbon.xrd.report import analyse_pattern

pattern = read_pattern("muestra.xye", anode="Cu")
xrd = analyse_pattern(pattern, preferred_axis=(0, 0, 1))
print(xrd.report())
print(xrd.refinement.weight_fractions())
```

```python
from ramancarbon.echem.curve import Electrode
from ramancarbon.echem.io import read_cv, read_gcd
from ramancarbon.echem.report import analyse_sample

electrode = Electrode(mass_mg=2.0, area_cm2=1.0, reference="Ag/AgCl_3M", ph=14.0)
sample = analyse_sample(
    "mi electrodo",
    cv=read_cv("cv.txt", scan_rate=0.02, electrode=electrode),
    gcd=read_gcd("gcd.txt", electrode=electrode),
)
print(sample.storage.summary())         # condensador, pseudo o batería
print(sample.report())
```

## Qué hace

| Módulo | Qué contiene |
|--------|--------------|
| `core` | Contenedor `Spectrum`, lectores de archivo, líneas base (asLS, arPLS, polinómica, banda elástica) con **elección automática de parámetros**, eliminación de rayos cósmicos, detección de picos |
| `models` | Lorentziana, gaussiana, pseudo-Voigt y **Breit-Wigner-Fano**; motor de ajuste con límites; preajustes de deconvolución (2 a 5 bandas, región G de nanotubo, RBM, 2D) |
| `database` | Bandas y dispersiones, 13 materiales de referencia, 5 parametrizaciones RBM, firmas de dopado y deformación — todo en JSON con su fuente |
| `analysis` | Asignación de bandas, cocientes, **índices estructurales**, diámetros, desplazamientos, clasificador, **combinación multiláser**, **calidad de la medida**, **lote en paralelo con estadística**, **bandas no carbonosas** (opcional), **exportación** |
| `gui` | Aplicación de escritorio Tkinter |
| `cli` | `ramancarbon analizar / lote / deconvolucionar / laseres / tmd / calibrar / bd / demo` |

### Bandas y cocientes

Identifica RBM, D4, D, D3, G⁻, G, G⁺, D', M, iTOLA, D+D'', 2D, D+D', 2D' y
la línea del diamante a 1332 cm⁻¹ — cada una con su ventana, su dispersión,
su anchura típica y su origen físico.

* **Γ_G (anchura de G)** → el índice que **sí** es monótono con el desorden
  en todo el rango, de ~15 cm⁻¹ en grafito a >150 en carbono amorfo. En
  material muy desordenado — MWCNT, nanofibras, muestras muy dopadas — es
  el número más útil y el que menos se publica.
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

* **R1 y R2 (Beyssac)** → parámetros de orden basados en áreas, mucho menos
  sensibles que I_D/I_G a cómo el ajuste repartió la intensidad. El
  termómetro geológico asociado a R2 se declara explícitamente **no
  aplicable** a nanotubos.
* **A_D3/A_G y A_D4/A_G** → fracción amorfa y fracción sp³/poliénica.
* **Etapa de amorfización** (Ferrari–Robertson) → lee ω_G junto a I_D/I_G
  para decir si un I_D/I_G de 1.0 significa «grafito con defectos» o «casi
  amorfo». Es lo que resuelve la ambigüedad de la rama.

Cada cociente lleva su valor en **las dos bases** (áreas y alturas), porque
la literatura cita unas veces una y otras veces la otra, y difieren en un
factor 2–3.

### Dos láseres: lo que 532 y 633 nm permiten y uno solo no

```bash
ramancarbon laseres muestra_532nm.txt muestra_633nm.txt
```

1. **Distinguir una banda real de una impostora.** La D se desplaza
   ~50 cm⁻¹/eV; casi nada más lo hace. Una línea fija cerca de 1332 cm⁻¹ es
   diamante, no la banda D.
2. **Detectar carbono amorfo.** La banda G **no** dispersa en grafito ni en
   grafito nanocristalino — es un modo del centro de zona. Solo dispersa
   (6–10 cm⁻¹/eV) cuando hay sp² amorfo. Ninguna medida a un solo láser
   puede ver esto, porque a una sola energía la G también se mueve por
   dopado y por deformación.
3. **Comprobar la corrección λ⁴.** I_D/I_G escala como λ⁴, así que el mismo
   material da 0.98 a 532 nm y 1.99 a 633 nm — pero el `L_a` deducido debe
   salir igual. Si no sale igual, o los puntos medidos no son el mismo, o
   las bases de los cocientes no coinciden.

### Preprocesado: automático, y editable

```bash
ramancarbon analizar muestra.txt --laser 532 --auto
```

El modo automático **escribe en los mismos campos que editarías a mano** y
explica cada decisión, en vez de esconderlas:

* **Suavizado**: no suaviza si la banda más intensa ya pasa de SNR 40, y
  cuando suaviza limita la ventana a un tercio de la banda más estrecha
  aunque el ruido justificara más — y lo dice.
* **Línea base**: la rigidez sale de la función de transferencia del
  suavizador de Whittaker, fijando el corte en cinco anchuras de banda. No
  hay búsqueda ni ajuste empírico, y escala correctamente con el paso de
  muestreo (el mismo material a 1 y a 4 cm⁻¹/punto necesita rigideces que
  difieren en 256×).
* Avisa cuando el fondo varía tan rápido como las bandas, caso en que
  ninguna línea base puede separarlos.

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

**Tú eliges cuántos picos van en cada región.** Si trabajas con tres
componentes en la D y dos en la G, que es una convención habitual:

```bash
ramancarbon deconvolucionar muestra.txt --picos-d 3 --picos-g 2
```

Eso da D4 + D + D3 + G + D′, es decir el modelo de Sadezky. Las componentes
van **con nombre mientras la física se lo dé**: las tres primeras de la
región D son D, D3 y D4, en el orden en que la literatura las añade; las
tres primeras de la G son G, D′ y G⁻. Si pides más, las extra salen como
`Dx1`, `Gx1`… repartidas en los huecos más anchos, y **no entran en los
cocientes ni en la clasificación**, porque una componente sin nombre no
tiene interpretación física.

También hay preajustes de 2, 3, 4 y 5 bandas, un modelo específico de
nanotubo con G⁻ y G⁺, y modelos para el RBM y la 2D.
El perfil se elige para todas las componentes a la vez: **pseudo-Voigt**,
gaussiana, lorentziana, o los valores por defecto de cada banda.

Pseudo-Voigt con η libre contiene la gaussiana (η=0) y la lorentziana (η=1)
como casos particulares, así que en vez de imponer la forma **el ajuste la
mide** y devuelve η como resultado. Cuesta un parámetro más por banda.

Todo se exporta: tabla de componentes con áreas, porcentajes e
incertidumbres; las curvas punto a punto con una columna por componente para
redibujarlas en Origin; y el JSON completo. Cada archivo lleva cabecera con
el láser, el preprocesado y el modelo.

```bash
ramancarbon analizar muestra.txt --laser 532 --perfil pseudo_voigt --exportar resultados/
```
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

### Calidad de la medida

Dos formas de arruinar un espectro que **no se ven en el informe final**, y
que ahora se detectan antes de analizar:

* **Detector saturado.** Recortar el 3 % superior de un espectro llevó
  I_D/I_G de 0.98 a 2.01. La saturación aplana primero la banda más intensa,
  que es la G, así que infla todo lo que se mide contra ella.
* **Muestreo demasiado grueso.** El mismo espectro a 16 cm⁻¹/punto dio
  I_D/I_G = 3.91 en lugar de 0.98. Por debajo de ~6 puntos por FWHM el
  ajuste no tiene con qué fijar la anchura.

Además detecta truncamiento, señal/ruido baja, intensidades negativas y
espectros sin variación. Y si hay una línea de silicio en el rango:

```bash
ramancarbon calibrar muestra.txt --corregir muestra_cal.txt
```

El fonón del silicio está en 520.7 cm⁻¹ exactos, así que su desviación es el
error de calibración de tu equipo ese día — del mismo tamaño que los
desplazamientos por dopado que luego quieres interpretar.

### Lotes: en paralelo, y con estadística honesta

```bash
ramancarbon lote datos/ --laser 532 --csv resultados.csv --procesos 4
```

Da la **mediana con dispersión robusta** junto a la media, y marca los
puntos atípicos por número. Un punto medido sobre un grumo de catalizador
mueve mucho la media y casi nada la mediana.

Sobre la paralelización hay un detalle que merece la pena saber, porque
afecta a cualquier programa científico en Python: el primer intento salió
**ocho veces más lento**. La causa es el BLAS de NumPy, que abre un hilo por
núcleo en cada proceso — cuatro procesos × cuatro hilos en cuatro núcleos.
Fijándolo a un hilo por proceso, 47 s pasaron a 1.1 s. El paquete lo hace
solo para sus procesos hijos, pero no puede hacerlo para el tuyo, así que
si procesas lotes grandes a menudo:

```bash
export OMP_NUM_THREADS=1
```

acelera todo, en paralelo o no.

### Incertidumbres realistas

Las barras de error que devuelve un ajuste vienen de la curvatura de χ² y
son **sistemáticamente cortas**: no incluyen la correlación entre
parámetros, ni la correlación de los residuos. Volver a ajustar sobre datos
remuestreados sí las incluye. En una banda G típica, el ajuste dice
±0.05 cm⁻¹ y el remuestreo dice ±0.24 — cinco veces más, y eso importa
cuando interpretas desplazamientos de pocos cm⁻¹.

### Bandas que no son carbono (opcional, desactivado por defecto)

```bash
ramancarbon analizar muestra.txt --laser 532 --interferencias
```

Óxidos de catalizador, precursor de dopante sin reaccionar, sustrato. Está
apagado por defecto a propósito: decirle a un buscador lo que podría
encontrar sesga lo que informa.

Una especie solo se acepta cuando aparecen **varias de sus líneas fuertes**,
no una. La primera versión emparejaba línea a línea con ±8 cm⁻¹ y, sobre una
muestra limpia de pared simple, descartó cuatro de sus cinco RBM auténticos
llamándolos óxido de cobalto y selenio — en la ventana 100–400 cm⁻¹ hay una
docena de líneas catalogadas y una coincidencia es más probable que la
especie. Un cristal tiene un espectro, no una línea.

Para muestras dopadas con S, Se o P importa por otra razón: si aparece la
línea de 219 cm⁻¹ del azufre elemental, el dopante **no** entró en la red,
que es lo contrario de lo que su presencia suele darse por demostrar.

## Dicalcogenuros (TMD)

```bash
ramancarbon tmd mos2.txt          # o la pestaña TMD de la interfaz
ramancarbon tmd --listar          # los materiales y sus modos
```

MoS₂, WS₂, MoSe₂, WSe₂ y MoTe₂. Física distinta de la del carbono, con la
misma maquinaria de ajuste y exportación.

**Las capas se cuentan por una separación, no por una posición.** Al apilar,
el modo E²g (en el plano) se ablanda y el A₁g (fuera del plano) se endurece,
así que la distancia entre ellos crece de forma monótona — en MoS₂, de
~19 cm⁻¹ en monocapa a ~25 en bulk. Al ser una **diferencia**, cualquier
error común de calibración se cancela, y por eso es mucho más robusta que
cualquier posición suelta.

Los dos van en direcciones opuestas por razones distintas: el A₁g mueve los
calcógenos perpendicularmente a la capa y la vecina se le opone; el E²g mueve
átomos dentro del plano, donde lo que manda es el apantallamiento dieléctrico
de la interacción de largo alcance, que lo ablanda.

**Donde el método no funciona, lo dice.** En WSe₂ los dos modos son casi
degenerados a ~250 cm⁻¹ y un espectrómetro normal ve una sola banda; allí se
cuenta por la presencia del modo B¹₂g, prohibido por simetría en monocapa. En
MoSe₂ y MoTe₂ la separación apenas cambia y se usa el mismo argumento.

**Fase 1T′.** La distorsión metálica dobla la celda y produce los modos J1,
J2 y J3, que no existen en la fase 2H. Verlos es prueba positiva de fase
metálica; no verlos es evidencia mucho más débil, porque una fracción pequeña
queda bajo el ruido — y el informe lo dice así.

**La resolución importa más que en carbono.** Estas bandas miden 2–6 cm⁻¹ y
las fronteras entre números de capa están a 2–3 cm⁻¹, así que un paso de
muestreo grueso hace imposible contar capas por muy bien que se vea el
ajuste. El programa avisa.

## Fases no carbonosas de la muestra

```bash
ramancarbon analizar nanotubos_FeSe.txt --laser 532
```

Un nanotubo decorado con FeSe pone modos que **no son carbono** en 181, 196,
237 y 254 cm⁻¹ — los cuatro dentro de la ventana del RBM. Convertidos con
`ω = A/d + B` dan cinco diámetros de nanotubo creíbles y falsos. Por eso este
escaneo **sí va encendido por defecto**, al contrario que el de
interferencias.

La base de datos trae β-FeSe tetragonal, δ-FeSe hexagonal, FeSe₂ marcasita,
selenio trigonal / monoclínico / amorfo, cementita Fe₃C y una entrada para el
hierro metálico **deliberadamente sin bandas**, porque es Raman-inactivo en
primer orden: lo que ves en una muestra con hierro nunca es el hierro.

Se informa la **familia** siempre que se pueda, y el **polimorfo** solo cuando
una línea discriminante o una prueba de anchura lo separa de sus hermanos. El
selenio amorfo y el monoclínico están en el mismo sitio (≈250 cm⁻¹) y lo único
que los distingue es el ancho de banda; el programa lo usa, y avisa de que un
límite *superior* de anchura no prueba nada porque lo cumple cualquier RBM.

Cada fase lleva la reflexión y el 2θ que zanjarían la duda en difracción:

```
Por composición:
  Seleniuro de hierro FeSe → β-FeSe tetragonal (tipo PbO)
    Resuelto: aparecen 2 línea(s) propias de este polimorfo.

Lo que zanjaría la duda en DRX:
  · δ-FeSe hexagonal: reflexión 101 a 2θ ≈ 32.4° (Cu Kα).
```

## Difracción de rayos X

```bash
ramancarbon drx patron.xye --cif mis_cifs/ --textura 001
ramancarbon drx --biblioteca
ramancarbon drx-lote datos/ --csv fases.csv
```

**Una fase es una estructura, no una lista de picos.** Cada posición e
intensidad sale del cálculo del factor de estructura, así que refinar un
parámetro de red mueve los picos como los movería la física. Las referencias
son archivos CIF: para añadir una fase, descárgala de la Crystallography Open
Database y suelta el archivo en una carpeta.

No hay búsqueda en línea en la COD, y es deliberado: un resultado que depende
de la red no se reproduce, y los ordenadores de los equipos suelen estar sin
conexión.

Vienen incluidas 14 estructuras de partida (grafito, las dos fases del FeSe,
FeSe₂, Se, Fe, Fe₃C, magnetita, hematita, MoS₂, MoSe₂, WS₂, MoO₂ y silicio
como patrón de calibración), **validadas de dos formas independientes**: la
densidad cristalográfica frente a la literatura, que comprueba celda y
contenido a la vez, y la posición de la reflexión más intensa frente a su
ficha.

Lo que hace además de identificar:

* **Ajusta el cero antes de juzgar nada.** Una muestra 0.1 mm alta desplaza
  todos los picos unas centésimas de grado, y un comparador con ventana
  estrecha rechaza entonces la fase correcta.
* **Pela el Kα₂** (Rachinger) antes de buscar picos. Sin eso, por encima de
  40° cada reflexión se encuentra dos veces y las sobrantes parecen una fase
  desconocida.
* **Separa posiciones de intensidades.** Las posiciones solo dependen de la
  red y son prueba fuerte; las intensidades las estropea la orientación
  preferente en cualquier material laminar, así que un acuerdo bajo de
  intensidades suele ser textura, no una fase equivocada.
* **Refina por etapas.** Escala y fondo, cero, celda, anchura, perfil,
  textura, y al final desplazamiento de muestra y U_iso. Soltarlo todo a la
  vez converge, da factores R plausibles y devuelve una estructura
  equivocada. También se puede refinar a mano, parámetro a parámetro.
* **Pone la curva diferencia antes que los factores R**, con el estadístico
  de rachas de signo. Un Rwp bonito con una ondulación sistemática en el
  residuo es peor ajuste que un Rwp feo con residuo sin estructura.

## Electroquímica

```bash
ramancarbon echem --cv cv.txt --velocidad 20 --masa 2 --area 1
ramancarbon echem --gcd gcd.txt --eis eis.txt --masa 2 --circuito "R0-(R1|Q1)-Q2"
ramancarbon echem --polarizacion lsv.txt --reaccion OER --ph 14 --resistencia 3
```

**Lo primero que dice es qué clase de material tienes**, porque eso decide si
las magnitudes de abajo son las correctas:

```
Mecanismo de almacenamiento: Tipo batería (faradaico con difusión) (confianza alta)
  Magnitud correcta para informar: capacidad (C/g o mAh/g), NUNCA F/g
  A favor:
    · picos redox estrechos (52 mV sobre una ventana de 604 mV) y 4.5 veces el fondo
    · la descarga tiene meseta (R² lineal = 0.859)
    · b = 0.52, cerca de 0.5: corriente controlada por difusión

  ⚠ Este electrodo NO debe informarse en F/g.
```

Lo demás:

* **Capacitancia en los dos convenios**, cada uno con su nombre. Integrar el
  lazo cerrado y dividir por 2νΔV, o integrar la rama anódica y dividir por
  νΔV: publicar sin decir cuál deja un factor 2 de ambigüedad.
* **Estudio de velocidad**: b de `i = a·ν^b` a lo largo de la ventana, reparto
  capacitivo/difusivo de Dunn, Trasatti, y C_dl → ECSA **con el factor de tres**
  que arrastra la Cs que se adopte.
* **Carga-descarga**: caída IR medida y quitada de la ventana, capacitancia y
  capacidad, eficiencia culómbica y energética, energía **integrada** (∫V dq,
  válida también para una meseta) y potencia.
* **Impedancia**: circuitos escritos como `R0-(R1|Q1)-Wo1`, ajuste no lineal
  complejo en espacio logarítmico con reinicios, y **Kramers-Kronig antes que
  el circuito** — un ajuste a datos que derivaron durante la medida da
  parámetros sin significado y el χ² no lo delata.
* **HER y OER**: sobrepotencial al punto de comparación, pendiente de Tafel
  con la región lineal **buscada** y no supuesta, corriente de intercambio,
  actividad másica y normalización por ECSA. Se niega a dar una pendiente
  ajustada sobre menos de una década.

## Limitaciones — léelas

* **Todo se ha validado contra datos sintéticos**, generados por el propio
  programa. Los espectros reales tienen ruido correlacionado, respuesta del
  instrumento, líneas del sustrato, inhomogeneidad de muestra y alas no
  lorentzianas. Nada de eso está en las pruebas. Contrasta con patrones
  propios antes de fiarte de un número para publicar.
* **No lee formatos binarios de fabricante** (Renishaw `.wxd`, Thermo
  `.spa`, Bruker `.opus`, Bruker `.raw`/`.brml` en difracción). Expórtalos
  como texto; el programa dice qué formato es y qué exportar.
* **En difracción no hay dispersión anómala (f′, f″).** El efecto en las
  intensidades calculadas es de un pequeño porcentaje. Lo que de verdad
  arruina una muestra de hierro con tubo de cobre es la fluorescencia, que
  sube el fondo y no toca los picos, y de eso sí avisa.
* **Las fracciones en peso de Rietveld son de la parte cristalina e
  identificada.** Una fase que no hayas modelado no baja el total de 100 %:
  su intensidad se reparte entre las que sí están. El amorfo no aparece.
* **El ECSA arrastra un factor de tres** por la capacitancia específica que
  se adopte. Sirve para comparar tus muestras entre sí, no contra la
  literatura.
* **Raman no ve topología.** Un MoO₃@MoSe₂ y una mezcla física de polvos dan
  el mismo espectro puntual. Lo que sí se mide es la deformación del
  calcogenuro, que es indicio y no prueba.
* **La asignación de quiralidad devuelve un conjunto, no una respuesta.**
  Fijar `(n,m)` requiere la condición de resonancia (Kataura), y eso
  necesita medir con varios láseres.
* **La separación deformación/dopado está calibrada para grafeno
  monocapa.** En nanotubos y multipared da números que parecen precisos y no
  lo son; el programa lo avisa cada vez.
* **Raman no distingue MWCNT de nanofibra de carbono con fiabilidad.** Los
  rangos de I_D/I_G y Γ_G se solapan casi por completo; la diferencia real
  está en si los planos grafénicos son cilindros concéntricos o están
  inclinados, y eso lo ve el TEM. El clasificador lo intenta con peso bajo y
  avisa cuando cae en la zona de solape.
* **Con dopantes grandes (S, P, Se) la lectura «la G sube = dopado»
  falla.** El enlace C–X es mucho más largo que el C–C (1.78 Å para C–S
  frente a 1.42), y la deformación local que eso introduce ablanda la G
  contra la transferencia de carga que la endurece. Las dos contribuciones
  son comparables y de signo contrario, así que los rangos de esos dopantes
  cruzan el cero y llevan confianza baja.
* **Las nanopartículas metálicas y el efecto SERS son opcionales y están
  apagados por defecto** (`--interferencias`). Cuando se activan, una especie
  solo se acepta si aparecen varias de sus líneas fuertes.
* **Los datos de MoSe₂, WSe₂ y MoTe₂ tienen confianza baja.** Hay mucha menos
  literatura cuantitativa que para MoS₂, y sus tablas de separación por capa
  son orientativas. MoS₂ es el único con confianza alta.
* **Raman no separa las configuraciones de un mismo dopante** (nitrógeno
  grafítico / piridínico / pirrólico; azufre tiofénico / sulfóxido) **y no
  cuantifica el contenido de dopante.** Eso es XPS. Lo que Raman aporta y
  XPS no es el **tipo** de defecto (por I_D/I_D′) y el **signo** de la
  transferencia de carga (por el par ΔG, Δ2D).
* **Calibra tu equipo** en la misma sesión (la línea de 520.7 cm⁻¹ del
  silicio) antes de interpretar desplazamientos de pocos cm⁻¹: la deriva
  típica de un espectrómetro es de ese mismo orden. Y comprueba la potencia
  del láser: unos pocos mW sobre un polvo negro lo calientan cientos de
  kelvin y bajan la G, imitando una deformación de tracción.

## Documentación

* [**Instalación**](INSTALACION.md) — paso a paso, sin dar nada por sabido.
* [**Guía rápida**](GUIA_RAPIDA.md) — cómo se usa.
* [**Guía de desarrollo**](DESARROLLO.md) — arquitectura, pruebas, dónde
  tocar cada cosa.

## Ejemplos ejecutables

Cada uno demuestra una cosa concreta, con números:

```bash
python -m ramancarbon.examples.ex01_analizar        # el análisis completo
python -m ramancarbon.examples.ex02_deconvolucion   # modelos de 2 a 5 bandas
python -m ramancarbon.examples.ex03_diametros       # RBM → diámetro
python -m ramancarbon.examples.ex04_dopado          # deformación vs dopado
python -m ramancarbon.examples.ex07_dos_laseres     # dispersión
python -m ramancarbon.examples.ex08_ruido_y_fluorescencia
python -m ramancarbon.examples.ex10_tmd             # contar capas
python -m ramancarbon.examples.ex11_fases           # FeSe: cinco diámetros falsos
python -m ramancarbon.examples.ex12_drx             # Rietveld de ida y vuelta
python -m ramancarbon.examples.ex13_echem           # F/g cuando no toca
```

`ex11`, `ex12` y `ex13` son los que más rápido explican por qué el programa
hace lo que hace:

* **ex11** analiza el mismo espectro con y sin el escaneo de fases. Sin él
  salen cinco diámetros de nanotubo creíbles en una muestra que no tiene ni
  uno.
* **ex12** calcula un difractograma de tres fases y lo refina: las celdas
  vuelven con seis cifras.
* **ex13** pone tres electrodos con ~25 F/g cada uno al lado y explica cuál
  de ellos no debe informarse así.

## Pruebas

```bash
pytest ramancarbon/tests -q            # ~620 pruebas
```

## Licencia

MIT.
