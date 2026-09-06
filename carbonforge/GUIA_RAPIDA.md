# Manual de carbonforge

---

## Empieza aquí: tres comandos

Si ya tienes el programa instalado (si no, ve a la [sección 2](#2-instalación)):

```bash
carbonforge-gui                    # interfaz gráfica
carbonforge presets                # ver las recetas disponibles
carbonforge ribbon --width 6 --length 3 --edge zigzag \
                   --preset bands --out mi_calculo
```

Ese último comando construye una nanocinta, **detecta que sus bordes son
magnéticos**, activa la polarización de espín, relaja la geometría, calcula
las bandas sobre la estructura relajada, elige cómo paralelizar, y te explica
cada decisión. Sin que tengas que saber nada de eso.

---

## ¿Qué quieres hacer?

| Quiero… | Ve a |
|---|---|
| Instalarlo | [§2](#2-instalación) |
| Usar la ventana en vez de la terminal | [§3](#3-la-interfaz-gráfica) |
| Que el programa elija la física por mí | [§4](#4-recetas-la-forma-fácil) |
| Poner grupos -NH₂, -OH, -COOH… | [§5](#5-grupos-funcionales-y-nitrógeno) |
| Estudiar nitrógeno dopante | [§5](#5-grupos-funcionales-y-nitrógeno) y [§10](#10-densidad-de-estados-de-dónde-salen-los-estados) |
| Calcular bandas, Raman, IR, espín-órbita | [§6](#6-calcular-espectros-bandas-y-espín-órbita) |
| Ver los resultados que me dio el clúster | [§8](#8-ver-los-resultados) |
| Saber qué pseudopotenciales bajar | [§9](#9-pseudopotenciales-los-archivos-que-tienes-que-descargar) |
| Asegurarme de que mis números están convergidos | [§11](#11-convergencia-los-valores-por-defecto-no-están-convergidos) |
| Hacer dinámica molecular | [§13](#13-qué-hacer-con-los-archivos-generados) |
| Resolver un error | [§14](#14-problemas-frecuentes) |
| Saber de qué NO fiarme | [§15](#15-límites-que-conviene-conocer) |

---

## Lo que este programa hace y lo que no

**Sí:** construye estructuras, decide los parámetros de la simulación, escribe
los archivos de entrada de Quantum ESPRESSO, SIESTA y LAMMPS, y lee y grafica
los resultados cuando terminan.

**No:** ejecutar los cálculos. Eso lo haces tú, en tu máquina o en un clúster.
Tampoco incluye pseudopotenciales.

---

## 1. Requisitos

### Python

Necesitas **Python 3.10 o superior**. Para comprobar si ya lo tienes, abre
una terminal y escribe:

```bash
python --version
```

Si responde `Python 3.10.x` o superior, ya está. Si dice que no existe el
comando, prueba con `python3 --version`.

Si no lo tienes:

| Sistema | Cómo instalarlo |
|---|---|
| **Windows** | Descarga desde [python.org](https://www.python.org/downloads/). **Importante:** marca la casilla *"Add Python to PATH"* al principio del instalador, y deja marcada *"tcl/tk and IDLE"* (es lo que dibuja la ventana). |
| **macOS** | Descarga desde [python.org](https://www.python.org/downloads/), o `brew install python-tk` si usas Homebrew. |
| **Linux (Ubuntu/Debian)** | `sudo apt install python3 python3-pip python3-tk` |
| **Linux (Fedora)** | `sudo dnf install python3 python3-pip python3-tkinter` |

> **Nota sobre Tkinter.** Es la librería que dibuja la ventana. Viene incluida
> con Python en Windows y macOS, pero en Linux suele ser un paquete aparte
> (`python3-tk`). **No se instala con pip.** Si te falta, el programa te lo
> dirá con instrucciones concretas.

---

## 2. Instalación

### Descargar

```bash
git clone https://github.com/jimenelouis808/Cc.git
cd Cc/carbonforge
```

Si no tienes `git`, puedes descargar el ZIP desde GitHub (botón verde
*Code → Download ZIP*), descomprimirlo y entrar en la carpeta
`Cc/carbonforge`.

---

### Instalar

Hay un instalador para cada sistema. Hace todo: crea el entorno aislado,
instala las dependencias, comprueba que Tkinter está disponible y lanza los
tests.

**Linux / macOS:**

```bash
./install.sh
```

**Windows:** haz doble clic en `install.bat`, o desde `cmd`:

```
install.bat
```

Si prefieres hacerlo a mano:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
pip install -e ".[dev]"
pytest -q
```

Deberías ver `515 passed`.

Sabrás que el entorno está activo porque el prompt de la terminal empieza
por `(.venv)`. **Tendrás que activarlo cada vez que abras una terminal
nueva** — es el paso que más se olvida.

---

## 3. La interfaz gráfica

```bash
carbonforge-gui
```

La ventana tiene **dos pestañas**: «Construir estructura» y «Analizar
resultados». La primera tiene esta disposición:

```
┌────────────────────────┬──────────────────────────────┐
│  Tipo de estructura    │                              │
│  [ Nanotubo (CNT)  ▾]  │      Vista previa 3D         │
│                        │   (rotar con el ratón)       │
│  Parámetros            │                              │
│   Índice quiral n [6]  │                              │
│   Índice quiral m [6]  ├──────────────────────────────┤
│   Longitud     [10.0]  │  Resumen y validación        │
│   ...                  │  Fórmula: C120               │
│                        │  Dimensionalidad: 1D         │
│  Dopaje y defectos     │  Coordinación media: 3.000   │
│  Formatos de salida    │  ✅ Validación superada      │
│                        │                              │
│ [Construir y previsualizar]                           │
│ [Exportar…]                                           │
└────────────────────────┴──────────────────────────────┘
```

**Flujo de trabajo:**

1. Elige el tipo de estructura en el desplegable de arriba.
2. Ajusta los parámetros. Cada campo lleva debajo una nota explicando qué
   significa y qué valores son razonables.
3. Pulsa **Construir y previsualizar**. La estructura aparece en 3D y el
   panel inferior te dice si pasa la validación.
4. Marca los formatos que quieras y pulsa **Exportar…** para elegir carpeta.

Puedes rotar, hacer zoom y desplazar la vista 3D con la barra de
herramientas bajo la figura.

En la pestaña **«Analizar resultados»** abres el archivo de salida de un
cálculo ya terminado —bandas o `dynmat.out`— y lo ves graficado ahí mismo,
con los mismos avisos que da la terminal (modos imaginarios, gap muestreado,
número de modos acústicos).

### Qué significa el panel de validación

| Símbolo | Significado |
|---|---|
| ✅ Validación superada | La estructura es físicamente razonable: sin átomos solapados, coordinaciones correctas, vacío suficiente. |
| ⚠️ Advertencias | Algo atípico pero no necesariamente erróneo (densidad inusual, átomos de borde sin saturar). |
| ❌ Validación fallida | Hay un problema real. Por defecto **no se exporta**; puedes forzarlo con la casilla correspondiente, que es lo normal en espumas antes de relajarlas. |

---

## 4. Recetas: la forma fácil

Montar bien un cálculo DFT significa acertar con una docena de decisiones a la
vez —funcional, dispersión, espín, malla, cutoff, si relajar antes— y
equivocarse en cualquiera puede pasar desapercibido. Una **receta** las toma
todas juntas, **adaptándose a tu estructura**, y te explica qué eligió.

```bash
carbonforge presets    # ver todas
```

| Receta | Para qué |
|---|---|
| `quick` | Ver que arranca. No publiques esto. |
| `geometry` | Relajar. Hazlo siempre antes de cualquier propiedad. |
| `bands` | Relajar → scf → bandas, con espín si hace falta. |
| `bands-hse` | Igual con HSE06: gaps mucho mejores, ~30× el coste. |
| `dos` | Densidad de estados proyectada por elemento. |
| `raman` | Espectro Raman e IR (solo sistemas con gap). |
| `phonon` | Frecuencias sin intensidades; vale en metálicos. |
| `adsorption` | Relajación con dispersión, para moléculas y apilamiento. |

```bash
carbonforge ribbon --width 6 --length 3 --edge zigzag \
                   --preset bands --cores 16 --out salida
```

### Lo que decide por ti, y por qué

Con una cinta **zigzag** te dirá esto:

```
  • Polarización de espín ACTIVADA con bordes antiferromagnéticos.
    Una cinta zigzag tiene estados de borde magnéticos; sin esto el SCF
    converge a un estado que no es el fundamental y las bandas salen mal.
```

Ese es el punto entero de las recetas. Los bordes zigzag son magnéticos y se
acoplan antiferromagnéticamente; es de los resultados más establecidos del
campo. Un cálculo sin `nspin=2` converge tranquilamente a un estado metálico
no magnético, **no da ningún error**, y las bandas que reporta están mal. Con
una cinta armchair, en cambio, no activa nada: no le hace falta.

También activa dispersión de van der Waals donde importa (espumas,
espirales, adsorción), aprieta la convergencia para fonones —que son
segundas derivadas y amplifican el ruido—, y restringe la celda al relajar
sistemas con vacío.

### Paralelización

Con `--cores` calcula cuántos *pools* de puntos k usar:

```
16 núcleos, ~8 puntos k
Se usarán 8 pools de 2 núcleos.
```

Quantum ESPRESSO paraleliza mucho mejor sobre puntos k (`-nk`) que sobre
ondas planas, y eso va en el script generado. Si puedes elegir el número de
núcleos, que sea divisible por el de puntos k.

### El encadenado

Las recetas con `relax_first` generan un `run_all.sh` que relaja, **extrae la
geometría relajada** y recalcula la propiedad sobre ella. Una banda calculada
sobre la geometría de partida es una propiedad de otra estructura.

Todas las decisiones quedan escritas en `DECISIONES.txt`, para que el
razonamiento sobreviva a la sesión de terminal.

---

## 5. Grupos funcionales y nitrógeno

Hay dos químicas distintas y conviene no mezclarlas:

Un **grupo anclado** cuelga del carbono: `-NH2`, `-NO2`, `-OH`, `-COOH`,
`-C≡N`, `-CONH2`, `-CHO`, `=O`, `-SH`, `-CH3`, y el epóxido puente.

Un **nitrógeno de red** va dentro de los anillos: grafítico, piridínico,
pirrólico, N-óxido. Se separan en el XPS del N 1s y dopan distinto, así que
decir «5 % de N» sin especificar cuál dice muy poco.

En la GUI tienen su propio panel. Desde la terminal:

```bash
carbonforge groups        # ver todos los grupos disponibles

# Amina en el borde de una nanocinta, y calcular sus bandas
carbonforge ribbon --width 6 --length 3 --group NH2 --group-count 2 \
                   --task bands --out salida/amino --format all

# Nitrógeno piridínico en grafeno (crea vacante y pone N en el borde)
carbonforge graphene --nx 6 --ny 6 --nitrogen pyridinic --nitrogen-count 2 \
                     --out salida/piridinico

# Óxido de grafeno: hidroxilos en el plano basal (fuerza sp3)
carbonforge graphene --nx 5 --ny 5 --group OH --group-count 3 \
                     --group-site basal --out salida/go

carbonforge nitrogen-report estructura.xyz
```

**El caso pirrólico, con honestidad.** Un sitio pirrólico real necesita un
anillo de **cinco** miembros, y esa reconstrucción la decide la energía, no
la geometría. Lo que genera el programa es un *precursor*: composición y
entorno correctos, pero anillos todavía de seis. Está etiquetado como tal.
Relájalo y comprueba con `ring_statistics` que apareció el pentágono.

**Todas las geometrías salen sin relajar.** Los grupos rotan en torno a sus
enlaces simples e interaccionan con lo que tengan cerca.

---

## 6. Calcular espectros, bandas y espín-órbita

En el panel «Cálculo» de la interfaz (o con `--task` en la terminal) eliges
qué quieres calcular, no solo qué estructura construir:

| Opción | Qué genera | Qué necesitas |
|---|---|---|
| `scf` | Un punto simple de energía | Nada especial |
| `relax` | Relajación de posiciones atómicas | Nada especial |
| `vc-relax` | Relajación de celda **y** posiciones | Fijar `cell-dofree` en 1D/2D |
| `bandas` | scf → bands → `bands.x` + script | Nada especial |
| `fonones` | scf → `ph.x` → `dynmat.x` (solo frecuencias) | Nada especial |
| `infrarrojo` | Lo anterior + cargas de Born | Que el sistema tenga **gap** |
| `raman` | Lo anterior + tensores Raman | Gap **y** pseudos norm-conserving |

### Las tres trampas que el programa detecta por ti

Estas no son detalles menores: cada una arruina un cálculo de forma que
cuesta horas descubrir.

**1. Raman o IR en un sistema metálico.** Quantum ESPRESSO necesita
`epsil=.true.` para las intensidades, y eso solo puede calcularlo si hay
gap. Los nanotubos armchair `(n,n)` son metálicos, igual que cualquier
`(n,m)` con `(n-m)` múltiplo de 3, y el grafeno prístino es un semimetal. El
programa lo predice de tu estructura y te lo dice antes de escribir nada.

**2. Raman con pseudopotenciales PAW.** El Raman por DFPT en QE **no**
admite PAW ni ultrasoft — y PAW es justo lo que usa la biblioteca estándar
por defecto. Necesitas norm-conserving (ONCV, SG15).

**3. Espín-órbita con pseudos escalares.** Este es el peor de los tres,
porque no falla: el cálculo termina bien y el desdoblamiento sale
exactamente cero, que es indistinguible de «aquí el SOC es despreciable».
Necesitas los pseudos `rel-`.

Sobre el espín-órbita, además, un aviso de magnitud: crece como Z⁴, así que
en carbono puro el efecto es de ~0.01 meV, muy por debajo de la precisión de
un DFT rutinario y de k_BT a temperatura ambiente (~25 meV). Solo se vuelve
interesante con adátomos pesados (Au, Bi, Pb) o un sustrato pesado.

**Si tu sistema es metálico y aun así quieres vibraciones**, usa `fonones`:
te da las frecuencias sin intensidades, y no tiene ninguna de estas
restricciones.

### SIESTA

SIESTA es una alternativa a Quantum ESPRESSO que usa orbitales localizados
en vez de ondas planas. Márcalo en los formatos de salida y obtendrás un
`.fdf` completo.

Dos diferencias importantes:

- **No hay `ecutwfc`.** La calidad de la base se controla con
  `PAO.BasisSize` (DZP es lo habitual). El `MeshCutoff` que verás en el
  archivo es para la malla real de la densidad, **no** es un cutoff de ondas
  planas: no copies ahí el valor de QE.
- **Los pseudopotenciales son distintos.** SIESTA usa `.psf` o `.psml`, no
  los `.UPF` de QE. No son intercambiables.
- **SIESTA no tiene DFPT.** Los fonones salen por constantes de fuerza
  congeladas (`MD.TypeOfRun FC` y luego la utilidad `vibra`), y **no hay
  Raman**. Para Raman usa Quantum ESPRESSO.

---

## 7. Si prefieres la línea de comandos

La GUI no es obligatoria: todo está disponible desde la terminal, y esto es
lo práctico para generar muchas estructuras de golpe.

```bash
# Nanotubo (6,6) de 10 Å, exportado a Quantum ESPRESSO y LAMMPS
carbonforge cnt --n 6 --m 6 --length 10 --out salida/cnt --format both

# Grafeno 4x4 con 3% de nitrógeno
carbonforge graphene --nx 4 --ny 4 --dopant N --dopant-conc 0.03 --out salida/gr

# Nanoespiral: radio 25 Å, paso 12 Å, vuelta y media
carbonforge nanocoil --n 6 --m 6 --coil-radius 25 --pitch 12 --turns 1.5 \
                    --out salida/coil --format both --force

# Estructura de bandas de un nanotubo semiconductor, para QE y SIESTA
carbonforge cnt --n 10 --m 0 --length 8 --task bands --out salida/bandas --format all

# Espectro Raman (aborta si la quiralidad es metálica, y te explica por qué)
carbonforge cnt --n 10 --m 0 --length 8 --task raman --out salida/raman

# Solo frecuencias: funciona también en metálicos
carbonforge cnt --n 6 --m 6 --length 8 --task phonon --out salida/fonones

# Con espín-órbita
carbonforge graphene --nx 4 --ny 4 --spinorbit --out salida/soc

# vc-relax en 2D: hay que fijar la celda o el vacío se comprime
carbonforge graphene --nx 4 --ny 4 --task vc-relax --cell-dofree 2Dxy --out salida/rel

# Ver todas las opciones
carbonforge --help
carbonforge cnt --help
```

---

## 8. Ver los resultados

carbonforge **no ejecuta nada**: escribe las entradas y lee las salidas.
Cuando tu cálculo haya terminado:

```bash
# Estructura de bandas (QE o SIESTA)
carbonforge plot-bands bands.dat --labels G,M,K,G --out bandas.png

# Espectro Raman, con corrección de láser y temperatura
carbonforge plot-spectrum dynmat.out --kind raman --laser 532 \
                          --temperature 300 --out raman.png
```

`plot-bands` te dice además el gap muestreado, o si el sistema es metálico.
Ojo: solo ve los puntos k del camino, así que un extremo de banda fuera de él
no aparece.

`plot-spectrum` te avisa de dos cosas que conviene pillar pronto:

- **Modos imaginarios** (frecuencias negativas): tu estructura no está en un
  mínimo sino en un punto de silla. El espectro no vale; relaja mejor.
- **Un número de modos acústicos distinto de tres**: suele significar que no
  se aplicó la regla de suma acústica.

Sobre las intensidades: lo que da el cálculo son **actividades**. Para
compararlas con un espectro experimental hacen falta el factor de Bose y el
prefactor `(ν_láser − ν)⁴`. Se aplican solo si los pides (`--laser`,
`--temperature`), y la etiqueta del eje cambia para que siempre sepas qué
estás mirando.

---

## 9. Pseudopotenciales: los archivos que tienes que descargar

carbonforge escribe los **nombres** de los pseudopotenciales en las entradas,
pero no puede incluir los archivos. Sin ellos, QE no arranca. Para saber
cuáles necesitas exactamente:

```bash
carbonforge pseudos estructura.xyz --dir ./pseudo
```

Te dice qué familia hace falta y por qué, los nombres de archivo, de dónde
bajarlos, y comprueba si ya los tienes.

La familia depende de lo que vayas a calcular:

| Cálculo | Familia | Por qué |
|---|---|---|
| Normal | PAW | Eficiente y preciso |
| Raman | Norm-conserving | `ph.x` no admite PAW ni ultrasoft para Raman |
| Espín-órbita | Relativista (`rel-`) | Con escalares el desdoblamiento sale cero |
| Raman + SOC | NC **y** relativista | La combinación más restrictiva |

Añade `--raman` o `--spinorbit` para que lo tenga en cuenta.

Si te falta un archivo pero hay otro de ese mismo elemento en la carpeta, te
lo señala como posible sustituto. **No lo usa por su cuenta**: puede ser
perfectamente válido, pero esa decisión es tuya.

---

## 10. Densidad de estados: ¿de dónde salen los estados?

Las bandas te dicen si hay gap. Pero si estás dopando con nitrógeno, la
pregunta que importa es **qué átomos** ponen estados en el nivel de Fermi. Eso
es la densidad de estados proyectada (PDOS), y es lo que distingue un
nitrógeno grafítico de uno piridínico.

```bash
# Generar el flujo (scf -> nscf -> dos.x -> projwfc.x)
carbonforge graphene --nx 5 --ny 5 --nitrogen graphitic --task dos --out salida/dos

cd salida/dos/qe && ./run_dos.sh

# Analizar: apunta a la CARPETA para el desglose por elemento
carbonforge plot-dos . --fermi <E_F que sale en pw.scf.out> --window -8 4
```

Te responde directamente:

```
En el nivel de Fermi (0.000 eV):
  C: 52.3 %
  N: 47.7 %
```

Dos detalles que el programa gestiona por ti en vez de dejártelos como
trampa:

- **El paso nscf usa una malla k más densa** (2× por defecto). Una malla que
  converge la densidad de carga es demasiado gruesa para resolver una curva
  de DOS: sale una fila de bultos de ensanchamiento en vez de una densidad.
- **La proyección no suma exactamente el DOS total.** Se proyecta sobre
  orbitales atómicos, que no cubren toda la base de ondas planas, así que
  falta un pequeño porcentaje. El programa te dice cuánto: entre 90 y 95 %
  es normal; bastante menos significa que las fracciones por elemento no son
  de fiar.

---

## 11. Convergencia: los valores por defecto NO están convergidos

Esto importa: un gap o una frecuencia sacados de un cálculo sin convergir
están mal, por muy cuidado que esté todo lo demás. Los 60 Ry por defecto son
un punto de partida razonable, nada más.

```bash
# 1. Genera el barrido
carbonforge converge estructura.xyz --parameter cutoff --out conv

# 2. Ejecútalo
cd conv && ./run_sweep.sh

# 3. Analiza
carbonforge converge-report conv --tolerance 1.0 --out conv.png
```

Te da una tabla como esta:

```
     valor      E/átomo (eV)     ΔE vs siguiente
--------------------------------------------------
        40        -72.212216          527.221 meV
        60        -72.739437            0.680 meV ✓
        80        -72.740117                   —

Convergido en ecutwfc (Ry) = 60
```

Compara cada punto con el **siguiente**, que es la pregunta real: «¿puedo
parar aquí?». Las energías van por átomo, así que la tolerancia significa lo
mismo en sistemas de cualquier tamaño.

Un matiz: converger la energía total no garantiza que estén convergidas otras
propiedades. Las frecuencias de fonones suelen necesitar más cutoff que la
energía. Si te importa una propiedad concreta, converge esa.

Con `--parameter kpoints` haces lo mismo para la malla de puntos k.

---

## 12. Desde Python

Para barridos o integrarlo en tus propios scripts:

```python
from carbonforge.builders import build_nanocoil
from carbonforge.dopants  import dope_random
from carbonforge.exports.qe import write_qe_input, QESettings
from carbonforge.validation import run_basic_checks

coil = build_nanocoil(n=6, m=6, coil_radius=25.0, pitch=12.0, n_turns=1.5)
coil = dope_random(coil, "N", 0.03, seed=42)   # seed = reproducible

print(run_basic_checks(coil).summary())
write_qe_input(coil, "salida/qe", settings=QESettings(calculation="relax"),
               force=True)
```

Generar muchas estructuras a la vez:

```python
from carbonforge.workflows import batch_cnt_sweep, write_dataset

jobs = batch_cnt_sweep(
    chiralities=[(6, 6), (8, 0)],
    lengths=[10.0, 20.0],
    dopant="N", dopant_concentrations=[0.0, 0.05],
    vacancies=[0, 1],
    seed=0,
)
write_dataset(jobs, "salida/dataset")   # 16 estructuras + dataset.json
```

---

## 13. Qué hacer con los archivos generados

### Quantum ESPRESSO

Obtienes un `pw.in` completo. **Te faltan los pseudopotenciales**, que no se
distribuyen con este proyecto: descárgalos de
[pseudopotentials.quantum-espresso.org](https://pseudopotentials.quantum-espresso.org/)
y ajusta la ruta `pseudo_dir` dentro del archivo.

```bash
pw.x -in pw.in > pw.out
```

Los nombres por defecto asumen PAW-PBE (`C.pbe-n-kjpaw_psl.1.0.0.UPF`). Si
usas otros, cámbialos en la sección `ATOMIC_SPECIES`.

### LAMMPS

Obtienes `data.lammps` (la estructura) e `in.lammps` (el script), con las
etapas separadas como corresponde:

```
1. Minimización
2. (opcional) Recocido: calentar y enfriar
3. Equilibración  <- se descarta
4. Producción     <- aquí se mide
```

Esa separación no es cosmética: promediar sobre el transitorio de
equilibración sesga cualquier magnitud que saques. Solo la etapa de
producción escribe trayectoria (`traj.lammpstrj`) y acumula promedios
(`averages.dat`).

Para espumas usa `mode="anneal"`: un empaquetado aleatorio no es una
estructura física hasta que se ha fundido y enfriado. Un enfriamiento más
lento da redes más ordenadas.

El paso de tiempo se rechaza por encima de 2 fs: la vibración C-C ronda los
1600 cm⁻¹ (~21 fs de periodo) y hacen falta ~20 pasos por periodo. Para
carbono puro usa AIREBO, cuyo archivo `CH.airebo` viene en el directorio
`potentials/` de tu instalación de LAMMPS.

```bash
lmp -in in.lammps
```

Si dopaste con N/B/S/P, el script marca `REPLACE_ME` en la línea
`pair_coeff`: ahí debes poner el potencial adecuado a esa mezcla de
elementos, porque AIREBO solo cubre C y H.

### XYZ / CIF

Para visualizar: **OVITO** y **VMD** leen XYZ; **VESTA** lee CIF.

---

## 14. Problemas frecuentes

**`command not found: carbonforge-gui`**
El entorno virtual no está activo. Ejecuta `source .venv/bin/activate`
(Linux/macOS) o `.venv\Scripts\activate` (Windows).

**`No se encontró Tkinter`**
Falta el paquete de Tk. En Linux: `sudo apt install python3-tk`. En
Windows/macOS: reinstala Python desde python.org marcando *"tcl/tk and
IDLE"*.

**La ventana se queda congelada al construir**
No debería: la construcción ocurre en un hilo aparte. Si tarda, es que la
estructura es grande — una nanoespiral de varias vueltas puede pasar de
5000 átomos. Baja el número de vueltas o el radio.

**La vista previa no muestra los enlaces**
Por encima de 4000 átomos se omiten a propósito: detectar enlaces es
O(N²) y la vista se volvería lenta. Los átomos sí se dibujan, y la
exportación no se ve afectada.

**La validación falla en una espuma**
Es lo esperado. Las espumas se generan colocando fragmentos al azar, así que
antes de relajarlas tienen coordinaciones imperfectas. Marca *"Exportar
aunque falle la validación"* y relaja con LAMMPS.

**`ModuleNotFoundError: No module named 'carbonforge'`**
No instalaste el paquete o estás en otra carpeta. Desde `Cc/carbonforge`
ejecuta `pip install -e .`.

---

## 15. Límites que conviene conocer

Estas no son pegas menores, son cosas que afectan a cómo interpretas los
resultados:

- **Las estructuras no están relajadas.** Los constructores generan
  geometrías de partida razonables, no mínimos de energía. Relaja siempre
  antes de sacar conclusiones físicas.
- **Las nanoespirales son una construcción geométrica.** Se doblan
  enrollando un nanotubo recto sobre una hélice, con defectos Stone–Wales
  opcionales. No son espirales topológicas puras (pares 5-7 periódicos tipo
  Dunlap/Ihara). La distorsión de enlaces es de pocos % con radio ≥ 25 Å,
  pero crece al reducir el radio.
- **Las espumas 3D son un punto de partida para MD**, no estructuras de
  equilibrio. Los enlaces colgantes no se saturan.
- **Los parámetros de DFT por defecto** (60/480 Ry, malla k automática) son
  puntos de partida sensatos, no valores convergidos. Haz tu propio estudio
  de convergencia.
- **La predicción metálico/semiconductor usa reglas de plegado de zona**
  (`(n-m) % 3`, tipo de borde), no un cálculo electrónico. Es fiable para
  estructuras prístinas; con dopaje fuerte o defectos abundantes el carácter
  puede cambiar y la predicción deja de valer.
- **La familia del pseudopotencial se deduce del nombre del archivo**, no
  leyendo su cabecera. Si usas nombres no estándar, el programa avisará de
  que no puede determinarla en vez de adivinar.
- **carbonforge no ejecuta nada.** Genera los archivos de entrada y los
  scripts; ejecutar `pw.x`, `siesta` o `lmp` es cosa tuya. Tampoco incluye
  pseudopotenciales.
- **Los lectores de resultados están probados con archivos sintéticos** que
  reproducen los formatos documentados, no con salidas de una instalación
  real de QE o SIESTA (no había ninguna disponible al desarrollarlos). La
  primera vez que los uses con tus datos, trátalo también como una prueba
  del lector: si algo no cuadra, dímelo.
- **Las recetas eligen ajustes razonables, no convergidos.** Cutoff y malla
  siguen siendo puntos de partida: converge antes de publicar.
- **El multiplicador de coste que reporta es orientativo** (2× por espín,
  ~30× por híbrido), para decidir si algo cabe en una cola, no un benchmark.
- **Los grupos funcionales y el nitrógeno de red salen sin relajar**, y las
  configuraciones pirrólicas son precursores, no sitios terminados.
- **El gap que estima `plot-dos` sale de una curva ensanchada**: el
  `degauss` rellena un gap pequeño. El valor fiable sale de las bandas.
- **El gap que reporta `plot-bands` es un gap muestreado**: solo ve los
  puntos k del camino. Un extremo de banda fuera de él no aparece.
