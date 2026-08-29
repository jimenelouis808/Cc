# Guía rápida — instalar y usar en tu laptop

Guía para dejar `carbonforge` funcionando desde cero, sin asumir
conocimientos de Python más allá de abrir una terminal.

---

## 1. Instalar Python

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

## 2. Descargar el proyecto

```bash
git clone https://github.com/jimenelouis808/Cc.git
cd Cc/carbonforge
```

Si no tienes `git`, puedes descargar el ZIP desde GitHub (botón verde
*Code → Download ZIP*), descomprimirlo y entrar en la carpeta
`Cc/carbonforge`.

---

## 3. Instalar el programa

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

Deberías ver `194 passed`.

Sabrás que el entorno está activo porque el prompt de la terminal empieza
por `(.venv)`. **Tendrás que activarlo cada vez que abras una terminal
nueva** — es el paso que más se olvida.

---

## 4. Abrir la interfaz gráfica

```bash
carbonforge-gui
```

Se abre una ventana con esta disposición:

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

### Qué significa el panel de validación

| Símbolo | Significado |
|---|---|
| ✅ Validación superada | La estructura es físicamente razonable: sin átomos solapados, coordinaciones correctas, vacío suficiente. |
| ⚠️ Advertencias | Algo atípico pero no necesariamente erróneo (densidad inusual, átomos de borde sin saturar). |
| ❌ Validación fallida | Hay un problema real. Por defecto **no se exporta**; puedes forzarlo con la casilla correspondiente, que es lo normal en espumas antes de relajarlas. |

---

## 5. Calcular espectros, bandas y espín-órbita

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

## 6. Si prefieres la línea de comandos

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

## 7. Desde Python

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

## 8. Qué hacer con los archivos generados

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

Obtienes `data.lammps` (la estructura) e `in.lammps` (el script). Para
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

## 9. Problemas frecuentes

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

## 10. Límites que conviene conocer

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
