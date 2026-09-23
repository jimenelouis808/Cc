# Instalación y primera prueba

Tres paquetes independientes en un mismo repositorio. Puedes instalarlos todos
de una vez o solo el que te interese.

## Requisitos

- **Python 3.11 o superior.** Comprueba con `python --version`.
- **Tkinter**, solo si vas a usar las interfaces gráficas. Viene incluido en los
  instaladores oficiales de Python para Windows y macOS. En Linux se empaqueta
  aparte: `sudo apt install python3-tk` (Debian/Ubuntu) o
  `sudo dnf install python3-tkinter` (Fedora).

No hace falta nada más. Ni compilador, ni Quantum ESPRESSO, ni LAMMPS: los
paquetes *generan* entradas para esos programas, no los ejecutan.

---

## Opción A — con uv (recomendada)

Usa el `uv.lock` del repositorio, así que instala exactamente las versiones con
las que se probó.

Instalar uv, si no lo tienes:

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Luego, dentro de la carpeta descomprimida:

```bash
uv sync --all-packages --extra dev
```

Eso crea `.venv/` con los tres paquetes instalados en modo editable. Para
ejecutar cualquier cosa, antepón `uv run`:

```bash
uv run ramancarbon --help
```

## Opción B — con pip

Si prefieres no instalar uv. No usa el lockfile, así que pip resolverá las
versiones por su cuenta.

```bash
python -m venv .venv
```
```powershell
.venv\Scripts\activate        # Windows
```
```bash
source .venv/bin/activate     # macOS / Linux
```
```bash
pip install -e packages/ramancarbon[dev]
pip install -e packages/carbonforge[dev]
pip install -e packages/nanocarbon_lab[dev]
```

Instala solo los que quieras; no dependen unos de otros.

---

## Comprobar que funciona

```bash
cd packages/ramancarbon
python -m pytest ramancarbon/tests -q -n 4 --dist loadscope
```

Debe decir **1137 passed**. Un `skipped` por tkinter es normal si no lo tienes.

Para los otros dos, lo mismo cambiando el nombre. En `nanocarbon_lab` añade
`-m "not slow"` para la versión rápida (~90 s en vez de ~10 min):

```bash
cd packages/nanocarbon_lab
python -m pytest nanocarbon_lab/tests -q -n 4 --dist loadscope -m "not slow"
```

En Linux y macOS, antepón `OMP_NUM_THREADS=1` a esos comandos: las matrices son
pequeñas y el paralelismo interno de numpy estorba más que ayuda. La diferencia
es de cinco minutos a noventa segundos.

---

## Qué probar primero

**Interfaces gráficas.** Es por donde tiene más sentido empezar:

```bash
ramancarbon-gui          # Raman, XRD, XPS y electroquímica
nanocarbon-gui           # generación de estructuras
python -m carbonforge.gui    # preparación de cálculos DFT/MD
```

**Línea de comandos.** Cada paquete trae la suya:

```bash
ramancarbon --help       # 19 subcomandos
carbonforge --help       # 18
nanocarbon --help        # 30
```

**Datos de ejemplo.** `ramancarbon` genera espectros sintéticos para que puedas
probar sin tus propios archivos, y trae veinte ejemplos ejecutables:

```bash
cd packages/ramancarbon
python ramancarbon/examples/ex01_analizar.py
python ramancarbon/examples/ex13_echem.py
python ramancarbon/examples/ex14_figuras.py
```

`carbonforge` y `nanocarbon_lab` traen los suyos en `*/examples/`.

**Un caso real de principio a fin**, sin GUI:

```bash
# construir un nanotubo (6,6) dopado con nitrógeno y exportarlo a Quantum ESPRESSO
nanocarbon cnt --n 6 --m 6 --length 12 --dopant N --dopant-conc 0.03 \
    --seed 42 --out salida/cnt --format qe
```

---

## GPAW para los cálculos IR de `carbonforge vibspec` (Ubuntu o WSL2)

Solo hace falta para **correr** los cálculos. Construir las cintas, preparar
los cálculos y analizar los espectros funciona en Windows sin GPAW.

GPAW solo se publica como código fuente y no compila en Windows de forma
nativa: usa Ubuntu, o WSL2 con Ubuntu en el mismo equipo. En Ubuntu 22.04/24.04:

```bash
# Compiladores (g++ es imprescindible: GPAW 26 se compila como C++), libxc y BLAS
sudo apt install build-essential libxc-dev libopenblas-dev
# Para correr en paralelo con MPI (opcional pero recomendado):
sudo apt install openmpi-bin libopenmpi-dev libscalapack-openmpi-dev libfftw3-dev

uv sync --all-packages --extra dev
CC=g++ uv pip install gpaw          # o: uv sync --package carbonforge --extra gpaw
uv run gpaw info                    # libxc: yes; MPI: yes si instalaste OpenMPI
```

`CC=g++` no es un capricho: con `gcc` la compilación de GPAW 26 se para en
`fatal error: algorithm: No such file or directory`. Desde GPAW 26 los datasets
PAW llegan como paquete (`gpaw-data`), así que `gpaw install-data` ya no hace
falta; `gpaw info` muestra dónde están.

**El flujo completo:**

```bash
carbonforge vibspec build --preset amine -o amina.xyz              # Windows o Ubuntu
carbonforge vibspec prepare amina.xyz -d calculos/amina            # valida y escribe el directorio
cd calculos/amina && mpiexec -n 4 gpaw python run.py               # Ubuntu, con GPAW
carbonforge vibspec show calculos/amina                            # de vuelta en Windows
carbonforge vibspec plot calculos/amina --ftir mi_ftir.csv --fit-scale -o amina.png
carbonforge vibspec gui                                            # lo mismo, con ventana
```

`run.py` se puede relanzar si se corta: la relajación terminada no se repite y
los desplazamientos ya calculados tampoco.

**Cuánto tarda:** una molécula de agua en serie, LCAO-dzp, ~10 min. Una cinta
de 80 átomos son ~480 cálculos SCF de un sistema bastante mayor: horas en un
PC de sobremesa, así que usa MPI (`mpiexec -n <núcleos>`) o el clúster.

---

## Si algo falla

**`ModuleNotFoundError: No module named 'tkinter'`** — solo afecta a las GUIs.
Instala el paquete del sistema (arriba, en Requisitos). Las CLIs y la biblioteca
funcionan sin él.

**`bpy` no instala** — es opcional, solo para el renderizador de Blender de
`nanocarbon_lab`. Existe una rueda por versión de Python y solo para 3.11.
Sáltalo: no lo necesitas para nada más.

**Los tests de `nanocarbon_lab` tardan muchísimo** — sin `-m "not slow"` la
suite completa son diez minutos largos. Son construcciones de mallas que tardan
minutos cada una; están marcadas como `slow` precisamente para poder saltarlas.

**`ImportError: GPAW no está instalado`** al hacer `vibspec run` — es lo
esperado en Windows. Prepara el cálculo ahí y córrelo en Ubuntu (arriba).

**El resultado de un test difiere en el último decimal** — varias rutinas usan
mínimos cuadrados y el resultado depende de la versión de scipy. La Opción A
evita esto porque fija las versiones.
