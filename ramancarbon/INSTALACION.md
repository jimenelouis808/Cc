# Instalación de ramancarbon

Guía completa, paso a paso, sin dar nada por sabido.

---

## 1. Comprueba que tienes Python 3.10 o más nuevo

Abre una terminal:

* **Windows**: tecla Windows → escribe `cmd` → Enter
* **macOS**: Cmd+Espacio → escribe `Terminal` → Enter
* **Linux**: Ctrl+Alt+T

Y escribe:

```
python --version
```

Si sale `Python 3.10`, `3.11`, `3.12` o `3.13`, sigue al paso 2.

Si sale `Python 2.7`, prueba con `python3 --version`.
Si dice que el comando no existe, o la versión es menor que 3.10, instala
Python desde <https://www.python.org/downloads/>. **Durante la instalación,
en Windows, marca las dos casillas:**

* ☑ **Add Python to PATH** (o *Add python.exe to PATH*)
* ☑ **tcl/tk and IDLE** — es lo que dibuja la ventana del programa

---

## 2. Descomprime el archivo

Descomprime `ramancarbon.zip` donde quieras tenerlo. Quedará una carpeta
llamada `ramancarbon` con esto dentro:

```
ramancarbon/
├── install.sh          ← instalador para Linux y macOS
├── install.bat         ← instalador para Windows
├── pyproject.toml
├── README.md
├── GUIA_RAPIDA.md      ← cómo se usa
├── INSTALACION.md      ← este archivo
└── ramancarbon/        ← el código
```

---

## 3. Instala

### Windows

Doble clic en **`install.bat`**.

Si Windows lo bloquea («Windows protegió su PC»), pulsa *Más información* →
*Ejecutar de todas formas*. Es un archivo de texto: puedes abrirlo con el
Bloc de notas y ver exactamente lo que hace.

### Linux y macOS

Abre una terminal **en la carpeta** `ramancarbon` y escribe:

```bash
./install.sh
```

Si dice «Permiso denegado»:

```bash
chmod +x install.sh
./install.sh
```

### Qué hace el instalador

1. Comprueba tu versión de Python.
2. Crea un entorno aislado en `.venv` — así no toca nada del Python de tu
   sistema y puedes borrarlo entero sin dejar rastro.
3. Instala el programa y sus tres dependencias (`numpy`, `scipy`,
   `matplotlib`).
4. Comprueba si tienes Tkinter (la ventana).
5. Ejecuta las 369 pruebas.

Si termina diciendo **`=== OK ===`**, ya está.

---

## 4. Úsalo

**Cada vez que abras una terminal nueva**, activa el entorno primero:

```bash
source .venv/bin/activate        # Linux y macOS
.venv\Scripts\activate           # Windows
```

Sabrás que está activo porque el prompt empieza por `(.venv)`.

Y luego:

```bash
ramancarbon-gui                  # la suite entera
```

Se abre una ventana con **cuatro secciones**: Raman de carbono, Raman de
dicalcogenuros, difracción de rayos X y electroquímica. Cada una tiene un
botón **Demo** que carga datos de prueba.

Para probarlo desde la terminal, sin tener datos tuyos todavía:

```bash
ramancarbon demo datos_prueba/          # 8 espectros Raman de carbono
ramancarbon demo datos_tmd/ --tmd       # dicalcogenuros, con óxidos
ramancarbon demo-datos drx datos_drx/   # difractogramas
ramancarbon demo-datos echem datos_ec/  # CV, carga-descarga, impedancia

ramancarbon analizar datos_prueba/demo_MWCNT_FeSe_532nm.txt --laser 532
ramancarbon drx datos_drx/demo_drx_CNT_FeSe.xye
ramancarbon echem --cv datos_ec/demo_cv_condensador_20mVs.txt --masa 2 --area 1
```

Son datos **sintéticos**, generados por el programa a partir de la física que
se quiere probar. Sirven para ver cómo funciona todo antes de meter tus
medidas.

---

## 5. Si algo falla

**«No se encontró Tkinter»** — todo lo demás funciona; solo te falta la
ventana.

| Sistema | Solución |
|---------|----------|
| Ubuntu / Debian | `sudo apt install python3-tk` |
| Fedora | `sudo dnf install python3-tkinter` |
| Arch | `sudo pacman -S tk` |
| Windows / macOS | Reinstala Python desde python.org marcando *tcl/tk and IDLE* |

Mientras tanto, la línea de comandos no necesita Tkinter:

```bash
ramancarbon analizar espectro.txt --laser 532 --auto
```

**«pip: command not found»** o falla la instalación de paquetes — comprueba
que tienes conexión. Si estás detrás de un proxy corporativo, pídele a
sistemas el valor de `HTTPS_PROXY`.

**«ramancarbon: command not found»** — no has activado el entorno. Vuelve al
paso 4. Como alternativa siempre funciona:

```bash
python -m ramancarbon.cli.main analizar espectro.txt --laser 532
python -m ramancarbon.gui
```

**Las pruebas fallan** — cópiame el mensaje. Que fallen significa que algo de
tu entorno no es el esperado, y es mejor saberlo ahora que descubrirlo en un
resultado.

---

## 6. Instalación manual (si prefieres controlarlo tú)

```bash
python -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate en Windows
pip install -e ".[dev]"
pytest ramancarbon/tests -q
```

---

## 7. Un ajuste que acelera todo

Si vas a procesar lotes grandes, añade esto antes de arrancar Python:

```bash
export OMP_NUM_THREADS=1           # Linux y macOS
set OMP_NUM_THREADS=1              # Windows cmd
```

Suena raro, pero es real y está medido: la biblioteca de álgebra de NumPy
abre un hilo por núcleo, y las matrices de un ajuste de picos son tan
pequeñas que sincronizar los hilos cuesta más de lo que ahorra. Con esta
variable, el modo serie va **1.6× más rápido** y el paralelo deja de pelearse
consigo mismo.

---

## 8. Desinstalar

Borra la carpeta. No hay nada fuera de ella.

---

## Requisitos, resumidos

| | |
|---|---|
| Python | 3.10 o más nuevo |
| Dependencias | `numpy` ≥ 1.24, `scipy` ≥ 1.10, `matplotlib` ≥ 3.7 |
| Interfaz gráfica | Tkinter (viene con Python; en Linux se instala aparte) |
| Espacio en disco | ~200 MB con el entorno virtual |
| Instrumentos | Raman (carbono y TMD), difracción de rayos X, electroquímica |
| Sistemas | Windows, macOS, Linux |
