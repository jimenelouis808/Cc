# Instalación

Instrucciones prácticas. La documentación científica está en `README.md`
(en inglés) y las notas de diseño en `CLAUDE.md`.

Probado en Python **3.11** y **3.12**. El mínimo es 3.10.

---

## 1. Requisito previo: tkinter

Es el único que `pip` **no** puede instalar, porque forma parte de la
biblioteca estándar pero se empaqueta aparte en muchos sistemas. Sin él
funciona todo menos la ventana gráfica.

Comprueba si ya lo tienes:

```bash
python -c "import tkinter; print('tkinter OK')"
```

Si falla:

| Sistema | Qué hacer |
|---------|-----------|
| **Windows** | Reinstala Python desde python.org y marca **"tcl/tk and IDLE"** en el instalador. Viene marcado por defecto. |
| **macOS** | El Python de python.org ya lo trae. Con Homebrew: `brew install python-tk` |
| **Ubuntu / Debian** | `sudo apt install python3-tk` |
| **Fedora** | `sudo dnf install python3-tkinter` |
| **conda** | `conda install tk` |

---

## 2. Instalar

Descomprime el zip y, desde la carpeta que contiene `pyproject.toml`:

### Windows (PowerShell)

```powershell
cd nanocarbon_lab
py -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[gui,dev]"
```

### macOS / Linux

```bash
cd nanocarbon_lab
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[gui,dev]"
```

`-e` (editable) deja el código a la vista y editable; sin él la
instalación también funciona. `[gui,dev]` añade matplotlib para la vista
3D y pytest/ruff para las pruebas.

**Entorno virtual, no `pip install` global.** Con `venv` puedes borrar la
carpeta y no queda rastro, y no chocas con otras versiones de numpy o ASE
que tengas para otros proyectos.

---

## 3. Comprobar que quedó bien

```bash
nanocarbon --help                 # el comando debe estar en el PATH
nanocarbon groups                 # la biblioteca de grupos funcionales
nanocarbon dopants                # los heteroátomos disponibles

# construir algo y volver a leerlo
nanocarbon cnt-cap --rings 8 --freq 3 --out out/tubo
nanocarbon analyse out/tubo.xyz
```

El último comando debe decir `1D tube`, `{5: 12, 6: 350}` y
`sum(6-n) = +12`. Si eso sale, la instalación está completa.

Y la ventana gráfica:

```bash
nanocarbon-gui
```

---

## 4. Pruebas (opcional)

```bash
pytest nanocarbon_lab/tests -q -m "not slow"
```

Unos 10 minutos y ~940 pruebas. Sin `-m "not slow"` añade las
construcciones de bobinas, que tardan varios minutos cada una.

Las pruebas de la interfaz necesitan tkinter y una pantalla; en un
servidor sin monitor:

```bash
xvfb-run -a pytest nanocarbon_lab/tests/test_gui.py -q
```

---

## 5. Blender (opcional, solo para renders de portada)

Dos caminos, y no hace falta ninguno para generar o exportar estructuras:

* **Blender instalado** — el botón *Locate Blender…* de la interfaz
  apunta al ejecutable. No hace falta instalar nada más.
* **Sin Blender** — `pip install "bpy>=4.0"` mete Blender como módulo de
  Python. La rueda es grande (~300 MB) y solo existe para la versión de
  Python contra la que se compiló cada versión de Blender, así que si pip
  no la encuentra, usa el primer camino.

---

## Problemas frecuentes

**`nanocarbon: command not found`** — el entorno virtual no está activado.
Repite `venv\Scripts\activate` (Windows) o `source venv/bin/activate`.

**`ModuleNotFoundError: No module named 'tkinter'`** — sección 1. Todo lo
demás sigue funcionando; solo se pierde `nanocarbon-gui`.

**`ModuleNotFoundError: No module named 'skimage'`** — la instalación
quedó a medias. scikit-image no es opcional: las uniones, las
schwarzitas, las redes de nanotubos y las bobinas relajadas se construyen
con marching cubes. Repite `pip install -e ".[gui,dev]"`.

**Un `import` falla dentro de la interfaz al construir** — la interfaz
lanza cada construcción en un **proceso** aparte, y en Windows y macOS
eso reimporta el módulo principal. Si escribes tu propio script que
construya `NanocarbonGUI`, necesita la guarda
`if __name__ == "__main__":`. `nanocarbon-gui` ya la tiene.

**La construcción tarda mucho** — es real y la interfaz lo avisa antes:
una jaula sale en décimas de segundo, una red de diamante tarda unos
cuatro minutos. El botón *Cancel* mata el proceso de verdad, no marca una
bandera que nadie mira.
