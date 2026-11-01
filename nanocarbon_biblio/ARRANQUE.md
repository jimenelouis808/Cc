# Arrancar la interfaz — guía literal

Si vuelves al proyecto después de un tiempo, **empieza por aquí**. No hace falta
recordar nada de lo anterior.

---

## En una línea

```bash
cd nanocarbon_biblio
python3 start.py
```

Eso es todo. `start.py` comprueba tu Python, crea el entorno virtual si no
existe, instala las dependencias, verifica que todo importa y abre la interfaz
en el navegador. La primera vez tarda un par de minutos; las siguientes, segundos.

En **Windows**: doble clic en `run_gui.bat`, o `py start.py` en la terminal.
En **macOS/Linux**: doble clic en `run_gui.sh`, o `./run_gui.sh`.

---

## La primera vez, con datos de prueba

Para ver la interfaz funcionando sin haber exportado nada de Scopus todavía:

```bash
python3 start.py --demo
```

Genera un corpus sintético de 900 artículos y lo deja en `data/raw/demo`. Luego,
en la pestaña **1 · Cargar**, pulsa **«Cargar carpeta»**. Son datos inventados:
sirven para aprender la interfaz, nunca para el manuscrito.

Recorrido recomendado la primera vez, cinco minutos:

1. **1 · Cargar** → «Cargar carpeta»
2. **2 · Deduplicar** → «Deduplicar». Mira el solapamiento Scopus/WoS.
3. **3 · Clasificar** → «Clasificar». Aparecen el tipo de estudio y la matriz dopante × aplicación.
4. **8 · RQ2 · RQ3** → el desfase teoría→experimento y los huecos de cobertura.

---

## Los comandos que existen

| Comando | Para qué |
|---|---|
| `python3 start.py` | Abre la interfaz (instalando lo que falte) |
| `python3 start.py --check` | Solo diagnostica. No instala ni abre nada |
| `python3 start.py --demo` | Genera corpus de prueba y abre la interfaz |
| `python3 start.py --tests` | Ejecuta los 68 tests |
| `python3 start.py --reinstall` | Borra el entorno virtual y lo rehace |

Cuando algo no funcione, el primero que hay que ejecutar es `--check`: dice
exactamente qué falta y cómo arreglarlo.

---

## Si algo falla

| Lo que ves | Qué pasa | Qué hacer |
|---|---|---|
| `python3: command not found` | Python no está instalado o no está en el PATH | macOS: `brew install python@3.12`. Windows: instala desde python.org **marcando «Add python.exe to PATH»**. Ubuntu: `sudo apt install python3` |
| `Python 3.9 es demasiado antiguo` | Versión por debajo de 3.10 | Instala una versión moderna y vuelve a ejecutar `python3 start.py` |
| `No pude crear el entorno virtual` | Falta el paquete `venv` (típico en Debian/Ubuntu) | `sudo apt install python3-venv` |
| `externally-managed-environment` | pip se niega a escribir en el Python del sistema | No es problema: `start.py` ya usa un entorno virtual. Lánzalo con `python3 start.py`, no con `pip install` a mano |
| `La instalación de dependencias falló` | Sin internet, o un proxy bloquea PyPI | Lee las líneas de pip que imprime. Con proxy corporativo, exporta `HTTPS_PROXY` antes |
| `Port 8501 is already in use` | Ya tienes otra interfaz abierta | No debería ocurrir: `start.py` busca un puerto libre solo. Si lo ves, cierra la otra con Ctrl+C |
| La web abre en blanco | El navegador cacheó una versión anterior | Recarga con Ctrl+Shift+R |
| `el paquete no importa` | El checkout está incompleto | `git status` y `git checkout claude/binance-scalping-bot-cut2op` |

Si nada de esto encaja, pega la salida completa de `python3 start.py --check`
junto con el error: con eso se diagnostica.

---

## Cerrar la interfaz

`Ctrl+C` en la terminal donde la lanzaste. Cerrar la pestaña del navegador no la
para: el servidor sigue corriendo.

---

## Después de la interfaz

Cuando ya tengas el corpus exportado y cribado, el paso siguiente es R:

```bash
Rscript R/99_check_r_setup.R     # comprueba que R y bibliometrix funcionan
Rscript R/00_build_M.R           # construye data/processed/M.rds
Rscript R/01_core_analyses.R     # análisis que van al paper
Rscript R/launch_biblioshiny.R   # abre biblioshiny
```

Los dos del medio también se lanzan desde la pestaña **6 · Exportar a R** de la
interfaz. Para instalar R: [cran.r-project.org](https://cran.r-project.org) y
después `Rscript R/install_deps.R`.

El recorrido completo del proyecto, de la consulta al manuscrito, está en
`docs/WORKFLOW.md`. Las decisiones de alcance, en `docs/PROTOCOL.md`.
