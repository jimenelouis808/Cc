# Guía de desarrollo — probar y mejorar carbonforge

Esta guía es para cuando quieras **cambiar** el programa, no solo usarlo.
Si solo quieres usarlo, mira [GUIA_RAPIDA.md](GUIA_RAPIDA.md).

---

## 1. Preparar el entorno de desarrollo

```bash
cd carbonforge
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows
pip install -e ".[dev]"
```

El `-e` (editable) es importante: instala el paquete **enlazado** a esta
carpeta, así los cambios que hagas en el código tienen efecto inmediato sin
reinstalar nada.

Comprueba que arranca:

```bash
pytest -q
```

Debes ver `278 passed`. Si no, algo está mal en la instalación y no tiene
sentido seguir.

---

## 2. Ejecutar los tests

```bash
# Todo
pytest -q

# Un archivo
pytest carbonforge/tests/test_builders.py -q

# Una clase o un test concreto
pytest carbonforge/tests/test_nanocoil.py::TestNanocoilGeometry -q
pytest carbonforge/tests/test_nanocoil.py::TestNanocoilGeometry::test_arc_length_matches_formula

# Todo lo que tenga "raman" en el nombre
pytest -k raman -q

# Con detalle cuando algo falla
pytest carbonforge/tests/test_results.py -v

# Parar en el primer fallo (útil al refactorizar)
pytest -x

# Ver qué líneas no están cubiertas
pytest --cov=carbonforge --cov-report=term-missing -q
```

### Qué cubre cada archivo de tests

| Archivo | Qué prueba |
|---|---|
| `test_builders.py` | CNT, grafeno, nanocintas, espuma 3D |
| `test_nanocoil.py` | Nanoespirales: geometría, distorsión de enlaces, rechazos |
| `test_dopants_defects.py` | Dopaje, vacancias, Stone-Wales, reproducibilidad |
| `test_validation_topology.py` | Grafo de enlaces, coordinación, checks geométricos |
| `test_calculations.py` | Caminos de banda, especificaciones de espectroscopía y SOC |
| `test_calc_validation.py` | **Las trampas físicas**: Raman en metales, SOC escalar, vc-relax |
| `test_exports.py` | Escritura de QE y LAMMPS |
| `test_exports_advanced.py` | SIESTA, bandas, espectroscopía, SOC |
| `test_pseudos.py` | Qué pseudopotenciales hacen falta y comprobación de carpeta |
| `test_results.py` | Lectura de bandas y espectros, ensanchamiento |
| `test_convergence.py` | Barridos de convergencia y su informe |
| `test_workflows.py` | Generación en lote y datasets |
| `test_relax_viz_ml.py` | Pre-relajación, visualización, features de ML |
| `test_gui.py` | Lógica de la GUI (sin widgets) |
| `test_gui_widgets.py` | Widgets, con Tk simulado y matplotlib real |

---

## 3. Prueba manual rápida (humo)

Después de un cambio grande, esto verifica que el programa sigue vivo de
punta a punta:

```bash
# 1. Construir y exportar a los tres formatos
carbonforge cnt --n 10 --m 0 --length 8 --out /tmp/smoke --format all

# 2. Un flujo completo de bandas
carbonforge cnt --n 10 --m 0 --length 8 --task bands --out /tmp/smoke_bands

# 3. Que las trampas siguen detectándose (esto DEBE fallar con código 1)
carbonforge cnt --n 6 --m 6 --length 8 --task raman --out /tmp/smoke_raman
echo "código de salida: $?"   # tiene que ser 1

# 4. Los ejemplos
for e in ex01_cnt_qe ex02_doped_graphene_lammps ex03_batch_sweep \
         ex04_foam_and_topology ex05_nanocoil ex06_spectra_and_bands \
         ex07_analyse_results; do
    python -m carbonforge.examples.$e >/dev/null && echo "$e OK" || echo "$e FALLO"
done

# 5. La interfaz gráfica
carbonforge-gui
```

---

## 4. Dónde tocar cada cosa

```
carbonforge/
├── builders/      ¿Nuevo tipo de estructura? Aquí.
├── dopants/       Química sustitucional
├── defects/       Vacancias, Stone-Wales, distorsión
├── topology/      Grafo de enlaces (networkx)
├── validation/
│   ├── checks.py        Geometría: distancias, coordinación, vacío
│   └── calculations.py  Física del cálculo: las trampas
├── calculations/  Caminos de banda, espectroscopía, espín-órbita
├── exports/       QE, SIESTA, LAMMPS, pseudopotenciales
├── results/       Lectura y graficado de salidas
├── workflows/     Lotes, convergencia, datasets ML
├── gui/
│   ├── params.py  Lógica (testeable sin pantalla)
│   └── app.py     Widgets Tk
└── cli/main.py    Línea de comandos
```

**Regla de oro de la GUI:** la lógica va en `params.py`, los widgets en
`app.py`. Si metes lógica en `app.py` deja de ser testeable sin pantalla.

---

## 5. Añadir cosas: ejemplos concretos

### Un tipo de estructura nuevo

Supongamos un nanotubo de doble pared.

**1.** Crea `carbonforge/builders/mwcnt.py`:

```python
"""Nanotubos de pared múltiple."""
from ase import Atoms
from .cnt import build_cnt

def build_double_wall_cnt(n_inner=5, m_inner=5, n_outer=10, m_outer=10,
                          length=10.0) -> Atoms:
    """Construye un nanotubo de doble pared.

    La separación entre paredes debe rondar 3.4 Å (distancia interplanar del
    grafito). Se rechazan combinaciones que den menos de 3.0 Å porque la
    repulsión de van der Waals las haría inestables.
    """
    inner = build_cnt(n_inner, m_inner, length=length, vacuum=0.0)
    outer = build_cnt(n_outer, m_outer, length=length, vacuum=0.0)
    gap = outer.info["radius"] - inner.info["radius"]
    if gap < 3.0:
        raise ValueError(
            f"Separación entre paredes {gap:.2f} Å < 3.0 Å: no es física."
        )
    ...
```

**2.** Expórtalo en `builders/__init__.py`.

**3.** Escribe el test en `tests/test_mwcnt.py` — incluyendo el caso que
debe fallar:

```python
def test_walls_too_close_rejected():
    with pytest.raises(ValueError, match="no es física"):
        build_double_wall_cnt(5, 5, 6, 6, length=6)
```

**4.** Si quieres que salga en la GUI, añádelo a `STRUCTURES` en
`gui/params.py`. Hay un test que comprueba que cada parámetro declarado
existe realmente en la firma del builder, así que un typo se caza solo.

### Una comprobación física nueva

Van en `validation/calculations.py` y devuelven un `ValidationReport`:

```python
def check_mi_cosa(atoms, spec) -> ValidationReport:
    report = ValidationReport()
    if <condición imposible>:
        report.errors.append("Explicación de por qué fallará y qué hacer.")
    elif <condición sospechosa>:
        report.warnings.append("Por qué preocuparse.")
    return report
```

Añádela a `check_full_setup` y escribe el test en `test_calc_validation.py`.

**Error vs advertencia:** error = el cálculo va a fallar o dará algo sin
sentido. Advertencia = probablemente quieras saberlo, pero puede ser
legítimo. Los errores abortan la exportación; las advertencias no.

### Un exportador nuevo

Crea `exports/<codigo>.py` con una función `write_<codigo>(atoms, outdir,
settings=None, force=False)` que:

1. Llame a `run_basic_checks(atoms)` y aborte si falla y `force` es falso.
2. Escriba los archivos.
3. Devuelva las rutas.

Mira `exports/siesta.py` como plantilla.

---

## 6. Antes de dar un cambio por bueno

```bash
pytest -q                    # todo verde
ruff check carbonforge       # estilo
```

Y pregúntate:

- ¿Escribí un test que **falla sin mi cambio**? Si el test pasa igual sin
  tocar nada, no está probando lo que crees.
- ¿Documenté el porqué físico, no solo el qué? Los docstrings de este
  proyecto explican *por qué* un valor es el que es.
- ¿Lo aleatorio acepta `seed`?
- Si toqué la GUI, ¿la lógica quedó en `params.py`?

---

## 7. Lo que MÁS falta validar (empieza por aquí)

Estas son las partes que **nunca se han probado contra la realidad**. Si
tienes acceso a Quantum ESPRESSO o SIESTA, validarlas es lo más valioso que
puedes hacer:

**1. Los lectores de resultados.** Están escritos contra los formatos
documentados y probados con archivos sintéticos, pero jamás han visto una
salida real. Coge un `bands.dat` y un `dynmat.out` de verdad y prueba:

```bash
carbonforge plot-bands tu_bands.dat --out test.png
carbonforge plot-spectrum tu_dynmat.out --kind raman --out test.png
```

Si algo revienta o sale raro, el archivo real es la referencia: manda un
fragmento y se ajusta el parser.

**2. Que las entradas generadas realmente corran.** Genera un caso pequeño,
consigue los pseudopotenciales y lánzalo:

```bash
carbonforge cnt --n 5 --m 0 --length 5 --out prueba --format qe
carbonforge pseudos prueba/... --dir ./pseudo
cd prueba/qe && pw.x -in pw.in > pw.out
```

**3. La interfaz gráfica, visualmente.** La lógica está probada, pero
**nadie ha visto nunca la ventana abierta**: se desarrolló en un entorno sin
Tkinter. Que los campos se lean bien, que nada se corte, que el scroll
funcione.

**4. `install.bat` en Windows.** Solo está revisado leyéndolo; nunca se ha
ejecutado.

**5. Los nombres de los pseudopotenciales.** Los genero siguiendo las
convenciones de PSLibrary y PseudoDojo, pero no he verificado archivo por
archivo que cada nombre exista tal cual en esas tablas.

---

## 8. Cosas que podrías añadir

Ideas ordenadas por relación valor/esfuerzo:

- **Densidad de estados (DOS/PDOS)**: complemento natural de las bandas.
  Sería `dos.x` y `projwfc.x` en QE, y en SIESTA sale casi gratis.
- **Nanotubos multipared y haces (bundles)**: hueco estructural claro.
- **Leer salidas de relajación** (`pw.out` de un `relax`) para recuperar la
  geometría final y volver a meterla en el ciclo.
- **Exportar a VASP** (POSCAR/INCAR/KPOINTS): otro código muy usado.
- **Fullerenos y cebollas de carbono**: 0D, que ahora solo aparece de refilón.
- **Un `Dockerfile`** con QE y SIESTA dentro, que resolvería de golpe el
  problema de no poder validar nada de verdad.

---

## 9. Depurar

```bash
# Entrar al depurador cuando un test falle
pytest carbonforge/tests/test_results.py --pdb

# Ver los print() (pytest los captura por defecto)
pytest -s

# Inspeccionar una estructura a mano
python -c "
from carbonforge.builders import build_cnt
from carbonforge.validation import run_basic_checks
a = build_cnt(6, 6, length=10)
print(a)
print(a.info)
print(run_basic_checks(a).summary())
"
```

Para mirar una estructura con los ojos:

```python
from carbonforge.viz import save_structure_png
save_structure_png(atoms, "debug.png")
```

---

## 10. Git

```bash
git checkout -b mi-mejora
# ... cambios ...
pytest -q
git add -A
git commit -m "Descripción de qué cambia y por qué"
git push -u origin mi-mejora
```

Los mensajes de commit de este proyecto explican el **porqué** y dejan
constancia de lo que no se pudo verificar. Merece la pena mantener esa
costumbre: dentro de seis meses lo agradecerás.
