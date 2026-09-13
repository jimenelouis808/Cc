# Fase 1 — Plan de migración acotado a la Opción 2

Estado: propuesto, sin ejecutar. Fecha: 2026-09-13.

## Alcance

**Dentro:** paraguas (workspace uv sobre los tres paquetes, sin mover código),
infraestructura compartida (LICENSE, CI, ruff, marcado de tests lentos) y
`nc-catalog` nuevo desde cero.

**Fuera, explícitamente:** `nc-core`, `nc-viz`, repartir `ramancarbon` en cuatro
paquetes, fusionar los dos linajes de `nc-build`, `apps/nc`, `apps/nc-gui`,
`blender_atomviz`, `nanocarbon_biblio`.

**Principio rector:** nada se mueve, nada se borra, los tres paquetes conservan
su nombre de import. El refactor es una fase posterior e independiente.

## Punto de partida medido

| Proyecto | Rama origen | LOC py | Tests | ruff |
|---|---|---:|---|---|
| `ramancarbon` v0.6.0 | `claude/raman-spectrum-analyzer-b462ke` | 60 108 | 1137 pasan, 1 skip (108 s) | limpio |
| `carbonforge` v0.1.0 | `claude/nanocarbon-framework-MIM17` | 22 224 | 639 pasan (26 s) | 40 avisos |
| `nanocarbon_lab` v0.1.0 | `claude/carbon-nanotube-generator-0dldiz` | 34 048 | 1138 pasan, 2 skip (28 min) | limpio |

Las copias de `nanocarbon_lab` en `cut2op` y `7tl3o3` están obsoletas: no se traen.
El `carbonforge` de la rama raman está congelado en `3c95508`: no se trae.

---

# TRAMO A — El paraguas (~150 k tokens)

## A1. Rama base y traída de los tres paquetes

```bash
git checkout -b nanocarbon origin/main          # main está vacía: 1 commit, 0 ficheros
git checkout origin/claude/raman-spectrum-analyzer-b462ke   -- ramancarbon
git checkout origin/claude/nanocarbon-framework-MIM17       -- carbonforge
git checkout origin/claude/carbon-nanotube-generator-0dldiz -- nanocarbon_lab
mkdir -p packages && git mv ramancarbon carbonforge nanocarbon_lab packages/
```

Las tres carpetas provienen de ramas distintas y no se solapan: sin conflictos y
sin `merge --allow-unrelated-histories`.

**Criterio de aceptación:** `uv pip install -e` de cada paquete y las tres suites
reproducen 1137 / 639 / 1138. Si alguna cifra baja, se para y se investiga.

## A2. Workspace raíz

`pyproject.toml` en la raíz, *virtual* (sin tabla `[project]`):

```toml
[tool.uv.workspace]
members = ["packages/*"]
```

Los `pyproject.toml` de cada miembro no se tocan en este paso. Un solo `uv.lock`.

**Riesgo medio-bajo:** el lock único debe resolver `ase`, `networkx`,
`scikit-image`, `matplotlib`, `numpy`, `scipy` a la vez. `scikit-image` (solo lo
usa `nanocarbon_lab` para marching cubes) es el candidato a dar problemas.

## A3. Infraestructura compartida

- **`LICENSE` (MIT).** Falta en las cinco ramas pese a que los cuatro
  `pyproject.toml` la declaran. Bloqueante para JOSS.
- **`ruff.toml` en la raíz.** Los tres ya usan `line-length = 100`. Unificar
  `target-version` en `py311`.
- **`.github/workflows/ci.yml`.** Matriz por paquete. En PR: `-m "not slow"`.
  Suite completa en nocturno.
- **`CLAUDE.md` raíz** describiendo la estructura y la regla "nada se mueve".

Conflictos medidos y su resolución:

| Conflicto | Estado | Decisión |
|---|---|---|
| `requires-python` | los 3 declaran `>=3.10`; la convención es 3.11+ | subir a `>=3.11` y verificar con los tests |
| `testpaths` | cada uno el suyo | se respeta, sin cambios |
| Scripts de consola | `ramancarbon`, `carbonforge`, `nanocarbon`, `nanocarbon-gui` | no colisionan, se quedan |
| ruff en `carbonforge` | 40 avisos (casi todos `--fix`) | arreglar en un commit aparte |

## A4. Acelerar la suite de `nanocarbon_lab`

**Medido, no supuesto.** La suite completa tarda 28:04. El marcador `slow` ya
existe en el `pyproject.toml` y marca 11 tests, pero excluirlos deja la suite
todavía por encima de 15 minutos: el marcado cubre una fracción pequeña del
tiempo real.

`pytest --durations=30` reparte así el tiempo (corrida de 47:16 que compitió con
otro proceso; el ranking es válido, los absolutos están inflados ~1,7×):

| Fichero | Suma | Tests en el top-30 |
|---|---:|---:|
| `test_capped_cnt.py` | 651,9 s | 3 |
| `test_swept.py` | 364,1 s | 1 (en **setup**: fixture de clase) |
| `test_analyse.py` | 331,3 s | 7 |
| `test_tmd_curved.py` | 329,2 s | 10 |
| `test_junction.py` | 227,4 s | 8 |
| `test_network.py` | 32,5 s | 1 |

Los dos más caros solos —`TestHelixDimensions::test_tube_is_sized_to_the_coil`
(519 s) y `TestRelaxedCoil::test_euler_budget_holds` (364 s en *setup*)— son el
31 % del total. El top-30 completo es el 68 %.

**Consecuencia: el objetivo de "suite rápida < 3 min" no se alcanza solo
marcando.** Corregido con los números reales: marcar el top-30 deja la suite
rápida en **~9 minutos**, no en 3. Llegar a 3 exigiría marcar del orden de 100
tests, y a partir de ahí se estaría escondiendo cobertura en vez de aislar
outliers.

Plan revisado, en dos movimientos que se suman:

1. **Marcar `slow` el top-30** (los seis ficheros de arriba). Suite rápida: ~9 min.
2. **Añadir `pytest-xdist`** y correr `-n 4 --dist loadscope`. `loadscope` es
   obligatorio, no opcional: el coste de `test_swept.py` está en un fixture de
   clase, y repartir por test lo recalcularía en cada worker.

Resultado esperado: **suite rápida ~2,5 min, suite completa ~7 min.** Es lo que
hace viable verificar en cada paso del tramo B.

Coste ~35 k tokens. Sigue siendo la tarea con mejor retorno del plan y sigue
yendo antes que ninguna otra.

**Entregable A:** rama `nanocarbon` con 2 914 tests verdes, un `uv sync`, CI en
verde, MIT. Útil por sí solo aunque el plan se detenga aquí.

---

# TRAMO B — `nc-catalog` (~300 k tokens)

## Principio de diseño

`nc-catalog` es **consumidor**: importa de los tres paquetes y ninguno importa de
él. Por eso no exige refactorizar nada.

Dependencias: solo stdlib (`sqlite3`, `hashlib`, `pathlib`, `json`). Los tres
paquetes entran como extras opcionales, de modo que el catálogo se instala y se
prueba sin `ase` ni `matplotlib`.

## B1. Esquema

```sql
samples(
  id TEXT PRIMARY KEY,            -- convención de nombre del laboratorio
  tipo, metodo_sintesis, precursor, dopante_nominal,
  temperatura_C REAL, flujos_json TEXT, tiempo_min REAL,
  lote, fecha, notas)

measurements(
  id INTEGER PRIMARY KEY,
  sample_id REFERENCES samples(id),
  tecnica,                        -- raman | xrd | xps | cv | gcd | eis | sem | eds
  instrumento, fecha,
  ruta_relativa TEXT,             -- relativa a la raíz de datos configurada
  sha256 TEXT,
  parametros_json TEXT,           -- parámetros de adquisición
  anadido_en)

results(                          -- formato largo clave-valor
  id INTEGER PRIMARY KEY,
  sample_id, measurement_id,
  nombre TEXT,                    -- ID_IG, N_at_pct, L_a, eta_10, tafel_slope…
  valor REAL, unidad TEXT, metodo TEXT,
  software_version TEXT, creado_en,
  UNIQUE(measurement_id, nombre, metodo))
```

Decisiones que hay que tomar explícitamente:

- **`UNIQUE(measurement_id, nombre, metodo)`** para que re-analizar **actualice**
  en vez de duplicar. Sin esto, el catálogo acumula basura en tres semanas.
- **Raíz de datos configurable** (tabla `config` o `NC_DATA_ROOT`), de modo que
  `ruta_relativa` sea portable entre máquinas.
- **`PRAGMA journal_mode=WAL`**, y aviso explícito si el `.db` vive en carpeta
  sincronizada (OneDrive/Drive): SQLite con dos escritores concurrentes sobre
  almacenamiento sincronizado se corrompe.
- **`schema_version`** + migraciones desde el día uno.

## B2. Integridad de los crudos

`verify()` recorre `measurements`, recalcula el SHA-256 y reporta
**OK / CAMBIADO / DESAPARECIDO**. Nunca mueve ni reescribe un archivo crudo.

## B3. Adaptadores — la pieza que conecta

Un módulo delgado por técnica que traduce el objeto de resultado *ya existente* a
filas de `results`:

| Técnica | Fuente | Nota |
|---|---|---|
| Raman | `AnalysisResult.to_dict()`, `ratios: dict[str, Ratio]` | `Ratio(name, value, basis)` **ya es una fila**: nombre, valor, método |
| Raman | `IndexSet.to_dict()` | índices estructurales |
| Echem | `analyse_cv`, `analyse_gcd`, `analyse_eis` | reexportados en `echem/__init__` |
| XRD | `refine`, `identify_phases` | reexportados en `xrd/__init__` |
| XPS | `XPSFitResult`, `quantify` | at% por estado químico |
| Build/Forge | manifest de `write_dataset` | opcional |

Esto es lo que abarata el Tramo B: los objetos de resultado ya exponen
`to_dict()` y ya llevan nombre, valor y base de cálculo. El adaptador es un bucle,
no un diseño nuevo.

## B4. CLI mínima

`nc-catalog init | add-sample | add-measurement | ingest | verify | query | export`

argparse, con el mismo patrón `build_parser` → `add_parser` → `_cmd_*` →
`set_defaults(func=…)` que ya usan los tres CLIs existentes.

## B5. Tests

pytest con base de datos temporal. Objetivo 60–80 tests. No toca ninguno de los
2 914 existentes.

---

## Orden de ejecución y puntos de corte

```
A4 → A1 → A2 → A3   |   B1 → B2 → B3 → B4 → B5
```

A4 va primero aunque pertenezca al tramo A: cada verificación posterior depende de
que la suite rápida exista. Cada paso se commitea por separado; si el plan se
detiene en cualquier punto, lo anterior queda funcionando.

## Riesgos

| # | Riesgo | Prob. | Mitigación |
|---|---|---|---|
| 1 | Subir a Python 3.11 rompe algo | baja | los 2 914 tests lo detectan |
| 2 | El lock único no resuelve `scikit-image` + `ase` + `matplotlib` | media-baja | fijar versiones; en el peor caso, dejar `scikit-image` como extra |
| 3 | La suite de 28 min hace lenta cada verificación | **confirmado** | A4 primero: marcado del top-30 + `pytest-xdist -n 4 --dist loadscope` |
| 4 | 40 avisos de ruff en `carbonforge` | baja | commit aparte, `--fix` |
| 5 | Pérdida de código | **nula** | nada se borra de las ramas originales |

## Preguntas abiertas

1. **¿Dónde viven los archivos crudos y respecto a qué raíz es la ruta relativa?**
   Disco local, NAS, o nube sincronizada. Si es nube sincronizada con más de un
   escritor, el diseño de la base de datos cambia (riesgo de corrupción).
2. **¿Cuál es la convención de nombre de muestra?** Un ejemplo real con su
   desglose. Determina si `samples.id` es parseable o una etiqueta opaca con los
   campos aparte.

Ninguna de las dos bloquea el Tramo A.
