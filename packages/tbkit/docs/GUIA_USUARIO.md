# Guía de usuario de tbkit

Cómo usar tbkit de principio a fin. Qué calcula cada cosa: [`METODOS.md`](METODOS.md).
Cuánto se equivoca: [`VALIDACION.md`](VALIDACION.md).

## 1. Instalación

```bash
uv sync --all-packages --extra dev               # núcleo y tests
uv sync --all-packages --extra dev --extra gui   # + ventana (PySide6, pyvista)
uv pip install 'tbkit[phonons]'                  # + phonopy (fonones en toda la zona)
uv pip install 'tbkit[plot]'                     # + matplotlib (figuras de `tbkit report`)
```

En Linux, la ventana necesita libEGL, libxkbcommon y las libxcb-* del sistema.
GPAW no es dependencia: solo lo usan las recetas que generan referencias (ver §6).

## 2. Elegir el modelo

| Tu estructura | Conjunto | Por qué |
|---|---|---|
| Solo bandas π cerca de E_F (grafeno, tubos, cintas, magnetismo de bordes) | `pi` | Rápido; sin energías ni fuerzas |
| Carbono puro, enlaces sp²/sp³ regulares | `sp3` (Xu) | Energías, fuerzas, fonones |
| Carbono con defectos, curvatura, anillos 5–7, amorfo | `tang` | Saltos que dependen del entorno; mejores fuerzas que Xu en 8 de 9 entornos |
| C con H y N | `chn` | SCC; H y N ajustados a GPAW |
| … más O | `chno` | Epóxido, OH, carbonilo, carboxilo… |
| … más B, S, P o Se (uno a la vez) | `chnob`, `chnos`, `chnop`, `chnose` | `chno` fijo más un elemento |
| Tus propios parámetros | `--parameters archivo.json` o `--skf carpeta` | Mismo formato que los incluidos |

La ventana sugiere el conjunto mínimo que cubre los elementos de la estructura.
Antes de confiar en un resultado, lee la `validity` del conjunto (está en su
archivo y en `VALIDACION.md`): dice para qué sistemas sirve y con qué error.

**SCC**: los `xu_ch*` se ajustaron con cargas autoconsistentes y se usan con
ellas por defecto; Xu, Tang y π, sin ellas. «Comparar SCC sí/no» (pestaña
Electrónica) muestra cuánto depende tu resultado de la SCC.

## 3. La ventana (`tbkit-gui`)

```bash
uv run tbkit-gui estructura.extxyz
```

Izquierda: estructura y modelo (carga, SCC, kT). Centro: vista 3D. Derecha,
una pestaña por tarea; cada control tiene una ayuda (pasa el ratón) y cada
pestaña un recuadro que explica para qué sirve (menú Ayuda para ocultarlos).

| Pestaña | Flujo típico |
|---|---|
| Electrónica | «Calcular estado fundamental» primero (lo usan las demás): niveles, gap, DOS/PDOS, bandas, cargas; «Comparar SCC sí/no» |
| Orbitales | Isosuperficie de un nivel (HOMO, LUMO, estados de borde) |
| Magnetismo | Hubbard de campo medio: momentos, m(E), puntos de partida, barridos de campo o dopaje |
| Geometría y modos | «Relajar» → «Modos en Γ» → ordenar por sitio (anillos 5/6/7, heteroátomo, C vecinos) → animar o ver flechas → frecuencia del modo con otro conjunto |
| Fonones (ZB) | phonopy: dispersión, DOS, simetría de cada modo en Γ, F/S/Cv; carpeta para reanudar |
| Espectros | Raman (no resonante, resonante), IR; fonones del modelo o de QE; escalar frecuencias (aparte); exportar CSV |
| Grafeno | G, 2D y 2D′ por doble resonancia y dispersión de la 2D con el láser |

Archivo → «Guardar registro del último cálculo» escribe un registro
reproducible (versión, commit, parámetros completos, ajustes, resultados).
Desde carbonforge, «Abrir en tbkit» pasa la estructura por archivo.

## 4. Línea de comandos

```bash
tbkit levels   molecula.xyz --model chn          # niveles, gap, cargas
tbkit bands    grafeno.extxyz --path GKMG -o bandas.csv
tbkit dos      tubo.extxyz --kmesh 60 --pdos element -o dos.csv
tbkit hubbard  cinta.extxyz --U 2.7 --kmesh 48 --m-energy m.csv
tbkit orbital  benceno.xyz --band homo -o homo.cube
tbkit relax    estructura.xyz --model tang -o relajada.extxyz
tbkit phonons  diamante.extxyz --model sp3 --kmesh 8
tbkit raman    piridina.xyz --model chn [--resonant 2.33 3.5 --eta 0.1] [--modes dynmat.out]
tbkit ir       piridina.xyz --model chn
tbkit graphene-raman --laser 1.96 2.41 2.80
tbkit run      simulacion.json                    # reproducible, guarda su registro
tbkit report   out/doped_raman [--ancho doble] [--figuras pdf,png]   # reporte, CSV y figuras
```

`tbkit <orden> --help` da todas las opciones.

### Reportes, datos y figuras para publicar

`tbkit report CARPETA_O_ESPECTRO` (en la ventana, «Exportar reporte…» en Espectros)
lee lo que dejó una receta y escribe junto a ella:

- `reporte_X.html`: una página que se abre sin internet, con gráficas interactivas,
  tablas, el método y sus límites; cada gráfica y tabla tiene un botón CSV.
- `reporte_X_datos/`: cada gráfica y tabla en CSV (UTF-8 con BOM, para que Excel lea
  los acentos; Origin, Igor y gnuplot también lo leen) e `indice.json`.
- `reporte_X_figuras/`: cada gráfica en PDF y SVG (vectoriales, fuentes incrustadas)
  y PNG a 600 dpi, a una columna de revista (85 mm) o dos (`--ancho doble`, 178 mm),
  fuente de 7 pt y paleta Okabe-Ito (legible con daltonismo y en gris). Necesita
  el extra `plot`.

Reconoce las carpetas de `doped_raman` y `doped_gpaw` y cualquier espectro de tbkit
(`.npz` con `grid`, o el CSV de `--out` y de «Exportar CSV…»). Para otra receta se
añade un adaptador en `tbkit/report.py` que convierta sus archivos en secciones,
gráficas y tablas; la página, los CSV y las figuras salen de eso. Ningún número de
un reporte se escribe a mano: todos se leen de los archivos.

## 5. Flujos de trabajo

**Molécula o copo (Raman/IR).** Relajar con el conjunto adecuado → modos en Γ
→ Raman no resonante (necesita gap) e IR. Compara frecuencias crudas y
escaladas (λ del conjunto, frente a GPAW). Las intensidades IR son
semicuantitativas (factor ~2 frente a GPAW).

**Cristal con gap (diamante, h-BN, grafano).** Relajar con una malla k suficiente →
modos en Γ o «Fonones (ZB)» → Raman con la α de cristal.

**Sistema sin gap (grafeno, tubos metálicos, coils, dopados metálicos).**
- Los modos avisan: converge la malla k antes de usar frecuencias
  (`tasks.kmesh_convergence`); en grafeno con Xu hacen falta ≥ 48 k por eje a
  kT = 0,05 eV (con 12, G sale 100 cm⁻¹ baja).
- El Raman no resonante no aplica: usa el resonante (pestaña Espectros, o
  `resonant_raman(select=..., cache_dir=...)` en celdas grandes) o, para el
  grafeno, la pestaña Grafeno.

**Dopaje sustitucional: ¿en qué sitio?**
`python -m tbkit.recipes.site_screening estructura.extxyz N salida/ --model chn --relax-top 5`
agrupa los C por entorno, calcula uno por clase, relaja los mejores y escribe
los primeros para comprobarlos con DFT. Grupos funcionales: constrúyelos con
nanocarbon_lab y compáralos como archivos.

**Comparar TB con DFT en un modo concreto.** `sites.projected_frequency(atoms,
modo, calculadora)` con una calculadora de GPAW (o de otro conjunto) da la
frecuencia de ese modo con esas fuerzas: 2 cálculos de fuerzas en lugar de una
hessiana. Es una cota superior si el modo no es propio de esa calculadora.

## 6. Cálculos largos y recetas

Las recetas de `tbkit/recipes/` son reproducibles y reanudables: cada punto es
un archivo en la carpeta de trabajo, y un cálculo interrumpido sigue donde
quedó. Para campañas de horas usa un supervisor como
`recipes/run_crystals.sh` (relanza si el proceso muere); nunca empieces de
cero tras un corte.

| Receta | Para qué |
|---|---|
| `xu_chn`, `xu_chno`, `xu_bsp`, `tang_fit` | Rehacer un ajuste (guarda un archivo nuevo) |
| `frequency_scaling` | Factor de escala de frecuencias de un conjunto frente a GPAW |
| `crystal_validation` | Un conjunto en cristales frente a GPAW |
| `cnt_validation` | Nanotubos prístinos (RBM, G) |
| `nanocoil` | La coil periódica de 204 átomos: TB, GPAW, Raman resonante |
| `site_screening` | Sitios de un dopante sustitucional |
| `structure_screening` | Ordenar estructuras ya construidas de una composición (grupos funcionales) |
| `doped_raman` | Raman resonante, banda D estimada e IR de la coil sin dopar, con N y con amina |
| `doped_gpaw` | Esos resultados frente a GPAW en los modos clave (frecuencia e IR) |
| `active_learning` | Buscar mínimos espurios de un conjunto con GPAW |
| `validation_report` | Regenerar `docs/VALIDACION.md` desde los datos |

GPAW (para las recetas que lo usan) va en un entorno aparte, sin MPI:
`apt install libxc-dev libopenblas-dev g++` y `CC=g++ uv pip install gpaw`.

## 7. Antes de publicar un número

- ¿La estructura está dentro de la `validity` del conjunto?
- ¿Relajaste con el mismo modelo y la misma malla k que usas para los modos?
- ¿Sistema sin gap? ¿Convergiste la malla k?
- ¿Raman resonante? Las energías de resonancia son del modelo (gaps TB/KS
  menores que los ópticos) y las alturas relativas dependen de η.
- ¿Frecuencias escaladas? Dilo, y da también las crudas.
- ¿Magnetismo? Los momentos son de campo medio.
- Guarda el registro (`tbkit run` o «Guardar registro»).
