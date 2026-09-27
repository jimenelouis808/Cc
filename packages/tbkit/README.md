# tbkit

Tight binding parametrizable para nanocarbonos: el cuarto paquete de la suite
nanocarbon. Independiente de los otros tres: intercambia estructuras por archivo
(extxyz de carbonforge o nanocarbon_lab) y no importa ninguno.

```bash
uv sync --all-packages --extra dev
tbkit levels benceno.xyz                      # niveles, gap, cargas
tbkit bands grafeno.extxyz --path GKMG -o bandas.csv
tbkit hubbard zgnr.extxyz --U 2.7 --kmesh 48 --m-energy m.csv
tbkit orbital coroneno.xyz --band homo -o homo.cube
tbkit relax cluster.xyz --model sp3 -o relajado.extxyz
tbkit phonons diamante.extxyz --model sp3 --kmesh 8
tbkit raman diamante.extxyz --model sp3 --kmesh 8 -o raman.csv
tbkit run simulacion.json          # reproducible desde el archivo
```

## Parámetros: fuera del código y con su fuente

Los conjuntos incluidos viven en `tbkit/parameters/*.json`. Cada número lleva
unidad, descripción y fuente, y cada archivo, su referencia, el sistema para el
que se obtuvo y su rango de validez. `model_to_dict` escribe un modelo (incluido
uno ajustado) en ese mismo formato, y `load_parameters(ruta)` lo lee.

## Qué hay

| Capa | Módulo | Qué hace |
|---|---|---|
| 1. Modelo π | `params.pi_model` | Orbital π local (normal a la superficie), t = −2,7 eV, heteroátomos N, B, O con parámetros de Hückel (Streitwieser); opción de hopping con deformación, t·exp(−β(d/1,42 − 1)) |
| 2. Slater–Koster sp³ | `params.xu_carbon`, `slater_koster` | ssσ, spσ, ppσ, ppπ con leyes constante, exponencial, Harrison, GSP o tabla; solapamiento opcional (H c = E S c). Carbono de Xu–Wang–Chan–Ho (1992) incluido |
| 3. Parámetros | `skf`, `fit` | Lector de archivos `.skf` de DFTB (formato simple; no se incluye ningún conjunto publicado); ajuste por mínimos cuadrados a niveles de referencia, con lector de `gpaw.txt` |
| 4. Cargas autoconsistentes | `scc` | DFTB2 con γ de Klopman–Ohno; solo sistemas finitos (sin Ewald) |
| 5. Espín | `hubbard` | Hubbard de campo medio; magnetización frente a la energía (m(E), dm/dE), frente al campo (M(h), χ) y frente al dopaje o el nivel de Fermi |
| 6. Periódico | `hamiltonian`, `kpoints` | H(k) por suma de Bloch; mallas Γ-centradas y caminos de bandas de ASE |
| 7. Parte repulsiva | `repulsive`, `forces`, `calculator` | Energía libre total (banda − TS + repulsión, + SCC), fuerzas de Hellmann–Feynman (ortogonal, no ortogonal, periódico, SCC), repulsión embebida de Xu y spline de los `.skf`; calculadora ASE para relajar y para fonones en Γ |
| Raman no resonante (fase E, paso 1) | `optics`, `raman` | Polarizabilidad por suma sobre estados (finitos y cristales, ε∞) o por respuesta lineal SCC con apantallamiento (finitos); tensores Raman dα/dQ sobre los fonones del modelo, actividades, razón de despolarización y espectro con factores de láser y Bose |
| Reproducibilidad | `record`, `tasks` | `tbkit run simulacion.json` guarda estructura, parámetros completos, versión, commit, fecha, ajustes y resultados; `record.replay` lo repite |

Salidas (`analysis`): matriz densidad P = Σ f c c†, poblaciones Mulliken y
Löwdin, cargas y momentos por átomo, órdenes de enlace de Mayer/Wiberg y de
Coulson (π), DOS y PDOS (por elemento, átomo u orbital; suman la DOS total),
bandas, y orbitales en una rejilla real con funciones de Slater → archivos
`.cube` (VESTA, VMD, Avogadro).

## Qué comprueban los tests

Todo contra resultados conocidos: bandas del grafeno (punto de Dirac, Γ = ±3|t|,
fórmula analítica), niveles del benceno y gaps de los acenos de Hückel, nanotubos
zigzag metálicos cuando n es múltiplo de 3, invariancia de los bloques
Slater–Koster ante rotaciones, dímero no ortogonal resuelto a mano, gap del
diamante, reglas de suma de DOS y PDOS, bordes de una cinta zigzag
antiparalelos (±0,24 μB con U = |t|), grafeno sin magnetismo por debajo de
U_c ≈ 2,2|t|, **teorema de Lieb** en el triangulene (M = |N_A − N_B| = 2),
m(E_F) = M, apantallamiento de la carga del N de la piridina con SCC, un `.skf`
generado a partir del modelo de Xu que lo reproduce (2 meV), y ajustes que
recuperan parámetros conocidos.

## Validación de la parte repulsiva (carbono de Xu, sin ajustar nada)

| Magnitud | tbkit | Experimento |
|---|---|---|
| Constante de red del diamante | ≈ 3,555 Å | 3,567 Å |
| Enlace C–C del grafeno | ≈ 1,42 Å | 1,42 Å |
| Energía de cohesión del diamante | ≈ 7,24 eV/átomo | 7,37 eV/átomo |
| Fonón óptico del diamante (Γ, línea Raman) | ≈ 1224 cm⁻¹ | 1332 cm⁻¹ (−8 %) |
| Banda G del grafeno | ≈ 1666–1682 cm⁻¹ | 1582 cm⁻¹ (+6 %) |

Las fuerzas se comparan con diferencias finitas de la energía en todos los
casos (ortogonal periódico con puntos k, no ortogonal finito y periódico, SCC
con solapamiento). La banda G tiene una anomalía de Kohn (se acopla al cono de
Dirac): su frecuencia depende de la malla de puntos k y del ensanchamiento, y
converge despacio; usa mallas densas (≥ 36×36) y comprueba la convergencia.

## Raman no resonante: cómo se valida

Primero por simetría, que no depende de los parámetros:

- **Diamante:** un solo triplete activo (T₂g), tensor con diagonal nula y
  ρ = 0,75.
- **C₆₀:** exactamente 10 frecuencias activas, 2 polarizadas (A_g, ρ ≈ 0) y 8
  despolarizadas y quíntuples (H_g, ρ = 0,75). Test marcado `slow`.

Luego consistencia (respuesta lineal = campo finito = suma sobre estados sin
apantallar) y experimento, diciendo el error:

| Magnitud | tbkit | Experimento |
|---|---|---|
| ε∞ del diamante (partículas independientes) | 4,75 | 5,7 (−17 %) |
| α de C₆₀ sin apantallar | ≈ 260 Å³ | 76,5 ± 8 Å³ |
| α de C₆₀ apantallado (SCC, U = 10 eV) | ≈ 62 Å³ | 76,5 ± 8 Å³ (−20 %) |

El apantallamiento no es un detalle: sin él, la polarizabilidad de una molécula
sale unas tres veces mayor. En cristales no hay apantallamiento SCC (haría falta
Ewald), así que su α es de partículas independientes.

## Lo que no hace, y dónde está la trampa

- **Raman solo no resonante y con gap**: metales, semimetales (el grafeno) y
  sistemas de capa abierta se rechazan, y también un láser a menos del 20 % del
  gap. El Raman del grafeno es siempre resonante; la banda 2D (segundo orden,
  doble resonancia) es el paso siguiente de la fase E, no este.

- **Energías y fuerzas solo con modelos que tienen parte repulsiva**: el de Xu
  (carbono puro) y los `.skf` con su spline. El modelo π no la tiene y lo dice.
  El modelo de Xu no describe H ni heteroátomos.
- **Campo medio no es correlación**: un copo con M = 0 y momentos locales es, en
  realidad, un singlete correlacionado; los momentos son el parámetro de orden
  de la aproximación.
- **SCC solo en sistemas finitos**: un sistema periódico necesitaría una suma de
  Ewald; se rechaza en vez de aproximarlo.
- **Parámetros π de Hückel para heteroátomos**: buenos para tendencias, no para
  niveles cuantitativos. Ajústalos a tu DFT con `fit`.
- **Convención `sp` de los `.skf` heteronucleares**: se toma Hsp0 de `A-B.skf`
  como ⟨s de A|H|p de B⟩. Compruébalo con un dímero frente a DFTB+ antes de
  confiar en un conjunto heteronuclear.
- **Ajuste alineado en mitad del gap**: las energías absolutas de DFT no tienen
  cero común con el modelo, así que las energías on-site quedan definidas salvo
  un desplazamiento y puede haber soluciones espejo; revisa el resultado.

Sin interfaz gráfica todavía: el toolkit (PySide6 + pyvista es el candidato) se
decide al empezarla (docs/PLAN_SUITE.md, decisión 3).
