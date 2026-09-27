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
```

## Qué hay

| Capa | Módulo | Qué hace |
|---|---|---|
| 1. Modelo π | `params.pi_model` | Orbital π local (normal a la superficie), t = −2,7 eV, heteroátomos N, B, O con parámetros de Hückel (Streitwieser); opción de hopping con deformación, t·exp(−β(d/1,42 − 1)) |
| 2. Slater–Koster sp³ | `params.xu_carbon`, `slater_koster` | ssσ, spσ, ppσ, ppπ con leyes constante, exponencial, Harrison, GSP o tabla; solapamiento opcional (H c = E S c). Carbono de Xu–Wang–Chan–Ho (1992) incluido |
| 3. Parámetros | `skf`, `fit` | Lector de archivos `.skf` de DFTB (formato simple; no se incluye ningún conjunto publicado); ajuste por mínimos cuadrados a niveles de referencia, con lector de `gpaw.txt` |
| 4. Cargas autoconsistentes | `scc` | DFTB2 con γ de Klopman–Ohno; solo sistemas finitos (sin Ewald) |
| 5. Espín | `hubbard` | Hubbard de campo medio; magnetización frente a la energía (m(E), dm/dE), frente al campo (M(h), χ) y frente al dopaje o el nivel de Fermi |
| 6. Periódico | `hamiltonian`, `kpoints` | H(k) por suma de Bloch; mallas Γ-centradas y caminos de bandas de ASE |

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

## Lo que no hace, y dónde está la trampa

- **Sin energías totales ni fuerzas**: no hay parte repulsiva (ni la de Xu ni la
  spline de los `.skf`). Sirve para estructura electrónica, no para relajar.
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
