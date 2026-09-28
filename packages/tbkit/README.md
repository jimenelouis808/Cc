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
| Raman no resonante y resonante (fase E) | `optics`, `raman`, `dipoles`, `resonance` |
| Infrarrojo | `infrared` | μ del modelo (cargas + dipolos intraatómicos), cargas de Born, km/mol | Polarizabilidad por suma sobre estados (finitos y cristales, ε∞) o por respuesta lineal SCC con apantallamiento (finitos); tensores Raman dα/dQ sobre los fonones del modelo, actividades, razón de despolarización y espectro con factores de láser y Bose |
| C, H y N (fase E, paso 2) | `parameters/xu_chn.json`, `references`, `recipes` | C–C de Xu intacto; H y N ajustados a GPAW (PBE, LCAO dzp) en niveles, fuerzas y energías; U de H, C, N calculadas con el átomo de GPAW; SCC. Receta reproducible y referencias incluidas |
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

## C, H y N: cómo se obtuvo `xu_chn` y cuánto vale

`--model chn` (o `load_parameters("xu_chn")`) añade H y N al carbono de Xu sin
tocar el C–C. Todo sale de dos recetas que cualquiera puede repetir:

1. `python -m tbkit.recipes.chn_references refs.json --frequencies` (necesita
   GPAW): relaja con GPAW (PBE, LCAO dzp, h = 0,2 Å) 16 moléculas de
   entrenamiento — hidrocarburos saturados, insaturados y aromáticos, aminas,
   iminas, nitrilos, piridina (N piridínico), pirrol (N pirrólico), N₂H₄, N₂ —
   y calcula cada una con cuatro desplazamientos al azar y dos escalados; guarda
   niveles, energías, fuerzas y frecuencias de CH₄, NH₃, HCN, benceno y
   piridina. Otras 6 moléculas, entre ellas un coroneno con N piridínico
   (C₂₃H₁₁N), son el conjunto de prueba, que el ajuste no ve. El resultado está
   en `parameters/references/gpaw_chn.json` (su SHA-256 va en el archivo de
   parámetros); las frecuencias, que validan y no se ajustan, en
   `gpaw_chn_frequencies.json`.
2. `python -m tbkit.recipes.xu_chn refs.json xu_chn.json`: primero ajusta los
   parámetros electrónicos a los niveles (todos los ocupados, el LUMO y el
   LUMO+1, con un desplazamiento común entre el cero de Xu y el vacío de GPAW);
   luego los reajusta a niveles, fuerzas y energías a la vez, resolviendo la
   repulsión por pares de forma exacta en cada paso (es lineal en sus
   coeficientes), como el ajuste conjunto de NRL-TB (Papaconstantopoulos et
   al., 2024).

Las U de Hubbard no se ajustan: son dε/dn del nivel de valencia del átomo libre
con el átomo de GPAW (PBE), la definición de DFTB, y coinciden con las de DFTB
mio (H 0,4195, C 0,3647, N 0,4309 Ha).

Resultado (todo frente a GPAW salvo α):

| Qué | Valor |
|---|---|
| Niveles (entrenamiento) | RMS 1,20 eV (los profundos pesan más; HOMO–LUMO mejor) |
| Fuerzas / energías relativas | RMS 0,48 eV/Å / 0,12 eV |
| Enlaces tras relajar con TB | C–H, N–H ≤ 0,02 Å; aromáticos C–C, C–N ≤ 0,02 Å; coroneno con N (prueba) ≤ 0,02 Å |
| Frecuencias (RMS) | CH₄ 52, NH₃ 55, benceno 50, piridina 69, HCN 156 cm⁻¹ (las de GPAW LCAO rompen degeneraciones hasta ~50 cm⁻¹ en los modos blandos: esa es la resolución de la comparación) |
| Raman del benceno | 2A₁g + 4E₂g + E₁g, como debe; respiración 1021 cm⁻¹ polarizada (exp. 992) |
| Raman de la piridina | modos de anillo polarizados a 992 y 1028 cm⁻¹ (exp. 991 y 1030) |
| α | con la polarizabilidad atómica extra, ±5 % de GPAW en entrenamiento y ±2 % en prueba, anisotropía incluida (ver abajo) |

Límites que el archivo declara en `validity`: solo sistemas finitos de capa
cerrada (SCC sin Ewald); energías comparables solo entre geometrías de la misma
composición (no se ajustaron atomizaciones); sin interacción H–H; y el C–C de
Xu falla en anillos tensos (aziridina, 0,22 Å), y los C–C y C–N simples junto
a un heteroátomo (aminas, nitrilos) se desvían ~0,08 Å.

**Dipolos intraatómicos (`dipoles`).** Con orbitales puntuales, la respuesta
solo mueve cargas atómicas y una carga no puede polarizarse perpendicular a
una cadena ni fuera del plano de una molécula plana: α⊥ = 0 exacto en HCN,
α_zz = 0 en el benceno, y ρ = 1/3 exacto en los modos totalmente simétricos de
una molécula lineal (0,125 en la respiración del benceno). Ahora el operador de
posición incluye ⟨s|r|p⟩ = d por elemento (`onsite_dipole`, calculado con el
átomo libre de GPAW: C 0,495 Å, N 0,413 Å; nada ajustado), su signo sale de la
convención de orbitales del propio modelo, y el apantallamiento incluye dipolos
atómicos con el mismo núcleo de Klopman–Ohno que las cargas (derivadas de γ;
sin parámetros nuevos). La respuesta lineal coincide con el campo finito del
mismo funcional y, sin apantallar, con la suma sobre estados.

**Polarizabilidad atómica extra.** Aun con esos dipolos, una base s+p no puede
polarizarse hacia las funciones difusas y de polarización que un átomo real
tiene. Lo que falta se da a cada átomo como un dipolo polarizable
(`extra_polarizability`: H 0,43, C 0,95, N 0,74 Å³), apantallado junto con las
cargas y los dipolos por el mismo núcleo (sin autointeracción). Los tres
números se ajustan a los tensores α completos, anisotropía incluida, de 12
moléculas calculadas con GPAW en rejilla real (FD, PBE; receta
`tbkit.recipes.xu_chn_alpha`, referencias en
`parameters/references/gpaw_chn_alpha.json`); nada más del modelo cambia.

| | puntual | + dipolos d | + α extra | referencia |
|---|---|---|---|---|
| α media, entrenamiento (vs GPAW) | | −41 a −73 % | ±5 % (NH₃ −11 %) | |
| α media, 5 moléculas de prueba (vs GPAW) | | −43 a −62 % | −2 a +1 % | |
| benceno, fuera / en el plano | 0 / 10,3 | 1,5 / 8,8 | 6,3 / 13,4 Å³ | GPAW 6,9 / 13,0 |
| C₆₀ | 61,7 | 69,2 | 84,5 Å³ | exp. 76,5 ± 8 |
| ε∞ del diamante | 4,75 | 2,96 | 5,07 | exp. 5,7 |
| ρ de la respiración del benceno | 0,125 | 0,092 | 0,071 | muy polarizada |

C₆₀ y el diamante no entran en el ajuste. GPAW-PBE ya sobreestima las α un
4–7 % frente al experimento (CH₄ 2,71 frente a 2,59; benceno 10,98 frente a
10,32). En cristales la polarizabilidad extra se suma por celda sin campos
locales (no hay apantallamiento periódico todavía).

## Raman resonante (primer orden)

`tbkit raman estructura.xyz --model chn --resonant 2.33 3.5 4.0 --eta 0.1`
(o `resonance.resonant_raman`) da la actividad de cada modo a cada energía de
láser: los tensores son ∂α(ω_L + iη)/∂Q con la polarizabilidad compleja
dependiente de la frecuencia (`optics.dynamic_polarizability_*`; en finitos,
apantallada con cargas, dipolos y α extra), el método de fonón congelado de
Gillet, Giantomassi y Gonze (PRB 88, 094305, 2013). η es el ensanchamiento de
las excitaciones. `perfil = resultado.profile(1690)` es el perfil de
excitación de un modo; `resultado.at(E)` alimenta `spectrum`.

Comprobado: con ω_L = η = 0 es exactamente la α estática; muy por debajo del
gap reproduce el Raman no resonante (2 %); Im α es positiva (absorción); los
modos prohibidos por simetría siguen prohibidos a cualquier láser; y el C=C
del butadieno se amplifica ~3000 veces cuando el láser alcanza su transición
π→π*, hasta dominar el espectro, como en los polienos reales.

Lo que no es: el Raman de resonancia vibrónico (Franck–Condon, término A de
Albrecht) con sobretonos; aquí los estados excitados entran solo por sus
energías y densidades de transición. Y las energías de resonancia son las del
modelo: la π→π* del butadieno sale a 4,25 eV (medida ~5,9 eV), porque los gaps
de un TB mínimo (y los de Kohn–Sham a los que se ajustó) son más pequeños que
los ópticos. Compara perfiles relativos, no energías absolutas. En semimetales
(grafeno) la suma en k converge mal con diferencias finitas: el grafeno tiene
su módulo perturbativo.

## Infrarrojo

`tbkit ir molecula.xyz --model chn` (o `infrared.infrared`): el dipolo del
modelo es μ = Σ Q_A R_A − Σ p_A, con las cargas de Mulliken (autoconsistentes)
y los dipolos intraatómicos del mismo operador de posición que da la α, de
modo que IR y Raman salen de una sola respuesta. Cargas de Born por
diferencias centrales, intensidades |∂μ/∂Q|² en km/mol, espectro ensanchado;
`--modes` acepta los modos de QE y `--charges-only` quita los dipolos
intraatómicos.

Comprobado: las cargas de Born de una molécula neutra suman cero; girar la
molécula no cambia nada; el benceno absorbe solo en A₂u y en los tres E₁u.

Frente a GPAW (PBE, LCAO dzp) en 8 moléculas, usando **los modos de GPAW**
para aislar el modelo de dipolo (`validation/ir_tb_vs_gpaw.json`, referencias
en `parameters/references/gpaw_chn_ir.json`):

| | solo cargas | cargas + dipolos intraatómicos |
|---|---|---|
| log₁₀(TB/GPAW), modos > 5 % del más intenso: media | −0,38 | −0,24 |
| mediana de \|log₁₀\| (factor típico) | 0,41 (×2,6) | 0,32 (×2,1) |

Con solo las cargas los **momentos dipolares** estáticos salen cerca del
experimento (HCN 2,93 frente a 2,98 D; CH₃CN 3,24 frente a 3,92 D), pero las
tensiones C–H de aromáticos y alquenos salen el doble de intensas; con los
dipolos intraatómicos esas mejoran y el dipolo estático queda alto (el término
de estado fundamental de los pares solitarios no se calibró). Un peso
intermedio no mejora (barrido de λ en μ = ΣQR − λΣp). Queda λ = 1 por defecto.

Alcance: el IR es **semicuantitativo**: los patrones y los modos intensos
aparecen donde deben, con errores típicos de un factor ~2 por modo y algunos
fallos claros (las tensiones C–H de la metilamina salen 10–50 veces débiles).
Solo sistemas finitos (sin polarización de Berry).

## Nanotubos prístinos: validación del modelo de Xu

`python -m tbkit.recipes.cnt_validation salida.json` relaja cada tubo (periodo
axial por barrido de energía), calcula sus modos en Γ y los identifica por
simetría (invariantes bajo la rotación C_g del tubo) y por carácter
(`modes`): la RBM por su solapamiento con la respiración pura, G⁺ y G⁻ entre
los modos ópticos (cada átomo contra sus vecinos) por su dirección. Resultado
con `xu_carbon`, sin ajustar nada (`validation/cnt_xu_carbon.json`):

| tubo | tipo | d (nm) | RBM TB | 227/d | error | 248/d | error | G⁺ | G⁻ (dir.) | ΔG TB | ΔG ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (8,0) | semic. | 0.636 | 349 | 357 | -2.1 % | 390 | -10.4 % | 1668 | 1623 (circ.) | 45 | 118 |
| (10,0) | semic. | 0.790 | 282 | 287 | -1.8 % | 314 | -10.1 % | 1669 | 1643 (circ.) | 25 | 76 |
| (12,0) | metálico | 0.945 | 235 | 240 | -2.3 % | 262 | -10.6 % | 1643 | 1626 (circ.) | 17 | 89 |
| (14,0) | semic. | 1.100 | 201 | 206 | -2.6 % | 225 | -10.9 % | 1687 | 1665 (circ.) | 21 | 39 |
| (17,0) | semic. | 1.334 | 165 | 170 | -3.3 % | 186 | -11.5 % | 1689 | 1671 (circ.) | 18 | 27 |
| (5,5) | metálico | 0.685 | 327 | 332 | -1.3 % | 362 | -9.7 % | 1655 | 1587 (axia.) | 67 | 170 |
| (6,6) | metálico | 0.819 | 271 | 277 | -2.4 % | 303 | -10.6 % | 1672 | 1621 (axia.) | 51 | 119 |
| (8,8) | metálico | 1.088 | 201 | 209 | -3.6 % | 228 | -11.8 % | 1684 | 1652 (axia.) | 32 | 67 |
| (10,10) | metálico | 1.358 | 160 | 167 | -4.0 % | 183 | -12.1 % | 1689 | 1665 (axia.) | 23 | 43 |
| (8,2) | metálico | 0.725 | 306 | 313 | -2.1 % | 342 | -10.4 % | 1652 | 1598 (axia.) | 54 | 151 |
| (6,3) | metálico | 0.630 | 352 | 361 | -2.3 % | 394 | -10.6 % | 1661 | 1623 (circ.) | 38 | 201 |

- **RBM**: entre −1,3 y −4,0 % de 227/d (Araujo et al., PRB 77, 241403, 2008,
  el límite sin entorno) en los once tubos; ~−11 % frente a 248/d (Jorio et
  al., PRL 86, 1118, 2001, tubos sobre Si/SiO₂, que el entorno endurece).
- **G**: todo el espectro va ~70–90 cm⁻¹ alto (la G del grafeno con Xu es
  1667 cm⁻¹ frente a 1582). El desdoblamiento ΔG = G⁺ − G⁻ sale 2–5 veces más
  pequeño que C/d² (Jorio et al., PRB 65, 155412, 2002). La dirección de G⁻
  es la medida en semiconductores (circunferencial) y en los armchair
  metálicos (axial), pero en metálicos falta el ablandamiento del modo LO por
  la anomalía de Kohn, que Xu no reproduce: en esos tubos la G⁻ calculada no
  es comparable con la medida (la línea BWF).

Conclusión: la RBM y su dependencia con el diámetro son fiables; la G sirve
para tendencias y asignaciones, no para posiciones absolutas ni para el
desdoblamiento.

## Grafeno: G, 2D y 2D′ por doble resonancia

`tbkit graphene-raman --laser 1.96 2.41 2.80` (o `graphene.graphene_raman`):
electrones del modelo π (t y β de su archivo), acoplamiento electrón–fonón
analítico (comprobado contra una suma explícita de ΔH), fonones en toda la
zona de Brillouin; G de tercer orden y 2D/2D′ de cuarto orden con los
procesos ee, hh, eh y he (Thomsen y Reich 2000; Venezuela, Lazzeri y Mauri
2011). Por defecto, fonones de GPAW (PBE, supercelda 6×6, regla acústica
impuesta; `parameters/references/gpaw_graphene_phonons.json`); también los de
Xu o cualquier archivo de constantes de fuerza (`validation/graphene_2d_gpaw.json`):

| láser (eV) | G (cm⁻¹) | 2D | 2D′ | I(2D)/I(G) |
|---|---|---|---|---|
| 1,96 | 1605 | 2819 | 3287 | 9,6 |
| 2,41 | 1605 | 2872 | 3313 | 12,0 |
| 2,80 | 1605 | 2913 | 3327 | 13,1 |

- **Dispersión de la 2D: 112 cm⁻¹/eV** (medida: ~100). El mecanismo es el
  correcto: la 2D sale de la rama TO cerca de K, en |q − K| ≈ E_L/ħv_F.
- **Posiciones absolutas altas** (2D a 2,41 eV: 2872 frente a ~2680 medidos;
  G 1605 frente a 1582): son las de los fonones PBE, que subestiman la
  anomalía de Kohn del TO en K. Con los fonones de Xu la 2D sale aún más alta
  (~2967 cm⁻¹).
- I(2D)/I(G) ~10 es del orden del grafeno suspendido; el valor depende de γ
  y del sustrato (no incluido).
- No incluye defectos (no hay banda D), excitones ni la renormalización
  electrón–electrón de v_F.

## Fonones de Quantum ESPRESSO con intensidades de tbkit

`tbkit raman estructura.xyz --model chn --modes dynmat.out` usa las
frecuencias y los modos de QE (más fiables que los del modelo) y calcula las
intensidades con la α del modelo. Lee los archivos de modos de `dynmat.x`
(`filout`, `fileig`) y `matdyn.x` (`flvec`, `fleig`) en q = 0; distingue solo
desplazamientos normalizados de autovectores (`--modes-kind auto`), fija la fase
global de cada modo y, en los conjuntos degenerados que QE imprime como
combinaciones complejas, toma una base real del mismo subespacio (la actividad
sumada no depende de ella). El modelo no necesita parte repulsiva para esto.
Lo que el usuario debe garantizar: la estructura es la del cálculo de QE, en el
mismo orden de átomos, y con las mismas masas (las de ASE si no se cambian).
El lector está comprobado con archivos escritos en el formato documentado; aún
no contra una instalación real de QE.

## Lo que no hace, y dónde está la trampa

- **Raman solo no resonante y con gap**: metales, semimetales (el grafeno) y
  sistemas de capa abierta se rechazan, y también un láser a menos del 20 % del
  gap. El Raman del grafeno es siempre resonante; la banda 2D (segundo orden,
  doble resonancia) es el paso siguiente de la fase E, no este.

- **Energías y fuerzas solo con modelos que tienen parte repulsiva**: el de Xu
  (carbono puro), `xu_chn` (C, H, N) y los `.skf` con su spline. El modelo π no
  la tiene y lo dice.
- **Campo medio no es correlación**: un copo con M = 0 y momentos locales es, en
  realidad, un singlete correlacionado; los momentos son el parámetro de orden
  de la aproximación.
- **SCC solo en sistemas finitos**: un sistema periódico necesitaría una suma de
  Ewald; se rechaza en vez de aproximarlo.
- **Parámetros π de Hückel para heteroátomos**: buenos para tendencias, no para
  niveles cuantitativos. Ajústalos a tu DFT con `fit`.
- **Convención `sp` de los `.skf` heteronucleares**: Hsp0 de `A-B.skf` es
  ⟨s de A|H|p de B⟩, comprobado en el lector de DFTB+ (`getFullTable` en
  `parser.F90`). Un conjunto `.skf` se marca `scc`: se usa con cargas
  autoconsistentes por defecto.
- **Ajuste alineado en mitad del gap**: las energías absolutas de DFT no tienen
  cero común con el modelo, así que las energías on-site quedan definidas salvo
  un desplazamiento y puede haber soluciones espejo; revisa el resultado.

Sin interfaz gráfica todavía: el toolkit (PySide6 + pyvista es el candidato) se
decide al empezarla (docs/PLAN_SUITE.md, decisión 3).
