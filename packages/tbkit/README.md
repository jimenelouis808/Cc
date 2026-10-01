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

## La ventana: `tbkit-gui`

```bash
uv sync --all-packages --extra dev --extra gui     # PySide6 + pyvista (opcionales)
uv run tbkit-gui [estructura.xyz]
```

Izquierda, la estructura (cualquier archivo que lea ASE) y el modelo (sugiere
el conjunto mínimo que cubre sus elementos; carga, SCC, kT). Centro, la vista
3D (átomos, enlaces, colores por carga o momento, isosuperficies, modos
animados). Derecha, una pestaña por capacidad:

| Pestaña | Qué hace |
|---|---|
| Electrónica | estado fundamental (SCC si el modelo lo pide), gap, niveles, DOS/PDOS por elemento, átomo u orbital, bandas (periódicos), cargas |
| Orbitales | isosuperficie de cualquier nivel, isovalor ajustable |
| Magnetismo | Hubbard de campo medio: momentos sobre la estructura, DOS de espín, m(E), comparación de puntos de partida, barridos en campo (T) y dopaje |
| Geometría y modos | relajación con las fuerzas del modelo (con vuelta a la original), modos en Γ con participación, DOS vibracional, animación y flechas, `modes.npz` |
| Espectros | Raman (láser, T), Raman resonante (láseres, η, perfiles de excitación), IR en km/mol; fonones de la pestaña anterior, del modelo o de un archivo de QE; CSV |
| Grafeno | G, 2D y 2D′ por láser con fonones GPAW, Xu o de archivo; dispersión de la 2D |

Archivo → «Guardar registro del último cálculo» escribe el registro
reproducible de `tbkit.record` (versión, commit, parámetros completos,
ajustes, resultados); «Abrir registro» recupera estructura y modelo. Desde
carbonforge, «Abrir en tbkit» (barra inferior) lanza esta ventana con la
estructura actual, por archivo: los paquetes no se importan.

Los cálculos corren en otro hilo; «Cancelar» descarta el resultado (un hilo de
Python no se puede matar). En Linux, Qt necesita libEGL, libxkbcommon y las
libxcb-* del sistema.

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
| Oxígeno | `parameters/xu_chno.json`, `recipes/xu_chno.py` | `xu_chn` más O (hidroxilo, epóxido, carbonilo, carboxilo, éter, furano, nitro), C–O, O–H, N–O, O–O; corrección de ángulos agudos para anillos de tres miembros; α extra del O ajustada a GPAW |
| B, S, P, Se | `parameters/xu_chnob.json`, `xu_chnos.json`, `xu_chnop.json`, `xu_chnose.json`; `recipes/xu_bsp.py` | Un elemento más sobre `xu_chno` fijo: el resto da exactamente lo de `xu_chno`; todos los pares del elemento con H, C, N, O y consigo mismo |
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

Límites que el archivo declara en `validity`: ajustado en moléculas de capa
cerrada (en cristales, lo medido frente a GPAW: ver «Cristales frente a GPAW»); energías comparables solo entre geometrías de la misma
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

## Oxígeno: `xu_chno` (carbono funcionalizado)

`--model chno` (o `load_parameters("xu_chno")`) extiende `xu_chn` con oxígeno
por la misma receta (`tbkit.recipes.xu_family`): C–C de Xu intacto; energías
de sitio del O, saltos C–O, O–H, N–O y O–O (dos O de un carboxilo o de un nitro
están a 2,2 Å, dentro del alcance), repulsiones de par, y H y N reajustados
desde sus valores de `xu_chn`. U_O = 13,48 eV (= DFTB mio) y d_O = 0,355 Å se
calculan con el átomo de GPAW, no se ajustan.

Referencias (`chno_references`, 127 estructuras GPAW): agua, metanol,
formaldehído, acetaldehído, ácidos fórmico y acético, CO, CO₂, dimetil éter,
oxirano, furano, H₂O₂, nitrometano, acetamida, ciclopropano; de prueba etanol,
acetona, formiato de metilo, glioxal y dos motivos de óxido de grafeno sobre
coroneno (epóxido basal y 1,4-diol).

**Anillos de tres miembros.** El C–C de Xu no cierra un epóxido: sin
corrección, el C–C bajo el O del epóxido sobre coroneno se abría a 2,16 Å
(GPAW 1,60). `repulsive.AcuteAngleTerm` añade un término de tres cuerpos que es
exactamente cero para ángulos ≥ 80° (grafeno, diamante, nanotubos, fullerenos y
aromáticos conservan los resultados de Xu; test) y se ajusta con la repulsión.
Dos lecciones del ajuste: el biciclobutano, extremadamente tenso, se llevaba la
mitad del error de fuerzas y queda fuera (`CHNO.held_out`, registrado en el
archivo); y las distorsiones aleatorias solo muestrean el mínimo, así que sin
**barridos de apertura del anillo** (C–C del anillo a 1,65, 1,80 y 1,95 Å en
ciclopropano, oxirano y aziridina) el ajuste dejaba los anillos sin barrera
(epóxido +0,57 Å).

| | resultado |
|---|---|
| ajuste conjunto | niveles 1,02 eV, fuerzas 0,79 eV/Å, energías 0,23 eV |
| epóxido basal sobre coroneno | C–C 0,03 Å, C–O 0,10 Å |
| carboxilo, carbonilo, éter, furano, nitro | enlaces a 0,01–0,03 Å; C–C/C–O simples junto al heteroátomo ~0,08 Å |
| 1,4-diol sobre coroneno | C–O 0,07 Å |
| oxirano / aziridina | C–C del anillo 0,12 / 0,11 Å |
| ciclopropano | C–C +0,19 Å, mínimo poco profundo (fuera del objetivo) |
| biciclobutano (fuera del ajuste) | se abre |
| α extra del O | 0,564 Å³ (`xu_chn_alpha --fit O`, H, C, N de `xu_chn`) |
| α media, entrenamiento / prueba (vs GPAW FD) | −9 a +10 % / −3 a +3 % |
| frecuencias frente a GPAW (hessianas en el ajuste) | RMS 38-180 cm⁻¹, media 126 (antes 189; metanol 491 → 120) |

Receta completa:

```bash
python -m tbkit.recipes.chno_references gpaw_chno.json
python -m tbkit.recipes.xu_chno gpaw_chn.json gpaw_chno.json xu_chno_fit.json
python -m tbkit.recipes.chn_polarizability gpaw_chno_alpha.json --set chno
python -m tbkit.recipes.xu_chn_alpha gpaw_chno_alpha.json gpaw_chn_alpha.json \
    xu_chno_fit.json xu_chno.json --fit O --fixed-from xu_chn.json \
    --geometries gpaw_chn.json gpaw_chno.json
```

## Boro, azufre y fósforo: `xu_chnob`, `xu_chnos`, `xu_chnop`

`--model chnob | chnos | chnop`: cada uno es `xu_chno` **fijo** más un
elemento (`XuFamily.base`). Una molécula sin B, S o P da exactamente el
resultado de `xu_chno` (test), y el desplazamiento de niveles es el suyo. Se
ajustan solo los 28 parámetros del elemento nuevo (energías s y p, saltos y
repulsiones con H, C, N, O y consigo mismo), con las moléculas que lo
contienen. Todos esos pares son obligatorios: un par sin ley de salto daría
cero en silencio, y la familia se niega a construirse sin él. U y ⟨ns|r|np⟩
salen del átomo de GPAW, como los demás (B 8,06, S 8,95, P 7,87 eV, iguales a
DFTB 3ob/matsci; d = 0,62, 0,55, 0,62 Å). Co-dopados B-N, N-S y N-P sí; B con S
o P en la misma estructura, no.

Referencias (`bsp_references`, geometrías de partida de RDKit guardadas en
`recipes/data/bsp_start.extxyz`, todas relajadas con GPAW):

| | entrenamiento | prueba | ajuste (niveles / fuerzas / energías) | enlaces relajados frente a GPAW |
|---|---|---|---|---|
| B | BH₃, BMe₃, B(OH)₃, B(OMe)₃, MeB(OH)₂, H₃B·NH₃, borazina, vinilborano, B₂(OH)₄ | PhB(OH)₂, BEt₃, par B-N en el anillo central del coroneno | 0,54 eV / 0,38 eV/Å / 0,11 eV | ≤ 0,05 Å; B-N en coroneno 0,037 Å |
| S | H₂S, CH₃SH, Me₂S, Me₂S₂, tiofeno, H₂CS, CS₂, OCS, DMSO, Me₂SO₂, SO₂, CH₃SO₃H, CH₃SO₂NH₂ | EtSH, PhSH, PhSO₃H, tiol en el borde del coroneno | 0,67 eV / 0,43 eV/Å / 0,11 eV | S=O, S-S, S-H, S-N y C-S aromático < 0,04 Å; C(sp³)-S simple 0,06-0,09 Å |
| P | PH₃, CH₃PH₂, PMe₃, OPMe₃, H₃PO₄, MePO₃H₂, PO(OMe)₃, fosfinina, P₂H₄, H₂PNH₂ | PhPH₂, PEt₃, ácido fosfónico en el borde del coroneno | 0,73 eV / 0,46 eV/Å / 0,12 eV | P-O, P=O, P-P, P-N, P-H y C-P aromático < 0,025 Å; C-P de P(III) 0,03-0,05 Å, de P(V) 0,07-0,14 Å (0,10 en el coroneno) |

Frecuencias frente a GPAW (hessianas de GPAW en el ajuste, colas de los pares
con H antes de los segundos vecinos): RMS por molécula B 61-141, S 54-142,
P 46-149 cm⁻¹. **Ésteres alquílicos de borato y fosfato (B/P–O–CH₃) no son
válidos**: los H de metilo colapsan sobre el O vecino (atracción de las cargas
SCC sin repulsión O···H a 1,6-2,4 Å; defecto de la base `xu_chno`, pendiente de
corregir añadiendo esas geometrías a los datos).

```bash
python -m tbkit.recipes.bsp_references S gpaw_s.json
python -m tbkit.recipes.xu_bsp S gpaw_s.json xu_chnos.json
```

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

## Factores de escala de frecuencias

Los conjuntos subestiman las frecuencias de GPAW de forma sistemática, ~3 %.
`recipes/frequency_scaling.py` ajusta un factor por conjunto (mínimos
cuadrados de Scott–Radom, como Witek y Morokuma 2004 para SCC-DFTB) sobre los
modos internos de sus moléculas de validación, frente a GPAW PBE (no frente al
experimento), y lo guarda en `frequency_scale` del archivo. `tasks.phonons`
devuelve `frequencies_scaled_cm1` junto a las crudas; el modelo no cambia. En la ventana,
la casilla «Escalar frecuencias» (pestaña Espectros) lo aplica a Raman, Raman
resonante e IR antes de ensanchar el espectro (los factores de sección eficaz
dependen de ω); nunca a modos importados de Quantum ESPRESSO.

| conjunto | λ | RMS antes → después (cm⁻¹) | dejando fuera cada molécula |
|---|---|---|---|
| xu_chn | 1,003 | 68 → 67 | 69 |
| xu_chno | 1,037 | 120 → 99 | 100 |
| xu_chnob | 1,030 | 97 → 82 | 84 |
| xu_chnos | 1,031 | 91 → 75 | 76 |
| xu_chnop | 1,025 | 82 → 71 | 71 |
| xu_chnose | 1,034 | 84 → 60 | 61 |

Un solo factor transfiere bien (dejar fuera la molécula apenas cambia el
RMS). Dos factores (encima y debajo de 2000 cm⁻¹) solo ayudaban a xu_chno
(99 → 88). `xu_chn` casi no tiene sesgo, y `xu_carbon` no lleva factor: el
carbono puro se valida contra experimento (RBM, G, C60) sin escalar.

## Cristales frente a GPAW: cuánto se extrapola

Los conjuntos `xu_ch*` se ajustaron en moléculas; en un cristal son una
extrapolación. `recipes/crystal_validation.py` la mide en 13 cristales, cada
uno con el conjunto mínimo que cubre sus elementos, con GPAW (PBE, LCAO dzp,
h 0,2 Å, espín apareado, Fermi–Dirac 0,1 eV) y TB en la misma celda, la misma
malla k y el mismo smearing: relajación con GPAW, tres distorsiones aleatorias
(σ = 0,05 Å) y ±2 % de deformación (con la rejilla de la celda relajada: GPAW
redondea la rejilla a múltiplos de 4 y una deformación la cambiaba, con saltos
de ~1 eV). Referencias en `parameters/references/gpaw_crystals.json`,
resultados en `validation/crystals_tb_vs_gpaw.json`. La escala es el error
del mismo conjunto en sus propias moléculas, medido igual (mediana por
estructura): en el mínimo de GPAW 0,26-0,54 eV/Å según el conjunto, casi todo
por el desplazamiento entre los mínimos de TB y GPAW.

| cristal | conjunto | F en el mínimo (eV/Å) | × moléculas | F distorsiones | enlace del dopante TB − GPAW |
|---|---|---|---|---|---|
| grafeno | xu_chno | 0,01 | 0,02 | 0,20 | — |
| grafeno + N grafítico | xu_chn | 0,18 | 0,49 | 0,27 | C–N −0,002 Å |
| grafeno + N3V piridínico | xu_chn | 0,42 | 1,18 | 0,47 | C–N +0,010 Å |
| grafeno + B | xu_chnob | 0,35 | 1,38 | 0,45 | B–C +0,027 Å |
| grafeno + S | xu_chnos | 0,20 | 0,50 | 0,25 | S–C +0,03 Å, altura +0,06 Å |
| grafeno + P | xu_chnop | 0,14 | 0,34 | 0,23 | P–C 0,00 Å, altura −0,12 Å |
| grafeno + Se | xu_chnose | 0,28 | 0,94 | 0,30 | Se–C +0,06 Å, altura +0,07 Å |
| grafeno + epóxido | xu_chno | 0,61 | 1,12 | 0,60 | C–O +0,09 Å |
| grafeno + OH | xu_chno | 0,20 | 0,36 | 0,26 | C–O −0,05 Å; el H gira 0,24 Å |
| grafano | xu_chn | 0,18 | 0,49 | 0,22 | — |
| h-BN | xu_chnob | 0,10 | 0,39 | 0,38 | red +1,1 %, 10-15 % más blando |
| CNT (8,0) + N | xu_chn | 0,27 | 0,77 | 0,35 | C–N ≤ 0,017 Å |
| diamante + N | xu_chn | 0,49 | 1,37 | 0,68 | C–N −0,08 Å |

- **La extrapolación aguanta**: el error de fuerzas en cristales va de 0,3 a
  1,4 veces el molecular, y cada desviación de enlace está dentro de lo que el
  conjunto ya declaraba para sus moléculas (epóxido C–O 0,10 Å, C–N simple
  ~0,08 Å, enlaces de Se < 0,08 Å, de B < 0,03 Å).
- **Lo nuevo en cristales**: la altura de S, P y Se sobre la hoja (0,06-0,12 Å;
  el P queda bajo con su enlace exacto, es decir, ángulos C–P–C abiertos) y la
  red, un 0,5-1 % más corta que la de GPAW-dzp en grafeno dopado (grafano
  igual; h-BN al revés, +1,1 %, y más blando).
- **Lo que se descubrió de paso en moléculas**: el C–O simple de alcoholes sale
  0,06-0,08 Å corto (metanol −0,08 Å); ahora figura en `validity` de xu_chno.
- Todo con espín apareado en ambos códigos: el N del diamante (centro P1) y el
  OH sobre grafeno son de capa abierta en la realidad.

Cada `validity` lo dice (campo `crystal_notes` de la receta; un test compara
cada archivo con su receta). Reproducir:

```bash
tbkit/recipes/run_crystals.sh out/crystals        # GPAW, reanudable; horas en serie
python -m tbkit.recipes.crystal_validation collect out/crystals gpaw_crystals.json
python -m tbkit.recipes.crystal_validation compare gpaw_crystals.json out.json
```

## Lo que no hace, y dónde está la trampa

- **Raman solo no resonante y con gap**: metales, semimetales (el grafeno) y
  sistemas de capa abierta se rechazan, y también un láser a menos del 20 % del
  gap. El Raman del grafeno es siempre resonante; la banda 2D (segundo orden,
  doble resonancia) es el paso siguiente de la fase E, no este.

- **Energías y fuerzas solo con modelos que tienen parte repulsiva**: el de Xu
  (carbono puro), `xu_chn` (C, H, N), `xu_chno` (C, H, N, O), `xu_chnob/s/p` y los `.skf` con su spline. El modelo π no
  la tiene y lo dice.
- **Campo medio no es correlación**: un copo con M = 0 y momentos locales es, en
  realidad, un singlete correlacionado; los momentos son el parámetro de orden
  de la aproximación.
- **Cristales = extrapolación medida**: el SCC periódico va por sumas de Ewald,
  pero los conjuntos se ajustaron en moléculas; lo que eso cuesta en 13
  cristales está medido (sección anterior). Un cristal fuera de esa lista
  (otras fases, dopajes altos, capa abierta) no está comprobado.
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
