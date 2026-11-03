# Validación de tbkit

Generado por `python -m tbkit.recipes.validation_report` a partir de los archivos de `validation/` y `tbkit/parameters/`. No editar a mano: un test comprueba que coincide con los datos.

## Conjuntos de parámetros

Lo que cada archivo declara (`system`, `validity`) y su factor de escala de frecuencias frente a GPAW (`frequency_scale`, receta `frequency_scaling`). Fuente: `tbkit/parameters/*.json`.

| conjunto | SCC | λ frecuencias | RMS (cm⁻¹) sin → con λ | modos / moléculas |
|---|---|---|---|---|
| `pi_huckel` | no | — | — | — |
| `xu_carbon` | no | — | — | — |
| `tang_carbon` | no | — | — | — |
| `xu_chn` | sí | 1,0033 | 67,5 → 67,2 | 76 / 5 |
| `xu_chno` | sí | 1,0365 | 115,8 → 95,0 | 162 / 14 |
| `xu_chnob` | sí | 1,0299 | 96,4 → 81,5 | 201 / 9 |
| `xu_chnos` | sí | 1,0305 | 91,2 → 75,4 | 184 / 13 |
| `xu_chnop` | sí | 1,0247 | 82,0 → 70,8 | 218 / 10 |
| `xu_chnose` | sí | 1,0337 | 83,0 → 60,0 | 150 / 12 |
| `c_bandas` | no | — | — | — |
| `c_capas` | no | — | — | — |
| `xu_capas` | no | — | — | — |

Validez declarada por cada archivo:

- **`pi_huckel`** (carbono sp2: grafeno, nanotubos, cintas, copos, fullerenos; N, B y O sustitucionales): solo estados π cerca del nivel de Fermi (±3 eV); enlaces C-C de 1.35 a 1.50 Å; sin energías totales
- **`xu_carbon`** (carbono: diamante, grafito, dímero C2, cadenas lineales y fases de alta coordinación): enlaces C-C de 1.2 a 2.6 Å; C puro (sin H ni heteroátomos); base ortogonal s+p
- **`tang_carbon`** (carbono puro: cadena, grafito, diamante y fases metálicas (sc, bcc, fcc) del ajuste original a LDA+GGA; aquí además validado frente a GPAW PBE en amorfos y defectos): solo carbono, sin SCC. Forma funcional y parte electrónica de Tang et al. 1996; repulsión, escala de Δe y corte ajustados a GPAW PBE. Error de fuerzas (RMS, eV/Å) frente a GPAW: diamond 0.18, graphene 0.14, chain 0.22, c60 0.42, vacancy 0.65, stone_wales 0.38, amorphous_2.0 0.87, amorphous_2.6 0.94, amorphous_3.2 0.98 (amorphous_3.2 y stone_wales fuera del ajuste). Energías comparables entre estructuras de carbono (una sola referencia por átomo). Diamante: mínimo en a = 3,515 Å (GPAW-PBE ~3,57; -1,5 %), B = 478 GPa. Energías entre fases muy distintas poco fiables (diamante 0,37 eV/átomo frente al resto); fuerzas y geometrías mejores que Xu en 8 de 9 entornos (grafeno 0,14 frente a 0,12 eV/Å).
- **`xu_chn`** (moléculas C/H/N de capa cerrada: hidrocarburos saturados, insaturados y aromáticos; aminas, iminas, nitrilos, N piridínico y pirrólico): ajustado en moléculas de capa cerrada; cristales (SCC periódico): extrapolación comprobada contra GPAW (validation/crystals_tb_vs_gpaw.json): N grafítico y piridínico (N3V) en grafeno, grafano y CNT (8,0) con N: error de fuerzas en el mínimo de GPAW 0,5-1,2 veces el molecular, C-N a menos de 0,02 Å; N sustitucional en diamante: C-N -0,08 Å (1,52 frente a 1,60, como los C-N simples de las moléculas) y fuerzas 1,4-1,7 veces el molecular (ambos con espín apareado: el centro P1 real es de capa abierta); la red de TB sale 0,5-1 % más corta que la de GPAW-dzp (el grafano, igual); enlaces dentro de lo muestreado (C-H 0.97-1.26 Å, C-N 1.11-1.57 Å, H-N 0.91-1.17 Å, N-N 1.07-1.52 Å); energías relativas solo entre estructuras de la misma composición y los mismos tipos de enlace (no se ajustaron energías de atomización; isómeros con enlaces distintos fallan: dimetil éter frente a etanol -1,44 eV con xu_chno, GPAW +0,38); C-C como en xu_carbon; falla en anillos tensos (aziridina); los C-C y C-N simples junto a un heteroátomo se desvían ~0.08 Å
- **`xu_chno`** (moléculas C/H/N/O de capa cerrada: las de xu_chn más alcoholes, éteres, epóxidos, aldehídos, cetonas, ácidos carboxílicos, ésteres, amidas, furano, nitro; motivos de óxido de grafeno (epóxido e hidroxilos basales)): ajustado en moléculas de capa cerrada; cristales (SCC periódico): extrapolación comprobada contra GPAW (validation/crystals_tb_vs_gpaw.json): epóxido y OH sobre grafeno: fuerzas 0,4-1,1 veces el error molecular; heredan los errores de las moléculas (epóxido C-O +0,09 Å; OH C-O -0,05 Å, el C bajo el OH 0,14 Å menos piramidal y el H gira, 0,24 Å); el C-O simple de los alcoholes sale 0,06-0,08 Å corto también en moléculas (metanol -0,08 Å); enlaces dentro de lo muestreado (C-H 0.97-1.26 Å, C-N 1.11-1.61 Å, C-O 1.10-1.58 Å, H-N 0.91-1.17 Å, H-O 0.89-1.09 Å, N-N 1.07-1.52 Å, N-O 1.14-1.31 Å, O-O 1.35-1.54 Å); energías relativas solo entre estructuras de la misma composición y los mismos tipos de enlace (no se ajustaron energías de atomización; isómeros con enlaces distintos fallan: dimetil éter frente a etanol -1,44 eV con xu_chno, GPAW +0,38); C-C como en xu_carbon; anillos de tres miembros con la corrección de ángulos agudos ajustada a barridos de apertura del anillo: epóxido basal sobre coroneno C-C 0.03 Å, C-O 0.10 Å; oxirano y aziridina C-C ~0.12 Å; ciclopropano C-C +0.19 Å con un mínimo poco profundo; el biciclobutano (excluido del ajuste) se abre; los C-C y C-N simples junto a un heteroátomo se desvían ~0.08 Å; frecuencias frente a GPAW (hessianas en el ajuste): RMS 38-180 cm⁻¹ por molécula, media 126 (agua 38, metanol 120, ácido fórmico 111)
- **`xu_chnob`** (xu_chno más boro: boranos, ésteres y ácidos bóricos y borónicos, amino-borano, borazina, B-N en grafeno): ajustado en moléculas de capa cerrada; cristales (SCC periódico): extrapolación comprobada contra GPAW (validation/crystals_tb_vs_gpaw.json): B grafítico en grafeno: fuerzas 1,4 veces el error molecular, B-C +0,027 Å; h-BN: red 1,1 % larga (2,53 frente a 2,50 Å) y 10-15 % más blanda en distorsiones; enlaces dentro de lo muestreado (B-B 1.64-1.81 Å, B-C 1.47-1.70 Å, B-H 1.10-1.37 Å, B-N 1.35-1.72 Å, B-O 1.30-1.46 Å; el resto, como xu_chno); energías relativas solo entre estructuras de la misma composición y los mismos tipos de enlace (no se ajustaron energías de atomización; isómeros con enlaces distintos fallan: dimetil éter frente a etanol -1,44 eV con xu_chno, GPAW +0,38); C-C como en xu_carbon; B junto con S o P en la misma estructura no está cubierto; enlaces de B a menos de 0.03 Å de GPAW; ésteres B(OCH3)n corregidos con active learning (antes colapsaban; RMS 949 -> 95 cm⁻¹), aunque al relajar giran sus metilos (desplazamiento ~1 Å, sin colapso); ácidos borónicos arílicos (PhB(OH)2): corregidos con el contacto H···H (torsión del B(OH)2 336° frente a 337° de GPAW; antes quedaba plano, 0,6 Å)
- **`xu_chnos`** (xu_chno más azufre: tioles, sulfuros, disulfuros, tiofeno, sulfóxidos, sulfonas, ácidos sulfónicos, sulfonamidas): ajustado en moléculas de capa cerrada; cristales (SCC periódico): extrapolación comprobada contra GPAW (validation/crystals_tb_vs_gpaw.json): S en grafeno: fuerzas 0,5-0,6 veces el error molecular, S-C +0,03 Å, S 0,06 Å más alto sobre la hoja; enlaces dentro de lo muestreado (C-S 1.40-1.94 Å, H-S 1.24-1.45 Å, N-S 1.62-1.79 Å, O-S 1.34-1.75 Å, S-S 1.96-2.17 Å; el resto, como xu_chno); energías relativas solo entre estructuras de la misma composición y los mismos tipos de enlace (no se ajustaron energías de atomización; isómeros con enlaces distintos fallan: dimetil éter frente a etanol -1,44 eV con xu_chno, GPAW +0,38); C-C como en xu_carbon; S junto con B o P en la misma estructura no está cubierto; los C-S simples con carbono sp3 se desvían 0.06-0.09 Å (S=O, S-S, S-H, S-N y C-S aromático a menos de 0.04 Å)
- **`xu_chnop`** (xu_chno más fósforo: fosfinas, óxidos de fosfina, ácidos fosfórico y fosfónicos, fosfatos, fosfinina): ajustado en moléculas de capa cerrada; cristales (SCC periódico): extrapolación comprobada contra GPAW (validation/crystals_tb_vs_gpaw.json): P en grafeno: fuerzas 0,3-0,6 veces el error molecular, P-C exacto (0,001 Å) pero P 0,12 Å más bajo sobre la hoja: ángulos C-P-C demasiado abiertos; enlaces dentro de lo muestreado (C-P 1.67-2.15 Å, H-P 1.30-1.57 Å, N-P 1.66-1.84 Å, O-P 1.39-1.74 Å, P-P 2.13-2.36 Å; el resto, como xu_chno); energías relativas solo entre estructuras de la misma composición y los mismos tipos de enlace (no se ajustaron energías de atomización; isómeros con enlaces distintos fallan: dimetil éter frente a etanol -1,44 eV con xu_chno, GPAW +0,38); C-C como en xu_carbon; P junto con B o S en la misma estructura no está cubierto; el C-P de P(V) (ácidos fosfónicos, óxidos de fosfina) sale 0.05-0.10 Å largo, el de P(III) 0.04-0.06 Å; P-O, P=O, P-P, P-N, P-H y C-P aromático a menos de 0.025 Å; ésteres P(OCH3)n corregidos con active learning (antes colapsaban); ácidos con P-OH (fosfórico, fosfónicos): el OH gira hacia el otro O al relajar, 0,4-0,9 Å; causa: la base mínima (GPAW con base sz también hunde el giro del OH: -0,35 eV a 180° frente a -0,23 con dzp; en TB es todo electrónico); en MePO3H2 los d del P también importan en DFT; úsense sus frecuencias O-H con cautela
- **`xu_chnose`** (xu_chno más selenio: selenoles, selenuros, diselenuros, selenofeno, selenóxidos, ácidos selenínicos, Se-N): ajustado en moléculas de capa cerrada; cristales (SCC periódico): extrapolación comprobada contra GPAW (validation/crystals_tb_vs_gpaw.json): Se en grafeno: fuerzas ~1 vez el error molecular, Se-C +0,06 Å, Se 0,07 Å más alto sobre la hoja; enlaces dentro de lo muestreado (C-Se 1.61-2.09 Å, H-Se 1.36-1.57 Å, N-Se 1.81-2.00 Å, O-Se 1.57-2.01 Å, Se-Se 2.23-2.47 Å; el resto, como xu_chno); energías relativas solo entre estructuras de la misma composición y los mismos tipos de enlace (no se ajustaron energías de atomización; isómeros con enlaces distintos fallan: dimetil éter frente a etanol -1,44 eV con xu_chno, GPAW +0,38); C-C como en xu_carbon; Se junto con B, S o P en la misma estructura no está cubierto; enlaces de Se a menos de 0.08 Å; ácidos seleníninicos (Se(=O)OH): el OH gira hacia el otro O al relajar, 0,4-0,9 Å; causa: la base mínima (en el giro rígido la repulsión no aporta y la parte electrónica pone 240-300° 0,1 eV bajo el mínimo; GPAW con base mínima sz hace lo mismo, -0,06 eV, y con dzp +0,03); no son los d del Se (GPAW sin ellos no gira); úsense sus frecuencias O-H con cautela
- **`c_bandas`** (carbono sp2: grafeno, nanotubos, coils; estructura electrónica a geometría dada): solo bandas, DOS y respuesta óptica a una geometría dada (de Xu o de DFT): sin repulsión, sin energías ni fuerzas; C-C de 1.37 a 1.48 Å; electrones a ±3 eV del nivel de Fermi (más lejos, las bandas vacías de GPAW tienen estados σ* y casi libres que una base mínima no describe)
- **`c_capas`** (carbono sp2 apilado: multicapas, grafito, tubos de varias paredes, coils; estructura electrónica a geometría dada): solo bandas, DOS y respuesta óptica a una geometría dada (de Xu o de DFT): sin repulsión, sin energías ni fuerzas; C-C de 1.37 a 1.48 Å; electrones a ±3 eV del nivel de Fermi (más lejos, las bandas vacías de GPAW tienen estados σ* y casi libres que una base mínima no describe); entre capas: 3.0 a 3.7 Å ajustado, más allá extrapolado; el acoplamiento lejano no lleva solapamiento
- **`xu_capas`** (carbono sp2/sp3 apilado: multicapas, grafito, tubos de varias paredes, coils): lo de Xu, más: separación entre capas 2.8 a 5.5 Å (curvas AB y AA); la corrugación AB/AA es la de un término de pares; el término de dispersión usa el grafo de enlaces de cada geometría (fijo dentro de una relajación sin rupturas)

## Infrarrojo frente a GPAW (8 moléculas C/H/N, modos de GPAW)

Fuente: `validation/ir_tb_vs_gpaw.json`. log₁₀(TB/GPAW) de las intensidades, modos internos (ω > 100 cm⁻¹: fuera las rotaciones y traslaciones, cuya intensidad en GPAW es ruido numérico) con más del 5 % del más intenso de su molécula.

|  | solo cargas | cargas + dipolos intraatómicos |
|---|---|---|
| media de log₁₀ | -0,38 | -0,24 |
| mediana de \|log₁₀\| | 0,41 | 0,32 |
| modos | 55 | 55 |

## Cristales frente a GPAW (extrapolación de conjuntos ajustados en moléculas)

Fuente: `validation/crystals_tb_vs_gpaw.json` (GPAW PBE lcao dzp, h = 0.2 Å). Errores de fuerza RMS en eV/Å; «razón» = error en el cristal / mediana del error molecular del mismo conjunto medido igual.

| cristal | conjunto | átomos | F mínimo | F distorsionado (mediana) | razón mínimo | razón distorsionado | red TB (%) | red GPAW (%) |
|---|---|---|---|---|---|---|---|---|
| graphene | `xu_chno` | 32 | 0,009 | 0,203 | 0,02 | 0,35 | -0,12 | 0,57 |
| graphene_N | `xu_chn` | 32 | 0,175 | 0,266 | 0,49 | 0,68 | -0,19 | 0,35 |
| graphene_N3V | `xu_chn` | 31 | 0,419 | 0,467 | 1,18 | 1,20 | -0,30 | 0,21 |
| graphene_B | `xu_chnob` | 32 | 0,354 | 0,447 | 1,38 | 1,46 | 0,51 | 1,06 |
| graphene_S | `xu_chnos` | 32 | 0,201 | 0,248 | 0,50 | 0,60 | 0,33 | 0,82 |
| graphene_P | `xu_chnop` | 32 | 0,141 | 0,229 | 0,34 | 0,63 | 0,16 | 0,80 |
| graphene_Se | `xu_chnose` | 32 | 0,277 | 0,302 | 0,94 | 1,04 | 0,36 | 0,85 |
| graphene_epoxide | `xu_chno` | 33 | 0,607 | 0,598 | 1,12 | 1,04 | 0,12 | 0,62 |
| graphene_OH | `xu_chno` | 34 | 0,196 | 0,259 | 0,36 | 0,45 | -0,10 | 0,50 |
| graphane | `xu_chn` | 16 | 0,175 | 0,222 | 0,49 | 0,57 | -0,68 | -0,71 |
| hBN | `xu_chnob` | 18 | 0,100 | 0,375 | 0,39 | 1,23 | 1,13 | 0,06 |
| cnt80_N | `xu_chn` | 64 | 0,273 | 0,348 | 0,77 | 0,89 | -0,58 | 0,46 |
| diamond_N | `xu_chn` | 64 | 0,486 | 0,675 | 1,37 | 1,73 | -0,48 | 0,38 |

Error molecular de referencia (medianas por estructura):

| conjunto | F en el mínimo | F distorsionado |
|---|---|---|
| `xu_chn` | 0,356 | 0,391 |
| `xu_chno` | 0,543 | 0,575 |
| `xu_chnob` | 0,257 | 0,306 |
| `xu_chnop` | 0,413 | 0,361 |
| `xu_chnos` | 0,406 | 0,414 |
| `xu_chnose` | 0,295 | 0,289 |

## Carbono dependiente del entorno (`tang_carbon`) frente a GPAW

Fuente: `tbkit/parameters/tang_carbon.json` (`fit`). Fuera del ajuste: amorphous_3.2, stone_wales.

| estructura | F RMSE (eV/Å) | F RMS GPAW | \|ΔE\| (eV/átomo) |  |
|---|---|---|---|---|
| diamond | 0,18 | 1,85 | 0,369 | ajuste |
| graphene | 0,14 | 2,26 | 0,058 | ajuste |
| chain | 0,22 | 2,18 | 0,081 | ajuste |
| c60 | 0,42 | 4,46 | 0,006 | ajuste |
| vacancy | 0,65 | 5,14 | 0,109 | ajuste |
| stone_wales | 0,38 | 3,96 | 0,100 | fuera |
| amorphous_2.0 | 0,87 | 5,49 | 0,235 | ajuste |
| amorphous_2.6 | 0,94 | 3,37 | 0,190 | ajuste |
| amorphous_3.2 | 0,98 | 3,39 | 0,021 | fuera |

## Bandas de grafeno y nanotubos frente a GPAW (ondas planas)

Fuente: `validation/bands_tb_vs_gpaw.json` (receta `band_validation`). Bands of graphene and nanotubes at one geometry per system (graphene a = 2.467 Å, tubes rolled from it, unrelaxed), every method relative to its own Fermi level or mid-gap. Reference: GPAW PBE plane waves. RMS over the 4 highest valence and 4 lowest conduction bands (index from the HOMO), at the k points where the reference band is within 1 or 2 eV.

Cada celda: RMS de bandas a ±1 eV (eV) / error del gap (eV) / v_F TB/GPAW (solo sin gap). `c_bandas` se ajustó a grafeno (tres parámetros de red), (8,0) y (5,5); 10,0, 9,0, 6,6 quedaron fuera del ajuste.

| sistema | gap PW (eV) | GPAW dzp | π | Xu | Tang | xu_chn | c_bandas |
|---|---|---|---|---|---|---|---|
| graphene | 0,000 | 0,005 / 0,00 / 1,01 | 0,052 / 0,00 / 1,07 | 0,123 / 0,00 / 0,77 | 0,065 / 0,00 / 0,88 | 0,123 / 0,00 / 0,77 | 0,018 / 0,00 / 0,97 |
| 8,0 | 0,553 | 0,015 / 0,01 | 0,357 / 0,71 | 0,166 / 0,42 | 0,188 / 0,42 | 0,166 / 0,42 | 0,170 / 0,28 |
| 10,0 (fuera) | 0,909 | 0,017 / 0,01 | 0,164 / 0,04 | 0,167 / -0,33 | 0,116 / -0,18 | 0,167 / -0,33 | 0,089 / 0,03 |
| 9,0 (fuera) | 0,094 | 0,016 / 0,02 | 0,503 / -0,09 | 0,203 / -0,01 | 0,175 / -0,09 | 0,203 / -0,01 | 0,211 / 0,10 |
| 5,5 | 0,000 | 0,012 / 0,00 / 1,02 | 0,359 / 0,00 / 1,10 | 0,164 / 0,00 / 0,77 | 0,122 / 0,00 / 0,91 | 0,164 / 0,00 / 0,77 | 0,058 / 0,00 / 0,93 |
| 6,6 (fuera) | 0,000 | 0,008 / 0,00 / 1,01 | 0,258 / 0,00 / 1,04 | 0,161 / 0,00 / 0,74 | 0,109 / 0,00 / 0,86 | 0,161 / 0,00 / 0,74 | 0,043 / 0,00 / 0,99 |

Ajuste de `c_bandas` en sus propias ventanas. Error de bandas frente a GPAW PBE (ondas planas) en las ventanas del ajuste: grafeno −9…+3 eV, tubos ±2.5 eV del nivel de referencia.

| sistema |  | RMS Xu (eV) | RMS c_bandas (eV) | gap GPAW | gap c_bandas |
|---|---|---|---|---|---|
| graphene@2.42 | ajuste | 2,854 | 0,111 | 0,000 | 0,000 |
| graphene | ajuste | 2,604 | 0,075 | 0,000 | 0,000 |
| graphene@2.52 | ajuste | 2,328 | 0,104 | 0,000 | 0,000 |
| 8,0 | ajuste | 0,314 | 0,185 | 0,553 | 0,833 |
| 5,5 | ajuste | 0,359 | 0,126 | 0,000 | 0,000 |
| 10,0 | fuera | 0,349 | 0,101 | 0,909 | 0,939 |
| 9,0 | fuera | 0,334 | 0,134 | 0,094 | 0,197 |
| 6,6 | fuera | 0,361 | 0,115 | 0,000 | 0,000 |

## Capas de grafeno: acoplamiento y enlace frente a GPAW

Fuente: `validation/interlayer_tb_vs_gpaw.json` (recetas `interlayer` e `interlayer_fit`). Bandas y enlace entre capas de grafeno frente a GPAW PBE (ondas planas; energías PBE+D3(BJ)). RMS de bandas en ±3 eV de E_F; niveles en K a ±1.5 eV de E_F.

Cada celda: RMS de bandas (eV) / RMS de los niveles en K (eV). `c_capas` y `xu_capas` añaden a `c_bandas` y a Xu (sin cambiarlos) V_ppσ y V_ppπ entre átomos a más de 6 enlaces; el RMS de bandas de Xu está dominado por su error en el plano (v_F 25 % baja), por eso `xu_capas` se ajustó solo a los niveles en K.

| sistema | niveles en K, GPAW (eV) | c_bandas | c_capas | xu_carbon | xu_capas |
|---|---|---|---|---|---|
| AB@3.0 | -0,63, -0,04, -0,04, 0,61 | 0,179 / 0,033 | 0,128 / 0,110 | 0,448 / 0,440 | 0,379 / 0,028 |
| AB@3.35 | -0,35, -0,02, -0,02, 0,33 | 0,227 / 0,242 | 0,133 / 0,027 | 0,388 / 0,242 | 0,356 / 0,018 |
| AB@3.7 | -0,19, -0,01, -0,01, 0,17 | 0,167 / 0,130 | 0,136 / 0,037 | 0,366 / 0,130 | 0,357 / 0,011 |
| AA@3.35 | -0,34, -0,34, 0,34, 0,34 | 0,385 / 0,342 | 0,149 / 0,031 | 0,480 / 0,342 | 0,370 / 0,028 |
| graphite@3.35 | -0,67, -0,05, -0,05, 0,69 | 0,404 / 0,484 | 0,151 / 0,052 | 0,472 / 0,484 | 0,377 / 0,033 |
| ABC@3.35 (fuera) | -0,35, -0,35, -0,01, 0,01, 0,33, 0,33 | 0,252 / 0,278 | 0,133 / 0,024 | 0,389 / 0,278 | 0,354 / 0,004 |
| graphene | — | 0,075 | 0,075 | 2,604 | 2,604 |

Enlace por átomo frente a capas separadas (meV/átomo), `xu_capas` frente a GPAW PBE+D3(BJ); PBE solo no une las capas:

| sistema | PBE | PBE+D3 | xu_capas |
|---|---|---|---|
| AB@2.8 | 53,3 | 7,8 | -0,4 |
| AB@3.0 | 27,0 | -13,3 | -10,7 |
| AB@3.2 | 12,9 | -22,1 | -20,4 |
| AB@3.35 | 7,2 | -24,1 | -23,8 |
| AB@3.5 | 3,6 | -24,0 | -24,3 |
| AB@3.7 | 1,2 | -22,1 | -22,6 |
| AB@4.0 | -0,3 | -18,3 | -18,6 |
| AB@4.5 | -0,7 | -12,3 | -11,4 |
| AB@5.5 | -0,2 | -5,4 | -4,7 |
| AA@2.8 | 92,3 | 47,0 | 52,0 |
| AA@3.0 | 48,5 | 8,3 | 11,5 |
| AA@3.2 | 24,9 | -10,0 | -10,7 |
| AA@3.35 | 14,5 | -16,7 | -18,6 |
| AA@3.5 | 8,0 | -19,7 | -21,3 |
| AA@3.7 | 3,4 | -20,0 | -21,0 |
| AA@4.0 | 0,4 | -17,5 | -17,9 |
| AA@4.5 | -0,5 | -12,1 | -11,5 |
| AA@5.5 | -0,2 | -5,4 | -4,7 |
| graphite@3.2 (fuera) | 26,2 | -50,5 | -47,0 |
| graphite@3.35 (fuera) | 14,4 | -53,5 | -51,4 |
| graphite@3.5 (fuera) | 7,6 | -52,4 | -51,6 |
| slide@0.5 (fuera) | 10,5 | -20,8 | -21,4 |
| slide@1.5 (fuera) | 7,8 | -23,4 | -23,2 |
| ABC@3.35 (fuera) | 9,6 | -33,5 | -33,1 |

Equilibrio (mínimo de un spline por las curvas):

| apilamiento | d PBE+D3 (Å) | d xu_capas (Å) | E PBE+D3 | E xu_capas |
|---|---|---|---|---|
| AB | 3,42 | 3,45 | -24,3 | -24,4 |
| AA | 3,61 | 3,57 | -20,2 | -21,5 |

## Espín: SCC con espín frente a GPAW PBE con espín

Fuente: `validation/spin_tb_vs_gpaw.json` (receta `spin_validation`). SCC con espín de tbkit (xu_chn; π con Hubbard U = 2,7 eV) frente a GPAW PBE con espín, misma geometría. Energías relativas al primer estado de cada sistema; momentos: Mulliken (TB) e integrados por átomo (GPAW).

W del carbono (átomo libre, GPAW): W_ss = -0,833 eV, W_sp = -0,682 eV, W_pp = -0,616 eV. λ = 1,658 (bisección sobre E_NM − E_AFM de zgnr6: GPAW 73.6 meV por celda, modelo xu_chn, kT 0.01 eV, 48 puntos k); todo lo demás queda fuera del ajuste. Cada celda: E − E(primer estado) en meV / M (μB) / gap (eV).

| sistema | estado | GPAW | TB, W del átomo | TB, W·λ | π, U = 2,7 |
|---|---|---|---|---|---|
| zgnr4 | AFM | 0 / -0,00 / 0,60 | 0 / -0,00 / 0,26 | 0 / -0,00 / 0,65 | 0 / 0,00 / 0,58 |
| zgnr4 | FM | 12 / 0,50 / 0,29 | 6 / 0,30 / 0,00 | 20 / 0,40 / 0,00 | 14 / 0,37 / 0,13 |
| zgnr4 | NM | 59 / 0,00 / 0,00 | 18 / 0,00 / 0,00 | 59 / 0,00 / 0,00 | 44 / 0,00 / 0,00 |
| zgnr6 | AFM | 0 / -0,00 / 0,56 | 0 / -0,00 / 0,25 | 0 / 0,00 / 0,55 | 0 / 0,00 / 0,51 |
| zgnr6 | FM | 5 / 0,50 / 0,48 | 5 / 0,44 / 0,00 | 13 / 0,51 / 0,00 | 9 / 0,46 / 0,17 |
| zgnr6 | NM | 74 / 0,00 / 0,00 | 26 / 0,00 / 0,00 | 74 / 0,00 / 0,00 | 60 / 0,00 / 0,00 |
| zgnr8 | AFM | 0 / -0,00 / 0,56 | 0 / 0,00 / 0,23 | 0 / 0,00 / 0,46 | 0 / 0,00 / 0,45 |
| zgnr8 | FM | 4 / 0,50 / 0,48 | 4 / 0,46 / 0,00 | 7 / 0,54 / 0,15 | 7 / 0,54 / 0,00 |
| zgnr8 | NM | 75 / 0,00 / 0,00 | 31 / 0,00 / 0,00 | 81 / 0,00 / 0,00 | 69 / 0,00 / 0,00 |
| triangulene | T | 0 / 2,00 / 0,71 | 0 / 2,00 / 0,31 | 0 / 2,00 / 0,70 | 0 / 2,00 / 0,64 |
| triangulene | BS | 338 / 0,00 / 0,00 | 154 / 0,00 / 0,00 | 352 / -0,00 / 0,00 | 319 / 0,00 / 0,00 |
| triangulene | NM | 338 / 0,00 / 0,00 | 154 / 0,00 / 0,00 | 352 / 0,00 / 0,00 | 319 / 0,00 / 0,00 |
| phenalenyl | D | 0 / 1,00 / 0,63 | 0 / 1,00 / 0,27 | 0 / 1,00 / 0,62 | 0 / 1,00 / 0,55 |
| phenalenyl | NM | 149 / 0,00 / 0,00 | 67 / 0,00 / 0,00 | 155 / 0,00 / 0,00 | 139 / 0,00 / 0,00 |
| h_graphene | M | 0 / 1,00 / 0,26 | 0 / 1,00 / 0,10 | 0 / 1,00 / 0,24 | — |
| h_graphene | NM | 72 / 0,00 / 0,00 | 36 / 0,00 / 0,00 | 84 / 0,00 / 0,00 | — |
| O2 | T | 0 / 2,00 / 2,26 | 0 / 2,00 / 1,52 | 0 / 2,00 / 2,51 | — |
| O2 | S | 1112 / 0,00 / 0,00 | 758 / 0,00 / 0,00 | 1257 / 0,00 / 0,00 | — |
| CH2 | T | 0 / 2,00 / 2,37 | 0 / 2,00 / 1,17 | 0 / 2,00 / 3,15 | — |
| CH2 | S | 1129 / 0,00 / 0,00 | -38 / 0,00 / 1,26 | 966 / 0,00 / 1,26 | — |

## tbkit frente a GFN1-xTB y GFN2-xTB (tblite)

Fuente: `validation/xtb_vs_tbkit.json` (receta `xtb_comparison`). tbkit (sus conjuntos ajustados a PBE) frente a GFN1-xTB y GFN2-xTB (tblite) sobre las mismas referencias GPAW PBE. xTB no se ajustó a PBE: parte de su diferencia es el error del propio PBE.

Moléculas, en las geometrías de GPAW: mediana por estructura del RMSE de fuerzas (eV/Å, equilibrio / distorsionadas), MAE de la energía relativa a la de equilibrio de cada molécula (eV) y MAE del gap HOMO-LUMO frente al de Kohn-Sham (eV). Celda: tbkit · GFN1 · GFN2.

| referencias | fuerzas | energía relativa | gap |
|---|---|---|---|
| `gpaw_chn.json` (xu_chn, 118) | 0,36/0,39 · 0,48/0,58 · 0,50/0,64 | 0,08 · 0,18 · 0,21 | 0,73 · 0,74 · 2,26 |
| `gpaw_chno.json` (xu_chno, 127) | 0,54/0,57 · 0,76/0,76 · 0,74/0,77 | 0,24 · 0,20 · 0,20 | 0,49 · 0,74 · 1,38 |
| `gpaw_b.json` (xu_chnob, 66) | 0,26/0,31 · 0,46/0,57 · 0,53/0,63 | 0,09 · 0,20 · 0,23 | 0,50 · 0,60 · 0,77 |
| `gpaw_p.json` (xu_chnop, 73) | 0,41/0,36 · 0,48/0,56 · 0,48/0,61 | 0,10 · 0,22 · 0,25 | 0,45 · 0,86 · 1,23 |
| `gpaw_s.json` (xu_chnos, 95) | 0,41/0,41 · 0,59/0,65 · 0,54/0,70 | 0,08 · 0,21 · 0,23 | 0,82 · 0,42 · 0,59 |
| `gpaw_se.json` (xu_chnose, 87) | 0,30/0,29 · 0,44/0,57 · 0,46/0,56 | 0,06 · 0,18 · 0,18 | 0,51 · 0,39 · 0,57 |

Frecuencias armónicas tras relajar con cada método (63 moléculas): mediana del RMS de los modos internos frente a GPAW: tbkit 92 cm⁻¹ (80 con el factor de escala de cada conjunto), GFN1 67, GFN2 56.

Cristales y láminas (xTB en superceldas, solo Γ): mediana del RMSE de fuerzas (eV/Å) / MAE de la energía de las distorsiones (eV por celda de GPAW). Celda: tbkit · GFN1 · GFN2.

| cristal | conjunto | fuerzas / energía |
|---|---|---|
| graphene | xu_chno | 0,09/0,21 · 0,14/0,43 · 0,15/0,46 |
| graphene_N | xu_chn | 0,21/0,12 · 0,20/0,44 · 0,23/0,47 |
| graphene_N3V | xu_chn | 0,44/0,38 · 0,25/0,47 · 0,36/0,58 |
| graphene_B | xu_chnob | 0,42/0,23 · 0,19/0,40 · 0,21/0,42 |
| graphene_S | xu_chnos | 0,22/0,12 · 0,20/0,41 · 0,20/0,44 |
| graphene_P | xu_chnop | 0,17/0,17 · 0,18/0,43 · 0,20/0,45 |
| graphene_Se | xu_chnose | 0,29/0,15 · 0,21/0,45 · 0,21/0,50 |
| graphene_epoxide | xu_chno | 0,60/0,33 · 0,38/0,29 · 0,34/0,35 |
| graphene_OH | xu_chno | 0,22/0,13 · 0,29/0,43 · 0,28/0,47 |
| graphane | xu_chn | 0,18/0,03 · 0,55/0,10 · 0,47/0,07 |
| hBN | xu_chnob | 0,23/0,23 · 0,22/0,29 · 0,24/0,30 |
| cnt80_N | xu_chn | 0,31/0,25 · 0,27/0,74 · 0,30/0,76 |
| diamond_N | xu_chn | 0,62/0,40 · 0,21/0,71 · 0,21/0,37 |

Enlace entre capas (meV/átomo) frente a PBE+D3(BJ):

| sistema | PBE+D3 | GFN1 | GFN2 | xu_capas |
|---|---|---|---|---|
| AB@3.35 | -24,1 | -25,9 | -26,4 | -23,8 |
| AB@3.0 | -13,3 | -25,8 | -32,3 | -10,7 |
| AB@3.7 | -22,1 | -21,3 | -17,8 | -22,6 |
| AA@3.35 | -16,7 | -27,7 | -28,6 | -18,6 |
| graphite@3.35 | -53,5 | -47,3 | -39,4 | -51,4 |
| ABC@3.35 | -33,5 | -34,9 | -31,8 | -33,1 |
| AB@2.8 | 7,8 | -17,6 | -28,9 | -0,4 |
| AB@3.2 | -22,1 | -27,0 | -29,8 | -20,4 |
| AB@3.5 | -24,0 | -24,1 | -22,7 | -24,3 |
| AB@4.0 | -18,3 | -17,2 | -11,6 | -18,6 |
| AB@4.5 | -12,3 | -11,5 | -4,3 | -11,4 |
| AB@5.5 | -5,4 | -4,3 | 2,5 | -4,7 |
| AA@2.8 | 47,0 | -16,3 | -27,5 | 52,0 |
| AA@3.0 | 8,3 | -27,9 | -34,9 | 11,5 |
| AA@3.2 | -10,0 | -29,3 | -32,6 | -10,7 |
| AA@3.5 | -19,7 | -25,2 | -24,1 | -21,3 |
| AA@3.7 | -20,0 | -21,8 | -18,5 | -21,0 |
| AA@4.0 | -17,5 | -17,4 | -11,7 | -17,9 |
| AA@4.5 | -12,1 | -11,5 | -4,2 | -11,5 |
| AA@5.5 | -5,4 | -4,3 | 2,5 | -4,7 |
| graphite@3.2 | -50,5 | -49,1 | -41,7 | -47,0 |
| graphite@3.5 | -52,4 | -43,9 | -34,8 | -51,6 |
| slide@0.5 | -20,8 | -26,9 | -27,7 | -21,4 |
| slide@1.5 | -23,4 | -26,2 | -26,8 | -23,2 |

E(M = 0) − E(M = 2), meV. xTB con polarización de espín (spGFN de tblite), uhf = 0 frente a uhf = 2; tbkit: SCC con espín (W del átomo y W·λ); GPAW: PBE con espín, M fija. Sin la polarización de espín GFN no separa triplete de singlete (≈ −70 meV en los tres sistemas). Fenalenilo fuera: su estado sin espín de GPAW (SOMO semiocupado) no tiene análogo directo en xTB.

| sistema | GPAW | spGFN1 | spGFN2 | tbkit W | tbkit W·λ |
|---|---|---|---|---|---|
| O2 | 1112 | 686 | 686 | 758 | 1257 |
| CH2 | 1129 | 717 | 859 | -38 | 966 |
| triangulene | 338 | 95 | 95 | 154 | 352 |

segundos de reloj por energía + fuerzas de un flake triangular H-saturado (OMP_NUM_THREADS=2), tbkit xu_chn con SCC (kT 0,1 eV) frente a tblite (300 K):

| átomos | tbkit (s) | GFN1 (s) | GFN2 (s) |
|---|---|---|---|
| 34 | 0,13 | 0,07 | 0,05 |
| 64 | 0,42 | 0,55 | 0,46 |
| 124 | 2,40 | 6,64 | 5,24 |
| 202 | 10,71 | 40,83 | 35,17 |

## Nanotubos prístinos con Xu

Fuente: `validation/cnt_xu_carbon.json`. RBM frente a las relaciones empíricas de Araujo y de Jorio; G⁺/G⁻ y su desdoblamiento frente a la referencia guardada. Malla k: la de la receta (`cnt_validation --kmesh`, 16 por defecto, kT 0,03 eV); el archivo no la guarda. Los tubos sin gap no se han comprobado en k después del hallazgo del paso 8 (`modes.GAPLESS_WARNING`): sus G pueden estar ablandadas por la malla, además de faltarles la anomalía de Kohn física.

| tubo | tipo | d (nm) | RBM | Araujo | Jorio | G⁺ | G⁻ | ΔG | ΔG ref. |
|---|---|---|---|---|---|---|---|---|---|
| (8,0) | semiconductor | 0,636 | 349 | 357 | 390 | 1668 | 1623 | 45 | 118 |
| (10,0) | semiconductor | 0,790 | 282 | 287 | 314 | 1669 | 1643 | 25 | 76 |
| (12,0) | metal | 0,945 | 235 | 240 | 262 | 1643 | 1626 | 17 | 89 |
| (14,0) | semiconductor | 1,100 | 201 | 206 | 225 | 1687 | 1665 | 21 | 39 |
| (17,0) | semiconductor | 1,334 | 165 | 170 | 186 | 1689 | 1671 | 18 | 27 |
| (5,5) | metal | 0,685 | 327 | 332 | 362 | 1655 | 1587 | 67 | 170 |
| (6,6) | metal | 0,819 | 271 | 277 | 303 | 1672 | 1621 | 51 | 119 |
| (8,8) | metal | 1,088 | 201 | 209 | 228 | 1684 | 1652 | 32 | 67 |
| (10,10) | metal | 1,358 | 160 | 167 | 183 | 1689 | 1665 | 23 | 43 |
| (8,2) | metal | 0,725 | 306 | 313 | 342 | 1652 | 1598 | 54 | 151 |
| (6,3) | metal | 0,630 | 352 | 361 | 394 | 1661 | 1623 | 38 | 201 |

## Grafeno: G, 2D y 2D′ por doble resonancia

Fuente: `validation/graphene_2d_gpaw.json` (fonones: GPAW PBE LCAO dzp 6x6, regla acústica impuesta; electrones: modelo π (t, β de pi_huckel.json); γ = 0.1 eV).

| láser (eV) | G | 2D | 2D′ | I(2D)/I(G) |
|---|---|---|---|---|
| 1,96 | 1605 | 2819 | 3287 | 9,6 |
| 2,41 | 1605 | 2872 | 3313 | 12,0 |
| 2,80 | 1605 | 2913 | 3327 | 13,0 |

Dispersión de la 2D: 112 cm⁻¹/eV.

## Nanocoil periódica de 204 átomos (anillos 5–7)

Fuente: `validation/nanocoil204_tb.json`.

| conjunto | topología | anillos | gap TB (eV) | enlaces vs PBE RMS (Å) | media (Å) |
|---|---|---|---|---|---|
| `xu_carbon` | OK | 5: 12, 6: 78, 7: 12 | 0,00 | 0,016 | -0,012 |
| `tang_carbon` | OK | 5: 12, 6: 78, 7: 12 | 0,00 | 0,015 | -0,012 |

Gap PBE: 0,22 eV (min over 11 k points Γ–Z, non-SCF bands on the kz=2 density; TB (Xu, Tang) gapless).

Frecuencias de modos TB con las constantes de fuerza PBE (cociente de Rayleigh, cota superior):

| conjunto | modo | TB (cm⁻¹) | PBE (cm⁻¹) |
|---|---|---|---|
| `xu_carbon` | most_pentagon | 1238 | 1209 |
| `xu_carbon` | highest | 1767 | 1670 |
| `tang_carbon` | most_pentagon | 1550 | 1465 |
| `tang_carbon` | highest | 1760 | 1681 |

Malla k de los fonones TB: de 4 a 16 k en z, los modos comprobados se mueven como máximo 0,00 cm⁻¹ (convergida con 4).

Raman resonante (Tang): picos a 532 nm (ensanchamiento 60 cm⁻¹): 320, 427, 475, 732, 939, 1283, 1476, 1586, 1737 cm⁻¹. Actividades de los 20 modos más intensos: ×0,96–1,04 al pasar de 4 a 8 k; ×0,11–0,71 al pasar η de 0,1 a 0,2 eV (posiciones robustas, alturas relativas no).

## Aprendizaje automático (paso 7): medido, no adoptado

Fuentes: `validation/mace_mp_vs_gpaw.json`, `validation/delta_learning.json`.

**MACE-MP-0 sin ajuste fino** frente a GPAW. Moléculas C/H/N/O (RMS por molécula, mediana): small 124,6, medium 116,9 cm⁻¹ (xu_chno, RMS conjunto 115,8). Coil de 204 átomos, modos TB proyectados con fuerzas PBE y MACE:

| modo | PBE | MACE small | MACE medium |
|---|---|---|---|
| xu_most_pentagon | 1209 | 1069 | 997 |
| xu_highest | 1670 | 1480 | 1381 |
| tang_most_pentagon | 1465 | 1301 | 1224 |
| tang_highest | 1681 | 1488 | 1381 |

Enlaces frente a PBE (RMS): Tang 0,015, MACE small 0,021, medium 0,025 Å.

**Δ-learning lineal (SOAP) sobre TB.** Errores de fuerza (eV/Å):

| estructura (tang_carbon) |  | TB | TB + Δ |
|---|---|---|---|
| diamond | ajuste | 0,178 | 0,190 |
| graphene | ajuste | 0,145 | 0,152 |
| chain | ajuste | 0,225 | 0,227 |
| c60 | ajuste | 0,424 | 0,348 |
| vacancy | ajuste | 0,653 | 0,631 |
| stone_wales | fuera | 0,375 | 0,370 |
| amorphous_2.0 | ajuste | 0,865 | 0,839 |
| amorphous_2.6 | ajuste | 0,938 | 0,872 |
| amorphous_3.2 | fuera | 0,982 | 0,934 |

xu_chno (23 moléculas, sin anillos tensos): TB 0,665; TB + Δ en validación cruzada (molécula no vista) 1,039.

With the stored GPAW data the correction does not generalise: held-out carbon structures improve 1-5 %, and on unseen C/H/N/O molecules the cross-validated force error (1.04 eV/Å at best) is worse than TB alone (0.67). Not used by default and not installed; it needs far more DFT data (an active-learning campaign) before it can help.
