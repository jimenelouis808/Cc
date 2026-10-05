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

Validez declarada por cada archivo:

- **`pi_huckel`** (carbono sp2: grafeno, nanotubos, cintas, copos, fullerenos; N, B y O sustitucionales): solo estados π cerca del nivel de Fermi (±3 eV); enlaces C-C de 1.35 a 1.50 Å; sin energías totales
- **`xu_carbon`** (carbono: diamante, grafito, dímero C2, cadenas lineales y fases de alta coordinación): enlaces C-C de 1.2 a 2.6 Å; C puro (sin H ni heteroátomos); base ortogonal s+p
- **`tang_carbon`** (carbono puro: cadena, grafito, diamante y fases metálicas (sc, bcc, fcc) del ajuste original a LDA+GGA; aquí además validado frente a GPAW PBE en amorfos y defectos): solo carbono, sin SCC. Forma funcional y parte electrónica de Tang et al. 1996; repulsión, escala de Δe y corte ajustados a GPAW PBE. Error de fuerzas (RMS, eV/Å) frente a GPAW: diamond 0.18, graphene 0.14, chain 0.22, c60 0.42, vacancy 0.65, stone_wales 0.38, amorphous_2.0 0.87, amorphous_2.6 0.94, amorphous_3.2 0.98 (amorphous_3.2 y stone_wales fuera del ajuste). Energías comparables entre estructuras de carbono (una sola referencia por átomo). Diamante: mínimo en a = 3,515 Å (GPAW-PBE ~3,57; -1,5 %), B = 478 GPa. Energías entre fases muy distintas poco fiables (diamante 0,37 eV/átomo frente al resto); fuerzas y geometrías mejores que Xu en 8 de 9 entornos (grafeno 0,14 frente a 0,12 eV/Å).
- **`xu_chn`** (moléculas C/H/N de capa cerrada: hidrocarburos saturados, insaturados y aromáticos; aminas, iminas, nitrilos, N piridínico y pirrólico): ajustado en moléculas de capa cerrada; cristales (SCC periódico): extrapolación comprobada contra GPAW (validation/crystals_tb_vs_gpaw.json): N grafítico y piridínico (N3V) en grafeno, grafano y CNT (8,0) con N: error de fuerzas en el mínimo de GPAW 0,5-1,2 veces el molecular, C-N a menos de 0,02 Å; N sustitucional en diamante: C-N -0,08 Å (1,52 frente a 1,60, como los C-N simples de las moléculas) y fuerzas 1,4-1,7 veces el molecular (ambos con espín apareado: el centro P1 real es de capa abierta); la red de TB sale 0,5-1 % más corta que la de GPAW-dzp (el grafano, igual); enlaces dentro de lo muestreado (C-H 0.97-1.26 Å, C-N 1.11-1.57 Å, H-N 0.91-1.17 Å, N-N 1.07-1.52 Å); energías relativas solo dentro de una misma composición (no se ajustaron energías de atomización); C-C como en xu_carbon; falla en anillos tensos (aziridina); los C-C y C-N simples junto a un heteroátomo se desvían ~0.08 Å
- **`xu_chno`** (moléculas C/H/N/O de capa cerrada: las de xu_chn más alcoholes, éteres, epóxidos, aldehídos, cetonas, ácidos carboxílicos, ésteres, amidas, furano, nitro; motivos de óxido de grafeno (epóxido e hidroxilos basales)): ajustado en moléculas de capa cerrada; cristales (SCC periódico): extrapolación comprobada contra GPAW (validation/crystals_tb_vs_gpaw.json): epóxido y OH sobre grafeno: fuerzas 0,4-1,1 veces el error molecular; heredan los errores de las moléculas (epóxido C-O +0,09 Å; OH C-O -0,05 Å, el C bajo el OH 0,14 Å menos piramidal y el H gira, 0,24 Å); el C-O simple de los alcoholes sale 0,06-0,08 Å corto también en moléculas (metanol -0,08 Å); enlaces dentro de lo muestreado (C-H 0.97-1.26 Å, C-N 1.11-1.61 Å, C-O 1.10-1.58 Å, H-N 0.91-1.17 Å, H-O 0.89-1.09 Å, N-N 1.07-1.52 Å, N-O 1.14-1.31 Å, O-O 1.35-1.54 Å); energías relativas solo dentro de una misma composición (no se ajustaron energías de atomización); C-C como en xu_carbon; anillos de tres miembros con la corrección de ángulos agudos ajustada a barridos de apertura del anillo: epóxido basal sobre coroneno C-C 0.03 Å, C-O 0.10 Å; oxirano y aziridina C-C ~0.12 Å; ciclopropano C-C +0.19 Å con un mínimo poco profundo; el biciclobutano (excluido del ajuste) se abre; los C-C y C-N simples junto a un heteroátomo se desvían ~0.08 Å; frecuencias frente a GPAW (hessianas en el ajuste): RMS 38-180 cm⁻¹ por molécula, media 126 (agua 38, metanol 120, ácido fórmico 111)
- **`xu_chnob`** (xu_chno más boro: boranos, ésteres y ácidos bóricos y borónicos, amino-borano, borazina, B-N en grafeno): ajustado en moléculas de capa cerrada; cristales (SCC periódico): extrapolación comprobada contra GPAW (validation/crystals_tb_vs_gpaw.json): B grafítico en grafeno: fuerzas 1,4 veces el error molecular, B-C +0,027 Å; h-BN: red 1,1 % larga (2,53 frente a 2,50 Å) y 10-15 % más blanda en distorsiones; enlaces dentro de lo muestreado (B-B 1.64-1.81 Å, B-C 1.47-1.70 Å, B-H 1.10-1.37 Å, B-N 1.35-1.72 Å, B-O 1.30-1.46 Å; el resto, como xu_chno); energías relativas solo dentro de una misma composición (no se ajustaron energías de atomización); C-C como en xu_carbon; B junto con S o P en la misma estructura no está cubierto; enlaces de B a menos de 0.03 Å de GPAW; ésteres B(OCH3)n corregidos con active learning (antes colapsaban; RMS 949 -> 95 cm⁻¹), aunque al relajar giran sus metilos (desplazamiento ~1 Å, sin colapso); ácidos borónicos arílicos (PhB(OH)2): corregidos con el contacto H···H (torsión del B(OH)2 336° frente a 337° de GPAW; antes quedaba plano, 0,6 Å)
- **`xu_chnos`** (xu_chno más azufre: tioles, sulfuros, disulfuros, tiofeno, sulfóxidos, sulfonas, ácidos sulfónicos, sulfonamidas): ajustado en moléculas de capa cerrada; cristales (SCC periódico): extrapolación comprobada contra GPAW (validation/crystals_tb_vs_gpaw.json): S en grafeno: fuerzas 0,5-0,6 veces el error molecular, S-C +0,03 Å, S 0,06 Å más alto sobre la hoja; enlaces dentro de lo muestreado (C-S 1.40-1.94 Å, H-S 1.24-1.45 Å, N-S 1.62-1.79 Å, O-S 1.34-1.75 Å, S-S 1.96-2.17 Å; el resto, como xu_chno); energías relativas solo dentro de una misma composición (no se ajustaron energías de atomización); C-C como en xu_carbon; S junto con B o P en la misma estructura no está cubierto; los C-S simples con carbono sp3 se desvían 0.06-0.09 Å (S=O, S-S, S-H, S-N y C-S aromático a menos de 0.04 Å)
- **`xu_chnop`** (xu_chno más fósforo: fosfinas, óxidos de fosfina, ácidos fosfórico y fosfónicos, fosfatos, fosfinina): ajustado en moléculas de capa cerrada; cristales (SCC periódico): extrapolación comprobada contra GPAW (validation/crystals_tb_vs_gpaw.json): P en grafeno: fuerzas 0,3-0,6 veces el error molecular, P-C exacto (0,001 Å) pero P 0,12 Å más bajo sobre la hoja: ángulos C-P-C demasiado abiertos; enlaces dentro de lo muestreado (C-P 1.67-2.15 Å, H-P 1.30-1.57 Å, N-P 1.66-1.84 Å, O-P 1.39-1.74 Å, P-P 2.13-2.36 Å; el resto, como xu_chno); energías relativas solo dentro de una misma composición (no se ajustaron energías de atomización); C-C como en xu_carbon; P junto con B o S en la misma estructura no está cubierto; el C-P de P(V) (ácidos fosfónicos, óxidos de fosfina) sale 0.05-0.10 Å largo, el de P(III) 0.04-0.06 Å; P-O, P=O, P-P, P-N, P-H y C-P aromático a menos de 0.025 Å; ésteres P(OCH3)n corregidos con active learning (antes colapsaban); ácidos con P-OH (fosfórico, fosfónicos): el OH gira hacia el otro O al relajar, 0,4-0,9 Å; causa: la base mínima (GPAW con base sz también hunde el giro del OH: -0,35 eV a 180° frente a -0,23 con dzp; en TB es todo electrónico); en MePO3H2 los d del P también importan en DFT; úsense sus frecuencias O-H con cautela
- **`xu_chnose`** (xu_chno más selenio: selenoles, selenuros, diselenuros, selenofeno, selenóxidos, ácidos selenínicos, Se-N): ajustado en moléculas de capa cerrada; cristales (SCC periódico): extrapolación comprobada contra GPAW (validation/crystals_tb_vs_gpaw.json): Se en grafeno: fuerzas ~1 vez el error molecular, Se-C +0,06 Å, Se 0,07 Å más alto sobre la hoja; enlaces dentro de lo muestreado (C-Se 1.61-2.09 Å, H-Se 1.36-1.57 Å, N-Se 1.81-2.00 Å, O-Se 1.57-2.01 Å, Se-Se 2.23-2.47 Å; el resto, como xu_chno); energías relativas solo dentro de una misma composición (no se ajustaron energías de atomización); C-C como en xu_carbon; Se junto con B, S o P en la misma estructura no está cubierto; enlaces de Se a menos de 0.08 Å; ácidos seleníninicos (Se(=O)OH): el OH gira hacia el otro O al relajar, 0,4-0,9 Å; causa: la base mínima (en el giro rígido la repulsión no aporta y la parte electrónica pone 240-300° 0,1 eV bajo el mínimo; GPAW con base mínima sz hace lo mismo, -0,06 eV, y con dzp +0,03); no son los d del Se (GPAW sin ellos no gira); úsense sus frecuencias O-H con cautela

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
