# Tablas — electrodo en H2SO4 0.5 M

Electrolito H2SO4 0.5 M, pH 0.3 (deducido de la molaridad), referencia Ag/AgCl KCl sat. Masa activa 20 µg, sobre un disco de carbón vítreo de 3 mm (0.07069 cm²). Carga: 283 µg/cm².

## Las ventanas frente al RHE (el agua es estable entre 0 y 1.229 V)

| experimento | vs Ag/AgCl sat. | vs RHE | margen inferior | margen superior |
|---|---|---|---|---|
| CV, serie ancha | -0.600 … +1.200 V | -0.385 … +1.415 V | -0.385 V | -0.186 V |
| CV, ECSA | -0.050 … +0.100 V | +0.165 … +0.315 V | +0.165 V | +0.914 V |
| GCD | -0.200 … +1.000 V | +0.015 … +1.215 V | +0.015 V | +0.014 V |

Un margen negativo es electrólisis: por debajo de 0 V se desprende hidrógeno y por encima de 1.229 V, oxígeno.

## Capacidad de velocidad del GCD

Ciclo representativo: mediana. Retención 78.5 % a 12× la corriente; sin descontar la caída IR saldría 64.0 %, así que esa diferencia es la resistencia de la celda y no el electrodo.

| I (mA) | j (mA/cm²) | C (µF) | C (F/g) | C (mF/cm²) | ventana (V) | IR (mV) | IR / excursión | ± entre ciclos | retención interna |
|---|---|---|---|---|---|---|---|---|---|
| 0.005 | 0.0707 | 1563 | 78.17 | 22.12 | 1.185 | 14.09 | 1.2 % | 13.5 % | 113.3 % |
| 0.01 | 0.141 | 1235 | 61.75 | 17.47 | 1.17 | 27.54 | 2.3 % | 3.0 % | 102.4 % |
| 0.02 | 0.283 | 1150 | 57.51 | 16.27 | 1.135 | 60.42 | 5.1 % | 0.7 % | 100.1 % |
| 0.04 | 0.566 | 1169 | 58.47 | 16.54 | 1.047 | 144.3 | 12.1 % | 7.2 % | 106.4 % |
| 0.06 | 0.849 | 1227 | 61.35 | 17.36 | 0.9558 | 230.1 | 19.4 % | 0.5 % | 99.8 % |

`C = a·I^-m` con m = 0.090 ± 0.048, R² = 0.534.
Los archivos GANAN un 23.3 % compuesto dentro de sus propios ciclos: activación, no degradación.

## Impedancia

| archivo | E (V vs Ag/AgCl) | R_s (Ω) | C′(0.1 Hz) (mF) | codo (Hz) | Kramers-Kronig | circuito | residuo | el que elegía «auto» |
|---|---|---|---|---|---|---|---|---|
| `PEIS_incial_C01` | +0.4176 | 105.8 | 0.4681 | 0.1829 | sí | `R0-(R1|Q1)-Q2` | 1.98 % | `R0-(R1|Q1)` a 17.25 % |
| `PEIS_despues_CV_mult_01_PEIS_C01` | +0.2753 | 87.54 | 0.7871 | 0.3782 | sí | `R0-(R1|Q1)-Q2` | 1.86 % | `R0-T1` a 11.04 % |
| `PEIS_despues_CV_mult_02_PEIS_C01` | +0.4997 | 84.99 | 0.7972 | 0.3156 | **no** | `R0-(R1|Q1)-Q2` | 1.54 % | `R0-(R1|Q1)-T1` a 1.54 % |

La R_ct no está en la tabla a propósito: ninguno de los espectros cierra el semicírculo, así que la que devuelva un ajuste la pone el modelo y no la medida.

## C_dl y ECSA (serie estrecha)

C_dl = 329.9 µF, ECSA = 8.25 cm² (entre 5.5 y 16.5 según la Cs que se adopte: ese factor de tres es irreducible y pasa entero al área). Contra los 0.07069 cm² geométricos del disco, un factor de rugosidad de **117** (entre 78 y 233).

| E (V) | b | error | R² |
|---|---|---|---|
| -0.028 | -0.013 | 0.471 | 0.0002 |
| -0.002 | 0.872 | 0.097 | 0.9526 |
| +0.025 | 0.962 | 0.073 | 0.9774 |
| +0.051 | 0.979 | 0.055 | 0.9873 |
| +0.077 | 0.977 | 0.045 | 0.9916 |

## Archivos ilegibles

- `GCPL__prueba_ventana_1-0_06_PEIS-FINAL.mpr: GCPL__prueba_ventana_1-0_06_PEIS-FINAL.mpr: la cabecera de datos declara 0 puntos y 16 columnas`

