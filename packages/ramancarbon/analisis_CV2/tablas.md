# Serie 5–100 mV/s — tablas

Muestra **S1** (FeSe/C a 900 °C). Carbono vítreo de 3 mm
(A = 0.07069 cm²), 20 µg de material activo, **KOH 0.5 M**, **Ag/AgCl de KCl
saturado**, contraelectrodo de Pt. Ventana **-0.601 a +0.400 V vs Ag/AgCl**
= **+0.399 a +1.399 V vs RHE**, con el límite positivo **169 mV por encima**
del equilibrio del oxígeno. Abanico de velocidades: **factor 20**.

De cada archivo se informa el último ciclo completo.

## Tabla 1 — Capacitancia y balance de carga

| ν / mV s⁻¹ | duración / s | Q_ox / µC | Q_red / µC | ΔQ / µC | Q_red/Q_ox | C_lazo / F g⁻¹ | **C_red / F g⁻¹** | C_red / mF cm⁻² |
|---|---|---|---|---|---|---|---|---|
| 5 | 400.0 | 305.5 | 147.8 | 157.8 | 0.484 | 11.33 | **7.39** | 2.09 |
| 10 | 200.0 | 219.0 | 150.4 | 68.6 | 0.687 | 9.23 | **7.52** | 2.13 |
| 20 | 100.0 | 165.8 | 135.5 | 30.3 | 0.817 | 7.53 | **6.77** | 1.92 |
| 50 | 40.0 | 122.1 | 110.4 | 11.7 | 0.904 | 5.81 | **5.52** | 1.56 |
| 100 | 20.0 | 99.4 | 94.7 | 4.7 | 0.953 | 4.85 | **4.73** | 1.34 |

Ajuste ΔQ = ⟨i_par⟩·t: **⟨i_par⟩ = 0.40 µA**, ordenada -6.7 µC, R² = 0.9958.
La fuga es el 53 % de Q_ox a 5 mV/s y el 8 % a 100.

## Tabla 2 — Las tres leyes de dos términos

| E / V vs Ag/AgCl | E / V vs RHE | b | R²(potencia) | R²(Dunn) | R²(fuga) | gana |
|---|---|---|---|---|---|---|
| -0.50 | 0.50 | 0.827 | 0.98825 | 0.99618 | 0.98734 | dunn |
| -0.40 | 0.60 | 0.902 | 0.97818 | 0.99466 | 0.98713 | dunn |
| -0.30 | 0.70 | 0.909 | 0.98309 | 0.99541 | 0.98927 | dunn |
| -0.20 | 0.80 | 0.825 | 0.99805 | 0.99815 | 0.99288 | **EMPATE** |
| -0.10 | 0.90 | 0.777 | 0.99995 | 0.99925 | 0.99468 | potencia |
| -0.00 | 1.00 | 0.722 | 0.99798 | 0.99981 | 0.99609 | dunn |
| +0.10 | 1.10 | 0.662 | 0.99387 | 0.99964 | 0.99676 | dunn |
| +0.20 | 1.20 | 0.594 | 0.98895 | 0.99852 | 0.99688 | dunn |
| +0.30 | 1.30 | 0.508 | 0.98482 | 0.99525 | 0.99544 | **EMPATE** |

Con un factor 20 la serie ya elige en 7 de los 9 potenciales, cosa que con
el factor 10 de la serie anterior no pasaba en ninguno. Pero elegir entre
PARES no es la pregunta correcta cuando se sabe que los tres términos
están: eso es la tabla 3.

## Tabla 3 — Los tres términos a la vez

`i = k_cap·ν + k_diff·√ν + i_par`, cinco velocidades, dos grados de libertad.

| E / V vs RHE | k_cap / µF | k_diff / µA (V/s)^-½ | i_par / µA | R² | determinados |
|---|---|---|---|---|---|
| 0.50 | 33.1 ± 8.2 | 24.2 ± 3.2 | -1.03 ± 0.26 | 0.99956 | k_cap k_diff i_par |
| 0.60 | 36.0 ± 1.6 | 28.4 ± 0.6 | -1.48 ± 0.05 | 0.99999 | k_cap k_diff i_par |
| 0.70 | 40.5 ± 2.0 | 24.8 ± 0.8 | -1.31 ± 0.06 | 0.99998 | k_cap k_diff i_par |
| 0.80 | 47.5 ± 3.7 | 18.2 ± 1.5 | -0.74 ± 0.12 | 0.99991 | k_cap k_diff i_par |
| 0.90 | 49.8 ± 4.1 | 14.4 ± 1.6 | -0.40 ± 0.13 | 0.99987 | k_cap k_diff i_par |
| 1.00 | 52.2 ± 4.3 | 11.6 ± 1.7 | -0.09 ± 0.14 | 0.99984 | k_cap k_diff |
| 1.10 | 53.6 ± 4.2 | 10.2 ± 1.7 | +0.21 ± 0.13 | 0.99984 | k_cap k_diff |
| 1.20 | 55.1 ± 3.2 | 10.4 ± 1.3 | +0.57 ± 0.10 | 0.99991 | k_cap k_diff i_par |
| 1.30 | 55.2 ± 3.9 | 14.0 ± 1.5 | +1.16 ± 0.12 | 0.99989 | k_cap k_diff i_par |

Los tres se determinan en casi toda la ventana, con R² de 0.9996 o mejor.
`i_par` **cambia de signo**: catódica en el extremo negativo (que está en
la región de reducción de oxígeno) y anódica en el positivo (que está por
encima del equilibrio del O₂), cruzando cero cerca de +0.03 V vs Ag/AgCl.

## Tabla 4 — Reparto de Dunn

| ν / mV s⁻¹ | capacitivo | difusivo |
|---|---|---|
| 5 | 23 % | 77 % |
| 10 | 29 % | 71 % |
| 20 | 37 % | 63 % |
| 50 | 48 % | 52 % |
| 100 | 58 % | 42 % |
