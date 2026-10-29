# PEIS de tres carbones en H₂SO₄ 0.5 M — tablas

Pirólisis a 800, 900 y 1020 °C. Barrido de 100 kHz a 0.0999 Hz, 37
frecuencias, 10 mV de amplitud, unos 82 s por espectro.

Lo que el usuario **no** dio y hace falta para leer estas tablas:

- **La masa.** No hay ni un F/g aquí. Una capacitancia específica con una
  masa supuesta está mal exactamente en esa proporción.
- **El área.** Se usa el disco de carbón vítreo de 3 mm que el usuario
  dice usar casi siempre: **0.0707 cm², suposición**, y además
  GEOMÉTRICA. La electroquímicamente activa de un carbón poroso es de
  diez a mil veces mayor, así que los mF/cm² de abajo sirven para
  comparar estas tres muestras entre sí y para nada más.

## 1. Las condiciones, leídas del propio archivo

| | 800 °C | 900 °C | 1020 °C |
|---|---|---|---|
| Polarización E (V vs Ag/AgCl sat) | +0.5241 | +0.4932 | +0.5218 |
| Polarización E (V vs RHE) | +0.739 | +0.708 | +0.737 |
| Amplitud \|ΔE\| (mV) | 10.7 | 10.3 | 10.1 |
| Duración (s) | 1.8 – 82.5 | 2.0 – 84.3 | 2.0 – 81.7 |

E(RHE) = E + 0.197 + 0.05916 × 0.3, con pH 0.3 para el H₂SO₄ 0.5 M de
`echem.json` (actividad, no −log c). Los tres están en la región de doble
capa del carbono en ácido, que es donde se mide una capacitancia.

## 2. Kramers-Kronig — antes del circuito, siempre

| | 800 °C | 900 °C | 1020 °C |
|---|---|---|---|
| ¿pasa? | sí | sí | sí |
| residuo RMS (%) | 0.093 | 0.067 | 0.095 |
| rachas de signo (× azar) | 0.84 | 1.08 | 1.03 |

Los tres son lineales, causales y estables. El residuo no tiene
estructura (las rachas salen donde el azar las pone). Cualquier circuito
que se ajuste abajo tiene sentido; si esto hubiera fallado, nada de lo
que sigue significaría nada por bien que ajustara.

## 3. La forma, leída de los datos sin ningún circuito

| | 800 °C | 900 °C | 1020 °C |
|---|---|---|---|
| R_s (Ω) | 11.36 | 20.53 | 25.49 |
| cómo | cota superior¹ | **cruce interpolado** | cota superior¹ |
| Z″ > 0 arriba (inductancia) | no | **sí** | no |
| ápice de arco | **no hay** | **no hay** | **no hay** |
| codo, fase = −45° (Hz) | 62.7 | 30.8 | 60.6 |
| τ del codo (ms) | 16.0 | 32.5 | 16.5 |
| décadas cerca de −45° | 0.67 | 0.50 | 0.50 |
| fase a 0.0999 Hz (°) | −80.0 | −81.5 | **−68.4** |

¹ el espectro no cruza el eje real, así que se informa Z′ a 100 kHz, que
es una COTA SUPERIOR. En el de 900 °C sí cruza porque la inductancia del
cableado le mete Z″ positivo arriba, y por eso ahí el número está
interpolado y es la R_s de verdad.

**Ninguno de los tres cierra un semicírculo.** −Z″ sube monótona hasta la
frecuencia más baja medida en los tres. Eso quiere decir que de estos
datos NO sale ninguna resistencia de transferencia de carga: la que
devuelva un ajuste sale del modelo.

## 4. El circuito, y cómo se eligió

Candidatos propuestos por la forma, ordenados por BIC, con la columna que
de verdad decide (cuántos parámetros vuelven determinados):

| circuito | 800 °C | 900 °C | 1020 °C |
|---|---|---|---|
| `R0-Q1` (edlc_cpe) | −314, 3/3 | −481, 3/3 | −502, 3/3 |
| `R0-(R1\|Q1)` (randles_cpe) | −310, **3/4** | −477, **3/4** | −505, 4/4 |
| `R0-T1` (poroso) | −469, 4/4 | −511, 4/4 | −577, 4/4 |
| `R0-(R1\|Q1)-Q2` | −784, 6/6 | −898, 6/6 | −521, **4/6** |
| **`R0-(R1\|Q1)-T1`** | −762, **7/7** | −893, **6/7** | **−811, 7/7** |
| **`L0-R0-(R1\|Q1)-T1`** | **−800**, **6/8** | **−931, 8/8** | −807, **7/8** |

El modelo común para comparar las tres muestras es **`R0-(R1|Q1)-T1`**:
una resistencia en serie, un arco interfacial y la línea de transmisión
del poro. Es el único que vuelve con TODOS sus parámetros determinados en
las tres. En la de 900 °C hay que añadirle la inductancia del montaje
(`L0-R0-(R1|Q1)-T1`), y no por pulcritud: ver §6.

`randles_cpe`, que era el circuito por defecto del programa, manda su R₁ a
la cota de 10⁹ Ω en las de 800 y 900 °C. Es el modelo diciendo «aquí no
hay arco» de la única forma que puede.

## 5. Los parámetros del ajuste

| parámetro | 800 °C | 900 °C | 1020 °C |
|---|---|---|---|
| circuito | `R0-(R1\|Q1)-T1` | `L0-R0-(R1\|Q1)-T1` | `R0-(R1\|Q1)-T1` |
| residuo medio (%) | 0.540 | 0.188 | 0.441 |
| L0 (nH) | — | 324 ± 12 % | — |
| R0 (Ω) | 10.66 ± 0.4 % | 20.31 ± 0.1 % | 24.89 ± 0.2 % |
| R1 (Ω) | 30.0 ± 4.3 % | 22.0 ± 3.5 % | **2978 ± 5.1 %** |
| Q1 (mS·sⁿ) | 1.233 ± 7.6 % | 3.854 ± 3.3 % | 0.490 ± 2.9 % |
| n₁ | **0.559 ± 2.6 %** | **0.604 ± 0.9 %** | 0.761 ± 0.2 % |
| **R_ion del poro (Ω)** | **6.02 ± 17 %** | **0.790 ± 13 %** | **16.03 ± 1.8 %** |
| Q_pared (mS·sⁿ) | 0.2047 ± 0.19 % | 0.3161 ± 0.06 % | 0.2469 ± 1.1 % |
| n_pared | 0.881 ± 0.11 % | 0.912 ± 0.05 % | 0.867 ± 0.46 % |

`R_ion` desplaza la vertical de baja frecuencia en **R_ion/3**: 2.01,
0.26 y 5.34 Ω.

## 6. La inductancia de la de 900 °C no es cosmética

Su Z″ es **positivo** a 100 kHz (+0.089 Ω): inductancia del cableado.
Ajustada sin una L en el circuito, R0 se la come:

| | con `L0` | sin `L0` |
|---|---|---|
| R_ion (Ω) | **0.790 ± 13 %** | 2.4·10⁻⁵ **± 3.5·10⁷ %** |
| R0 (Ω) | 20.31 ± 0.1 % | 20.46 **± 13.6 %** |
| residuo (%) | 0.188 | 0.242 |

La resistencia que limita la potencia del electrodo deja de ser medible,
y el residuo apenas se mueve: 0.19 → 0.24 %. Nada en la calidad del
ajuste lo delata.

## 7. Capacitancia — cuatro números que no son el mismo

| | 800 °C | 900 °C | 1020 °C |
|---|---|---|---|
| **C′ (0.0999 Hz), entregable** | **0.2117 mF** | **0.3244 mF** | **0.1964 mF** |
| C_s = −1/ωZ″ | 0.2183 mF | 0.3316 mF | 0.2272 mF |
| C de la pared del poro (elemento) | 0.2164 mF | 0.3293 mF | 0.2626 mF |
| C del ARCO, Hsu-Mansfeld | 0.0918 mF | 0.7645 mF | 0.5516 mF |
| C del ARCO, Brug faradaico | 0.0319 mF | 0.4721 mF | 0.1225 mF |
| C′ / área geométrica supuesta | 2.995 mF/cm² | 4.589 mF/cm² | 2.779 mF/cm² |
| τ₀ (máx. de C″) | 7.38 ms | 10.84 ms | **no determinada** |
| f₀ (Hz) | 21.6 | 14.7 | — |

**La que se cita es C′.** Las otras tres responden otras preguntas:

- **C_s frente a C′.** Difieren un 3 % en las de 800 y 900 °C y un **16 %**
  en la de 1020. Esa diferencia es exactamente la parte REAL de la
  impedancia a esa frecuencia: donde la fase llega a −80° las dos
  fórmulas son casi la misma y donde se queda en −68° no.
- **La C del elemento frente a C′.** +2.2 %, +1.5 % y **+33.7 %**. El
  ajuste reproduce C′ a mejor del 1 % en las tres (ver §8), así que ese
  34 % no es un fallo del modelo: es que C′ lleva dentro la resistencia
  de toda la cadena y la de 1020 °C no ha llegado al régimen capacitivo.
- **La C del arco no es la del electrodo.** Es del `R1|Q1`, que describe
  la interfase y no lo que almacena. Y Brug devuelve siempre menos que
  Hsu-Mansfeld porque R_s ∥ R_ct < R_ct; las dos están en la tabla porque
  la literatura las usa indistintamente y no dan lo mismo.

## 8. La comprobación que hace falta hacer

**Contra la propia EC-Lab.** El instrumento calcula sus columnas `Cp` y
`Cs` por su cuenta. Lo que el paquete calcula de Z sin tocar esas
columnas:

| | Cp de EC-Lab (µF) | C′ nuestra (µF) | Cs de EC-Lab (µF) | C_s nuestra (µF) |
|---|---|---|---|---|
| 800 °C | 211.7311 | **211.7311** | 218.2865 | **218.2865** |
| 900 °C | 324.3567 | **324.3567** | 331.6434 | **331.6434** |
| 1020 °C | 196.4450 | **196.4450** | 227.1514 | **227.1514** |

Coinciden en las siete cifras. Eso comprueba de una vez la columna de
frecuencia, las de Re(Z) y −Im(Z), el convenio de signo con el que se
guardan y la fórmula C(ω) = 1/(jωZ) — que es justo lo que el lector de
`.mpr`, al ser ingeniería inversa, necesita contrastar.

**Contra los datos, el ajuste.** C′ del circuito frente a C′ medida:
+0.2 %, +0.3 % y +0.9 %. El modelo describe el extremo bajo del barrido
en las tres.

## 9. La DRT, y por qué sus picos no son procesos

| | 800 °C | 900 °C | 1020 °C |
|---|---|---|---|
| λ | 1.11·10⁻⁵ | 1.11·10⁻⁵ | 4.96·10⁻⁶ |
| residuo (%) | 1.59 | 1.80 | 0.28 |
| R óhmica (Ω) | 8.99 | 20.4 | 24.1 |
| picos (τ, R) | 0.047 s, 120 Ω<br>0.382 s, 484 Ω | 0.029 s, 41 Ω<br>0.302 s, 218 Ω | 0.047 s, 171 Ω<br>0.190 s, 434 Ω<br>0.769 s, 1764 Ω |

Los tres disparan el mismo aviso: **la fase pasa 0.5–0.7 décadas cerca de
−45°, que es un elemento DISTRIBUIDO y no tiene constante de tiempo
propia.** Un poro, un Warburg y un Gerischer son distribuciones continuas;
la DRT las reparte en una joroba ancha y el buscador de picos la trocea
en arcos. Esos picos no se cuentan como procesos ni se traducen a ZARC.

Los tres avisan además de peso en el extremo lento: falta frecuencia baja.

## 10. Lo que se puede decir de las tres muestras

| | 800 °C | 900 °C | 1020 °C |
|---|---|---|---|
| C′ entregable | 0.212 mF | **0.324 mF** | 0.196 mF |
| R_s | **11.4 Ω** | 20.5 Ω | 25.5 Ω |
| R_ion del poro | 6.0 Ω | **0.79 Ω** | **16.0 Ω** |
| codo (potencia) | 62.7 Hz | 30.8 Hz | 60.6 Hz |
| fase final | −80.0° | **−81.5°** | **−68.4°** |
| n de la pared | 0.881 | **0.912** | 0.867 |

- **La de 900 °C es la mejor de las tres por todos los criterios que
  importan a un condensador**: la mayor capacitancia (+53 % sobre la de
  800, +65 % sobre la de 1020), la menor resistencia iónica dentro del
  poro (0.79 Ω, ocho veces menos que la de 800 y veinte veces menos que
  la de 1020), la pared más próxima a ideal (n = 0.912) y la única que
  llega a −81.5° de fase.
- **La de 1020 °C es la peor, y de forma coherente**: R_ion de 16 Ω, fase
  que se para en −68° y τ₀ que ni siquiera cae dentro del barrido. Su
  R₁ de 3 kΩ dice que no hay reacción faradaica apreciable; lo que hay es
  un electrodo al que el ion le cuesta entrar. Encaja con que una
  pirólisis más alta cierre la microporosidad.
- **El codo no ordena las muestras igual que R_ion**, y no es
  contradicción: el codo es del DISPOSITIVO y lleva dentro R_s, que es la
  menor justo en la de 800 °C.

Lo que NO se puede decir con estos tres espectros:

- **Nada por gramo.** Falta la masa.
- **Nada de área superficial.** mF/cm² está sobre el área geométrica
  supuesta del disco; el área activa de un carbón poroso es otra cosa.
- **Nada de transferencia de carga medida.** Ninguno cierra un arco.
- **Nada sobre a qué se debe cada constante de tiempo.** Una τ es un
  número y los procesos se solapan en él. Para asignarlos hay que mover
  algo —el potencial, la temperatura, la concentración— y ver cómo se
  mueve el pico.
