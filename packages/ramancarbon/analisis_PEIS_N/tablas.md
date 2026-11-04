# PEIS de carbones Fe/Fe₃C según el precursor de nitrógeno — tablas

Etanol como fuente de carbono en las cuatro muestras; lo que cambia es
el precursor que aporta el nitrógeno. **EB está dos veces, a dos
potenciales**, y esa pareja es el control interno de todo el análisis.

| clave | precursor de N | qué nitrógeno aporta |
|---|---|---|
| **E** | — (etanol solo) | ninguno: es el control |
| **EA** | acetonitrilo, CH₃CN | nitrilo, N sp, relación C/N baja |
| **EB** | bencilamina, PhCH₂NH₂ | amina primaria sp³ sobre un anillo |
| **EP** | piridina, C₅H₅N | N aromático, ya dentro de un anillo |

## 0. Las condiciones de celda

Todas dadas por el usuario.

| | valor | nota |
|---|---|---|
| Electrolito | **H₂SO₄ 0.5 M** | |
| pH | **0.3** | de la tabla del paquete, y sale de la **actividad**, no de −log c: el primer protón está disociado del todo y el segundo sólo en parte (Ka₂ = 0.012) |
| Electrodo | **disco de carbón vítreo de 3 mm**, 0.0707 cm² | área **GEOMÉTRICA** |
| Referencia | **Ag/AgCl saturado** | confirmada. E(RHE) = E + 0.2147 V a pH 0.3 |
| Masa | **0.909 mg** | 20 µL de una tinta de 5 mg en 100 µL de isopropanol + 10 µL de Nafion: 20/110 × 5 mg |
| Carga | **12.9 mg cm⁻²** | **de diez a cien veces lo habitual sobre un carbón vítreo** |

Dos consecuencias que hay que tener delante al leer el resto:

- **La masa es NOMINAL.** Sale del volumen depositado, no de pesar el
  electrodo seco. Si parte de la gota se salió del disco, la masa real
  es menor y los F/g de abajo son todavía más bajos de lo que deberían.
- **12.9 mg cm⁻² es una película gruesa**, de decenas de micras. Una
  película así tiene una resistencia iónica EN PROFUNDIDAD que el ion
  no atraviesa en los 100 s de un punto a 0.01 Hz. Eso explica por sí
  solo lo que la tabla 4 muestra: que ninguna muestra llega al régimen
  capacitivo, y que los F/g salgan dos órdenes por debajo de lo que
  publica cualquier carbón poroso.

## 1. Las condiciones, leídas de los propios archivos

| | E | EA | **EB cat.** | **EB an.** | EP |
|---|---|---|---|---|---|
| Polarización (V vs Ag/AgCl sat) | +0.4489 | +0.3105 | **−0.4043** | **+0.3249** | +0.4853 |
| **Polarización (V vs RHE)** | +0.664 | +0.525 | **−0.190** | **+0.540** | +0.700 |
| Amplitud \|ΔE\| (mV) | 10.7 | 10.4 | 10.3 | 10.3 | 10.6 |
| Frecuencias | 43 | 43 | 43 | 43 | 43 |
| Barrido (Hz) | 10⁵ → 0.01 | 10⁵ → 0.01 | 10⁵ → 0.01 | 10⁵ → 0.01 | 10⁵ → 0.01 |
| Duración (s) | 678 | 673 | 676 | 674 | 672 |
| Corriente media \|I\| (µA) | 0.354 | 0.265 | **15.9** | **2.47** | 0.538 |
| \|j\| geométrica (µA cm⁻²) | 5.0 | 3.8 | **225** | **34.9** | 7.6 |

**Los cuatro precursores se midieron a potenciales distintos.** Los
nombres dicen «PEIS 0 V», que en EC-Lab es 0 V *frente al potencial de
circuito abierto*, así que el continuo de cada medida es el OCP de su
propia muestra. Con el segundo espectro de EB, **cuatro de los cinco
caen ya dentro de 0.18 V entre sí** (+0.525 a +0.700 V vs RHE), y EB an.
y EA están a **13 mV**: ésa es una comparación directa.

**El primer EB estaba por debajo del equilibrio del hidrógeno.** −0.190 V
vs RHE con 0.225 mA cm⁻²: desprendimiento de hidrógeno corriendo durante
los once minutos del barrido. El segundo espectro, **la misma muestra en
la misma celda** (R_s 85.01 y 85.22 Ω, 0.25 % de diferencia), pasa seis
veces menos corriente.

## 2. Kramers-Kronig — antes del circuito, siempre

| | residuo RMS | rachas de signo | veredicto |
|---|---|---|---|
| E | 0.25 % | 0.71× | **pasa** |
| EA | 0.14 % | 0.88× | **pasa** |
| **EB cat. (−0.19 V)** | **6.57 %** | **0.24×** | **NO pasa** |
| **EB an. (+0.54 V)** | **0.73 %** | **0.72×** | **pasa** |
| EP | **1.65 %** | **0.41×** | **NO pasa** |

**Las dos filas de EB son el resultado más limpio de esta tabla.** Misma
muestra, misma celda, mismo barrido: lo único que cambia es el
potencial, y con él el veredicto. El fallo de Kramers-Kronig de EB no
era de la muestra ni del equipo — era la reacción.

Las rachas son lo que decide, no el residuo: un residuo del 1.5 % puede
ser ruido, pero un residuo que cambia de signo cuatro veces menos de lo
que cambiaría por azar es **estructura**, y la estructura significa que
el sistema no era el mismo al principio y al final del barrido.

> **Consecuencia práctica:** de **EB cat.** y **EP** se pueden publicar la
> forma, R_s, el codo y C′(f) —que son datos— y **no** los parámetros de
> ningún circuito ajustado a ellos. De **EB an.** sí.

## 3. La forma, medida sin ajustar nada

| | E | EA | EB cat. | **EB an.** | EP |
|---|---|---|---|---|---|
| R_s (Ω) | 68.12 | 47.10 | 85.01 | **85.22** | 58.81 |
| ¿cruza el eje real? | sí, interpolado | sí, interpolado | no | no | no |
| Z″ > 0 arriba (inductancia de cables) | sí | sí | no | no | no |
| ápice de arco | — | — | **0.032 Hz** | — | — |
| codo (fase = −45°), Hz | 10.86 | 41.0 | 46.88 | **25.16** | **712** |
| τ del codo (s) | 0.092 | 0.024 | 0.021 | 0.040 | **0.0014** |
| fase a 0.01 Hz (°) | −55.6 | **−71.1** | −20.1 | **−47.0** | −22.1 |
| tramo a 45° (décadas) | 1.83 | 0.50 | 1.00 | 1.00 | 0.67 |

Dos lecturas de la pareja de EB:

- **El ápice de arco a 0.032 Hz desaparece.** Ese arco, el único de las
  cinco medidas que se cerraba, era el proceso farádico. A +0.54 V no
  hay ninguno.
- **La fase a 0.01 Hz pasa de −20.1° a −47.0°.** El electrodo no había
  cambiado; lo que a −0.19 V era una resistencia de reacción en paralelo
  tapando la doble capa, a +0.54 V deja de estar.

Ninguna de las cinco cierra un semicírculo dentro del barrido salvo el
roce de EB cat., así que **ninguna resistencia de transferencia de carga
sale de los datos**: la que devuelva un ajuste sale del modelo. Y en EB
y EP, que no cruzan el eje real, **R_s es una cota superior**.

## 4. Capacitancias — todas las bases, lado a lado

`C′(0.01 Hz)` sale de los datos sin ajustar nada (`C = 1/(jωZ)`); el
resto pasa por un modelo, y todas las filas de arco salen del MISMO
modelo, el circuito común de dos arcos de la tabla 5.

| | E | EA | EB cat. | **EB an.** | EP |
|---|---|---|---|---|---|
| **C′(0.01 Hz)** (mF) | **0.827** | 0.162 | 0.0844 | **0.200** | 0.287 |
| C_s(0.01 Hz) (mF) | 1.214 | 0.181 | 0.713 | 0.375 | 2.024 |
| C del elemento de almacenamiento (mF) | 1.012 | 0.171 | — | — | — |
| C del arco rápido (Hsu-Mansfeld, mF) | 0.238 | 0.0992 | 0.0448 | 0.345 | 0.00714 |
| τ de ese arco (s) | 0.036 | 0.0017 | 0.0060 | **6.8** | 0.038 |
| C del arco lento (Hsu-Mansfeld, mF) | **1.331** | 0.216 | 0.0732 | 0.320 | 0.368 |
| τ de ese arco (s) | **86** | **145** | 5.3 | **24** | 7.3 |
| mF cm⁻² (área geométrica) | 11.70 | 2.29 | 1.19 | 2.84 | 4.06 |
| **F g⁻¹** (masa nominal 0.909 mg) | **0.910** | 0.178 | 0.0929 | **0.220** | 0.316 |
| fase a 0.01 Hz (°) | −55.6 | −71.1 | −20.1 | −47.0 | −22.1 |

Cuatro cosas que esta tabla dice:

1. **Las C′ son LÍMITES INFERIORES.** A 0.01 Hz la fase está entre −20°
   y −71° y un condensador está en −90°: el barrido no llegó al régimen
   capacitivo en ninguna. Bajar a 1 mHz es lo que las convierte en
   medidas.
2. **Los F/g son dos órdenes de magnitud bajos.** Un carbón poroso
   publica de 50 a 300 F/g; aquí el mejor da 0.91. Con 12.9 mg cm⁻² de
   carga, lo que estos espectros miden no es la capacitancia del
   material sino **la fracción de una película gruesa a la que el ion
   llega en 100 segundos**. El número es correcto; lo que mide no es lo
   que uno querría que midiera.
3. **A potencial comparable, EB no es la peor.** EB an. (+0.540 V) y EA
   (+0.525 V) están a 13 mV, y EB da **0.220 F/g contra 0.178** de EA. La
   lectura de que «EB almacena menos» era un artefacto del potencial.
4. **El arco lento es el que almacena**, y su C sigue el orden de C′.
   Pero **en E y EA está extrapolado**: τ = 86 y 145 s, o sea un ápice en
   1.9 y 1.1 mHz, dos décadas por debajo de lo medido.

La diferencia C_s − C′: 47 % en E, 12 % en EA, 745 % en EB cat., **87 %**
en EB an. y 606 % en EP. Donde la respuesta todavía es resistiva las dos
definiciones no se parecen — y el salto de 745 % a 87 % al cambiar el
potencial de EB es la misma historia por tercera vez.

## 5. Los ajustes — parámetros, con su incertidumbre

### 5.1 Por qué este circuito común

El automático elige el circuito que la forma de CADA espectro pide, que
es lo correcto para describir una muestra y lo incorrecto para comparar
cinco. Barrido el catálogo entero, el que gana en todas a la vez es
**`R0-(R1|Q1)-(R2|Q2)`**, dos arcos:

| | automático (por espectro) | residuo | **común, dos arcos** |
|---|---|---|---|
| E | R0-(R1\|Q1)-T1 | 2.38 % | **0.71 %** |
| EA | R0-T1 | 3.18 % | **0.54 %** |
| EB cat. | R0-(R1\|Q1) | 9.75 % | **5.79 %** |
| EB an. | R0-(R1\|Q1) | 3.05 % | **1.83 %** |
| EP | R0-(R1\|Q1) | **17.83 %** | **5.11 %** |

**Es el circuito de la figura y el de todas las columnas de arco de la
tabla 4.** La inductancia del cableado (`L0-R0-(R1|Q1)-(R2|Q2)`) se
ajusta ADEMÁS donde la forma la pide, no en lugar del común: en E no
cambia nada (0.706 % con y sin) y vuelve indeterminada (±468 %); en EA
paga, 0.541 → 0.497 % con los ocho parámetros determinados. Las dos
filas están en `ajustes.csv`.

### 5.2 El circuito común, parámetro a parámetro

| | E | EA | EB cat. | EB an. | EP |
|---|---|---|---|---|---|
| residuo medio (%) | **0.71** | **0.54** | 5.79 | **1.83** | 5.11 |
| R0 (Ω) | 68.35 ± 0.13 % | 46.87 ± 0.25 % | 84.64 ± 3.0 % | 85.29 ± 0.60 % | 56.98 ± 1.6 % |
| R1 (Ω) | 152 ± 3.3 % | 17.5 ± 7.5 % | 134 ± **74 %** | 19 671 ± **81 %** | 5 287 ± 7.8 % |
| Q1 (S·sⁿ) | 3.18·10⁻⁴ ± 2.8 % | 1.46·10⁻³ ± 18 % | 3.28·10⁻⁴ ± **88 %** | 1.85·10⁻⁴ ± 28 % | 1.16·10⁻⁵ ± 5.7 % |
| n₁ | 0.912 ± 0.90 % | 0.577 ± 4.1 % | 0.611 ± 20 % | 0.674 ± 4.1 % | 0.852 ± 2.1 % |
| R2 (Ω) | 64 542 ± 2.6 % | 6.70·10⁵ ± 2.8 % | 72 591 ± 5.6 % | 75 338 ± 27 % | 19 734 ± 9.9 % |
| Q2 (S·sⁿ) | 4.60·10⁻⁴ ± 0.27 % | 1.14·10⁻⁴ ± 0.12 % | 5.59·10⁻⁵ ± 2.5 % | 2.75·10⁻⁴ ± **49 %** | 1.97·10⁻⁴ ± 10 % |
| n₂ | 0.761 ± 0.27 % | 0.872 ± 0.07 % | 0.839 ± 1.9 % | 0.953 ± 8.4 % | 0.685 ± 6.6 % |
| **determinados** | **7/7** | **7/7** | 5/7 | 5/7 | **7/7** |

**Un residuo bajo no garantiza parámetros determinados.** EB an. ajusta
al 1.83 % y devuelve cinco de siete: sus dos arcos están en τ = 6.8 y
24 s, a un factor 3.5, y a esa distancia el reparto entre ellos lo
decide el punto de partida más que los datos. El residuo mide
descripción; la columna de la derecha mide medida.

**La Q de un CPE no son faradios** y n no es un número de nada: la
conversión está en la tabla 4 y necesita la resistencia en paralelo.

### 5.3 El circuito FÍSICO del electrodo poroso

Se ajusta además del común porque es el único que devuelve la
**resistencia iónica del poro**, `T1.Ri`, que es la magnitud que limita
la potencia de un electrodo poroso y la que un `R-(R|Q)` entierra.

| | circuito | residuo (%) | **T1.Ri (Ω)** |
|---|---|---|---|
| E | L0-R0-(R1\|Q1)-T1 | 5.19 | 127.8 ± **146 %** |
| **EA** | L0-R0-(R1\|Q1)-T1 | **0.51** | **43.87 ± 4.7 %** |
| EB cat. | R0-(R1\|Q1)-T1 | 6.32 | 65 564 ± **2·10⁸ %** |
| EB an. | R0-(R1\|Q1)-T1 | 1.80 | 1.4·10⁻³ ± **5·10⁸ %** |
| EP | R0-(R1\|Q1)-T1 | 12.67 | 38 574 ± **76 %** |

**Un parámetro con un 10⁸ % de incertidumbre no es una medida, es un
hueco**, y pasa cuando la línea de transmisión no tiene nada que
describir y colapsa. Ocurre en los dos EB, con buen residuo en uno de
ellos: **el residuo no lo delata, la incertidumbre sí.** Sólo EA
devuelve una resistencia de poro medida.

## 6. DRT — qué constantes de tiempo hay

λ elegido por la curva L. Sólo cuentan los picos dentro del intervalo
medido de τ (1.6 µs a 16 s).

| | τ (s) | R del pico (Ω) | C = τ/R (mF) | residuo DRT |
|---|---|---|---|---|
| **E** | 0.276 / 1.43 / 7.45 | 557 / 867 / 4 610 | 0.495 / 1.65 / 1.62 | 0.51 % |
| **EA** | 0.441 / 2.29 / 11.9 | 1 185 / 4 855 / 16 657 | 0.372 / 0.473 / 0.716 | 0.42 % |
| **EB cat.** | 1.43 / 5.88 | 20 452 / 46 307 | 0.070 / 0.127 | **9.97 %** |
| **EB an.** | 0.707 | 29 788 | 0.024 | 6.33 % |
| **EP** | 0.0671 | 17 876 | 0.0038 | 4.67 % |

Tres avisos antes de usar esta tabla:

1. **Los picos del extremo lento no son procesos**, son la acumulación
   de lo que no se midió: falta frecuencia por debajo de 0.01 Hz. Es
   también por lo que el panel (e) de la figura sale dominado por el
   borde derecho — ahí está el peso, y eso ES el resultado.
2. **La DRT no tiene picos donde hay un elemento distribuido.** Las
   cinco tienen entre 0.5 y 1.8 décadas de fase cerca de −45°, que es
   una línea de transmisión o un Warburg: la DRT los reparte en una
   joroba que esta tabla trocearía en arcos que no existen.
3. **La DRT de EB cat. no reproduce su espectro** (10.0 %), que es lo
   mismo que ya dijo Kramers-Kronig.

## 7. Los archivos para hacer los plots

| archivo | qué lleva |
|---|---|
| `espectro_E.csv`, `_EA`, `_EB_cat`, `_EB_an`, `_EP` | f, Z′, Z″, \|Z\|, fase, Z′ y Z″ **del ajuste común**, residuo %, C′, C″ |
| `drt_*.csv` | τ, γ(τ), y si ese τ cae dentro del intervalo medido |
| `comparacion.csv` | una fila por muestra: R_s, R_ion, codo, fase, C, circuito, residuo |
| `ajustes.csv` | todos los parámetros de todos los ajustes (común, con L, poroso, automático), con error absoluto y en % |
| `capacitancias.csv` | la tabla 4 entera, con RHE, los dos arcos y los F/g |
| `resumen.json` | todo lo anterior más la forma, el ranking de circuitos, KK y los avisos |
| `informe_*.txt` | el informe en prosa de cada espectro, con sus avisos |

**Z″ va con su signo FÍSICO** (negativo si es capacitivo) en todos los
CSV. El −Z″ del diagrama de Nyquist es convenio de dibujo; si tu script
espera el de EC-Lab, cámbiale el signo tú y que se vea.
