# PEIS de cuatro carbones Fe/Fe₃C según el precursor de nitrógeno — tablas

Etanol como fuente de carbono en las cuatro; lo que cambia es el
precursor que aporta el nitrógeno.

| clave | precursor de N | qué nitrógeno aporta |
|---|---|---|
| **E** | — (etanol solo) | ninguno: es el control |
| **EA** | acetonitrilo, CH₃CN | nitrilo, N sp, relación C/N baja |
| **EB** | bencilamina, PhCH₂NH₂ | amina primaria sp³ sobre un anillo |
| **EP** | piridina, C₅H₅N | N aromático, ya dentro de un anillo |

Lo que el usuario **no** dio y hace falta para leer estas tablas:

- **El electrolito.** No se supone ninguno. Nada de lo medido depende de
  él; la interpretación sí, y lo dice donde toca.
- **La masa.** No hay ni un F/g aquí. Una capacitancia específica con una
  masa supuesta está mal exactamente en esa proporción.
- **El área.** Se usa el disco de carbón vítreo de 3 mm que el usuario
  emplea casi siempre: **0.0707 cm², suposición**, y además GEOMÉTRICA.
  La electroquímicamente activa de un carbón poroso es de diez a mil
  veces mayor, así que los mF/cm² de abajo sirven para comparar estas
  cuatro muestras entre sí y para nada más.

## 1. Las condiciones, leídas del propio archivo

| | E | EA | EB | EP |
|---|---|---|---|---|
| Polarización (V, continuo) | **+0.4489** | **+0.3105** | **−0.4043** | **+0.4853** |
| Amplitud \|ΔE\| (mV) | 10.67 | 10.39 | 10.30 | 10.62 |
| Frecuencias | 43 | 43 | 43 | 43 |
| Barrido (Hz) | 10⁵ → 0.01 | 10⁵ → 0.01 | 10⁵ → 0.01 | 10⁵ → 0.01 |
| Duración (s) | 678 | 673 | 676 | 672 |
| Corriente media \|I\| (µA) | 0.354 | 0.265 | **15.9** | 0.538 |

**Los cuatro espectros NO se midieron al mismo potencial.** Los nombres
dicen «PEIS 0 V», que en EC-Lab es 0 V *frente al potencial de circuito
abierto*, así que el continuo de cada medida es el OCP de su propia
muestra, y entre EB y EP hay **0.89 V**. Una resistencia de
transferencia de carga depende exponencialmente del sobrepotencial: a
esa distancia no se puede atribuir a la química lo que puede ser el
punto de polarización. Es la primera cosa que habría que rehacer.

**Y la corriente de EB es treinta a sesenta veces la de las otras tres.**
Con la misma amplitud de 10 mV, eso no es ruido: a −0.40 V hay un
proceso farádico corriendo en esa celda.

## 2. Kramers-Kronig — antes del circuito, siempre

| | residuo RMS | rachas de signo | veredicto |
|---|---|---|---|
| E | 0.25 % | 0.71× | **pasa** |
| EA | 0.14 % | 0.88× | **pasa** |
| EB | **6.57 %** | **0.24×** | **NO pasa** |
| EP | **1.65 %** | **0.41×** | **NO pasa** |

Las rachas son lo que decide, no el residuo: un residuo del 1.5 % puede
ser ruido, pero un residuo que cambia de signo cuatro veces menos de lo
que cambiaría por azar es **estructura**, y la estructura significa que
el sistema no era el mismo al principio y al final del barrido. Once
minutos por espectro es tiempo de sobra para que se mueva.

> **Consecuencia práctica, y no es un matiz:** de EB y EP se pueden
> publicar la forma, R_s, el codo y C′(f) —que son datos— y **no** los
> parámetros de ningún circuito ajustado a ellos. Un circuito flexible
> ajusta bien datos que han derivado y el χ² no lo delata.

## 3. La forma, medida sin ajustar nada

| | E | EA | EB | EP |
|---|---|---|---|---|
| R_s (Ω) | 68.12 | 47.10 | 85.01 | 58.81 |
| ¿cruza el eje real? | no | no | no | no |
| Z″ > 0 arriba (inductancia de cables) | **sí**, 2 pts | **sí**, 1 pt | no | no |
| ápice de arco | — | — | 0.032 Hz | — |
| codo (fase = −45°), Hz | 10.86 | 41.0 | 46.88 | **712** |
| τ del codo (s) | 0.092 | 0.024 | 0.021 | **0.0014** |
| fase a 0.01 Hz (°) | −55.6 | **−71.1** | −20.1 | −22.1 |
| tramo a 45° (décadas) | 1.83 | 0.50 | 1.00 | 0.67 |

Ninguno de los cuatro cierra un semicírculo dentro del barrido (EB roza
un ápice en el último punto), así que **ninguna resistencia de
transferencia de carga sale de los datos**: la que devuelva un ajuste
sale del modelo. Y R_s es una **cota superior** en los cuatro, porque
ninguno cruza el eje real.

## 4. Capacitancias — todas las bases, lado a lado

Unidades mF salvo donde se diga. `C′(0.01 Hz)` sale de los datos sin
ajustar nada (`C = 1/(jωZ)`); el resto pasa por un modelo.

| | E | EA | EB | EP |
|---|---|---|---|---|
| **C′(0.01 Hz)** | **0.827** | 0.162 | 0.0844 | 0.287 |
| C_s(0.01 Hz) | 1.214 | 0.181 | 0.713 | 2.024 |
| C del elemento de almacenamiento | 1.012 | 0.171 | — | — |
| C del arco (Hsu-Mansfeld) | 0.351 | — | 0.0893 | 0.0128 |
| C del arco (Brug) | 0.0997 | 0.0599 | 0.0260 | 0.00473 |
| mF cm⁻² (área **supuesta**, geométrica) | 11.70 | 2.29 | 1.19 | 4.06 |
| fase a esa frecuencia (°) | −55.6 | −71.1 | −20.1 | −22.1 |

**Las cuatro C′ son LÍMITES INFERIORES.** A 0.01 Hz la fase está entre
−20° y −71° y un condensador está en −90°: el barrido no llegó al
régimen capacitivo en ninguna de las cuatro. Bajar a 1 mHz es lo que
convierte estos números en medidas.

La diferencia C_s − C′ mide lo mismo por otro lado: 47 % en E, 12 % en
EA, **745 %** en EB y **606 %** en EP. Donde la respuesta todavía es
resistiva las dos definiciones no se parecen en nada, y esa es
exactamente la situación de EB y EP.

## 5. Los ajustes — parámetros, con su incertidumbre

Circuito común a todas: `R0-(R1|Q1)-T1`, y `L0-R0-(R1|Q1)-T1` donde el
extremo alto se va inductivo (E, EA). `T` es la línea de transmisión de
De Levie: `T.Ri` es la **resistencia iónica del poro**, que es la que
limita la potencia del electrodo.

### 5.1 Circuito común

| | E | EA | EB | EP |
|---|---|---|---|---|
| circuito | L0-R0-(R1\|Q1)-T1 | L0-R0-(R1\|Q1)-T1 | R0-(R1\|Q1)-T1 | R0-(R1\|Q1)-T1 |
| residuo medio (%) | 5.19 | **0.51** | 6.32 | 12.67 |
| L0 (µH) | 1.11 ± 308 % | 0.972 ± 27 % | — | — |
| R0 (Ω) | 65.67 ± 1.4 % | **46.48 ± 0.2 %** | 82.08 ± 3.7 % | 48.70 ± 5.5 % |
| R1 (Ω) | 64 070 ± 147 % | 502 371 ± 8.6 % | 64 900 ± 17 % | 5 671 ± 6.7 % |
| Q1 (S·sⁿ) | 4.62·10⁻⁴ ± 4.9 % | 1.30·10⁻⁴ ± 2.4 % | 6.07·10⁻⁵ ± 8.7 % | 1.11·10⁻⁵ ± 12 % |
| n₁ | 0.695 ± 3.0 % | 0.868 ± 0.4 % | 0.862 ± 4.0 % | 0.898 ± 3.2 % |
| **T1.Ri (Ω)** | 127.8 ± **146 %** | **43.87 ± 4.7 %** | 65 564 ± **2·10⁸ %** | 38 574 ± 76 % |
| T1.Q (S·sⁿ) | 6.05·10⁻³ ± 100 % | 9.28·10⁻⁴ ± 18 % | 3.60·10⁻² ± 2·10⁸ % | 3.40·10⁻³ ± 84 % |
| T1.n | 1.000 ± 19 % | 0.895 ± 2.6 % | 0.968 ± 23 % | 1.000 ± 3.2 % |

### 5.2 El circuito que la forma de cada una pide

| | circuito | residuo (%) | lo que sale determinado |
|---|---|---|---|
| E | R0-(R1\|Q1)-T1 | 2.38 | R1, Q1, n₁, T1.Q, T1.n; **T1.Ri = 1.5·10⁻⁵ Ω ± 4·10⁸ %** |
| EA | R0-T1 | 3.18 | todo: R0 ± 1 %, T1.Ri = **21.9 Ω ± 11 %**, T1.Q ± 0.6 %, T1.n ± 0.2 % |
| EB | R0-(R1\|Q1) | 9.75 | los cuatro, pero el espectro **no pasa KK** |
| EP | R0-(R1\|Q1) | 17.83 | los cuatro, pero el espectro **no pasa KK** |

**Un parámetro con un 10⁸ % de incertidumbre no es una medida, es un
hueco.** Pasa en dos sitios y por la misma razón: la línea de
transmisión no tiene nada que describir en ese espectro y colapsa.

## 6. DRT — qué constantes de tiempo hay

λ elegido por la curva L. Sólo cuentan los picos dentro del intervalo
medido de τ.

| | τ (s) | R del pico (Ω) | C = τ/R (mF) |
|---|---|---|---|
| **E** | 0.276 | 557 | 0.495 |
| | 1.43 | 867 | 1.65 |
| | 7.45 | 4 610 | 1.62 |
| **EA** | 0.441 | 1 185 | 0.372 |
| | 2.29 | 4 855 | 0.473 |
| | 11.9 | 16 660 | 0.716 |
| **EB** | 1.43 | 20 450 | 0.070 |
| | 5.88 | 46 310 | 0.127 |
| **EP** | 0.0671 | 17 880 | 0.0038 |

Tres avisos que hay que leer antes de usar esta tabla:

1. **Los picos del extremo lento no son procesos**, son la acumulación
   de lo que no se midió: falta frecuencia por debajo de 0.01 Hz y la
   difusión suele estar ahí. Afecta a E, EA y EP.
2. **La DRT no tiene picos donde hay un elemento distribuido.** Las
   cuatro tienen entre 0.5 y 1.8 décadas de fase cerca de −45°, que es
   una línea de transmisión o un Warburg: la DRT los reparte en una
   joroba que esta tabla trocearía en arcos que no existen.
3. **La DRT de EB no reproduce su espectro** (residuo 10.0 %), que es lo
   mismo que ya dijo Kramers-Kronig.

## 7. Los archivos para hacer los plots

| archivo | qué lleva |
|---|---|
| `espectro_E.csv`, `_EA`, `_EB`, `_EP` | f, Z′, Z″, \|Z\|, fase, Z′ y Z″ **del ajuste**, residuo %, C′, C″ |
| `drt_E.csv`, … | τ, γ(τ), y si ese τ cae dentro del intervalo medido |
| `comparacion.csv` | una fila por muestra: R_s, R_ion, codo, fase, C, circuito, residuo |
| `ajustes.csv` | todos los parámetros de todos los ajustes, con error absoluto y en % |
| `capacitancias.csv` | la tabla 4 entera, en crudo |
| `resumen.json` | todo lo anterior más la forma, el ranking de circuitos, KK y los avisos |
| `informe_*.txt` | el informe en prosa de cada muestra, con sus avisos |

**Z″ va con su signo FÍSICO** (negativo si es capacitivo) en todos los
CSV. El −Z″ del diagrama de Nyquist es convenio de dibujo; si tu script
espera el de EC-Lab, cámbiale el signo tú y que se vea.
