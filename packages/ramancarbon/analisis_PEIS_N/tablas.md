# PEIS de cuatro carbones Fe/Fe₃C según el precursor de nitrógeno — tablas

Etanol como fuente de carbono en las cuatro; lo que cambia es el
precursor que aporta el nitrógeno.

| clave | precursor de N | qué nitrógeno aporta |
|---|---|---|
| **E** | — (etanol solo) | ninguno: es el control |
| **EA** | acetonitrilo, CH₃CN | nitrilo, N sp, relación C/N baja |
| **EB** | bencilamina, PhCH₂NH₂ | amina primaria sp³ sobre un anillo |
| **EP** | piridina, C₅H₅N | N aromático, ya dentro de un anillo |

## 0. Las condiciones de celda, y cuáles son suposiciones

| | valor | de dónde |
|---|---|---|
| Electrolito | **H₂SO₄ 0.5 M** | **dado por el usuario** |
| pH | **0.3** | de la tabla del paquete, y sale de la **actividad**, no de −log c: el primer protón está disociado del todo y el segundo sólo en parte (Ka₂ = 0.012) |
| Electrodo | **disco de carbón vítreo de 3 mm**, 0.0707 cm² | **dado por el usuario**, pero es área **GEOMÉTRICA** |
| Referencia | Ag/AgCl saturado | **SUPUESTA**: el usuario no la nombró en esta serie, se usa la de la anterior |
| Masa | — | no dada, así que **no hay ni un F/g** en ninguna tabla |

Dos consecuencias que hay que tener delante al leer el resto:

- **La conversión a RHE depende de la referencia supuesta.**
  E(RHE) = E(Ag/AgCl sat) + 0.2147 V a pH 0.3. Si la referencia era
  otra, toda la columna RHE de la tabla 1 se desplaza en bloque.
- **El área activa de un carbón poroso es de diez a mil veces la
  geométrica.** Los mF/cm² de abajo sirven para comparar estas cuatro
  muestras entre sí y para nada más.

## 1. Las condiciones, leídas del propio archivo

| | E | EA | EB | EP |
|---|---|---|---|---|
| Polarización (V vs Ag/AgCl sat) | **+0.4489** | **+0.3105** | **−0.4043** | **+0.4853** |
| **Polarización (V vs RHE)** | **+0.664** | **+0.525** | **−0.190** | **+0.700** |
| Amplitud \|ΔE\| (mV) | 10.7 | 10.4 | 10.3 | 10.6 |
| Frecuencias | 43 | 43 | 43 | 43 |
| Barrido (Hz) | 10⁵ → 0.01 | 10⁵ → 0.01 | 10⁵ → 0.01 | 10⁵ → 0.01 |
| Duración (s) | 678 | 673 | 676 | 672 |
| Corriente media \|I\| (µA) | 0.354 | 0.265 | **15.9** | 0.538 |
| \|j\| con el área geométrica (µA cm⁻²) | 5.0 | 3.8 | **225** | 7.6 |

**Los cuatro espectros NO se midieron al mismo potencial.** Los nombres
dicen «PEIS 0 V», que en EC-Lab es 0 V *frente al potencial de circuito
abierto*, así que el continuo de cada medida es el OCP de su propia
muestra, y entre EB y EP hay **0.89 V**. Una resistencia de
transferencia de carga depende exponencialmente del sobrepotencial: a
esa distancia no se puede atribuir a la química lo que puede ser el
punto de polarización. Es la primera cosa que habría que rehacer.

**Y con el electrolito puesto, EB deja de ser una rareza y pasa a tener
nombre.** Su continuo está en **−0.190 V vs RHE**, es decir **por debajo
del equilibrio del hidrógeno**, con 190 mV de sobrepotencial catódico, y
pasa **0.225 mA cm⁻²**. Eso no es una doble capa medida con 10 mV: es
**desprendimiento de hidrógeno corriendo durante los once minutos del
barrido**, que es exactamente lo que hace derivar un espectro — y lo que
Kramers-Kronig dice en la tabla siguiente.

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
minutos por espectro es tiempo de sobra para que se mueva, y en EB
sabemos ya qué lo movía.

> **Consecuencia práctica, y no es un matiz:** de EB y EP se pueden
> publicar la forma, R_s, el codo y C′(f) —que son datos— y **no** los
> parámetros de ningún circuito ajustado a ellos. Un circuito flexible
> ajusta bien datos que han derivado y el χ² no lo delata.

## 3. La forma, medida sin ajustar nada

| | E | EA | EB | EP |
|---|---|---|---|---|
| R_s (Ω) | 68.12 | 47.10 | 85.01 | 58.81 |
| ¿cruza el eje real? | sí, interpolado | sí, interpolado | **no** | **no** |
| Z″ > 0 arriba (inductancia de cables) | **sí** | **sí** | no | no |
| ápice de arco | — | — | 0.032 Hz | — |
| codo (fase = −45°), Hz | 10.86 | 41.0 | 46.88 | **712** |
| τ del codo (s) | 0.092 | 0.024 | 0.021 | **0.0014** |
| fase a 0.01 Hz (°) | −55.6 | **−71.1** | −20.1 | −22.1 |
| tramo a 45° (décadas) | 1.83 | 0.50 | 1.00 | 0.67 |

Ninguno de los cuatro cierra un semicírculo dentro del barrido (EB roza
un ápice en el último punto), así que **ninguna resistencia de
transferencia de carga sale de los datos**: la que devuelva un ajuste
sale del modelo. Y en EB y EP, que no cruzan el eje real, **R_s es una
cota superior**.

## 4. Capacitancias — todas las bases, lado a lado

Unidades mF salvo donde se diga. `C′(0.01 Hz)` sale de los datos sin
ajustar nada (`C = 1/(jωZ)`); el resto pasa por un modelo, y desde esta
versión **todas las filas de arco salen del MISMO modelo**, el circuito
común de dos arcos de la tabla 5.

| | E | EA | EB | EP |
|---|---|---|---|---|
| **C′(0.01 Hz)** | **0.827** | 0.162 | 0.0844 | 0.287 |
| C_s(0.01 Hz) | 1.214 | 0.181 | 0.713 | 2.024 |
| C del elemento de almacenamiento | 1.012 | 0.171 | — | — |
| C del arco RÁPIDO (Hsu-Mansfeld) | 0.238 | 0.0992 | 0.0448 | 0.00714 |
| τ de ese arco (s) | 0.036 | 0.0017 | 0.0060 | 0.038 |
| C del arco LENTO (Hsu-Mansfeld) | **1.331** | 0.216 | 0.0732 | 0.368 |
| τ de ese arco (s) | **86** | **145** | 5.3 | 7.3 |
| C del arco rápido (Brug) | 0.212 | 0.0786 | 0.0245 | 0.00325 |
| mF cm⁻² (área **geométrica**) | 11.70 | 2.29 | 1.19 | 4.06 |
| fase a 0.01 Hz (°) | −55.6 | −71.1 | −20.1 | −22.1 |

Tres cosas que esta tabla dice y que no se ven en una sola fila:

1. **Las cuatro C′ son LÍMITES INFERIORES.** A 0.01 Hz la fase está
   entre −20° y −71° y un condensador está en −90°: el barrido no llegó
   al régimen capacitivo en ninguna de las cuatro. Bajar a 1 mHz es lo
   que convierte estos números en medidas.
2. **El arco LENTO es el que almacena**, y su capacitancia sigue el
   mismo orden que C′ (1.33 / 0.22 / 0.07 / 0.37 frente a 0.83 / 0.16 /
   0.08 / 0.29). Es la comprobación de que el circuito común no está
   inventando el almacenamiento.
3. **Pero en E y EA ese arco está EXTRAPOLADO.** Su τ vale 86 y 145 s,
   o sea un ápice en 1.9 y 1.1 mHz: dos décadas por debajo de la
   frecuencia más baja medida. R₂ y la C del arco lento de esas dos
   muestras son del modelo, no de la medida. En EB y EP (τ = 5.3 y
   7.3 s) el ápice cae justo en el borde del barrido.

La diferencia C_s − C′ mide lo mismo por otro lado: 47 % en E, 12 % en
EA, **745 %** en EB y **606 %** en EP. Donde la respuesta todavía es
resistiva las dos definiciones no se parecen en nada, y esa es
exactamente la situación de EB y EP.

## 5. Los ajustes — parámetros, con su incertidumbre

### 5.1 Por qué este circuito común

El automático elige el circuito que la forma de CADA espectro pide, y
eso es lo correcto para describir una muestra y lo incorrecto para
comparar cuatro: los parámetros de dos circuitos distintos no son los
mismos parámetros. Barrido el catálogo entero sobre las cuatro, el que
gana en las cuatro a la vez es **`R0-(R1|Q1)-(R2|Q2)`**, dos arcos:

| | automático (por espectro) | residuo | **común, dos arcos** | residuo |
|---|---|---|---|---|
| E | R0-(R1\|Q1)-T1 | 2.38 % | R0-(R1\|Q1)-(R2\|Q2) | **0.71 %** |
| EA | R0-T1 | 3.18 % | R0-(R1\|Q1)-(R2\|Q2) | **0.54 %** |
| EB | R0-(R1\|Q1) | 9.75 % | R0-(R1\|Q1)-(R2\|Q2) | **5.79 %** |
| EP | R0-(R1\|Q1) | 17.83 % | R0-(R1\|Q1)-(R2\|Q2) | **5.11 %** |

El 17.8 % de EP es lo que se veía en la figura anterior como «la línea
no sigue a los puntos». **Es el circuito de la figura y el de todas las
columnas de arco de la tabla 4.**

La inductancia del cableado (`L0-R0-(R1|Q1)-(R2|Q2)`) se ajusta ADEMÁS
donde la forma la pide, no en lugar del común: en E no cambia nada
(0.706 % con y sin) y vuelve indeterminada (±468 %); en EA paga,
0.541 → 0.497 % con los ocho parámetros determinados. Las dos filas
están en `ajustes.csv`.

### 5.2 El circuito común, parámetro a parámetro

| | E | EA | EB | EP |
|---|---|---|---|---|
| residuo medio (%) | **0.71** | **0.54** | 5.79 | 5.11 |
| R0 (Ω) | 68.35 ± 0.13 % | 46.87 ± 0.25 % | 84.64 ± 3.0 % | 56.98 ± 1.6 % |
| R1 (Ω) | 151.9 ± 3.3 % | 17.53 ± 7.5 % | 133.6 ± **74 %** | 5 287 ± 7.8 % |
| Q1 (S·sⁿ) | 3.18·10⁻⁴ ± 2.8 % | 1.46·10⁻³ ± 18 % | 3.28·10⁻⁴ ± **88 %** | 1.16·10⁻⁵ ± 5.7 % |
| n₁ | 0.912 ± 0.90 % | 0.577 ± 4.1 % | 0.611 ± 20 % | 0.852 ± 2.1 % |
| R2 (Ω) | 64 542 ± 2.6 % | 6.70·10⁵ ± 2.8 % | 72 591 ± 5.6 % | 19 734 ± 9.9 % |
| Q2 (S·sⁿ) | 4.60·10⁻⁴ ± 0.27 % | 1.14·10⁻⁴ ± 0.12 % | 5.59·10⁻⁵ ± 2.5 % | 1.97·10⁻⁴ ± 10 % |
| n₂ | 0.761 ± 0.27 % | 0.872 ± 0.07 % | 0.839 ± 1.9 % | 0.685 ± 6.6 % |
| parámetros determinados | **7/7** | **7/7** | 5/7 | **7/7** |

**La Q de un CPE no son faradios** y n no es un número de nada: la
conversión está en la tabla 4 y necesita la resistencia en paralelo.

### 5.3 El circuito FÍSICO del electrodo poroso

Se ajusta además del común porque es el único que devuelve la
**resistencia iónica del poro**, `T1.Ri`, que es la magnitud que limita
la potencia de un electrodo poroso y la que un `R-(R|Q)` entierra.

| | circuito | residuo (%) | **T1.Ri (Ω)** | T1.n |
|---|---|---|---|---|
| E | L0-R0-(R1\|Q1)-T1 | 5.19 | 127.8 ± **146 %** | 1.000 ± 19 % |
| EA | L0-R0-(R1\|Q1)-T1 | **0.51** | **43.87 ± 4.7 %** | 0.895 ± 2.6 % |
| EB | R0-(R1\|Q1)-T1 | 6.32 | 65 564 ± **2·10⁸ %** | 0.969 ± 23 % |
| EP | R0-(R1\|Q1)-T1 | 12.67 | 38 574 ± **76 %** | 1.000 ± 3.2 % |

**Un parámetro con un 10⁸ % de incertidumbre no es una medida, es un
hueco**, y pasa cuando la línea de transmisión no tiene nada que
describir en ese espectro y colapsa. **Sólo EA devuelve una resistencia
de poro medida.** Todos los parámetros de los cuatro ajustes, con error
absoluto y porcentual, están en `ajustes.csv`.

## 6. DRT — qué constantes de tiempo hay

λ elegido por la curva L. Sólo cuentan los picos dentro del intervalo
medido de τ (1.6 µs a 16 s).

| | τ (s) | R del pico (Ω) | C = τ/R (mF) |
|---|---|---|---|
| **E** | 0.276 | 557 | 0.495 |
| | 1.43 | 867 | 1.65 |
| | 7.45 | 4 610 | 1.62 |
| **EA** | 0.441 | 1 185 | 0.372 |
| | 2.29 | 4 855 | 0.473 |
| | 11.9 | 16 657 | 0.716 |
| **EB** | 1.43 | 20 452 | 0.070 |
| | 5.88 | 46 307 | 0.127 |
| **EP** | 0.0671 | 17 876 | 0.0038 |

Tres avisos que hay que leer antes de usar esta tabla:

1. **Los picos del extremo lento no son procesos**, son la acumulación
   de lo que no se midió: falta frecuencia por debajo de 0.01 Hz y la
   difusión suele estar ahí. Es también por lo que el panel (e) de la
   figura sale dominado por el borde derecho — ahí está el peso, y eso
   ES el resultado. Afecta a E, EA y EP.
2. **La DRT no tiene picos donde hay un elemento distribuido.** Las
   cuatro tienen entre 0.5 y 1.8 décadas de fase cerca de −45°, que es
   una línea de transmisión o un Warburg: la DRT los reparte en una
   joroba que esta tabla trocearía en arcos que no existen.
3. **La DRT de EB no reproduce su espectro** (residuo 10.0 %), que es lo
   mismo que ya dijo Kramers-Kronig.

## 7. Los archivos para hacer los plots

| archivo | qué lleva |
|---|---|
| `espectro_E.csv`, `_EA`, `_EB`, `_EP` | f, Z′, Z″, \|Z\|, fase, Z′ y Z″ **del ajuste común**, residuo %, C′, C″ |
| `drt_E.csv`, … | τ, γ(τ), y si ese τ cae dentro del intervalo medido |
| `comparacion.csv` | una fila por muestra: R_s, R_ion, codo, fase, C, circuito, residuo |
| `ajustes.csv` | todos los parámetros de todos los ajustes (común, con L, poroso, automático), con error absoluto y en % |
| `capacitancias.csv` | la tabla 4 entera, con la columna RHE y los dos arcos |
| `resumen.json` | todo lo anterior más la forma, el ranking de circuitos, KK y los avisos |
| `informe_*.txt` | el informe en prosa de cada muestra, con sus avisos |

**Z″ va con su signo FÍSICO** (negativo si es capacitivo) en todos los
CSV. El −Z″ del diagrama de Nyquist es convenio de dibujo; si tu script
espera el de EC-Lab, cámbiale el signo tú y que se vea.
