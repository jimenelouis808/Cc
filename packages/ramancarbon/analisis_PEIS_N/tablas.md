# PEIS de cuatro carbones Fe/Fe₃C según el precursor de nitrógeno — tablas

Etanol como fuente de carbono en las cuatro; lo que cambia es el
precursor que aporta el nitrógeno.

| clave | precursor de N | qué nitrógeno aporta |
|---|---|---|
| **E** | — (etanol solo) | ninguno: es el control |
| **EA** | acetonitrilo, CH₃CN | nitrilo, N sp, relación C/N baja |
| **EB** | bencilamina, PhCH₂NH₂ | amina primaria sp³ sobre un anillo |
| **EP** | piridina, C₅H₅N | N aromático, ya dentro de un anillo |

**El EB que entra aquí es el medido a +0.32 V vs Ag/AgCl** (+0.54 V vs
RHE), dentro de la ventana de las otras tres. El primero, a −0.40 V
(−0.19 V vs RHE, por debajo del equilibrio del hidrógeno, con
0.225 mA cm⁻² de HER corriendo durante los once minutos del barrido) no
pasa Kramers-Kronig y **se ha retirado del análisis**. Las dos medidas
—misma muestra, misma celda, R_s 85.01 y 85.22 Ω— sirvieron para
demostrar que el fallo de KK era el potencial y no la muestra; hecho
eso, el espectro que se compara es el bueno.

## 0. Las condiciones de celda

Todas dadas por el usuario.

| | valor | nota |
|---|---|---|
| Electrolito | **H₂SO₄ 0.5 M** | |
| pH | **0.3** | de la tabla del paquete, y sale de la **actividad**, no de −log c |
| Electrodo | **disco de carbón vítreo de 3 mm**, 0.0707 cm² | área **GEOMÉTRICA** |
| Referencia | **Ag/AgCl saturado** | **confirmada**. E(RHE) = E + 0.2147 V a pH 0.3 |
| Masa | **0.909 mg** | 20 µL de una tinta de 5 mg en 100 µL de isopropanol + 10 µL de Nafion: 20/110 × 5 mg |
| Carga | **12.9 mg cm⁻²** | **de diez a cien veces lo habitual sobre un carbón vítreo** |

- **La masa es NOMINAL.** Sale del volumen depositado, no de pesar el
  electrodo seco.
- **12.9 mg cm⁻² es una película gruesa**, de decenas de micras. El ion
  no la atraviesa en los 100 s de un punto a 0.01 Hz, y eso explica por
  sí solo que ninguna muestra llegue al régimen capacitivo y que los
  F/g de la tabla 4 salgan dos órdenes por debajo de lo publicado.

## 1. Las condiciones, leídas de los propios archivos

| | E | EA | EB | EP |
|---|---|---|---|---|
| Polarización (V vs Ag/AgCl sat) | +0.4489 | +0.3105 | +0.3249 | +0.4853 |
| **Polarización (V vs RHE)** | **+0.664** | **+0.525** | **+0.540** | **+0.700** |
| Amplitud \|ΔE\| (mV) | 10.7 | 10.4 | 10.3 | 10.6 |
| Frecuencias | 43 | 43 | 43 | 43 |
| Barrido (Hz) | 10⁵ → 0.01 | 10⁵ → 0.01 | 10⁵ → 0.01 | 10⁵ → 0.01 |
| Duración (s) | 678 | 673 | 674 | 672 |
| Corriente media \|I\| (µA) | 0.354 | 0.265 | 2.47 | 0.538 |
| \|j\| geométrica (µA cm⁻²) | 5.0 | 3.8 | 34.9 | 7.6 |

**Las cuatro caen ahora dentro de 0.18 V** (+0.525 a +0.700 V vs RHE), y
**EB y EA están a 13 mV**: ésa es una comparación directa. En E, EA y EP
el continuo es el OCP de cada muestra —sus nombres dicen «PEIS 0 V», que
en EC-Lab es 0 V *frente al circuito abierto*— y en EB es un potencial
**aplicado**. La etiqueta de la figura no distingue las dos cosas porque
no cabe; esta tabla sí.

## 2. Kramers-Kronig — antes del circuito, siempre

| | residuo RMS | rachas de signo | veredicto |
|---|---|---|---|
| E | 0.25 % | 0.71× | **pasa** |
| EA | 0.14 % | 0.88× | **pasa** |
| EB | 0.73 % | 0.72× | **pasa** |
| **EP** | **1.65 %** | **0.41×** | **NO pasa** |

Las rachas son lo que decide, no el residuo: un residuo del 1.5 % puede
ser ruido, pero un residuo que cambia de signo menos de la mitad de lo
que cambiaría por azar es **estructura**, y la estructura significa que
el sistema no era el mismo al principio y al final del barrido.

> **De EP se pueden publicar la forma, R_s, el codo y C′(f) —que son
> datos— y NO los parámetros de ningún circuito ajustado a él.**

## 3. La forma, medida sin ajustar nada

| | E | EA | EB | EP |
|---|---|---|---|---|
| R_s (Ω) | 68.12 | 47.10 | 85.22 | 58.81 |
| ¿cruza el eje real? | sí, interpolado | sí, interpolado | no | no |
| Z″ > 0 arriba (inductancia de cables) | **sí** | **sí** | no | no |
| ápice de arco | — | — | — | — |
| codo (fase = −45°), Hz | 10.86 | 41.0 | 25.16 | **712** |
| τ del codo (s) | 0.092 | 0.024 | 0.040 | **0.0014** |
| fase a 0.01 Hz (°) | −55.6 | **−71.1** | −47.0 | −22.1 |
| tramo a 45° (décadas) | 1.83 | 0.50 | 1.00 | 0.67 |

**Ninguna de las cuatro cierra un semicírculo dentro del barrido**, así
que **ninguna resistencia de transferencia de carga sale de los datos**:
la que devuelva un ajuste sale del modelo. Y en EB y EP, que no cruzan
el eje real, **R_s es una cota superior**.

## 4. Capacitancias — todas las bases, lado a lado

`C′(0.01 Hz)` sale de los datos sin ajustar nada (`C = 1/(jωZ)`); el
resto pasa por el circuito común de dos arcos de la tabla 5.

| | E | EA | EB | EP |
|---|---|---|---|---|
| **C′(0.01 Hz)** (mF) | **0.827** | 0.162 | 0.200 | 0.287 |
| C_s(0.01 Hz) (mF) | 1.214 | 0.181 | 0.375 | 2.024 |
| C del elemento de almacenamiento (mF) | 1.012 | 0.171 | — | — |
| C del arco rápido (Hsu-Mansfeld, mF) | 0.238 | 0.0992 | 0.345 | 0.00714 |
| τ de ese arco (s) | 0.036 | 0.0017 | **6.8** | 0.038 |
| C del arco lento (Hsu-Mansfeld, mF) | **1.331** | 0.216 | 0.320 | 0.368 |
| τ de ese arco (s) | **86** | **145** | **24** | 7.3 |
| mF cm⁻² (área geométrica) | 11.70 | 2.29 | 2.84 | 4.06 |
| **F g⁻¹** (masa nominal 0.909 mg) | **0.910** | 0.178 | **0.220** | 0.316 |
| fase a 0.01 Hz (°) | −55.6 | −71.1 | −47.0 | −22.1 |

Cuatro cosas que esta tabla dice:

1. **Las C′ son LÍMITES INFERIORES.** A 0.01 Hz la fase está entre −22°
   y −71° y un condensador está en −90°: el barrido no llegó al régimen
   capacitivo en ninguna.
2. **Los F/g son dos órdenes de magnitud bajos.** Un carbón poroso
   publica de 50 a 300 F/g; aquí el mejor da 0.91. Con 12.9 mg cm⁻² de
   carga, lo que estos espectros miden es **la fracción de la película a
   la que el ion llega en 100 segundos**, no la capacitancia del
   material.
3. **A potencial comparable, EB está por encima de EA.** +0.540 contra
   +0.525 V vs RHE —13 mV— y 0.220 contra 0.178 F/g.
4. **Los arcos lentos de E y EA están EXTRAPOLADOS**: τ = 86 y 145 s, o
   sea ápices en 1.9 y 1.1 mHz, dos décadas por debajo de lo medido. Los
   de EB (24 s) y EP (7.3 s) caen en el borde del barrido o dentro.

La diferencia C_s − C′: 47 % en E, 12 % en EA, **87 %** en EB y **606 %**
en EP. Donde la respuesta todavía es resistiva, las dos definiciones no
se parecen.

## 5. Los ajustes

### 5.1 El circuito común, y por qué no se cambió por uno mejor

| | automático (por espectro) | residuo | **común, dos arcos** |
|---|---|---|---|
| E | R0-(R1\|Q1)-T1 | 2.38 % | **0.71 %** |
| EA | R0-T1 | 3.18 % | **0.54 %** |
| EB | R0-(R1\|Q1) | 3.05 % | **1.83 %** |
| EP | R0-(R1\|Q1) | **17.83 %** | **5.11 %** |

Barrido el catálogo entero buscando bajar el residuo de EP y de EB,
**tres arcos** (`R0-(R1|Q1)-(R2|Q2)-(R3|Q3)`) los baja a 2.50 % y 1.33 %.
**No se ha adoptado, y la razón es una medida, no una preferencia.**

`residual_against_kk` mide **dónde** cae el residuo del ajuste respecto
al de Kramers-Kronig: si falta un elemento, las frecuencias donde el
circuito falla no tienen por qué ser aquéllas donde los datos no son
consistentes consigo mismos, y la fracción sale cerca de la mitad; si lo
que queda es deriva, las dos coinciden.

| | circuito | residuo | share | correlación | |
|---|---|---|---|---|---|
| **E** | dos arcos | 0.71 % | 0.51 | −0.02 | modelo |
| | tres arcos | 0.66 % | 0.48 | −0.00 | modelo |
| **EA** | dos arcos | 0.54 % | 0.65 | +0.46 | modelo |
| | tres arcos | **0.25 %** | 0.60 | +0.31 | modelo |
| **EB** | dos arcos | 1.83 % | **0.77** | **+0.77** | **deriva** |
| | tres arcos | 1.33 % | **0.78** | **+0.72** | **deriva** |
| **EP** | dos arcos | 5.11 % | 0.64 | +0.62 | **deriva** |
| | tres arcos | 2.50 % | **0.77** | **+0.79** | **deriva** |

**En EP el arco nuevo no encontró un proceso: siguió la deriva.** El
residuo se parte por dos mientras el share sube de 0.64 a 0.77 y la
correlación de +0.62 a +0.79, es decir el ajuste pasa a seguir aún más
de cerca justo la parte del espectro que Kramers-Kronig dice que no es
fiable. **En EB** sus dos arcos lentos salen a τ = 2.3 y 4.3 s, un
factor **1.9** — por debajo del factor 3 que este paquete llama
degenerado — con R a ±128 % y ±225 %.

> **Conclusión: el ajuste de EP y de EB no se puede mejorar cambiando de
> circuito.** Lo que queda no es error de modelo. Lo que lo arregla es
> volver a medir. (El umbral está calibrado: cuarenta espectros
> sintéticos con el residuo puramente de modelo dan share 0.52 ± 0.05,
> máximo 0.61, y correlación +0.15 ± 0.19, máximo +0.53.)

Los tres ajustes de cada muestra —automático, común y tres arcos— están
completos en `ajustes.csv`.

### 5.2 El circuito común, parámetro a parámetro

`R0-(R1|Q1)-(R2|Q2)` en las cuatro.

| | E | EA | EB | EP |
|---|---|---|---|---|
| residuo medio (%) | **0.71** | **0.54** | 1.83 | 5.11 |
| R0 (Ω) | 68.35 ± 0.13 % | 46.87 ± 0.25 % | 85.29 ± 0.60 % | 56.98 ± 1.6 % |
| R1 (Ω) | 152 ± 3.3 % | 17.5 ± 7.5 % | 19 671 ± **81 %** | 5 287 ± 7.8 % |
| Q1 (S·sⁿ) | 3.18·10⁻⁴ ± 2.8 % | 1.46·10⁻³ ± 18 % | 1.85·10⁻⁴ ± 28 % | 1.16·10⁻⁵ ± 5.7 % |
| n₁ | 0.912 ± 0.90 % | 0.577 ± 4.1 % | 0.674 ± 4.1 % | 0.852 ± 2.1 % |
| R2 (Ω) | 64 542 ± 2.6 % | 6.70·10⁵ ± 2.8 % | 75 338 ± 27 % | 19 734 ± 9.9 % |
| Q2 (S·sⁿ) | 4.60·10⁻⁴ ± 0.27 % | 1.14·10⁻⁴ ± 0.12 % | 2.75·10⁻⁴ ± **49 %** | 1.97·10⁻⁴ ± 10 % |
| n₂ | 0.761 ± 0.27 % | 0.872 ± 0.07 % | 0.953 ± 8.4 % | 0.685 ± 6.6 % |
| **determinados** | **7/7** | **7/7** | 6/7 | **7/7** |

**Un residuo bajo no garantiza parámetros determinados.** EB ajusta al
1.83 % y devuelve seis de siete: sus dos arcos están a un factor 3.5 en
τ, y a esa distancia el reparto lo decide el punto de partida más que
los datos.

**La Q de un CPE no son faradios** y n no es un número de nada: la
conversión está en la tabla 4 y necesita la resistencia en paralelo.

La inductancia del cableado (`L0-R0-(R1|Q1)-(R2|Q2)`) se ajusta ADEMÁS
donde la forma la pide, no en lugar del común: en E no cambia nada
(0.706 % con y sin) y vuelve indeterminada (±468 %); en EA paga,
0.541 → 0.497 % con los ocho parámetros determinados.

### 5.3 El circuito FÍSICO del electrodo poroso

El único que devuelve la **resistencia iónica del poro**, `T1.Ri`, que
es la magnitud que limita la potencia y la que un `R-(R|Q)` entierra.

| | circuito | residuo (%) | **T1.Ri (Ω)** |
|---|---|---|---|
| E | L0-R0-(R1\|Q1)-T1 | 5.19 | 127.8 ± **146 %** |
| **EA** | L0-R0-(R1\|Q1)-T1 | **0.51** | **43.87 ± 4.7 %** |
| EB | R0-(R1\|Q1)-T1 | 1.80 | 1.4·10⁻³ ± **5·10⁸ %** |
| EP | R0-(R1\|Q1)-T1 | 12.67 | 38 574 ± **76 %** |

**Un parámetro con un 10⁸ % de incertidumbre no es una medida, es un
hueco**, y en EB pasa con un residuo excelente: **el residuo no lo
delata, la incertidumbre sí.** Sólo EA devuelve una resistencia de poro
medida.

## 6. DRT — qué constantes de tiempo hay

λ elegido por la curva L. Sólo cuentan los picos dentro del intervalo
medido de τ (1.6 µs a 16 s).

| | τ (s) | R del pico (Ω) | C = τ/R (mF) | residuo DRT |
|---|---|---|---|---|
| **E** | 0.276 / 1.43 / 7.45 | 557 / 867 / 4 610 | 0.495 / 1.65 / 1.62 | 0.51 % |
| **EA** | 0.441 / 2.29 / 11.9 | 1 185 / 4 855 / 16 657 | 0.372 / 0.473 / 0.716 | 0.42 % |
| **EB** | 0.707 | 29 788 | 0.024 | 6.33 % |
| **EP** | 0.0671 | 17 876 | 0.0038 | 4.67 % |

Tres avisos antes de usar esta tabla:

1. **Los picos del extremo lento no son procesos**, son la acumulación
   de lo que no se midió: falta frecuencia por debajo de 0.01 Hz. Es
   también por lo que el panel (e) de la figura sale dominado por el
   borde derecho — ahí está el peso, y eso ES el resultado.
2. **La DRT no tiene picos donde hay un elemento distribuido.** Las
   cuatro tienen entre 0.5 y 1.8 décadas de fase cerca de −45°, que es
   una línea de transmisión o un Warburg: la DRT los reparte en una
   joroba que esta tabla trocearía en arcos que no existen.
3. **Las DRT de EB y EP reproducen peor sus espectros** (6.3 y 4.7 %)
   que las de E y EA (0.5 y 0.4 %).

## 7. Las figuras y los archivos

| archivo | qué es |
|---|---|
| `peis_precursores.png` / `.pdf` | los seis paneles. **Sin recuadro en (a)**: el rincón tiene su propio panel en (b), y la misma década dos veces gasta un cuarto del Nyquist |
| `peis_nyquist_lineal_vs_log.png` / `.pdf` | el mismo Nyquist en las dos escalas, lado a lado |
| `espectro_*.csv` | f, Z′, Z″, \|Z\|, fase, Z′ y Z″ **del ajuste común**, residuo %, C′, C″ |
| `drt_*.csv` | τ, γ(τ), y si ese τ cae dentro del intervalo medido |
| `comparacion.csv` | una fila por muestra: R_s, R_ion, codo, fase, C, circuito, residuo |
| `ajustes.csv` | los parámetros de los cuatro ajustes de cada muestra (automático, común, con L, tres arcos), con error absoluto y en % |
| `capacitancias.csv` | la tabla 4 entera, con RHE, los dos arcos y los F/g |
| `resumen.json` | todo lo anterior más la forma, el ranking de circuitos, KK y el veredicto de deriva por circuito |
| `informe_*.txt` | el informe en prosa de cada espectro |

### Sobre el Nyquist en logaritmos

Lo que **gana**: cada década de |Z| recibe el mismo espacio, así que la
resistencia en serie, el tramo a 45° y la cola de baja frecuencia son
legibles en una sola caja, sin recuadro, y una muestra dos órdenes por
debajo de otra sigue estando en el gráfico.

Lo que **cuesta**, y no es poco — un Nyquist en log **ya no es un mapa
del plano complejo**:

- **Un semicírculo no es un semicírculo**, así que **no se puede leer un
  ángulo de depresión** ni un diámetro: la distancia entre dos puntos ya
  no es una impedancia.
- **El ápice de un arco se mueve**, así que señalar la cima de una curva
  ahí y llamarla R_ct está mal.
- **Z′ y −Z″ ya no suman como vectores**, así que nada se mide restando
  un corte de otro.

Lo que **sobrevive es la PENDIENTE**: una ley de potencia −Z″ ∝ Z′^m es
una recta de pendiente m sea cual sea su prefactor, así que el tramo a
45° de una línea de transmisión o de un Warburg es pendiente 1 y lo
sigue siendo. Por eso la referencia del panel es **la línea de trazos**,
no la proporción de los ejes: una curva paralela a ella tiene pendiente
1 a cualquier proporción.

**Tres puntos inductivos se quedan fuera** (dos de E, uno de EA): un eje
logarítmico necesita −Z″ > 0 y la inductancia del cableado lo pone por
debajo de cero arriba del barrido. El panel lo dice en vez de enseñar un
espectro más corto que el lineal de al lado.

**Z″ va con su signo FÍSICO** (negativo si es capacitivo) en todos los
CSV. El −Z″ del diagrama de Nyquist es convenio de dibujo.
