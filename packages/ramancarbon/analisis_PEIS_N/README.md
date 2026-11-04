# Cuatro carbones Fe/Fe₃C, cuatro precursores de nitrógeno — lectura del PEIS

Etanol como fuente de carbono en las cuatro muestras. Lo que cambia es
el precursor que aporta el nitrógeno: ninguno (**E**), acetonitrilo
(**EA**), bencilamina (**EB**) y piridina (**EP**). PEIS de 100 kHz a
0.01 Hz, 43 frecuencias, ~10.5 mV y unos once minutos por espectro, en
**H₂SO₄ 0.5 M** sobre el **disco de carbón vítreo de 3 mm**, frente a
**Ag/AgCl saturado**, con **0.909 mg** de material depositado.

Los números están en [`tablas.md`](tablas.md) y en los CSV; esto es qué
significan.

```bash
uv run python -I analisis_PEIS_N/exportar.py E.mpr EA.mpr EB.mpr EP.mpr
uv run python -I analisis_PEIS_N/figuras.py  E.mpr EA.mpr EB.mpr EP.mpr
```

---

## Las cuatro ya son comparables

| | V vs Ag/AgCl sat | **V vs RHE** | KK |
|---|---|---|---|
| EA | +0.3105 | +0.525 | pasa |
| **EB** | +0.3249 | **+0.540** | **pasa** |
| E | +0.4489 | +0.664 | pasa |
| EP | +0.4853 | +0.700 | **no pasa** |

Las cuatro caen dentro de **0.18 V**, y **EB y EA están a 13 mV**. Eso
desbloquea la comparación que antes no se podía hacer: el recorrido era
de 0.89 V y una resistencia de transferencia de carga depende
exponencialmente del sobrepotencial.

**El EB que entra aquí es el segundo**, a +0.54 V vs RHE. El primero, a
−0.19 V, estaba por debajo del equilibrio del hidrógeno pasando
0.225 mA cm⁻²: desprendimiento de hidrógeno corriendo durante los once
minutos del barrido. Las dos medidas, misma muestra y misma celda (R_s
85.01 y 85.22 Ω, 0.25 % de diferencia), demostraron que el fallo de
Kramers-Kronig era **el potencial y no la muestra**:

| | EB a −0.19 V | EB a +0.54 V |
|---|---|---|
| Kramers-Kronig | **no pasa** (6.57 %, rachas 0.24×) | **pasa** (0.73 %, 0.72×) |
| \|j\| | 0.225 mA cm⁻² | 0.035 mA cm⁻² |
| fase a 0.01 Hz | −20.1° | **−47.0°** |
| C′(0.01 Hz) | 0.084 mF | **0.200 mF** |
| ápice de arco | **0.032 Hz** | ninguno |

Demostrado eso, **el espectro catódico se ha retirado del análisis**: lo
que queda son cuatro muestras en una ventana común. Que el OCP de EB
caiga medio voltio por debajo del de las otras tres sigue siendo un
resultado del material — en una muestra con mucho Fe y Fe₃C apunta a
**hierro accesible al electrolito** — y es otro experimento: un LSV, no
un PEIS.

### EP es el que queda sin explicar

Residuo KK 1.65 % con rachas 0.41×, sin corriente anómala (0.54 µA) y en
el potencial más anódico de los cuatro. La sospecha razonable es
oxidación lenta de la superficie del carbón a +0.700 V vs RHE, y **no
está demostrada**. **De EP se pueden usar la forma, R_s, el codo y C′(f).
No los parámetros de ningún circuito.**

---

## El ajuste: por qué no se puede mejorar más

| | automático | **común, dos arcos** |
|---|---|---|
| E | 2.38 % | **0.71 %** |
| EA | 3.18 % | **0.54 %** |
| EB | 3.05 % | **1.83 %** |
| EP | **17.83 %** | **5.11 %** |

Un circuito común —`R0-(R1|Q1)-(R2|Q2)`— porque parámetros de circuitos
distintos no son los mismos parámetros. Buscando bajar el residuo de EP
y de EB, **tres arcos** los lleva a 2.50 % y 1.33 %. No se ha adoptado, y
la razón es una medida:

| | circuito | residuo | share | correlación | |
|---|---|---|---|---|---|
| **E** | dos arcos | 0.71 % | 0.51 | −0.02 | modelo |
| **EA** | tres arcos | **0.25 %** | 0.60 | +0.31 | modelo |
| **EB** | dos arcos | 1.83 % | **0.77** | **+0.77** | **deriva** |
| **EP** | dos arcos | 5.11 % | 0.64 | +0.62 | **deriva** |
| **EP** | tres arcos | **2.50 %** | **0.77** | **+0.79** | **deriva** |

El *share* es qué fracción del residuo del ajuste cae en la mitad de
frecuencias donde el residuo de Kramers-Kronig es peor. Si falta un
elemento, las frecuencias donde el circuito falla no tienen por qué ser
aquéllas donde los datos no son consistentes consigo mismos, y sale
cerca de la mitad. Si lo que queda es deriva, coinciden.

**En EP el arco nuevo no encontró un proceso: siguió la deriva.** El
residuo se parte por dos mientras el share sube de 0.64 a 0.77: el
ajuste pasa a seguir aún más de cerca justo la parte del espectro que KK
dice que no es fiable. **En EB** los dos arcos lentos del modelo de tres
salen a τ = 2.3 y 4.3 s, un factor **1.9** —por debajo del factor 3 que
este paquete llama degenerado— con R a ±128 % y ±225 %.

> **El ajuste de EP y de EB no se mejora cambiando de circuito.** Lo que
> queda no es error de modelo. Lo que lo arregla es volver a medir: cada
> espectro **dos veces, de alta a baja frecuencia y al revés**. Si las
> dos pasadas no coinciden, ahí está la deriva con nombre y apellidos.

El umbral está calibrado sobre cuarenta espectros sintéticos cuyo
residuo es puramente de modelo: share 0.52 ± 0.05 con un máximo de 0.61,
correlación +0.15 ± 0.19 con un máximo de +0.53. Los cuatro ajustes de
cada muestra están completos en `ajustes.csv`.

**Sólo EA tiene una resistencia iónica de poro medida**: 43.87 Ω ± 4.7 %
con el modelo físico `L0-R0-(R1|Q1)-T1` al 0.51 % de residuo. En EB la
línea de transmisión colapsa con ±5·10⁸ % **y un residuo excelente, del
1.80 %** — que es exactamente por lo que se mira la incertidumbre y no
basta el residuo.

---

## La capacitancia, y el problema que la masa destapó

| | V vs RHE | C′(0.01 Hz) | **F g⁻¹** | fase ahí |
|---|---|---|---|---|
| **E** | +0.664 | 0.827 mF | **0.910** | −56° |
| EP | +0.700 | 0.287 mF | 0.316 | −22° |
| **EB** | +0.540 | 0.200 mF | **0.220** | −47° |
| EA | +0.525 | 0.162 mF | 0.178 | −71° |

**Un carbón poroso publica de 50 a 300 F/g. Aquí el mejor da 0.91.** Dos
órdenes de magnitud, y no es un error de cuenta.

0.909 mg sobre 0.0707 cm² son **12.9 mg cm⁻²**, de diez a cien veces la
carga habitual sobre un carbón vítreo: una película de decenas de
micras. Una película así tiene una resistencia iónica **en profundidad**
que el ion no atraviesa en los 100 s que dura un punto a 0.01 Hz. Lo que
estos espectros miden es **la fracción de la película a la que el ion
llega en 100 segundos**, no la capacitancia del material.

Encaja con todo lo demás: ninguna muestra alcanza el régimen capacitivo
(la fase se queda entre −22° y −71°, un condensador está en −90°), las
cuatro C′ son cotas inferiores, y las cuatro tienen entre 0.5 y 1.8
décadas de fase cerca de −45°, que es la firma de una película
cargándose por difusión.

> **El experimento que lo arregla es una tinta más diluida.** Con
> 0.05–0.1 mg cm⁻² (unos 2–5 µL de una tinta diez veces más diluida) la
> película baja a micras y el barrido actual ya la atraviesa. Eso, más
> bajar a 1 mHz, convierte esta tabla en capacitancias de verdad.

La masa es además **nominal**: sale del volumen depositado, no de pesar
el electrodo seco. 20 µL sobre un disco de 3 mm es mucho líquido para
ese diámetro, así que si parte se salió, la masa real es menor y los F/g
algo mayores. Sigue sin cambiar el orden de magnitud.

---

## Lo que sí se puede leer, muestra por muestra

### Ninguna cierra un semicírculo

En ninguna de las cuatro. **Ninguna resistencia de transferencia de
carga sale de los datos**: la que devuelva un ajuste sale del modelo. Y
en EB y EP, que además no cruzan el eje real, **R_s es una cota
superior**. En E y EA sí hay cruce interpolado, y por encima de él Z″ se
va positivo: inductancia del cableado, no de la muestra.

### EA es la única con un electrodo poroso bien medido

Su fase llega a −71° a 0.01 Hz, la más capacitiva de las cuatro, y su
Nyquist (panel a) es la única rama casi vertical. **Sólo EA se comporta
como el electrodo poroso que las cuatro pretenden ser** — aunque
almacene menos que EB al mismo potencial, que son dos cosas distintas:
una es cómo entra el ion, la otra cuánta carga cabe.

### El codo ordena las muestras, y mide el dispositivo

| | codo (Hz) | τ (s) |
|---|---|---|
| E | 10.9 | 0.092 |
| EB | 25.2 | 0.040 |
| EA | 41.0 | 0.024 |
| **EP** | **712** | **0.0014** |

El codo es la frecuencia a la que la fase pasa por −45°. Lleva dentro la
R_s de la celda a propósito —es del **dispositivo**, no del material— así
que no tiene por qué ordenar igual que la resistencia del poro. EP está
un factor 65 por encima de E, y eso suena a electrodo rápido: hay que
resistirse. Con una fase que a 0.01 Hz sigue en −22°, lo que EP tiene no
es una respuesta capacitiva rápida sino **poca capacitancia** repartida
sobre una resistencia grande.

---

## La lectura química, con lo que aguanta y lo que no

**El control sin nitrógeno (E) es el que más carga almacena**, y ahora
la comparación es limpia: las cuatro dentro de 0.18 V. Si el objetivo
del dopaje era subir la capacitancia, estos espectros no lo apoyan. Dos
cosas lo bloquean antes de escribirlo:

1. **Ninguno llegó al régimen capacitivo**, y la causa probable es la
   película gruesa, no el material.
2. **La masa es la misma para las cuatro por protocolo, no por pesada.**
   Si el depósito varió entre electrodos, una diferencia de carga se
   come un factor de tres en F/g sin que nada lo delate.

**EA es la que se comporta como un carbón poroso.** El acetonitrilo es
el precursor de relación C/N más baja y el único cuyo nitrógeno entra
como nitrilo; que su electrodo sea el único con una resistencia de poro
medible y la fase más capacitiva es **consistente** con una porosidad
mejor desarrollada — consistente, no probado: esto es impedancia, no
área. **La prueba que lo zanja es fisisorción de N₂** (BET y
distribución de tamaño de poro), contrastada contra la `T1.Ri` de EA.

**EB, a potencial comparable, almacena más que EA** (0.220 contra
0.178 F/g a 13 mV de distancia) y su OCP está medio voltio por debajo.
Eso último apunta a **hierro accesible al electrolito** — Fe⁰ o Fe₃C sin
recubrir. La bencilamina es la única amina alifática primaria del grupo
y el ligando que más razonablemente complejaría hierro durante la
síntesis; una partícula peor encapsulada es coherente con lo medido.
**No está demostrado aquí.** Lo demostrarían XPS del Fe 2p y un barrido
anódico lento buscando el pico de disolución.

**EP tiene el codo más rápido y poca capacitancia**, la combinación de
un electrodo poco poroso con una interfase pequeña. Su DRT da un único
pico resuelto en τ = 0.067 s con 3.8 µF asociados — dos órdenes por
debajo de las demás. Eso es una interfase pequeña, no una rápida. Y es
además el único que no pasa KK, así que nada de su circuito se cita.

---

## Lo que NO se puede decir con estos datos

- **Nada sobre la configuración del nitrógeno en el carbón.** El
  precursor dice qué nitrógeno entró al horno, no cuál salió: la
  pirólisis lo rehace. Eso es **XPS del N 1s** (`ramancarbon xps`).
- **Nada cuantitativo sobre Fe/Fe₃C.** La impedancia ve una interfase,
  no fases. Eso es DRX con Rietveld, que también está aquí.
- **Ninguna capacitancia específica del MATERIAL.** Los F/g son de la
  película tal como está depositada, y la película es gruesa.
- **Ninguna comparación de R_ct**, porque ninguna sale de los datos.
- **Nada de EP que venga de un circuito.**
- **Ningún ángulo de depresión ni diámetro de arco leído del panel
  logarítmico** — ver la nota al final de `tablas.md`.

---

## Qué mediría a continuación, por orden de rendimiento

1. **Una tinta diez veces más diluida**, 2–5 µL, para bajar a
   0.05–0.1 mg cm⁻². Es lo único que puede convertir estos F/g en una
   capacitancia del material, y cuesta una tarde.
2. **Cada espectro en los dos sentidos de frecuencia.** Es lo que
   convierte «EP no pasa KK» en una deriva medida, y el análisis ya
   demostró que ningún circuito va a arreglarlo.
3. **Repetir E, EA y EP a +0.54 V vs RHE**, donde ya hay dos medidas
   comparables.
4. **Bajar a 1 mHz.** Convierte las cotas inferiores en medidas y mete
   dentro del barrido los arcos lentos de E y EA (τ = 86 y 145 s), que
   ahora están extrapolados.
5. **Pesar el electrodo seco**, antes y después, en vez de contar el
   volumen de tinta.
6. **Un LSV de HER sobre EB**, y sobre las otras tres para comparar. El
   PEIS ya enseñó que esa muestra hace hidrógeno a −0.19 V vs RHE.
7. **BET** sobre las cuatro, para contrastar contra la `T1.Ri` de EA.
8. **XPS N 1s y Fe 2p**, que es lo único que contesta la pregunta que
   motivó el experimento.

Con 1, 2 y 4 hechos, estos mismos scripts vuelven a correr sin tocar una
línea.

---

## Las figuras

**`peis_precursores`** — seis paneles: (a) Nyquist a escala uno a uno,
(b) el rincón de alta frecuencia, (c) Bode con el codo marcado, (d) C′ y
C″ sin ningún circuito de por medio, (e) DRT normalizada y (f) el
residuo de cada ajuste.

El Nyquist **ya no lleva recuadro**: el rincón tiene su propio panel en
(b), y la misma década dos veces en una figura gasta un cuarto del panel
en algo que el lector tiene al lado — en un Nyquist, justo el cuarto
donde va la rama de baja frecuencia.

En el panel (e) el peso se amontona contra el borde derecho. **Eso no es
un fallo del dibujo: es el resultado.** Falta barrido por debajo de
0.01 Hz, y la DRT acumula ahí todo lo que no se midió.

**`peis_nyquist_lineal_vs_log`** — el mismo Nyquist en las dos escalas,
lado a lado, que es la comparación dibujada. A la izquierda se pueden
leer ángulos y diámetros y no se ve el rincón; a la derecha se ven las
cuatro décadas y **no** se puede leer ni un ángulo de depresión ni un
diámetro. Lo único que sobrevive al logaritmo es la **pendiente**, y por
eso la referencia es la línea de trazos (pendiente 1) y no la proporción
de los ejes. La discusión completa está al final de
[`tablas.md`](tablas.md).

---

## Los archivos

| archivo | qué es |
|---|---|
| `peis_precursores.png` / `.pdf` | la figura de seis paneles |
| `peis_nyquist_lineal_vs_log.png` / `.pdf` | lineal contra logarítmico |
| `tablas.md` | todas las tablas, con las unidades y los avisos |
| `espectro_*.csv` | f, Z′, Z″, \|Z\|, fase, ajuste, residuo, C′, C″ |
| `drt_*.csv` | τ, γ(τ), y si cae dentro del intervalo medido |
| `comparacion.csv`, `ajustes.csv`, `capacitancias.csv` | las tablas en crudo |
| `informe_*.txt` | el informe en prosa de cada espectro |
| `resumen.json` | todo, incluidos el ranking de circuitos y el veredicto de deriva por circuito |
| `exportar.py`, `figuras.py` | lo que lo produce |

**Z″ va con su signo físico** (negativo si es capacitivo) en todos los
CSV, no con el −Z″ del Nyquist, que es convenio de dibujo.
