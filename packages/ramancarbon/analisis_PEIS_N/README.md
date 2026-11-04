# Cuatro carbones Fe/Fe₃C, cuatro precursores de nitrógeno — lectura del PEIS

Etanol como fuente de carbono en las cuatro muestras. Lo que cambia es
el precursor que aporta el nitrógeno: ninguno (**E**), acetonitrilo
(**EA**), bencilamina (**EB**) y piridina (**EP**). Las medidas son
PEIS de 100 kHz a 0.01 Hz, 43 frecuencias, ~10.5 mV de amplitud y unos
once minutos por espectro.

Los números están en [`tablas.md`](tablas.md) y en los CSV; esto es qué
significan.

```bash
uv run python -I analisis_PEIS_N/exportar.py E.mpr EA.mpr EB.mpr EP.mpr
uv run python -I analisis_PEIS_N/figuras.py  E.mpr EA.mpr EB.mpr EP.mpr
```

---

## Lo primero, porque cambia qué se puede concluir

### Dos de los cuatro espectros no son consistentes con Kramers-Kronig

| | residuo RMS | rachas de signo | |
|---|---|---|---|
| E | 0.25 % | 0.71× | pasa |
| EA | 0.14 % | 0.88× | pasa |
| **EB** | **6.57 %** | **0.24×** | **no pasa** |
| **EP** | **1.65 %** | **0.41×** | **no pasa** |

Kramers-Kronig no comprueba el modelo: comprueba que el sistema fuera
**el mismo** del principio al final del barrido. Lo que lo decide no es
el tamaño del residuo sino su **estructura**: un residuo que cambia de
signo cuatro veces menos de lo que cambiaría por azar no es ruido.

Once minutos a un potencial fijo dan de sobra para que una celda se
mueva, y en EB hay además la evidencia directa: **su corriente media es
15.9 µA contra 0.27–0.54 µA de las otras tres**, con la misma amplitud
de 10 mV. Treinta a sesenta veces más corriente significa que a −0.40 V
hay un proceso farádico corriendo — algo se reduce, se disuelve o se
deposita — y eso es exactamente lo que hace derivar un espectro.

**Consecuencia:** de EB y EP se pueden usar la forma, R_s, el codo y
C′(f), que son datos. **No** se pueden citar los parámetros de ningún
circuito ajustado a ellos, por bien que ajuste. Está en la pantalla
del residuo (panel f de la figura): EP llega al 54 % y EB al 28 %.

### Y los cuatro se midieron a potenciales distintos

+0.449, +0.311, **−0.404** y +0.485 V. Los nombres dicen «PEIS 0 V»,
que en EC-Lab es 0 V *frente al OCP*, así que cada medida está en el
OCP de su propia muestra y entre EB y EP hay **0.89 V**.

Una resistencia de transferencia de carga depende **exponencialmente**
del sobrepotencial. A 0.89 V de distancia, una diferencia de un orden
de magnitud en R_ct no dice nada sobre la química del nitrógeno: dice
dónde estaba polarizada cada celda. Que el OCP mismo difiera *sí* es un
resultado —es el potencial mixto del material— pero es otro resultado.

> **El experimento que arregla las dos cosas es el mismo:** repetir los
> cuatro PEIS al **mismo potencial** frente a la misma referencia, y
> medir cada uno **dos veces, de alta a baja frecuencia y al revés**. Si
> las dos pasadas no coinciden, ahí está la deriva, con nombre y
> apellidos.

---

## Lo que sí se puede leer, muestra por muestra

### Ninguna cierra un semicírculo

En las cuatro, −Z″ sube hasta la última frecuencia medida (EB roza un
ápice en el último punto, a 0.032 Hz). Dos consecuencias que no son
matices:

- **Ninguna resistencia de transferencia de carga sale de los datos.**
  La que devuelva un ajuste sale del modelo. Los 11 kΩ de EP y los
  82 kΩ de EB son extrapolaciones de un arco que la medida no vio.
- **R_s es una cota superior en las cuatro**, porque ninguna cruza el
  eje real. En E y EA hay además Z″ positivo en el extremo alto:
  inductancia del cableado, no de la muestra.

### EA es la única con un electrodo poroso bien medido

Es el único espectro donde todo el modelo vuelve determinado:

```
L0-R0-(R1|Q1)-T1     residuo 0.51 %
  R0    = 46.48 Ω ± 0.2 %
  T1.Ri = 43.87 Ω ± 4.7 %      ← la resistencia iónica del poro
  T1.n  = 0.895  ± 2.6 %
```

`T1.Ri` es la magnitud que limita la potencia de un electrodo poroso, y
es la que un ajuste con `R-(R|Q)` entierra. En EA está medida. En E el
mismo circuito la devuelve con un **146 %** de incertidumbre, y sin la
inductancia en serie colapsa a 1.5·10⁻⁵ Ω **± 4·10⁸ %** — que no es un
valor pequeño, es un hueco. En EB y EP la línea de transmisión no tiene
nada que describir y degenera igual.

Dicho de otro modo: **sólo EA se comporta como el electrodo poroso que
las cuatro pretenden ser.** Su fase llega a −71° a 0.01 Hz, la más
capacitiva de las cuatro, y su Nyquist (panel a) es la única rama casi
vertical.

### El codo ordena las muestras, y mide el dispositivo

| | codo (Hz) | τ (s) |
|---|---|---|
| E | 10.9 | 0.092 |
| EA | 41.0 | 0.024 |
| EB | 46.9 | 0.021 |
| **EP** | **712** | **0.0014** |

El codo es la frecuencia a la que la fase pasa por −45°: por encima, el
electrodo deja de comportarse como un condensador. Lleva dentro la R_s
de la celda a propósito —es del **dispositivo**, no del material— así
que no tiene por qué ordenar igual que la resistencia del poro, y aquí
no lo hace.

EP tiene el codo a 712 Hz, **un factor 65 por encima de E**. Eso suena
a electrodo rápido y hay que resistirse: con una fase que a 0.01 Hz
sigue en −22°, lo que EP tiene no es una respuesta capacitiva rápida
sino **poca capacitancia** repartida sobre una resistencia grande. El
codo baja cuando τ = RC sube, y EP tiene C pequeña.

### Las capacitancias son todas límites inferiores

| | C′(0.01 Hz) | fase ahí |
|---|---|---|
| **E** | **0.827 mF** | −56° |
| EP | 0.287 mF | −22° |
| EA | 0.162 mF | −71° |
| EB | 0.084 mF | −20° |

A 0.01 Hz ninguna ha llegado al régimen capacitivo (un condensador está
en −90°), así que las cuatro son **cotas inferiores**. La distancia
entre C_s y C′ lo dice por otro lado: 12 % en EA, 47 % en E, **606 %**
en EP y **745 %** en EB. Donde la respuesta todavía es resistiva, las
dos definiciones de capacitancia no se parecen.

**Bajar a 1 mHz es lo que convierte estos cuatro números en medidas.**
Con el barrido actual, lo honesto es decir que E almacena al menos
cinco veces más que EA y diez veces más que EB, y que el orden final
puede cambiar.

---

## La lectura química, con lo que aguanta y lo que no

**El control sin nitrógeno (E) es el que más carga almacena** de los
cuatro, al menos a 0.01 Hz. Si el objetivo del dopaje era subir la
capacitancia, estos espectros no lo apoyan. Pero antes de escribir eso
en un artículo hay tres cosas que lo bloquean, y las tres son
arreglables:

1. Los cuatro están a potenciales distintos.
2. Ninguno llegó al régimen capacitivo.
3. No hay masa, así que no hay forma de saber si E almacena más **por
   gramo** o simplemente hay más material en ese electrodo. Con las
   cuatro C′ dentro de un factor diez, una diferencia de carga de
   electrodo se las come enteras.

**EA es la que se comporta como un carbón poroso.** De los cuatro
precursores, el acetonitrilo es el de relación C/N más baja y el único
cuyo nitrógeno entra como nitrilo. Que su electrodo sea el único con
una resistencia de poro medible y la fase más capacitiva es consistente
con una porosidad mejor desarrollada — y es consistente, no probado:
esto es una medida de impedancia, no de área. **La prueba que lo zanja
es fisisorción de N₂** (BET y distribución de tamaño de poro), y
compararla con `T1.Ri` es justo el contraste que hace falta.

**EB a −0.40 V está haciendo otra cosa.** Su OCP está 0.85 V por debajo
del de EA y pasa 15.9 µA. En una muestra con alto contenido de Fe y
Fe₃C, un OCP tan negativo y una corriente farádica a ese potencial
apuntan a **hierro accesible al electrolito** — Fe⁰ o Fe₃C sin
recubrir, oxidándose. La bencilamina es la única amina alifática
primaria del grupo y el ligando que más razonablemente complejaría
hierro durante la síntesis; una partícula peor encapsulada es una
explicación coherente con todo lo medido. **No está demostrado aquí.**
Lo que lo demostraría: XPS del Fe 2p (¿hay Fe oxidado en superficie?),
y un barrido anódico lento buscando el pico de disolución.

**EP tiene el codo más rápido y la capacitancia casi más baja**, que es
la combinación de un electrodo poco poroso con una interfase pequeña.
Su DRT da un único pico resuelto en **τ = 0.067 s**, el más rápido de
los cuatro, con una capacitancia asociada de 3.8 µF — dos órdenes por
debajo de las demás. Eso es una interfase pequeña, no una rápida.

---

## Lo que NO se puede decir con estos datos

- **Nada sobre la configuración del nitrógeno en el carbón.** El
  precursor dice qué nitrógeno entró al horno, no cuál salió: la
  pirólisis lo rehace. Piridínico, pirrólico y grafítico se separan con
  **XPS del N 1s**, y el paquete lo hace (`ramancarbon xps`).
- **Nada cuantitativo sobre Fe/Fe₃C.** La impedancia ve una interfase,
  no fases. Eso es DRX con Rietveld, que también está aquí.
- **Ninguna capacitancia específica.** Sin masa no hay F/g, y
  suponerla es equivocarse exactamente en esa proporción.
- **Ninguna comparación de R_ct entre muestras**, por los potenciales.
- **Nada de EB y EP que venga de un circuito.**

---

## Qué mediría a continuación, por orden de rendimiento

1. **Repetir los cuatro PEIS al mismo potencial**, y cada uno en los dos
   sentidos de frecuencia. Cuesta una tarde y desbloquea la mitad de
   esta tabla.
2. **Bajar a 1 mHz.** Convierte cuatro cotas inferiores en cuatro
   capacitancias.
3. **Pesar los electrodos.** Convierte mF en F/g.
4. **BET** sobre las cuatro, para contrastar contra la `T1.Ri` de EA.
5. **XPS N 1s y Fe 2p**, que es lo único que contesta la pregunta que
   motivó el experimento.

Con 1 y 2 hechos, este mismo script vuelve a correr sin tocar una línea
y las tablas salen comparables.

---

## Los archivos

| archivo | qué es |
|---|---|
| `peis_precursores.png` / `.pdf` | la figura de seis paneles |
| `tablas.md` | todas las tablas, con las unidades y los avisos |
| `espectro_*.csv` | f, Z′, Z″, \|Z\|, fase, ajuste, residuo, C′, C″ |
| `drt_*.csv` | τ, γ(τ), y si cae dentro del intervalo medido |
| `comparacion.csv`, `ajustes.csv`, `capacitancias.csv` | las tablas en crudo |
| `informe_*.txt` | el informe en prosa de cada muestra |
| `resumen.json` | todo, incluidos el ranking de circuitos y los avisos |
| `exportar.py`, `figuras.py` | lo que lo produce |

**Z″ va con su signo físico** (negativo si es capacitivo) en todos los
CSV, no con el −Z″ del Nyquist, que es convenio de dibujo.

Los seis paneles de la figura: (a) Nyquist a escala uno a uno con el
rincón en un recuadro, (b) el rincón ampliado, (c) Bode con el codo
marcado, (d) C′ y C″ sin ningún circuito de por medio, (e) DRT
normalizada y (f) el residuo de cada ajuste — que es el panel donde se
ve de un vistazo cuáles dos no hay que creerse.
