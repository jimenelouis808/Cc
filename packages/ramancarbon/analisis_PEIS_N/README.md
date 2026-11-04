# Cuatro carbones Fe/Fe₃C, cuatro precursores de nitrógeno — lectura del PEIS

Etanol como fuente de carbono en las cuatro muestras. Lo que cambia es
el precursor que aporta el nitrógeno: ninguno (**E**), acetonitrilo
(**EA**), bencilamina (**EB**) y piridina (**EP**). Las medidas son
PEIS de 100 kHz a 0.01 Hz, 43 frecuencias, ~10.5 mV de amplitud y unos
once minutos por espectro, en **H₂SO₄ 0.5 M** sobre el **disco de
carbón vítreo de 3 mm**.

Los números están en [`tablas.md`](tablas.md) y en los CSV; esto es qué
significan.

```bash
uv run python -I analisis_PEIS_N/exportar.py E.mpr EA.mpr EB.mpr EP.mpr
uv run python -I analisis_PEIS_N/figuras.py  E.mpr EA.mpr EB.mpr EP.mpr
```

---

## Lo primero, porque cambia qué se puede concluir

### EB no estaba midiendo una doble capa: estaba haciendo hidrógeno

Con el electrolito puesto, el pH es 0.3 y los continuos se pueden pasar
a la escala del hidrógeno (suponiendo Ag/AgCl saturado, que es la
referencia de la serie anterior del usuario — **no la dio para ésta**):

| | V vs Ag/AgCl sat | **V vs RHE** | \|j\| (µA cm⁻², geométrica) |
|---|---|---|---|
| E | +0.4489 | +0.664 | 5.0 |
| EA | +0.3105 | +0.525 | 3.8 |
| **EB** | **−0.4043** | **−0.190** | **225** |
| EP | +0.4853 | +0.700 | 7.6 |

**EB está 190 mV por DEBAJO del equilibrio del hidrógeno**, y pasa
0.225 mA cm⁻² — de treinta a sesenta veces la corriente de las otras
tres, con la misma amplitud de 10 mV. Eso es **desprendimiento de
hidrógeno corriendo durante los once minutos del barrido**: una
reacción farádica, burbujas, y una interfase que cambia mientras se la
mide. No es ruido ni un artefacto del equipo, y explica por sí solo por
qué ese espectro no pasa Kramers-Kronig.

Es, además, un resultado por derecho propio y no sólo un problema: el
OCP de EB está donde el hidrógeno se desprende, que es lo que hace un
electrodo con hierro accesible al electrolito.

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

**Consecuencia:** de EB y EP se pueden usar la forma, R_s, el codo y
C′(f), que son datos. **No** se pueden citar los parámetros de ningún
circuito ajustado a ellos, por bien que ajuste. Está en la pantalla
del residuo (panel f de la figura).

### Y los cuatro se midieron a potenciales distintos

Entre EB y EP hay **0.89 V**. Una resistencia de transferencia de carga
depende **exponencialmente** del sobrepotencial: a esa distancia, una
diferencia de un orden de magnitud en R_ct no dice nada sobre la
química del nitrógeno, dice dónde estaba polarizada cada celda. Que el
OCP mismo difiera *sí* es un resultado —es el potencial mixto del
material— pero es otro resultado.

> **El experimento que arregla las dos cosas es el mismo:** repetir los
> cuatro PEIS al **mismo potencial** frente a la misma referencia, y
> medir cada uno **dos veces, de alta a baja frecuencia y al revés**. Si
> las dos pasadas no coinciden, ahí está la deriva, con nombre y
> apellidos.

---

## El ajuste: un circuito para las cuatro, y cuál

La primera versión de esta figura dejaba que el automático eligiera el
circuito de cada espectro, que es lo correcto para describir una
muestra y lo incorrecto para comparar cuatro. Y además ajustaba mal:
la curva de EP no seguía a los puntos.

| | automático | residuo | **común: `R0-(R1\|Q1)-(R2\|Q2)`** |
|---|---|---|---|
| E | R0-(R1\|Q1)-T1 | 2.38 % | **0.71 %** |
| EA | R0-T1 | 3.18 % | **0.54 %** |
| EB | R0-(R1\|Q1) | 9.75 % | **5.79 %** |
| EP | R0-(R1\|Q1) | **17.83 %** | **5.11 %** |

Dos arcos ganan en las cuatro a la vez y devuelven los siete parámetros
determinados en tres de ellas (cinco de siete en EB, que es la que no
pasa KK). Es el circuito de la figura y el de todas las columnas de
arco de las tablas.

**El arco lento es el que almacena**: su capacitancia sigue el mismo
orden que la C′ medida (1.33 / 0.22 / 0.07 / 0.37 mF frente a 0.83 /
0.16 / 0.08 / 0.29), que es la comprobación de que el modelo no está
inventando el almacenamiento. Pero **en E y EA ese arco está
extrapolado**: su τ vale 86 y 145 s, o sea un ápice en 1.9 y 1.1 mHz,
dos décadas por debajo de lo medido.

---

## Lo que sí se puede leer, muestra por muestra

### Ninguna cierra un semicírculo

En las cuatro, −Z″ sube hasta la última frecuencia medida (EB roza un
ápice en el último punto, a 0.032 Hz). **Ninguna resistencia de
transferencia de carga sale de los datos**: la que devuelva un ajuste
sale del modelo. Y en EB y EP, que además no cruzan el eje real, **R_s
es una cota superior**. En E y EA sí hay cruce interpolado, y por
encima de él Z″ se va positivo: inductancia del cableado, no de la
muestra.

### EA es la única con un electrodo poroso bien medido

Es el único espectro donde el modelo FÍSICO del poro vuelve entero
determinado:

```
L0-R0-(R1|Q1)-T1     residuo 0.51 %
  R0    = 46.48 Ω ± 0.2 %
  T1.Ri = 43.87 Ω ± 4.7 %      ← la resistencia iónica del poro
  T1.n  = 0.895  ± 2.6 %
```

`T1.Ri` es la magnitud que limita la potencia de un electrodo poroso, y
es la que un ajuste con `R-(R|Q)` entierra. En EA está medida. En E el
mismo circuito la devuelve con un **146 %** de incertidumbre; en EB y
EP la línea de transmisión no tiene nada que describir y colapsa
(±2·10⁸ % y ±76 %) — que no son valores, son huecos.

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

| | C′(0.01 Hz) | fase ahí | mF cm⁻² (geométrica) |
|---|---|---|---|
| **E** | **0.827 mF** | −56° | 11.7 |
| EP | 0.287 mF | −22° | 4.1 |
| EA | 0.162 mF | −71° | 2.3 |
| EB | 0.084 mF | −20° | 1.2 |

A 0.01 Hz ninguna ha llegado al régimen capacitivo (un condensador está
en −90°), así que las cuatro son **cotas inferiores**. La distancia
entre C_s y C′ lo dice por otro lado: 12 % en EA, 47 % en E, **606 %**
en EP y **745 %** en EB. Donde la respuesta todavía es resistiva, las
dos definiciones de capacitancia no se parecen.

Y los mF cm⁻² usan el área **geométrica** del disco: la activa de un
carbón poroso es de diez a mil veces mayor, así que esa columna compara
estas cuatro muestras entre sí y nada más.

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

1. Los cuatro están a potenciales distintos, y uno de ellos (EB) ni
   siquiera estaba midiendo una doble capa.
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

**EB tiene el OCP en la región del hidrógeno, y eso apunta al hierro.**
Su potencial mixto está 0.85 V por debajo del de EA y 190 mV por debajo
del equilibrio del H₂, con 0.225 mA cm⁻² corriendo. En una muestra con
alto contenido de Fe y Fe₃C, un OCP tan catódico y una corriente
farádica a ese potencial apuntan a **hierro accesible al electrolito**
— Fe⁰ o Fe₃C sin recubrir, catalizando HER y oxidándose. La bencilamina
es la única amina alifática primaria del grupo y el ligando que más
razonablemente complejaría hierro durante la síntesis; una partícula
peor encapsulada es una explicación coherente con todo lo medido.
**No está demostrado aquí.** Lo que lo demostraría: XPS del Fe 2p (¿hay
Fe oxidado en superficie?), y un barrido anódico lento buscando el pico
de disolución. Y si EB resulta ser el mejor catalizador de HER del
grupo, eso es un experimento aparte que esta misma celda ya está
pidiendo: un LSV, no un PEIS.

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
- **Nada que dependa de la referencia sin decir que es supuesta.** Toda
  la columna RHE —y con ella la lectura de EB— se desplaza en bloque si
  la referencia de esta serie no era Ag/AgCl saturado. **Confírmala.**

---

## Qué mediría a continuación, por orden de rendimiento

1. **Confirmar la referencia** de esta serie. Cuesta cero y es de lo
   que cuelga la conclusión más fuerte del análisis.
2. **Repetir los cuatro PEIS al mismo potencial**, y cada uno en los dos
   sentidos de frecuencia. Cuesta una tarde y desbloquea la mitad de
   esta tabla.
3. **Bajar a 1 mHz.** Convierte cuatro cotas inferiores en cuatro
   capacitancias, y mete dentro del barrido el arco lento que ahora
   está extrapolado en E y EA.
4. **Pesar los electrodos.** Convierte mF en F/g.
5. **Un LSV de HER sobre EB** (y sobre las otras tres para comparar).
   El PEIS ya enseñó que esa muestra hace hidrógeno; medirlo es un
   barrido.
6. **BET** sobre las cuatro, para contrastar contra la `T1.Ri` de EA.
7. **XPS N 1s y Fe 2p**, que es lo único que contesta la pregunta que
   motivó el experimento.

Con 2 y 3 hechos, este mismo script vuelve a correr sin tocar una línea
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

En el panel (e) el peso se amontona contra el borde derecho en tres de
las cuatro. **Eso no es un fallo del dibujo: es el resultado.** Falta
barrido por debajo de 0.01 Hz, y la DRT acumula ahí todo lo que no se
midió.
