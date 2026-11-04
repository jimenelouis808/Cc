# Carbones Fe/Fe₃C y el precursor de nitrógeno — lectura del PEIS

Etanol como fuente de carbono en las cuatro muestras. Lo que cambia es
el precursor que aporta el nitrógeno: ninguno (**E**), acetonitrilo
(**EA**), bencilamina (**EB**) y piridina (**EP**). PEIS de 100 kHz a
0.01 Hz, 43 frecuencias, ~10.5 mV y unos once minutos por espectro, en
**H₂SO₄ 0.5 M** sobre el **disco de carbón vítreo de 3 mm**, frente a
**Ag/AgCl saturado**, con **0.909 mg** de material depositado.

**EB está dos veces, a dos potenciales.** Esa pareja es el control
interno de todo el análisis y es de donde sale casi todo lo que sigue.

Los números están en [`tablas.md`](tablas.md) y en los CSV; esto es qué
significan.

```bash
uv run python -I analisis_PEIS_N/exportar.py \
    E.mpr EA.mpr EB_catodico.mpr EB_anodico.mpr EP.mpr
uv run python -I analisis_PEIS_N/figuras.py \
    E.mpr EA.mpr EB_catodico.mpr EB_anodico.mpr EP.mpr
```

---

## El resultado: el mismo electrodo, dos potenciales, dos medidas distintas

| | **EB a −0.19 V vs RHE** | **EB a +0.54 V vs RHE** |
|---|---|---|
| Kramers-Kronig | **NO pasa** (rms 6.57 %, rachas 0.24×) | **pasa** (0.73 %, 0.72×) |
| Corriente media | 15.9 µA (**0.225 mA cm⁻²**) | 2.47 µA (0.035 mA cm⁻²) |
| Fase a 0.01 Hz | −20.1° | **−47.0°** |
| C′(0.01 Hz) | 0.084 mF | **0.200 mF** |
| ápice de arco | **0.032 Hz** | ninguno |
| Residuo del ajuste común | 5.79 % | **1.83 %** |
| R_s | 85.01 Ω | 85.22 Ω |

Misma muestra, misma celda —las dos R_s coinciden al 0.25 %, que es lo
que dice que el electrodo no se tocó entre medidas— y mismo barrido. Lo
único que cambia es el potencial. Y con él cambia el veredicto.

A −0.190 V vs RHE el electrodo está **por debajo del equilibrio del
hidrógeno**, con 190 mV de sobrepotencial catódico y 0.225 mA cm⁻²
pasando: eso es **desprendimiento de hidrógeno corriendo durante los
once minutos del barrido**. Una reacción farádica, burbujas, y una
interfase que cambia mientras se la mide. El arco que se cerraba a
0.032 Hz era ese proceso; a +0.54 V no está.

**Tres conclusiones, y las tres cambian lo que se puede escribir:**

1. **El fallo de Kramers-Kronig de EB no era de la muestra ni del
   equipo. Era el potencial.** Es la demostración, no la hipótesis.
2. **La capacitancia de EB estaba subestimada más del doble.** A
   potencial comparable da 0.200 mF, no 0.084.
3. **Y el OCP de EB sigue siendo un resultado.** Que el potencial mixto
   de esa muestra caiga en la región del hidrógeno mientras el de las
   otras tres está medio voltio más arriba es una propiedad del
   material, no un accidente de la medida. En una muestra con mucho Fe
   y Fe₃C, eso apunta a **hierro accesible al electrolito**.

---

## Lo que esto arregla, y lo que sigue abierto

### Los potenciales, casi resueltos

| | V vs Ag/AgCl sat | **V vs RHE** |
|---|---|---|
| EA | +0.3105 | +0.525 |
| **EB an.** | +0.3249 | **+0.540** |
| E | +0.4489 | +0.664 |
| EP | +0.4853 | +0.700 |
| ~~EB cat.~~ | −0.4043 | −0.190 |

Cuatro de los cinco caen dentro de **0.18 V** entre sí, y **EB an. y EA
están a 13 mV**: ésa es una comparación directa, sin excusas. Antes la
tabla tenía 0.89 V de recorrido y nada era comparable.

### EP sigue sin pasar Kramers-Kronig

Residuo 1.65 % con rachas 0.41×. No hay corriente anómala que lo
explique (0.54 µA) y el potencial es el más anódico de los cinco:
+0.700 V vs RHE. La sospecha razonable es oxidación lenta de la
superficie del carbón a ese potencial, pero **no está demostrada**, y el
experimento que lo zanjaría es el mismo que para todo lo demás: medir de
alta a baja frecuencia y otra vez al revés.

**De EP y de EB cat. se pueden usar la forma, R_s, el codo y C′(f), que
son datos. No los parámetros de ningún circuito.**

---

## El ajuste: un circuito para todas

| | automático | residuo | **común: `R0-(R1\|Q1)-(R2\|Q2)`** |
|---|---|---|---|
| E | R0-(R1\|Q1)-T1 | 2.38 % | **0.71 %** |
| EA | R0-T1 | 3.18 % | **0.54 %** |
| EB cat. | R0-(R1\|Q1) | 9.75 % | **5.79 %** |
| EB an. | R0-(R1\|Q1) | 3.05 % | **1.83 %** |
| EP | R0-(R1\|Q1) | **17.83 %** | **5.11 %** |

Dos arcos ganan en todas a la vez. Es el circuito de la figura y el de
todas las columnas de arco de las tablas.

**Pero un residuo bajo no garantiza parámetros determinados.** EB an.
ajusta al 1.83 % y devuelve cinco de siete: sus dos arcos están a un
factor 3.5 en τ, y a esa distancia el reparto lo decide el punto de
partida más que los datos. El residuo mide descripción; la columna
«determinados» de `tablas.md` mide medida.

**Sólo EA tiene una resistencia iónica de poro medida**: 43.87 Ω ± 4.7 %
con el modelo físico `L0-R0-(R1|Q1)-T1` al 0.51 % de residuo. En los dos
EB la línea de transmisión colapsa con ±10⁸ % — en uno de ellos con un
residuo excelente, que es exactamente por lo que la incertidumbre se
mira y el residuo no basta.

---

## La capacitancia, y el problema que la masa destapó

Con los 0.909 mg ya se pueden dar F/g. El resultado es incómodo y hay
que decirlo:

| | V vs RHE | C′(0.01 Hz) | **F g⁻¹** | fase ahí |
|---|---|---|---|---|
| **E** | +0.664 | 0.827 mF | **0.910** | −56° |
| EP | +0.700 | 0.287 mF | 0.316 | −22° |
| **EB an.** | +0.540 | 0.200 mF | **0.220** | −47° |
| EA | +0.525 | 0.162 mF | 0.178 | −71° |
| EB cat. | −0.190 | 0.084 mF | 0.093 | −20° |

**Un carbón poroso publica de 50 a 300 F/g. Aquí el mejor da 0.91.** Eso
son dos órdenes de magnitud, y no es un error de cuenta: es lo que la
medida mide.

La causa está en la propia preparación. 0.909 mg sobre 0.0707 cm² son
**12.9 mg cm⁻²**, de diez a cien veces la carga habitual sobre un carbón
vítreo, o sea una película de decenas de micras. Una película así tiene
una resistencia iónica **en profundidad** que el ion no atraviesa en los
100 s que dura un punto a 0.01 Hz. Lo que estos espectros miden no es la
capacitancia del material: es **la fracción de la película a la que el
ion llega en 100 segundos**.

Y encaja con todo lo demás: ninguna muestra alcanza el régimen
capacitivo (la fase se queda entre −20° y −71°, un condensador está en
−90°), las cinco C′ son cotas inferiores, y las cinco tienen entre 0.5 y
1.8 décadas de fase cerca de −45°, que es justo la firma de un poro —o
de una película— que se está cargando por difusión.

> **El experimento que lo arregla es una tinta más diluida.** Con
> 0.05–0.1 mg cm⁻² (unos 2–5 µL de una tinta diez veces más diluida) la
> película baja a micras y el barrido actual ya la atraviesa. Eso, más
> bajar a 1 mHz, convierte esta tabla en capacitancias de verdad.

La masa es además **nominal**: sale del volumen depositado, no de pesar
el electrodo seco. 20 µL sobre un disco de 3 mm es mucho líquido para
ese diámetro, así que si parte se salió, la masa real es menor y los F/g
reales algo mayores. Sigue sin cambiar el orden de magnitud.

---

## Lo que sí se puede leer, muestra por muestra

### Ninguna cierra un semicírculo

Salvo el roce de EB cat. a 0.032 Hz —que era la reacción— en ninguna.
**Ninguna resistencia de transferencia de carga sale de los datos**: la
que devuelva un ajuste sale del modelo. Y en EB y EP, que además no
cruzan el eje real, **R_s es una cota superior**. En E y EA sí hay cruce
interpolado, y por encima de él Z″ se va positivo: inductancia del
cableado, no de la muestra.

### EA es la única con un electrodo poroso bien medido

```
L0-R0-(R1|Q1)-T1     residuo 0.51 %
  R0    = 46.48 Ω ± 0.2 %
  T1.Ri = 43.87 Ω ± 4.7 %      ← la resistencia iónica del poro
  T1.n  = 0.895  ± 2.6 %
```

Su fase llega a −71° a 0.01 Hz, la más capacitiva de las cinco, y su
Nyquist (panel a) es la única rama casi vertical. **Sólo EA se comporta
como el electrodo poroso que todas pretenden ser** — aunque almacene
menos que EB al mismo potencial, que son dos cosas distintas: una es
cómo entra el ion, la otra cuánta carga cabe.

### El codo ordena las muestras, y mide el dispositivo

| | codo (Hz) | τ (s) |
|---|---|---|
| E | 10.9 | 0.092 |
| EB an. | 25.2 | 0.040 |
| EA | 41.0 | 0.024 |
| EB cat. | 46.9 | 0.021 |
| **EP** | **712** | **0.0014** |

El codo es la frecuencia a la que la fase pasa por −45°. Lleva dentro la
R_s de la celda a propósito —es del **dispositivo**, no del material— así
que no tiene por qué ordenar igual que la resistencia del poro, y aquí
no lo hace. EP está un factor 65 por encima de E, y eso suena a
electrodo rápido: hay que resistirse. Con una fase que a 0.01 Hz sigue
en −22°, lo que EP tiene no es una respuesta capacitiva rápida sino
**poca capacitancia** repartida sobre una resistencia grande.

---

## La lectura química, con lo que aguanta y lo que no

**El control sin nitrógeno (E) sigue siendo el que más carga almacena**,
y ahora la comparación es más limpia que antes: cuatro de los cinco
espectros están dentro de 0.18 V. Si el objetivo del dopaje era subir la
capacitancia, estos espectros no lo apoyan. Pero antes de escribirlo hay
dos cosas que lo bloquean:

1. **Ninguno llegó al régimen capacitivo**, y la causa probable es la
   película gruesa, no el material.
2. **La masa es la misma para los cinco por protocolo, no por pesada.**
   Si el depósito varió entre electrodos, una diferencia de carga se come
   un factor de tres en F/g sin que nada lo delate.

**EA es la que se comporta como un carbón poroso.** El acetonitrilo es
el precursor de relación C/N más baja y el único cuyo nitrógeno entra
como nitrilo; que su electrodo sea el único con una resistencia de poro
medible y la fase más capacitiva es **consistente** con una porosidad
mejor desarrollada — consistente, no probado: esto es impedancia, no
área. **La prueba que lo zanja es fisisorción de N₂** (BET y
distribución de tamaño de poro), contrastada contra la `T1.Ri` de EA.

**EB tiene el OCP en la región del hidrógeno, y eso apunta al hierro.**
Su potencial mixto está medio voltio por debajo del de las otras tres y
0.19 V por debajo del equilibrio del H₂. En una muestra con alto
contenido de Fe y Fe₃C, eso apunta a **hierro accesible al electrolito**
— Fe⁰ o Fe₃C sin recubrir, catalizando HER. La bencilamina es la única
amina alifática primaria del grupo y el ligando que más razonablemente
complejaría hierro durante la síntesis; una partícula peor encapsulada
es coherente con todo lo medido. **No está demostrado aquí.** Lo
demostrarían XPS del Fe 2p y un barrido anódico lento buscando el pico
de disolución.

Y hay una lectura al revés que vale la pena: **si EB cataliza HER a
0.225 mA cm⁻² con 190 mV de sobrepotencial, eso es un resultado de
catálisis, no un problema de impedancia.** Medirlo cuesta un LSV, y esta
misma celda ya lo está pidiendo.

**EP tiene el codo más rápido y poca capacitancia**, que es la
combinación de un electrodo poco poroso con una interfase pequeña. Su
DRT da un único pico resuelto en τ = 0.067 s con 3.8 µF asociados — dos
órdenes por debajo de las demás. Eso es una interfase pequeña, no una
rápida.

---

## Lo que NO se puede decir con estos datos

- **Nada sobre la configuración del nitrógeno en el carbón.** El
  precursor dice qué nitrógeno entró al horno, no cuál salió: la
  pirólisis lo rehace. Piridínico, pirrólico y grafítico se separan con
  **XPS del N 1s**, y el paquete lo hace (`ramancarbon xps`).
- **Nada cuantitativo sobre Fe/Fe₃C.** La impedancia ve una interfase,
  no fases. Eso es DRX con Rietveld, que también está aquí.
- **Ninguna capacitancia específica del MATERIAL.** Los F/g de arriba
  son de la película tal como está depositada, y la película es gruesa.
- **Ninguna comparación de R_ct**, porque ninguna sale de los datos.
- **Nada de EB cat. ni de EP que venga de un circuito.**

---

## Qué mediría a continuación, por orden de rendimiento

1. **Una tinta diez veces más diluida**, 2–5 µL, para bajar a
   0.05–0.1 mg cm⁻². Es lo único que puede convertir estos F/g en una
   capacitancia del material, y cuesta una tarde.
2. **Repetir E, EA y EP al potencial de EB an.** (+0.54 V vs RHE), que
   es donde ya hay dos medidas comparables. Y cada espectro **en los dos
   sentidos de frecuencia**: si las dos pasadas no coinciden, ahí está la
   deriva de EP con nombre y apellidos.
3. **Bajar a 1 mHz.** Convierte las cotas inferiores en medidas y mete
   dentro del barrido el arco lento que ahora está extrapolado en E y EA.
4. **Pesar el electrodo seco**, antes y después, en vez de contar el
   volumen de tinta.
5. **Un LSV de HER sobre EB**, y sobre las otras tres para comparar. El
   PEIS ya enseñó que esa muestra hace hidrógeno.
6. **BET** sobre las cuatro, para contrastar contra la `T1.Ri` de EA.
7. **XPS N 1s y Fe 2p**, que es lo único que contesta la pregunta que
   motivó el experimento.

Con 1, 2 y 3 hechos, estos mismos scripts vuelven a correr sin tocar una
línea y las tablas salen comparables.

---

## Los archivos

| archivo | qué es |
|---|---|
| `peis_precursores.png` / `.pdf` | la figura de seis paneles, cinco espectros |
| `tablas.md` | todas las tablas, con las unidades y los avisos |
| `espectro_*.csv` | f, Z′, Z″, \|Z\|, fase, ajuste, residuo, C′, C″ |
| `drt_*.csv` | τ, γ(τ), y si cae dentro del intervalo medido |
| `comparacion.csv`, `ajustes.csv`, `capacitancias.csv` | las tablas en crudo |
| `informe_*.txt` | el informe en prosa de cada espectro |
| `resumen.json` | todo, incluidos el ranking de circuitos y los avisos |
| `exportar.py`, `figuras.py` | lo que lo produce |

Los sufijos `_EB_cat` y `_EB_an` son los dos potenciales de EB:
catódico (−0.19 V vs RHE) y anódico (+0.54 V).

**Z″ va con su signo físico** (negativo si es capacitivo) en todos los
CSV, no con el −Z″ del Nyquist, que es convenio de dibujo.

Los seis paneles: (a) Nyquist a escala uno a uno con el rincón en un
recuadro, (b) el rincón ampliado, (c) Bode con el codo marcado, (d) C′ y
C″ sin ningún circuito de por medio, (e) DRT normalizada y (f) el
residuo de cada ajuste.

En el panel (e) el peso se amontona contra el borde derecho. **Eso no es
un fallo del dibujo: es el resultado.** Falta barrido por debajo de
0.01 Hz, y la DRT acumula ahí todo lo que no se midió.
