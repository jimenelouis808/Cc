# Serie 5–200 mV/s — S1 (900 °C) en KOH 3 M

Carbono vítreo de 3 mm (0.0707 cm²), 20 µg de S1 (FeSe/C a 900 °C),
**KOH 3 M**, Ag/AgCl de KCl saturado, contraelectrodo de Pt. Seis
velocidades: 5, 10, 20, 50, 100 y 200 mV/s — **un factor 40**.

## Dos correcciones sobre lo anterior

**El electrolito es 3 M, no 0.5 M**, y eso no es un detalle de la ficha:

| | KOH 0.5 M | KOH 3 M |
|---|---|---|
| pH (de la actividad) | 13.56 | **14.51** |
| E(RHE) = E + … | 0.999 V | **1.055 V** |
| ventana vs RHE | 0.399 – 1.399 V | **0.455 – 1.455 V** |
| η en el límite positivo | +169 mV | **+225 mV** |

Son 57 mV en toda la escala RHE y por tanto 57 mV en el sobrepotencial,
que es el número del que cuelga la lectura del límite positivo. Con el
electrolito correcto, ese límite está **225 mV por encima** del
equilibrio del oxígeno, no 169.

**Y las velocidades medidas son 5–200**, no 10–200: los archivos 02 a 07
dan 5, 10, 20, 50, 100 y 200 mV/s, medido como `|dE/dt|` de las propias
columnas (ciclos de 400, 200, 100, 40, 20 y 10 s sobre 1.000 V).

## Las gráficas, panel por panel

La figura es `cv_serie.{png,pdf}`. Los cuatro paneles contestan cuatro
preguntas distintas y están puestos en el orden en que hay que leerlos.

---

### (a) Los seis voltamperogramas

**Qué hay.** El último ciclo completo de cada archivo, con la corriente
dividida por la masa activa (20 µg), así que el eje y es A/g y las seis
curvas son comparables aunque abarquen un factor 40 en corriente. Eje x
inferior: el potencial **tal como se midió**, frente a Ag/AgCl. Eje x
superior: el mismo potencial en la escala **RHE**, que es la que tiene
sentido físico fijo (los equilibrios del H₂ y del O₂ no se mueven con el
pH en esa escala). La **línea roja a trazos** es el equilibrio del
oxígeno, 1.23 V vs RHE.

**Cómo se lee.** Un condensador ideal da un rectángulo: corriente
constante en cada rama, y las dos ramas simétricas respecto de cero.
Aquí:

- La forma es **de caja redondeada**, no de rectángulo: hay resistencia
  en serie (las esquinas se curvan) y los extremos de la ventana suben.
- Las dos ramas **no son simétricas**. La rama catódica (la de abajo) es
  mucho más profunda en el extremo negativo que la anódica en el
  positivo, y eso es lo que el panel (b) cuantifica.
- **No hay picos redox definidos**, sólo jorobas anchas. Por eso la
  clasificación de mecanismo no dice «batería» por la forma.
- La curva a 200 mV/s cruza a la derecha de la línea roja con corriente
  anódica creciente: ahí se está evolucionando oxígeno.

**Lo que hay que mirar.** Si las seis curvas se superpusieran al dividir
por ν (hay una casilla en la ventana que lo hace: «Dividir la corriente
por la velocidad»), el electrodo sería puramente capacitivo. Lo que no se
superponga es lo que no es capacitivo.

---

### (b) Capacitancia y eficiencia contra la velocidad

**Qué hay.** Tres series sobre un eje x logarítmico de velocidad:

- **Negro, `lazo ÷ 2νΔV`**: el convenio habitual, `∮|i|dt / (2νΔV)`.
  Integra el valor absoluto de la corriente sobre el ciclo entero.
- **Morado, `carga de reducción ÷ ΔV`**: sólo la carga que VUELVE,
  `Q_red/ΔV`. Es la que no contamina una corriente parásita anódica.
- **Rojo a trazos, eje derecho**: la eficiencia coulómbica `Q_red/Q_ox`.

**Cómo se lee.** En un condensador los dos convenios coinciden y la
eficiencia vale 1. Aquí:

| ν / mV s⁻¹ | C_lazo | C_red | Q_red/Q_ox |
|---|---|---|---|
| 5 | 11.33 | 7.39 | **0.484** |
| 50 | 5.81 | 5.52 | 0.904 |
| 200 | 4.15 | **4.09** | **0.971** |

A 5 mV/s los dos convenios difieren un 53 % y **más de la mitad de la
carga de oxidación no vuelve**. A 200 mV/s difieren un 1.5 % y vuelve el
97 %. Eso es la firma de una corriente parásita: el ciclo a 200 mV/s dura
10 s y la parásita casi no tiene tiempo de correr; el de 5 mV/s dura
400 s y le da de sobra.

**Lo que hay que mirar.** La distancia entre la curva negra y la morada
ES la fuga. Si se tocan, el convenio da igual; si se separan, hay que
citar la morada y declarar la eficiencia al lado.

---

### (c) El balance de carga contra la duración del ciclo

Este panel es el que mide la fuga, y es el único de los cuatro que no
tiene nada que ver con el potencial.

**Qué hay.** Para el ciclo informado de cada velocidad, tres cargas
contra la **duración** de ese ciclo (no contra ν: la duración es lo que
importa para algo que corre con el tiempo):

- **Círculos negros, `Q_ox`**: ∫i dt sobre los tramos de corriente
  positiva.
- **Cuadrados morados, `Q_red`**: lo mismo con la corriente negativa.
  La línea gris de puntos es su media.
- **Triángulos rojos, `ΔQ = Q_ox − Q_red`**: la carga NETA del ciclo.
- **Recta roja**: el ajuste `ΔQ = ⟨i_par⟩·t`.

**Por qué ΔQ es la magnitud correcta.** `ΔQ = ∮ i dt` es, por
definición, la carga que pasó y no volvió. Una carga almacenada no
aporta nada a esa integral sea cual sea su mecanismo —entra y sale—, así
que **sólo un proceso irreversible contribuye**. La pendiente contra el
tiempo es por tanto la corriente parásita promediada sobre el ciclo, sin
suponer nada sobre cómo escala con ν.

**El resultado.** ⟨i_par⟩ = **0.40 µA**, R² = 0.995, con una ordenada en
el origen de −5 µC sobre cargas de 5 a 300 µC. **Esa ordenada tiene que
ser cero**, y que lo sea es la comprobación del modelo, no un detalle
estético: si saliera grande, la carga neta no sería proporcional al
tiempo y el número no significaría nada.

**Lo que hay que mirar.** Que `Q_red` (morado) sea casi plano mientras
`Q_ox` (negro) crece con la duración. Eso dice que lo que se almacena no
depende de cuánto dure el ciclo —como debe ser— y que lo que crece es
otra cosa.

---

### (d) Las tres contribuciones a la corriente, potencial a potencial

Es el panel nuevo, y el que la serie de factor 10 no permitía dibujar.

**Qué hay.** En nueve potenciales se ajusta, con las seis velocidades:

    i(ν) = k_cap·ν + k_diff·√ν + i_par

y se dibujan los tres coeficientes con su barra de error:

- **Círculos negros, `k_cap` (µF)**: el término proporcional a ν. Es la
  parte **superficial** — doble capa más cualquier redox confinado en la
  superficie, que es lo que significa «pseudocapacitivo».
- **Cuadrados morados, `k_diff`**: el término en √ν, la parte
  **limitada por difusión**.
- **Triángulos rojos a trazos, eje derecho, `i_par` (µA)**: el término
  que **no depende de ν**. La línea roja de puntos marca el cero.

**Por qué los tres juntos y no el b ni Dunn.** El b-value ajusta
`i = a·ν^b` y el reparto de Dunn ajusta `k₁ν + k₂√ν`: los dos suponen
que hay dos contribuciones. Una corriente constante no es ninguna de las
dos, y sobre una década de velocidad una constante y un `k₂√ν` son casi
la misma función, así que el término difusivo se la traga y la devuelve
como difusión. Con un factor 40 y seis puntos ya hay con qué ajustar los
tres a la vez.

**El resultado.** R² ≥ 0.9996 en los nueve potenciales, y los tres
coeficientes salen **determinados** (mayores que el doble de su propio
error) en ocho de los nueve:

| E / V vs RHE | k_cap / µF | k_diff | i_par / µA |
|---|---|---|---|
| 0.555 | 48.8 ± 4.7 | 18.5 ± 2.4 | −0.62 ± 0.25 |
| 0.655 | 48.1 ± 3.0 | 24.0 ± 1.6 | **−1.17 ± 0.16** |
| 0.955 | 45.5 ± 1.7 | 16.0 ± 0.9 | −0.51 ± 0.09 |
| 1.155 | 48.9 ± 1.8 | 12.0 ± 0.9 | +0.09 ± 0.09 *(cruza cero)* |
| 1.355 | 53.3 ± 1.3 | 14.7 ± 0.7 | **+1.11 ± 0.07** |

**Cómo se lee, y es lo más importante de toda la figura:**

1. **`k_cap` es casi plano**, 45–53 µF en todo el recorrido. Una
   capacitancia de doble capa debe serlo, y que lo sea dice que el
   ajuste está separando bien.
2. **`k_diff` no es cero** y está determinado en los nueve potenciales:
   hay un término difusivo de verdad. Es máximo en 0.655 V y cae hacia
   el límite positivo.
3. **`i_par` cambia de signo.** Es catódica (−1.2 µA) en el extremo
   negativo, cruza cero cerca de **1.13 V vs RHE** y es anódica
   (+1.1 µA) en el positivo. El único potencial donde queda indeterminado
   es, precisamente, el del cruce.

**Y eso tercero es un resultado físico, no un artefacto.** La ventana va
de 0.455 a 1.455 V vs RHE. El extremo negativo está en plena región de
**reducción de oxígeno** y el positivo está 225 mV por encima del
equilibrio del O₂, o sea en **evolución de oxígeno**. La corriente
parásita no es «una fuga»: son dos reacciones distintas, una en cada
extremo, y el cruce por cero está donde ninguna de las dos corre con
fuerza. El cruce medido (1.13 V) queda 100 mV por debajo del equilibrio
del O₂ (1.23 V), lo cual es compatible con que haya además una oxidación
del propio hierro de la muestra empezando por ahí — pero eso es una
hipótesis que esta medida no zanja.

---

## El efecto sobre la clasificación de mecanismo

El b baja a 0.55 en el extremo positivo, y 0.55 se lee tradicionalmente
como «controlado por difusión, tipo batería, informar en mAh/g». El
programa **no** lo hace, y dice por qué: en ese mismo potencial el
ajuste de tres términos determina una corriente parásita que es el
**31 %** de la corriente total a la velocidad más lenta. Una constante
dobla el b hacia abajo sin que haya difusión de por medio, así que con
esa fracción el b está midiendo dos cosas a la vez.

El veredicto es **pseudocapacitivo, F/g con reservas**, y el término
difusivo se informa por su propio nombre (`k_diff`) con su
incertidumbre, que es donde sí es una medida.

## Lo que sigue sin poder separarse

La serie **no tiene velocidad repetida**, así que no hay control de
deriva. Se sabe por la tanda anterior que el electrodo perdió un 16 %
entre las dos medidas a 50 mV/s, y el 200 mV/s se midió el último, así
que parte de su caída es degradación y no cinética. Repetir 5 o 10 mV/s
al final lo cierra.

## Qué pediría ahora

1. **Repetir la primera velocidad al final.**
2. **Bajar el límite positivo** a ≈ −0.25 V vs Ag/AgCl (0.80 V vs RHE):
   sin OER posible, la mitad anódica de `i_par` desaparece.
3. **Purgar con N₂ o Ar**, que quita la otra mitad.
4. **Cambiar a Hg/HgO**: un Ag/AgCl en KOH 3 M deriva, y es la referencia
   adecuada en alcalino.
5. Con eso, el mismo ajuste de tres términos debería dar `i_par`
   compatible con cero, y entonces `k_diff` es una medida de difusión sin
   reservas.

## Los archivos

| archivo | qué es |
|---|---|
| `exportar.py` | carga los `.mpr` por la sesión de la ventana y escribe todo lo demás |
| `figuras.py` | la figura de cuatro paneles |
| `tablas.md` | las cuatro tablas |
| `resumen.json` | todos los números, condiciones incluidas |
| `ciclos.csv`, `velocidades.csv`, `leyes.csv`, `dunn.csv` | lo que escribe «Exportar tablas…» |
| `ciclo_NmVs.csv` | el ciclo informado de cada velocidad: t, E, i, i/m |
| `estudio_de_velocidad.txt` | la salida completa del estudio y de Dunn |

```bash
cd packages/ramancarbon
uv run python -I analisis_CV2/exportar.py <los seis .mpr>
uv run python    analisis_CV2/figuras.py
uv run python    analisis_CV2/tablas.py > analisis_CV2/tablas.md
```
