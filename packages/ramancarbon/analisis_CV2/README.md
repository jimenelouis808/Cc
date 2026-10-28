# Serie 5–100 mV/s — S1 (900 °C) en KOH 0.5 M

Prueba de las configuraciones nuevas de la ventana sobre la serie
ampliada. Mismo electrodo y mismo material que `analisis_CV/`: carbono
vítreo de 3 mm, 20 µg de S1 (FeSe/C a 900 °C), KOH 0.5 M, Ag/AgCl de KCl
saturado, contraelectrodo de Pt.

## Antes que nada, dos cosas sobre los archivos

**Las velocidades son 5, 10, 20, 50 y 100 mV/s**, no 10–200. Es lo que
mide `|dE/dt|` de las propias columnas, que no depende de ningún ajuste:
duraciones de ciclo de 400, 200, 100, 40 y 20 s sobre una ventana de
1.000 V. Si en EC-Lab pusiste 10 y 200, algo se quedó por el camino —
merece una mirada antes de citar nada.

**Y 02, 03, 04 y 05 son byte a byte los mismos archivos de la vez
anterior.** El único nuevo es el 06. Eso no resta nada: lo que hace falta
para separar las contribuciones no es medir otra vez, es el brazo de
palanca, y el 06 lo dobla de un factor 10 a un factor 20. Pero conviene
saberlo, porque significa que esta serie y la anterior **no son medidas
independientes** y sus números no se promedian.

Lo que no hay en esta serie es una velocidad repetida, así que **no tiene
control de deriva**. El de la vez anterior sigue valiendo: entre las dos
medidas a 50 mV/s, el electrodo había perdido el 16 %.

## Lo que la configuración nueva hizo sola

Todo lo de abajo salió de cargar los cinco archivos y pulsar Analizar.
De la tarjeta Electrodo no hubo que tocar nada salvo la masa: el disco de
3 mm, el Ag/AgCl saturado y el KOH 0.5 M con su pH de 13.56 ya son los
valores de partida. El eje quedó en Ag/AgCl, que es el defecto, y la
figura lleva el eje RHE arriba porque lo pide el argumento.

## El resultado: ahora sí se separan los tres términos

Tablas en [`tablas.md`](tablas.md), figura en `cv_5a100.{png,pdf}`.

Con la serie anterior (factor 10) escribí que el b y el reparto de Dunn
**no se podían obtener**: las tres leyes de dos términos —`a·ν^b`, la de
Dunn `k₁ν + k₂√ν`, y `k_cap·ν + i_par`— ajustaban igual de bien en todos
los potenciales, y restar la fuga medida llevaba el b por encima de 1 en
un extremo y lo dejaba corto en el otro. **Con el factor 20 eso cambia**:
la ley de Dunn gana en 7 de los 9 potenciales.

Pero elegir entre pares sigue siendo la pregunta equivocada cuando se
sabe que los tres términos están. La pregunta correcta es ajustarlos a la
vez, y con cinco velocidades ya hay con qué:

    i = k_cap·ν + k_diff·√ν + i_par

y sale determinado, con R² de 0.9996 o mejor en toda la ventana:

| E / V vs RHE | k_cap / µF | k_diff / µA (V/s)^−½ | i_par / µA |
|---|---|---|---|
| 0.60 | 36.0 ± 1.6 | 28.4 ± 0.6 | **−1.48 ± 0.05** |
| 0.80 | 47.5 ± 3.7 | 18.2 ± 1.5 | −0.74 ± 0.12 |
| 1.00 | 52.2 ± 4.3 | 11.6 ± 1.7 | −0.09 ± 0.14 |
| 1.20 | 55.1 ± 3.2 | 10.4 ± 1.3 | +0.57 ± 0.10 |
| 1.30 | 55.2 ± 3.9 | 14.0 ± 1.5 | **+1.16 ± 0.12** |

Tres cosas de ahí, y la tercera es la buena:

1. **`k_cap` sube con el potencial** y `k_diff` baja: la parte superficial
   crece hacia el límite positivo y la difusiva domina en el negativo.
2. **`k_diff` NO es cero y está determinado** en los nueve potenciales.
   Hay un término en √ν de verdad, cosa que la serie de factor 10 no podía
   afirmar.
3. **`i_par` cambia de signo.** Es catódica en el extremo negativo
   (−1.5 µA) y anódica en el positivo (+1.2 µA), cruzando cero cerca de
   +0.03 V vs Ag/AgCl, o sea **1.03 V vs RHE**. Eso encaja exactamente con
   dónde está la ventana: el límite negativo (0.40 V vs RHE) está en plena
   región de reducción de oxígeno y el positivo (1.40 V vs RHE) está
   169 mV por encima del equilibrio del O₂. La parásita no es «una fuga»:
   son **ORR abajo y OER arriba**, y el cruce por cero está donde ninguna
   de las dos corre.

Esa tercera es, además, lo que hizo falta arreglar en el programa: yo
había puesto `i_par ≥ 0`, y forzar una corriente parásita a ser anódica
hacía que el ajuste pusiera 11 µF de capacitancia donde hay 50 y llamara
fuga al resto.

### Lo que sigue sin poder separarse

`k_cap` baja un 23 % al añadir el punto de 100 mV/s —de 64 a 50 µF en
0.90 V vs RHE— mientras el ajuste declara ±8 %. El 100 mV/s se midió el
último, así que eso es degradación entrando en la serie, y es **mayor que
la incertidumbre que el ajuste se atribuye**. `k_diff` e `i_par` apenas
se mueven entre las dos series: lo que el electrodo pierde es
almacenamiento, no el proceso difusivo ni la reacción parásita.

Sin una velocidad repetida al final no hay forma de corregirlo desde
dentro de la serie.

### La capacitancia

| convenio | 5 mV/s | 100 mV/s |
|---|---|---|
| lazo ÷ 2νΔV | 11.33 | 4.85 |
| **carga de reducción ÷ ΔV** | **7.39** | **4.73** |

La eficiencia coulómbica va de **0.48 a 0.95**: a 100 mV/s el ciclo dura
20 s y la parásita casi no tiene tiempo de correr, así que los dos
convenios convergen. Ése es el argumento más limpio de toda la serie para
que la fuga sea lo que separa los dos números.

## Qué pediría ahora

1. **Repetir 5 o 10 mV/s al final.** Es lo único que separa la
   degradación de la cinética, y sin ello el `k_cap` de la tabla lleva la
   deriva dentro.
2. **Bajar el límite positivo** a ≈ −0.2 V vs Ag/AgCl (0.80 V vs RHE): sin
   OER posible, la mitad anódica de `i_par` desaparece.
3. **Purgar con N₂ o Ar**, que es lo que quita la otra mitad — la ORR del
   extremo negativo, que es el `i_par` negativo de la tabla.
4. Con eso, **el mismo ajuste de tres términos** debería devolver `i_par`
   compatible con cero en toda la ventana, y entonces `k_diff` sí es una
   medida de difusión y no una conjetura.

## Los archivos

| archivo | qué es |
|---|---|
| `exportar.py` | carga los `.mpr` por la sesión de la ventana y escribe todo lo demás |
| `figuras.py` | la figura de cuatro paneles |
| `resumen.json` | todos los números, condiciones incluidas |
| `ciclos.csv`, `velocidades.csv`, `leyes.csv`, `dunn.csv` | lo que escribe «Exportar tablas…» |
| `ciclo_NmVs.csv` | el ciclo informado de cada velocidad: t, E, i, i/m |
| `estudio_de_velocidad.txt` | la salida completa del estudio y de Dunn |

```bash
cd packages/ramancarbon
uv run python -I analisis_CV2/exportar.py <los cinco .mpr>
uv run python    analisis_CV2/figuras.py
```
