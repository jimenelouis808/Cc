# HER: FeSe@C (carbón dopado con N y Se) a 900 °C, en H₂SO₄ 0.5 M

Dos barridos lineales del mismo electrodo, a 1 y 5 mV/s, 20 µg sobre un
disco de carbón vítreo de 3 mm contra Ag/AgCl (KCl sat.).

```bash
uv run python analisis_HER/exportar.py <carpeta con los .mpr>
```

## Lo que la impedancia y las retenciones zanjaron

**R_s = 1.88–1.93 Ω** en los tres espectros, así que la corrección óhmica
a 10 mA/cm² vale **1.3 mV**. El aviso de los «43 Ω» queda contestado por
la medida. Con la corrección puesta: **η₁₀ = 29 mV**, η₅₀ = 53, η₁₀₀ = 70,
Tafel **36.7 mV/dec**.

**Y las retenciones matan la hipótesis del platino**: a 10 mA/cm² el
electrodo deriva **+57 y +92 mV/h**, con la segunda más rápida que la
primera. El platino en ácido a esa corriente aguanta cientos de horas sin
moverse. Lo que queda es un material muy activo al principio y poco
estable, que es el patrón conocido de los calcogenuros de hierro en
ácido.

**No termina una prueba de diez horas**: a 92 mV/h gana otros 100 mV en
1.1 h.

## Dos artefactos, medidos

**Hay una corriente catódica sostenida por encima del equilibrio del
hidrógeno**, donde no puede circular ninguna: 0.31 mA/cm² a 1 mV/s y 0.24
a 5. En una celda acuosa sin desairear eso es **reducción del oxígeno
disuelto**, cuya corriente límite en un disco de milímetros en ácido al
aire son justo unas décimas de mA/cm². Se suma al barrido, así que la
curva llega a cualquier densidad antes de lo que llegaría la reacción
sola. Explica entero el sobrepotencial de inicio negativo.

**Y el η a 10 mA/cm² cae en el intervalo del platino** (20–35 mV). Para
un catalizador no noble eso se explica casi siempre por **platino
disuelto del contraelectrodo** y redepositado en el de trabajo: bastan
nanogramos, y después ninguna medida electroquímica lo ve, porque lo que
se mide es platino de verdad. Si el contraelectrodo era de Pt, hay que
descartarlo antes de citar nada.

## Léelos en este orden

**1. El sobrepotencial de inicio sale NEGATIVO** en las dos curvas: −11 mV
a 1 mV/s y −18 mV a 5. Eso es imposible. Un sobrepotencial se mide desde
el potencial termodinámico de la reacción, así que uno negativo dice que
el hidrógeno se desprende antes de que la termodinámica lo permita. Lo
que informa de verdad es un **desplazamiento del eje de potencial**, y lo
heredan todos los demás números: η₁₀, la pendiente de Tafel y j₀.

Tres causas, en orden de probabilidad:

- el electrodo de referencia no está donde dice la tabla (0.197 V para
  Ag/AgCl KCl sat.). Se contrasta contra uno nuevo, o mejor contra el
  potencial de equilibrio del hidrógeno en un hilo de platino en este
  mismo electrolito, que es la calibración que vale para esto;
- el pH no es el nominal 0.3 que la base de datos da para H₂SO₄ 0.5 M;
- lo que se mide cerca del inicio no es la reacción sino un fondo. Hay
  unos 20 µA (≈0.28 mA/cm²) corriendo desde +0.48 V, y el inicio está
  definido a 1 mA/cm².

**2. No hay corrección de caída óhmica, y aquí no es un detalle.** A
10 mA/cm² circulan 0.707 mA, así que **bastarían 43 Ω** de resistencia no
compensada para explicar entero el η de 30 mV que se informa. Una celda
de laboratorio con el capilar de Luggin cerca anda justo por ahí. Sin
medirla, este número no distingue un catalizador bueno de una celda con
poca caída.

Hace falta la impedancia de **esta misma celda**: el corte de alta
frecuencia. Con eso el análisis la aplica y los números pasan a
significar algo.

**3. Las dos velocidades no caen una sobre otra.** Una curva de
polarización se supone que ES el estado estacionario, así que tienen que
coincidir. Separan 5 mV a 10 mA/cm², 5 a 50 y 8 a 100, y **la lenta
siempre informa MÁS sobrepotencial**.

Eso tiene dirección y la dirección dice qué es. Un barrido rápido lleva
más corriente capacitiva, así que llega antes a una densidad dada y lee
MENOS sobrepotencial — eso explicaría el signo. Pero la separación
**crece** con la corriente (5 → 8 → 15 mV a 10, 100 y 200 mA/cm²), y un
desplazamiento capacitivo haría lo contrario. Lo que sí crece con la
corriente es el tiempo que el electrodo pasa desprendiendo gas: el
barrido de 1 mV/s dura 350 s contra 166 s, y llega a 236 mA/cm² donde el
rápido llega a 315 **al mismo potencial final**. La lectura honrada es
que el electrodo se degrada durante la medida, y que la de 1 mV/s, por
ser más lenta, sale peor.

Es la misma barrera que ya tienen el lado de CV y el de GCD: sin volver a
la condición inicial, una limitación cinética y una degradación dan la
misma curva. Aquí falta el barrido de vuelta a 1 mV/s al final.

## Los números, con todo lo anterior puesto

| v (mV/s) | η₁₀ | η₅₀ | η₁₀₀ | Tafel | η inicio |
|---|---|---|---|---|---|
| 1 | 30 mV | 59 mV | 83 mV | 42.4 mV/dec (1.85 déc, R² 0.981) | −11 mV |
| 5 | 26 mV | 55 mV | 75 mV | rechazada: 0.98 décadas | −18 mV |
| **dispersión** | **5 mV** | **5 mV** | **8 mV** | | |

La pendiente de Tafel de 5 mV/s se **rechaza**, no se informa: su tramo
lineal no llega a una década, y una pendiente de Tafel sobre menos de una
década no es una pendiente de Tafel.

Los 42.4 mV/dec de la curva lenta, **si el desplazamiento del eje no los
toca** —y sí los toca—, estarían en el régimen de Volmer-Heyrovsky. No lo
cites hasta arreglar los puntos 1 y 2.

## La escala de corriente está comprobada

Los 16.7 y 22.3 mA máximos son reales: integrando `I dt` contra la
columna `(Q-Qo)/C` que el propio archivo trae, coinciden al **0.1 %** en
los dos. Así que las densidades de 236 y 315 mA/cm² no son un error de
unidades ni de área — son lo que se midió.

## Qué hay aquí

| archivo | qué es |
|---|---|
| `tablas.md` | la tabla por velocidad con su dispersión |
| `velocidades.csv` | lo mismo, separado por comas |
| `curva_*.csv` | η contra \|j\| de cada barrido |
| `informe.txt` | el análisis completo con todos sus avisos |
| `resumen.json` | para leerlo desde otro guion |
| `her.png/.pdf` | los tres paneles |
| `figura/` | los números dibujados en cada panel |
