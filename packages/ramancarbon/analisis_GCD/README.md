# GCD: FeSe a 900 °C en KOH 3 M

Cinco archivos GCPL de la misma muestra, 20 µg sobre un disco de carbón
vítreo de 3 mm, ventana −0.6 a +0.4 V frente a Ag/AgCl saturado.

```bash
uv run python analisis_GCD/exportar.py <carpeta con los .mpr>
```

## Lo primero: qué son estos archivos

Cada uno tiene **1001 ciclos completos** y **tres puntos por semiciclo**.
EC-Lab guardó con condición de potencial, así que de cada rama están el
principio, el final y poco más. Eso decide qué se puede medir y qué no:

- la carga `I·Δt`, la ventana `ΔV` y por tanto la capacitancia `I·Δt/ΔV`
  **sí**, porque sólo necesitan los dos extremos;
- la pendiente local, la linealidad y la capacitancia por energía **no**,
  y se niegan con su motivo en vez de calcularse de dos puntos.

Y los cinco corren **seguidos en el reloj del potenciostato**, de 0 a
57 386 s. El de 5 µA fue el primero y el de 40 µA el último, así que la
corriente sube a la vez que el tiempo acumulado: **velocidad y
degradación apuntan en la misma dirección y una sola serie no las
separa**. Lo que sí las separa son los ciclos de frontera, que están en
`ciclos_*.csv`: el último del archivo anterior y el primero del
siguiente son adyacentes en el tiempo y difieren sólo en la corriente.

## El archivo asimétrico

El quinto carga a **+60 µA** y descarga a **−20**. Un par así no tiene UNA
densidad de corriente, así que por defecto se analiza, sale en las tablas
con su marca, y se queda fuera de la curva contra corriente y del Ragone.
`tablas.md` lo compara donde sí se puede: a 20 µA de descarga da
**1.977 F/g** frente a los **2.270** que interpolan los puntos simétricos,
un **−12.9 %**. El horario roto sí cambió la respuesta.

Su resistencia, en cambio, sale bien: el salto abarca la SUMA de las dos
corrientes (80 µA), no el doble de la de descarga, y con esa cuenta los
cinco archivos coinciden en 72–86 Ω.

## La fuga

La eficiencia culómbica está por encima del 100 % en los cinco, de 101 a
111 %. Eso es carga que sale sin haber entrado, y como porcentaje no se
puede comparar entre corrientes. Dividido por el ciclo entero da una
corriente parásita de **0.19 a 0.25 µA** a lo largo de un factor de ocho
en la corriente aplicada: es la misma fuga vista cinco veces.

## El salto del archivo de 5 µA

Pierde el 80 % de su capacitancia (391 → 77 µF), y **no gradualmente**:
en 1001 ciclos hay exactamente dos caídas de más del 20 %, el ciclo 1
—que es formación— y el **ciclo 319**, que se lleva el **67 %** de una vez
(298.7 → 99.9 µF) mientras la eficiencia culómbica apenas se mueve. Eso
no es degradación: es material que se desprende, un contacto que cambia
o una burbuja. El modelo de desvanecimiento devuelve «raíz de ciclos»
para esa curva, y lo que está describiendo es el escalón.

## Las figuras de artículo: cuáles de estas medidas las soportan

Las dos figuras que se piden de un electrodo —la **capacidad de
velocidad** y el **ciclado**— salen de estos archivos y están en
`gcd_articulo.png`. Una tercera, muy publicada, **no sale**, y conviene
saber por qué antes de intentarlo.

### (a) Retención contra corriente, normalizada

El panel que se compara entre muestras: el % de lo que dio la corriente
más baja. Esta muestra retiene el **50 %** al pasar de 0.25 a 2 A/g, un
factor de ocho en corriente.

**El eje es una CORRIENTE, no una «C».** Una C está definida contra una
capacidad nominal —1C es la corriente que vacía el electrodo en una
hora— así que presupone un electrodo de batería con meseta y con una
capacidad declarada. Un carbón capacitivo no tiene esa capacidad, y
poner microamperios en un eje rotulado «C» se la inventa. En A/g
cuando hay masa, en µA cuando no.

### (b) Capacidad y eficiencia culómbica contra ciclo

Las dos en una caja con eje gemelo, que es como se imprime. No es
decoración: capacidad que cae con la eficiencia clavada en 100 % es
material que se pierde; capacidad que aguanta con la eficiencia fuera
del 100 % es carga que se va a otro sitio cada ciclo. En dos paneles
separados, el ojo tiene que llevar una al otro.

Aquí la eficiencia está **por encima** del 100 % en los cinco archivos,
de 101 a 111 %, que es la fuga de la sección anterior.

### La que NO se puede hacer: E contra capacidad específica

Las curvas de voltaje frente a capacidad para el ciclo 4, 25, 50, 75 y
100 —el panel que acompaña a toda figura de celda— **no se pueden
dibujar con estos archivos**, y no es una limitación del programa:

| archivo | puntos | semiciclos | puntos por semiciclo |
|---|---|---|---|
| GCPL_01 | 8 565 | 2 002 | 4 (hasta 6) |
| GCPL_02–05 | 6 006 | 2 002 | **3** |

Tres puntos por rama son el principio, el final y uno en medio. Con eso
hay carga y ventana —y por tanto capacitancia— pero no hay FORMA: ni
meseta, ni curvatura, ni el perfil de la caída óhmica. Un panel de E
contra capacidad trazado con tres puntos es tres segmentos rectos que
no dicen nada del material.

**Lo que lo arregla es la condición de guardado de EC-Lab**: grabar por
tiempo (o cada pocos mV) en vez de por condición de potencial. Con 50–100
puntos por rama ese panel sale solo, con este mismo código.

## Y una advertencia sobre la figura de celda completa

Las figuras de celda —N/P, Wh/kg de cátodo+ánodo, bolsa— **no se pueden
obtener de una medida a tres electrodos**, por mucho que se procesen.
Una celda de tres electrodos mide el potencial de UN electrodo frente a
una referencia: no tiene voltaje de celda, no tiene masa de
contraelectrodo y por tanto no tiene relación N/P ni densidad de
energía de dispositivo. El Ragone que sale de aquí (panel f de
`gcd_serie`) es **por gramo del material activo de este electrodo** y
lleva esa base escrita; no es comparable con un Wh/kg de celda, que
incluye el otro electrodo, el separador, el electrolito y el empaque, y
suele ser de tres a cinco veces menor.

Para una figura de celda hace falta montar la celda: dos electrodos,
las dos masas, y el voltaje entre ellos.

## Qué hay aquí

| archivo | qué es |
|---|---|
| `tablas.md` | las tres tablas en markdown |
| `serie.csv` | un punto por corriente |
| `ciclado.csv` | lo que mil ciclos le hacen a cada archivo |
| `ciclos_*.csv` | **cada ciclo** de cada archivo: q, C y eficiencia |
| `informe.txt` | el análisis completo con todos sus avisos |
| `resumen.json` | lo mismo, para leerlo desde otro guion |
| `gcd_serie.png/.pdf` | la figura de seis paneles |
| `gcd_articulo.png/.pdf` | los dos paneles de artículo: retención contra corriente y capacidad con eficiencia |
| `figura/`, `figura_articulo/` | los números DIBUJADOS en cada panel de cada figura |

La figura no está escrita a mano: los paneles salen del catálogo de la
ventana (`gui/figures_echem.py`), así que lo que se publica y lo que se
ve en pantalla no pueden separarse.
