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
| `figura/` | los números DIBUJADOS en cada panel |

La figura no está escrita a mano: los paneles salen del catálogo de la
ventana (`gui/figures_echem.py`), así que lo que se publica y lo que se
ve en pantalla no pueden separarse.
