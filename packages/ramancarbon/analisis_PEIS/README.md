# Impedancia de tres carbones en H₂SO₄ 0.5 M

Pirolizados a **800, 900 y 1020 °C**. Tres espectros PEIS de 100 kHz a
0.0999 Hz, 37 frecuencias, 10 mV de amplitud.

```
uv run python -I analisis_PEIS/exportar.py <800>.mpr <900>.mpr <1020>.mpr
uv run python -I analisis_PEIS/figuras.py
```

`exportar.py` pasa por `EchemSession`, el mismo camino que usa la ventana,
así que lo que escribe aquí y lo que se ve en la pestaña Impedancia no
pueden separarse. Los números están en [`tablas.md`](tablas.md); esto
explica la figura.

## Lo primero: dos cosas que el usuario no dio

- **La masa.** Así que **no hay ni un F/g** en ningún sitio de este
  análisis. Una capacitancia específica con la masa supuesta está mal
  exactamente en la proporción que se suponga, y nada aguas abajo lo
  delata.
- **El área.** Se usa el disco de carbón vítreo de 3 mm (0.0707 cm²) que
  el usuario dice usar casi siempre, y va marcado como **suposición** en
  cada tabla. Además es área GEOMÉTRICA: la electroquímicamente activa de
  un carbón poroso es de diez a mil veces mayor, así que los mF/cm² solo
  comparan estas tres muestras entre sí.

Si tienes los dos números, están en la barra lateral de la sección de
electroquímica y todas las tablas se rehacen.

---

# `peis_serie.png` — los seis paneles

> **La hace la ventana, no un script.** `figuras.py` ya no dibuja nada:
> monta la misma `EchemSession` que usa la sección de electroquímica,
> elige seis paneles del catálogo y llama a `save_figure` — que es el
> botón «Guardar figura…» de la pestaña Figura. Las etiquetas de los
> ejes van en inglés y la ventana sigue en español: son dos públicos
> distintos.
>
> **Y el formato compacto que usabas también está ahí**, como panel
> «Nyquist y Bode en la misma caja». No es el que se usa aquí, y la
> razón está al final de este archivo.

## (a) Los tres Nyquist completos, en log-log

**Qué hay.** Z′ contra −Z″ de los tres espectros (círculos, los datos) con
el ajuste encima (línea continua). Los dos ejes son logarítmicos porque el
espectro recorre cuatro décadas de |Z|: en lineal, los cuatro puntos del
extremo bajo ocuparían la figura entera y el resto sería un punto en el
origen. La recta discontinua gris es pendiente 45°, que es lo que hace un
RC distribuido.

**Cómo se lee.** De derecha a izquierda es de baja a alta frecuencia. El
tramo final, casi vertical en lineal, aquí sale como la parte de arriba
donde las curvas se separan de la recta de 45° y suben más deprisa: eso es
la cola capacitiva. El tramo que corre paralelo a la recta gris es el
poro llenándose.

**Lo que hay que mirar.** Que **ninguna de las tres curva hacia el eje
real**. Un semicírculo cerrado bajaría de vuelta; −Z″ aquí sube monótona
hasta el último punto medido en las tres muestras. Eso quiere decir que de
estos datos NO sale ninguna resistencia de transferencia de carga: la que
devuelva un ajuste sale del modelo, no de la medida, y el informe lo dice
en voz alta.

> Esto era, además, un fallo del programa. La «frecuencia de codo» se
> calculaba como el máximo global de −Z″, que en cualquier cosa con cola
> capacitiva es **el último punto del barrido**. En estos tres archivos
> devolvía 0.0999 Hz y τ = 10 s, las tres veces.

## (b) El rincón de alta frecuencia, lineal y cuadrado

**Qué hay.** La esquina de abajo a la izquierda del Nyquist, hasta seis
veces la mayor R_s. Ejes lineales, misma escala en los dos y caja
cuadrada: un diagrama de Nyquist se lee por la FORMA, y una relación de
aspecto distinta de uno convierte una recta de 45° en el ángulo que las
dimensiones de la caja quieran.

**Cómo se lee.** Los triángulos ▼ sobre el eje son la R_s **medida de los
datos** (cruce interpolado con el eje real, o Z′ a 100 kHz cuando no
cruza), no la R0 del ajuste. La recta gris vuelve a ser pendiente 45°.

**Lo que hay que mirar.** Tres cosas:

1. **El orden de las R_s**: 11.4, 20.5 y 25.5 Ω. Es contacto y colector,
   no muestra; la disolución es la misma para las tres.
2. **El tramo a 45°** antes de que la curva se enderece. Esa es la región
   de De Levie: a alta frecuencia el ion solo ha entrado en la boca del
   poro. Mide 0.67, 0.50 y 0.50 décadas.
3. **Dónde se endereza cada una.** La verde (900 °C) sube casi inmediata
   después de su R_s; la roja (1020 °C) arrastra mucho más. Esa diferencia
   es la resistencia iónica dentro del poro, y es la que el ajuste cifra
   en 0.79 Ω contra 16.0 Ω.

## (c) Bode: |Z| y la fase, con el codo marcado

**Qué hay.** |Z| contra la frecuencia (continua, eje izquierdo, log) y la
fase (discontinua, eje derecho, lineal). La línea de puntos horizontal es
−45°, y el ● de cada color es donde **esa** curva la cruza.

**Cómo se lee.** El codo, fase = −45°, es donde la respuesta deja de ser
resistiva y empieza a ser capacitiva. Es la definición que cita la
literatura de supercondensadores y se interpola en log f entre dos puntos
medidos, así que no es ninguno de ellos. Está marcado porque es la medida
de potencia del dispositivo: por encima de su codo, un condensador ya no
es un condensador.

**Lo que hay que mirar.** El extremo izquierdo de la fase. La azul
(800 °C) llega a −80.0°, la verde a −81.5° y **la roja se para en
−68.4°**. Un condensador está en −90°. La de 1020 °C **no ha llegado al
régimen capacitivo** dentro del barrido, así que su capacitancia es una
cota INFERIOR y hay que bajar la frecuencia para medirla. Esto es lo que
explica el único desacuerdo entre capacitancias de la tabla 7, y la
explicación no es que el ajuste falle (ver el panel f).

El codo, además, **no ordena las muestras igual que la resistencia del
poro**, y no es una contradicción: el codo es del DISPOSITIVO y lleva R_s
dentro, y la R_s más baja es la de la de 800 °C.

## (d) C′(f) y C″(f): la capacitancia sin ningún circuito

**Qué hay.** C(ω) = 1/(jωZ), partida en su parte real (continua, eje
izquierdo) e imaginaria (discontinua, eje derecho). Sale directa de la
medida, sin ajustar nada. El ■ marca el máximo de C″.

**Cómo se lee.** C′ es la capacitancia que de verdad se puede entregar a
esa frecuencia; C″ es la parte disipativa, y su máximo da la constante de
tiempo τ₀ = 1/(2πf₀), que no necesita saber la masa ni el área. A la
izquierda del todo está el número que se cita: C′ a la frecuencia más
baja medida.

**Lo que hay que mirar.**

- **Las tres C′ saturan**… salvo la roja. La azul y la verde se aplanan;
  la roja todavía sube en el último punto, que es el mismo hecho que la
  fase de −68°.
- **El máximo de C″ de la de 1020 °C no está en la figura.** Cae fuera del
  barrido, así que su τ₀ no la determina esta medida. Las otras dos:
  7.4 ms (f₀ = 21.6 Hz) y 10.8 ms (14.7 Hz).
- **La verde está por encima de las otras dos en todo el intervalo.** La
  de 900 °C almacena más a cualquier frecuencia, no solo al final.

**Y aquí está la comprobación que vale la pena.** EC-Lab calcula sus
propias columnas `Cp` y `Cs` dentro del `.mpr`. Las C′ y C_s que el
paquete saca de Z, sin mirar esas columnas, coinciden con ellas en **las
siete cifras** en los tres archivos (tabla 8). Eso valida de una vez la
columna de frecuencias, las de Re(Z) y −Im(Z), el convenio de signo y la
fórmula — que es exactamente lo que un lector de `.mpr` hecho por
ingeniería inversa necesita contrastar.

## (e) La DRT, normalizada

**Qué hay.** La distribución de constantes de tiempo γ(τ), cada curva
dividida por su propio máximo (que va en la leyenda, en ohmios). Solo se
dibuja el intervalo de τ que el barrido mide. Los ▲ del eje son los picos
que el buscador resuelve.

**Cómo se lee.** Normalizada y lineal a propósito. Los tres γ difieren en
un orden de magnitud, así que en un eje lineal compartido el más alto
aplastaría a los otros dos contra el cero; y en escala logarítmica la
dispersión de la solución no negativa —la mayoría de las casillas son
exactamente cero— se dibuja como un peine de púas que no dice nada. Lo
que el panel sirve para ver es DÓNDE está el peso.

**Lo que hay que mirar: que estos picos no son procesos.** Los tres
espectros disparan el mismo aviso, con su número: *la fase pasa 0.5–0.7
décadas cerca de −45°, que es un elemento DISTRIBUIDO y no tiene constante
de tiempo propia*. Un poro, un Warburg y un Gerischer son distribuciones
continuas; la DRT las reparte en una joroba ancha y el buscador de picos
la trocea en arcos. Los dos o tres picos de cada muestra son esa
distribución troceada, no dos o tres procesos. No se cuentan y no se
traducen a ZARC.

Lo que sí se lee es que **todo el peso está en el extremo lento** (τ > 10
ms) en las tres, y que la de 1020 °C lo tiene todavía más lejos — el aviso
de «falta frecuencia baja» salta en las tres.

## (f) El residuo del ajuste, frecuencia a frecuencia

**Qué hay.** |Z − Z_ajuste| / |Z| en por ciento, contra la frecuencia, con
el circuito usado y el residuo medio en la leyenda.

**Cómo se lee.** Es el equivalente de la curva diferencia de un Rietveld:
un residuo medio bonito con una ondulación sistemática es peor ajuste que
uno feo sin estructura. Aquí lo que se busca es que el residuo esté
repartido y no concentrado en una región de frecuencia.

**Lo que hay que mirar.** Los tres están por debajo del 1 % en casi todo
el barrido (medias de 0.54, 0.19 y 0.44 %). Los dos sitios donde sube:

- **La azul en el extremo bajo** (2.0 % a 0.1 Hz) y **en el alto** (1.3 %
  a 100 kHz). En el alto, es que la de 800 °C es la única sin L en su
  circuito cuyo extremo superior se resiste un poco.
- **El resto es ruido**, sin estructura de rachas, que es lo que tiene que
  pasar.

Y la comprobación que importa de verdad no está en este panel sino en la
tabla 8: **C′ del circuito frente a C′ medida** sale a +0.2 %, +0.3 % y
+0.9 %. El modelo describe el extremo bajo del barrido en las tres. Por
eso el desacuerdo del 34 % entre la capacitancia del elemento y C′ en la
de 1020 °C **no es un fallo del ajuste**: es que C′ lleva dentro la parte
real de toda la cadena, y mientras la fase no se acerque a −90° las dos
magnitudes no son la misma.

---

# El circuito

El modelo común a las tres es **`R0-(R1|Q1)-T1`**:

```
 ──[ R0 ]──┬──[ R1 ]──┬──[T T1 ]──
           └──[/Q1/]──┘
```

- `R0` — disolución, contactos y colector.
- `R1|Q1` — el arco interfacial. **No se cierra dentro del barrido**, así
  que R1 sale del modelo y no de los datos, y eso va dicho.
- `T1` — la línea de transmisión de De Levie/Bisquert del poro
  bloqueante. `T1.Ri` es la resistencia IÓNICA dentro del poro, la que
  limita la potencia, y desplaza la vertical de baja frecuencia en
  **R_ion/3**.

Elegido por comparación, no por defecto: es el único que devuelve TODOS
sus parámetros determinados en las tres muestras. En la de 900 °C lleva
además `L0` delante, y eso **no es pulcritud**: sin la L, R0 se come la
inductancia del cableado y `T1.Ri` se desploma de 0.79 Ω (± 13 %) a
2·10⁻⁵ Ω (± 3.5·10⁷ %) mientras el residuo apenas pasa de 0.19 a 0.24 %.
La resistencia que limita la potencia deja de ser medible y nada en la
calidad del ajuste lo delata.

Una advertencia sobre `Q1.n`: vale **0.56, 0.60 y 0.76**. Por debajo de
0.8 un CPE ya no describe una doble capa distribuida, y 0.56 está en el
valor de un Warburg. Es decir: ese `R1|Q1` está absorbiendo comportamiento
distribuido del extremo alto, no midiendo un arco limpio. Los números de
R1 y Q1 se informan, y no se citan como transferencia de carga.

# Las tres muestras

| | 800 °C | 900 °C | 1020 °C |
|---|---|---|---|
| C′ entregable (0.1 Hz) | 0.212 mF | **0.324 mF** | 0.196 mF |
| R_s | **11.4 Ω** | 20.5 Ω | 25.5 Ω |
| R_ion del poro | 6.0 Ω | **0.79 Ω** | **16.0 Ω** |
| codo (fase −45°) | 62.7 Hz | 30.8 Hz | 60.6 Hz |
| fase a 0.1 Hz | −80.0° | **−81.5°** | **−68.4°** |
| n de la pared | 0.881 | **0.912** | 0.867 |

**La de 900 °C gana por todos los criterios que le importan a un
condensador**: más capacitancia (+53 % sobre la de 800, +65 % sobre la de
1020), veinte veces menos resistencia iónica que la de 1020, la pared más
cercana a ideal y la única que pasa de −81°.

**La de 1020 °C es la peor y de forma coherente**: R_ion de 16 Ω, fase que
se para en −68°, τ₀ fuera del barrido, y una R₁ de 3 kΩ que dice que no
hay reacción faradaica apreciable. Lo que hay es un electrodo al que al
ion le cuesta entrar — que es lo que hace una pirólisis más alta al cerrar
la microporosidad.

# Lo que estos tres espectros NO dicen

- **Nada por gramo.** Falta la masa.
- **Nada de área activa.** mF/cm² está sobre el área geométrica supuesta.
- **Ninguna transferencia de carga medida.** Ninguno cierra un arco.
- **Nada sobre a qué corresponde cada constante de tiempo.** Una τ es un
  número y los procesos se solapan en él. Para asignar uno hay que mover
  algo —el potencial mueve la transferencia de carga, la temperatura mueve
  lo que tenga energía de activación, la concentración mueve el
  transporte— y ver cómo se mueve el pico. Un solo espectro no lo
  sustituye.
- **Y falta frecuencia baja.** Los tres lo avisan. Bajar a 10 mHz cerraría
  la cola capacitiva, pondría el máximo de C″ de la de 1020 °C dentro del
  barrido y daría τ₀ para las tres.

# Qué formato usar para presentar: el combinado o el Nyquist cuadrado

Las dos presentaciones están en el catálogo de paneles y las dos se
usan en la literatura. Lo que las separa:

**El combinado** —Nyquist y Bode en una caja, frecuencia arriba y fase a
la derecha— es compacto: dos representaciones por muestra en un panel.
Tiene dos costes, y los dos son reales:

1. **El Nyquist no puede ir a escala uno a uno** porque la caja es
   compartida. Con los ejes a 0–1400 Ω contra 0–8000 Ω, el tramo a 45°
   de un electrodo poroso se dibuja casi vertical: quien lea de ahí un
   ángulo de depresión o un régimen de transporte está leyendo los ejes,
   no los datos. La forma ES el contenido de un diagrama de Nyquist.
2. **Las dos curvas comparten marco y nada más.** El Nyquist va contra
   Z′ en el eje de abajo y la fase contra f en el de arriba, así que un
   punto de una no tiene contrapartida en la otra — y el ojo la busca
   igualmente.

Y como es un panel por muestra, tres muestras son tres paneles entre
los que hay que ir y venir para compararlas.

**Lo que se usa aquí por defecto** es el Nyquist a escala uno a uno con
el rincón de alta frecuencia en un recuadro, más el Bode aparte. Es la
figura más impresa del campo de los supercondensadores, y por las dos
razones que el combinado pierde: conserva la forma y superpone las tres
muestras, que es para lo que se midieron tres.

**Elige el que quieras**: las dos están a una casilla de distancia en la
pestaña Figura, y lo que se exporta son los mismos números.

# Los archivos

| archivo | qué lleva |
|---|---|
| `peis_serie.png` / `.pdf` | la figura de seis paneles |
| `tablas.md` | todas las tablas, con sus avisos |
| `comparacion.csv` | una fila por muestra, lo esencial |
| `ajustes.csv` | cada parámetro con su incertidumbre, por muestra y circuito |
| `capacitancias.csv` | C′, C_s, la del elemento, la del arco por los dos métodos |
| `espectro_*.csv` | medida, ajuste, residuo, C′ y C″ frecuencia a frecuencia |
| `drt_*.csv` | γ(τ), marcando lo que cae fuera del intervalo medido |
| `informe_*.txt` | el informe completo de cada muestra, avisos incluidos |
| `resumen.json` | todo lo anterior en un solo sitio, para rehacer la figura |
